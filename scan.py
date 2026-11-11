#!/usr/bin/env python3
"""
Bebas - Bursa Malaysia candlestick scanner with a built-in backtest.

Runs once a day after market close (GitHub Actions does this for you):
  1. Downloads ~3 years of daily prices for every stock in config.json
  2. Finds candlestick patterns, but only when the chart context makes sense
     (uptrend pullback or oversold at support)
  3. Backtests every pattern on the same stocks, with costs, so you can see
     which patterns actually made money
  4. Builds a trade plan (buy-stop trigger, stop-loss, target, position size)
     and, if you list call warrants, a matching warrant plan
  5. Writes signals.json for the web app and sends a Telegram alert

Usage:
  python scan.py            # normal run
  python scan.py --no-send  # build signals.json without sending Telegram
"""
from __future__ import annotations

import datetime as dt
import html
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
S = CFG["strategy"]
A = CFG["alerts"]
ACC = CFG["account"]
MYT = dt.timezone(dt.timedelta(hours=8))

# --------------------------------------------------------------------------
# Bursa price ticks
# --------------------------------------------------------------------------

def tick_size(p: float) -> float:
    if p < 1.0:
        return 0.005
    if p < 10.0:
        return 0.01
    if p < 100.0:
        return 0.02
    return 0.10


def round_tick(p: float, mode: str = "near") -> float:
    t = tick_size(p)
    q = p / t
    q = math.ceil(q - 1e-9) if mode == "up" else math.floor(q + 1e-9) if mode == "down" else round(q)
    return round(q * t, 3)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna()
    df = df[(df["high"] > 0) & (df["close"] > 0)]
    # Yahoo sometimes returns non-trading days as flat zero-volume bars
    df = df[~((df["volume"] <= 0) & (df["high"] == df["low"]))]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def fetch_data(tickers: list[str], period: str, min_len: int = 120) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), 20):
        chunk = tickers[i : i + 20]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, period=period, interval="1d", group_by="ticker",
                                  auto_adjust=True, threads=True, progress=False)
                break
            except Exception as e:  # network hiccup / rate limit
                print(f"download retry {attempt + 1}: {e}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        else:
            continue
        for t in chunk:
            try:
                sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = _clean(sub)
                if len(df) >= min_len:
                    out[t] = df
            except Exception:
                pass
        time.sleep(1)
    return out


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    df["ema20"], df["ema50"], df["ema200"] = ema(c, 20), ema(c, 50), ema(c, 200)
    df["rsi"] = rsi(c)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["vol20"] = v.rolling(20).mean()
    df["value20"] = (c * v).rolling(20).mean()
    df["low60"] = l.rolling(60).min()
    df["high60_prev"] = h.shift(1).rolling(60).max()
    df["body"] = (c - o).abs()
    df["rng"] = h - l
    df["upper"] = h - np.maximum(o, c)
    df["lower"] = np.minimum(o, c) - l
    df["avg_body"] = df["body"].rolling(14).mean()
    df["bull"] = c > o
    df["bear"] = c < o
    df["vr"] = v / df["vol20"]
    df["close_pos"] = (c - l) / df["rng"].replace(0, np.nan)
    df["high20_prev"] = h.shift(1).rolling(20).max()
    df["high30_prev"] = h.shift(1).rolling(30).max()
    df["low30_prev"] = l.shift(1).rolling(30).min()
    df["high250"] = h.rolling(250, min_periods=120).max()
    df["low250"] = l.rolling(250, min_periods=120).min()
    df["obv"] = (np.sign(c.diff()).fillna(0) * v).cumsum()
    return df


# --------------------------------------------------------------------------
# Candlestick patterns   (each returns a boolean Series, evaluated on the last bar)
# --------------------------------------------------------------------------

def _p(df, k, n=1):
    return df[k].shift(n)


def bull_engulfing(d):
    return (_p(d, "bear") == True) & d["bull"] & (d["open"] <= _p(d, "close")) & \
        (d["close"] >= _p(d, "open")) & (d["body"] > _p(d, "body")) & (d["body"] > 0.5 * d["avg_body"])


def hammer(d):
    return (d["rng"] >= 0.6 * d["atr"]) & (d["lower"] >= 2 * d["body"]) & (d["lower"] >= 0.55 * d["rng"]) & \
        (d["upper"] <= 0.15 * d["rng"]) & (d["close"] < d["close"].shift(5))


def morning_star(d):
    return (_p(d, "bear", 2) == True) & (_p(d, "body", 2) > d["avg_body"]) & \
        (_p(d, "body", 1) < 0.5 * d["avg_body"]) & d["bull"] & \
        (d["close"] > (_p(d, "open", 2) + _p(d, "close", 2)) / 2)


def piercing(d):
    mid = (_p(d, "open") + _p(d, "close")) / 2
    return (_p(d, "bear") == True) & (_p(d, "body") > d["avg_body"]) & d["bull"] & \
        (d["open"] <= _p(d, "close")) & (d["close"] > mid) & (d["close"] < _p(d, "open"))


def three_white_soldiers(d):
    ok = d["bull"] & (d["body"] > 0.6 * d["avg_body"]) & (d["upper"] < 0.35 * d["body"])
    rising = (d["close"] > _p(d, "close")) & (d["open"] >= _p(d, "open"))
    return ok & ok.shift(1).fillna(False).astype(bool) & ok.shift(2).fillna(False).astype(bool) & \
        rising & rising.shift(1).fillna(False).astype(bool)


def bull_harami(d):
    return (_p(d, "bear") == True) & (_p(d, "body") > 1.2 * d["avg_body"]) & d["bull"] & \
        (d["open"] >= _p(d, "close")) & (d["close"] <= _p(d, "open")) & (d["body"] < 0.6 * _p(d, "body"))


def bear_engulfing(d):
    return (_p(d, "bull") == True) & d["bear"] & (d["open"] >= _p(d, "close")) & \
        (d["close"] <= _p(d, "open")) & (d["body"] > _p(d, "body")) & (d["body"] > 0.5 * d["avg_body"])


def shooting_star(d):
    return (d["rng"] >= 0.6 * d["atr"]) & (d["upper"] >= 2 * d["body"]) & (d["upper"] >= 0.55 * d["rng"]) & \
        (d["lower"] <= 0.15 * d["rng"]) & (d["close"] > d["close"].shift(5))


def evening_star(d):
    return (_p(d, "bull", 2) == True) & (_p(d, "body", 2) > d["avg_body"]) & \
        (_p(d, "body", 1) < 0.5 * d["avg_body"]) & d["bear"] & \
        (d["close"] < (_p(d, "open", 2) + _p(d, "close", 2)) / 2)


def dark_cloud(d):
    mid = (_p(d, "open") + _p(d, "close")) / 2
    return (_p(d, "bull") == True) & (_p(d, "body") > d["avg_body"]) & d["bear"] & \
        (d["open"] >= _p(d, "close")) & (d["close"] < mid) & (d["close"] > _p(d, "open"))


# --- Price-action strategies (carry their own context, no candlestick needed) ---

def _uptrend(d):
    return (d["close"] > d["ema50"]) & (d["ema50"] > d["ema50"].shift(10))


def breakout_20(d):
    """Close above the last 20 days' high on strong volume, closing near the day's high."""
    return (d["close"] > d["high20_prev"]) & (d["vr"] >= 1.5) & (d["close_pos"] >= 0.6) & \
        _uptrend(d) & (d["close"] > d["ema200"])


def base_breakout(d):
    """Breaks out of a tight 30-day base (range <= 15%) while volatility was contracting."""
    tight = (d["high30_prev"] - d["low30_prev"]) / d["close"] <= 0.15
    contracting = d["atr"].shift(1) < d["atr"].shift(31) * 0.9
    return (d["close"] > d["high30_prev"]) & tight & contracting & (d["vr"] >= 1.5) & \
        (d["close_pos"] >= 0.5) & (d["close"] > d["ema50"])


def swing_pullback(d):
    """Uptrend, 3-4 day dip to the 20-day EMA, then a day that closes above the prior high."""
    up = _uptrend(d) & (d["ema20"] > d["ema50"])
    dipped = (d["close"].shift(1) < d["close"].shift(4)) & (d["low"].rolling(4).min() <= d["ema20"] * 1.02)
    reversal = (d["close"] > d["high"].shift(1)) & d["bull"]
    return up & dipped & reversal & d["rsi"].between(40, 65)


def volume_surge(d):
    """Institutional-style buying: 2.5x volume, +3% day, closing in the top third."""
    return (d["vr"] >= 2.5) & (d["close"] >= d["close"].shift(1) * 1.03) & (d["close_pos"] >= 0.66) & \
        (d["close"] > d["ema50"])


def pocket_pivot(d):
    """Up day whose volume beats every down day of the last 10, near the 20-day EMA in an uptrend."""
    down_vol = d["volume"].where(d["close"] < d["close"].shift(1))
    max_down = down_vol.shift(1).rolling(10, min_periods=1).max()
    return (d["close"] > d["close"].shift(1)) & (d["volume"] > max_down) & (d["close"] > d["ema20"]) & \
        (d["low"] <= d["ema20"] * 1.03) & _uptrend(d) & d["bull"]


PATTERNS = {
    # key: (label, direction, bars used for the stop, detector, family)
    "bull_engulfing": ("Bullish Engulfing", "bull", 2, bull_engulfing, "Candlestick"),
    "hammer": ("Hammer", "bull", 1, hammer, "Candlestick"),
    "morning_star": ("Morning Star", "bull", 3, morning_star, "Candlestick"),
    "piercing": ("Piercing Line", "bull", 2, piercing, "Candlestick"),
    "three_white_soldiers": ("Three White Soldiers", "bull", 3, three_white_soldiers, "Candlestick"),
    "bull_harami": ("Bullish Harami", "bull", 2, bull_harami, "Candlestick"),
    "breakout_20": ("20-day Volume Breakout", "bull", 2, breakout_20, "Breakout"),
    "base_breakout": ("Tight Base Breakout", "bull", 2, base_breakout, "Breakout"),
    "swing_pullback": ("Swing Pullback Reversal", "bull", 4, swing_pullback, "Swing"),
    "volume_surge": ("Volume Surge", "bull", 1, volume_surge, "Volume"),
    "pocket_pivot": ("Pocket Pivot", "bull", 3, pocket_pivot, "Volume"),
    "bear_engulfing": ("Bearish Engulfing", "bear", 2, bear_engulfing, "Candlestick"),
    "shooting_star": ("Shooting Star", "bear", 1, shooting_star, "Candlestick"),
    "evening_star": ("Evening Star", "bear", 3, evening_star, "Candlestick"),
    "dark_cloud": ("Dark Cloud Cover", "bear", 2, dark_cloud, "Candlestick"),
}

FAMILY_REASON = {
    "Breakout": "Breaking out to a new high on heavy volume",
    "Swing": "Uptrend swing: dipped to the 20-day EMA and turned back up",
    "Volume": "Heavy buying volume in an uptrend",
}


def volume_profile(d: pd.DataFrame, i: int) -> dict:
    lo = max(0, i - 19)
    chg = d["close"].diff().iloc[lo: i + 1]
    vol = d["volume"].iloc[lo: i + 1]
    up, down = float(vol[chg > 0].sum()), float(vol[chg < 0].sum())
    updown = up / down if down > 0 else 3.0
    obv = d["obv"]
    obv_up = bool(obv.iloc[i] > obv.iloc[max(0, i - 20)])
    label = "Accumulation" if updown >= 1.2 and obv_up else "Distribution" if updown <= 0.8 and not obv_up else "Neutral"
    return {"ratio": round(float(d["vr"].iloc[i]), 2), "updown_20d": round(updown, 2),
            "obv_trend": "rising" if obv_up else "falling", "label": label}


def setups(d: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Context filters. A pattern only counts if it appears in one of these places."""
    uptrend = (d["close"] > d["ema50"]) & (d["ema20"] > d["ema50"]) & (d["ema50"] > d["ema50"].shift(10))
    pullback = d["low"].rolling(3).min() <= d["ema20"] * 1.01
    trend_pullback = uptrend & pullback & d["rsi"].between(35, 62)
    oversold = (d["rsi"].rolling(3).min() < 35) & (d["low"] <= d["low60"] + 0.5 * d["atr"])
    liquid = (d["value20"] >= S["min_avg_value_rm"]) & (d["close"] >= S["min_price"])
    return trend_pullback.fillna(False), oversold.fillna(False), liquid.fillna(False)


# --------------------------------------------------------------------------
# Trade plan + simulation
# --------------------------------------------------------------------------

def plan_at(d: pd.DataFrame, i: int, nbars: int) -> dict | None:
    row = d.iloc[i]
    trigger = round_tick(row["high"] + tick_size(row["high"]), "up")
    pat_low = d["low"].iloc[max(0, i - nbars + 1): i + 1].min()
    stop = round_tick(pat_low - S["stop_buffer_atr"] * row["atr"], "down")
    risk = trigger - stop
    if stop <= 0 or risk <= 0 or risk / trigger * 100 > S["max_stop_pct"]:
        return None
    target = round_tick(trigger + S["reward_r"] * risk, "up")
    return {"trigger": trigger, "stop": stop, "target": target}


def simulate(d: pd.DataFrame, i: int, plan: dict) -> dict:
    """Buy-stop entry within the entry window, then stop / target / time exit.
    If stop and target are hit on the same day we assume the stop (conservative)."""
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    n = len(d)
    trig, stop, tgt = plan["trigger"], plan["stop"], plan["target"]
    entry_i = None
    for j in range(i + 1, min(i + 1 + S["entry_window_days"], n)):
        if h[j] >= trig:
            entry_i, entry = j, max(o[j], trig)
            break
        if c[j] < stop:
            return {"status": "cancelled"}
    if entry_i is None:
        return {"status": "pending" if i + S["entry_window_days"] >= n else "not_triggered"}
    risk = entry - stop
    if risk <= 0:
        return {"status": "cancelled"}
    exit_px, exit_i, how = None, None, None
    if l[entry_i] <= stop:
        exit_px, exit_i, how = stop, entry_i, "stop"
    else:
        for k in range(entry_i + 1, min(entry_i + 1 + S["max_hold_days"], n)):
            if o[k] <= stop:
                exit_px, exit_i, how = o[k], k, "stop"
            elif l[k] <= stop:
                exit_px, exit_i, how = stop, k, "stop"
            elif o[k] >= tgt:
                exit_px, exit_i, how = o[k], k, "target"
            elif h[k] >= tgt:
                exit_px, exit_i, how = tgt, k, "target"
            if exit_px is not None:
                break
        if exit_px is None:
            if entry_i + S["max_hold_days"] < n:
                exit_i = entry_i + S["max_hold_days"]
                exit_px, how = c[exit_i], "time"
            else:
                return {"status": "open", "entry": float(entry), "entry_i": entry_i}
    cost = S["cost_pct_round_trip"] / 100 * entry
    r = (exit_px - entry - cost) / risk
    return {"status": "closed", "r": float(r), "how": how, "hold": int(exit_i - entry_i),
            "entry_date": d.index[entry_i].strftime("%Y-%m-%d"), "entry": float(entry)}


# --------------------------------------------------------------------------
# Warrants (Black-Scholes, for call warrants)
# --------------------------------------------------------------------------

def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(S0, K, T, sigma, r=0.03):
    if T <= 0 or sigma <= 0:
        return max(S0 - K, 0.0), (1.0 if S0 > K else 0.0)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2), _ncdf(d1)


def implied_vol(price_per_share, S0, K, T):
    lo, hi = 0.01, 3.0
    if bs_call(S0, K, T, hi)[0] < price_per_share:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_call(S0, K, T, mid)[0] > price_per_share:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def warrant_plan(w: dict, stock_df: pd.DataFrame, plan: dict | None, wprice: float | None) -> dict:
    S0 = float(stock_df["close"].iloc[-1])
    K, ratio = float(w["exercise"]), float(w["ratio"])
    expiry = dt.date.fromisoformat(w["expiry"])
    days = (expiry - dt.datetime.now(MYT).date()).days
    T = max(days, 0) / 365
    hv = float(np.log(stock_df["close"]).diff().tail(60).std() * math.sqrt(250))
    iv = implied_vol(wprice * ratio, S0, K, T) if wprice and T > 0 else None
    vol = iv or hv
    theo, delta = bs_call(S0, K, T, vol)
    W = wprice if wprice else theo / ratio
    out = {
        "code": w["code"], "exercise": K, "ratio": ratio, "expiry": w["expiry"], "days_left": days,
        "price": round(W, 3), "price_source": "market" if wprice else "theoretical",
        "vol_used": round(vol * 100, 1), "vol_type": "implied" if iv else "historical",
        "delta": round(delta, 3),
        "gearing": round(S0 / (W * ratio), 2) if W > 0 else None,
        "effective_gearing": round(delta * S0 / (W * ratio), 2) if W > 0 else None,
        "premium_pct": round(((W * ratio + K) - S0) / S0 * 100, 1),
        "breakeven": round(W * ratio + K, 3),
        "moneyness": "ITM" if S0 > K else "OTM",
        "warnings": [],
    }
    if days < 90:
        out["warnings"].append("Less than 3 months to expiry: time decay is fast")
    if out["premium_pct"] > 25:
        out["warnings"].append("High premium: stock must rise a lot just to break even")
    if W < 0.02:
        out["warnings"].append("Very low price: one tick is a big % move, spreads hurt")
    if plan:
        t_exit = max(T - 10 / 365, 0)  # assume ~2 weeks to reach target/stop
        for k in ("stop", "target"):
            out[f"at_stock_{k}"] = round(bs_call(plan[k], K, t_exit, vol)[0] / ratio, 3)
    return out


# --------------------------------------------------------------------------
# Main scan
# --------------------------------------------------------------------------

def position_size(entry: float, stop: float, capital: float, risk_pct: float, max_pos_pct: float) -> dict:
    risk_rm = capital * risk_pct / 100
    per_share = entry - stop
    shares = int(risk_rm / per_share // 100 * 100) if per_share > 0 else 0
    cap_shares = int(capital * max_pos_pct / 100 / entry // 100 * 100)
    capped = shares > cap_shares
    shares = min(shares, cap_shares)
    return {"shares": shares, "lots": shares // 100, "cost_rm": round(shares * entry, 2),
            "risk_rm": round(shares * per_share, 2), "capped": capped}


def stats_from(rs: list[dict]) -> dict:
    if not rs:
        return {"n": 0}
    r = np.array([t["r"] for t in rs])
    wins, losses = r[r > 0], r[r <= 0]
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    return {
        "n": int(len(r)),
        "win": round(float((r > 0).mean() * 100), 1),
        "avg_r": round(float(r.mean()), 3),
        "pf": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
        "avg_win_r": round(float(wins.mean()), 2) if len(wins) else 0,
        "avg_loss_r": round(float(losses.mean()), 2) if len(losses) else 0,
        "max_losing_streak": int(best),
        "avg_hold": round(float(np.mean([t["hold"] for t in rs])), 1),
        "targets_hit": round(float(np.mean([t["how"] == "target" for t in rs]) * 100), 1),
    }


def curve(rs: list[dict], points: int = 80) -> list[float]:
    rs = sorted(rs, key=lambda t: t["entry_date"])
    cum = np.cumsum([t["r"] for t in rs])
    if len(cum) > points:
        idx = np.linspace(0, len(cum) - 1, points).astype(int)
        cum = cum[idx]
    return [round(float(x), 2) for x in cum]


def candles(d: pd.DataFrame, n: int = 70) -> list:
    t = d.tail(n)
    return [[ix.strftime("%Y-%m-%d"), round(r.open, 3), round(r.high, 3), round(r.low, 3),
             round(r.close, 3), int(r.volume)] for ix, r in t.iterrows()]


def market_overview(rows: list[dict], index_df: pd.DataFrame | None) -> dict:
    """Breadth, sector rotation and a 0-100 sentiment gauge built from the scanned stocks."""
    n = max(len(rows), 1)
    adv = sum(r["r1"] > 0 for r in rows)
    dec = sum(r["r1"] < 0 for r in rows)
    pct = lambda k: round(sum(r[k] for r in rows) / n * 100, 1)
    highs, lows = sum(r["new_high"] for r in rows), sum(r["new_low"] for r in rows)
    out = {"advancers": adv, "decliners": dec, "unchanged": len(rows) - adv - dec,
           "pct_above_ema20": pct("a20"), "pct_above_ema50": pct("a50"), "pct_above_ema200": pct("a200"),
           "new_52w_highs": highs, "new_52w_lows": lows}
    sect: dict[str, list] = {}
    for r in rows:
        sect.setdefault(r["sector"], []).append(r)
    out["sectors"] = sorted([{
        "sector": s, "n": len(v),
        "r1": round(float(np.mean([x["r1"] for x in v])), 2),
        "r5": round(float(np.mean([x["r5"] for x in v])), 2),
        "r20": round(float(np.mean([x["r20"] for x in v])), 2),
        "pct_above_ema50": round(sum(x["a50"] for x in v) / len(v) * 100)} for s, v in sect.items()],
        key=lambda x: x["r20"], reverse=True)
    brief = lambda r: {k: r[k] for k in ("ticker", "code", "name", "close", "r1", "vr")}
    out["gainers"] = [brief(r) for r in sorted(rows, key=lambda r: r["r1"], reverse=True)[:5] if r["r1"] > 0]
    out["losers"] = [brief(r) for r in sorted(rows, key=lambda r: r["r1"])[:5] if r["r1"] < 0]
    out["unusual_volume"] = [brief(r) for r in sorted(rows, key=lambda r: r["vr"], reverse=True) if r["vr"] >= 2.5][:8]

    trend = 50.0
    if index_df is not None and len(index_df) > 60:
        ic = index_df["close"]
        e50, e200 = ema(ic, 50), ema(ic, 200)
        chg = lambda k: round(float((ic.iloc[-1] / ic.iloc[-1 - k] - 1) * 100), 2) if len(ic) > k else None
        out["index"] = {"close": round(float(ic.iloc[-1]), 2), "chg_1d": chg(1), "chg_5d": chg(5), "chg_20d": chg(20),
                        "above_ema50": bool(ic.iloc[-1] > e50.iloc[-1]),
                        "above_ema200": bool(ic.iloc[-1] > e200.iloc[-1]) if len(ic) >= 200 else None,
                        "rsi": round(float(rsi(ic).iloc[-1]), 1),
                        "closes": [round(float(x), 2) for x in ic.tail(120)]}
        trend = 30 * out["index"]["above_ema50"] + 30 * bool(out["index"]["above_ema200"]) + \
            40 * min(max(((out["index"]["chg_20d"] or 0) + 5) / 10, 0), 1)
    hl = highs / (highs + lows) * 100 if highs + lows else 50
    ad = adv / (adv + dec) * 100 if adv + dec else 50
    mom = min(max(50 + float(np.mean([r["r20"] for r in rows])) * 5, 0), 100) if rows else 50
    parts = {"Stocks above 50-day EMA": out["pct_above_ema50"], "Stocks above 200-day EMA": out["pct_above_ema200"],
             "52-week highs vs lows": round(hl), "Advancers vs decliners": round(ad),
             "KLCI trend": round(trend), "20-day momentum": round(mom)}
    w = [0.25, 0.15, 0.15, 0.15, 0.20, 0.10]
    score = int(round(sum(v * k for v, k in zip(parts.values(), w))))
    label = ("Extreme fear" if score < 20 else "Fear" if score < 40 else "Neutral" if score < 60
             else "Greed" if score < 80 else "Extreme greed")
    out["sentiment"] = {"score": score, "label": label, "components": parts}
    lead = out["sectors"][0]["sector"] if out["sectors"] else "-"
    lag = out["sectors"][-1]["sector"] if out["sectors"] else "-"
    out["summary"] = (f"{label} ({score}/100). {adv} up, {dec} down today; {out['pct_above_ema50']}% of stocks are "
                      f"above their 50-day EMA. {highs} new 52-week highs vs {lows} lows. "
                      f"Strongest sector over 20 days: {lead}. Weakest: {lag}.")
    return out


def run(data: dict[str, pd.DataFrame], index_df: pd.DataFrame | None = None) -> dict:
    names = CFG["universe"]
    sectors = CFG.get("sectors", {})
    bull_keys = [k for k, v in PATTERNS.items() if v[1] == "bull"]
    bear_keys = [k for k, v in PATTERNS.items() if v[1] == "bear"]
    trades: dict[str, list] = {k: [] for k in bull_keys}
    trades_by_setup: dict[str, list] = {}
    trades_by_stock: dict[tuple, list] = {}
    bear_fwd: dict[str, list] = {k: [] for k in bear_keys}
    candidates, warnings, exits, rows = [], [], [], []
    holdings = {h["ticker"]: h for h in CFG.get("holdings", [])}
    warrants_by_mother: dict[str, list] = {}
    for w in CFG.get("warrants", []):
        warrants_by_mother.setdefault(w["mother"], []).append(w)

    last_date = max(df.index[-1] for df in data.values())

    for t, raw in data.items():
        d = add_indicators(raw)
        tp, os_, liquid = setups(d)
        n = len(d)
        stale = d.index[-1] < last_date
        c = d["close"]
        last = d.iloc[-1]
        ret = lambda k: round(float((c.iloc[-1] / c.iloc[-1 - k] - 1) * 100), 2) if n > k else 0.0
        if not stale and t in names:
            rows.append({"ticker": t, "code": t.replace(".KL", ""), "name": names.get(t, t),
                         "sector": sectors.get(t, "Other"), "close": round(float(last["close"]), 3),
                         "r1": ret(1), "r5": ret(5), "r20": ret(20), "r250": ret(250) if n > 250 else None,
                         "a20": bool(last["close"] > last["ema20"]), "a50": bool(last["close"] > last["ema50"]),
                         "a200": bool(last["close"] > last["ema200"]), "rsi": round(float(last["rsi"]), 1),
                         "vr": round(float(last["vr"]), 2) if not np.isnan(last["vr"]) else 0.0,
                         "value20_m": round(float(last["value20"]) / 1e6, 2),
                         "new_high": bool(last["high"] >= last["high250"]), "new_low": bool(last["low"] <= last["low250"]),
                         "pct_from_high": round(float((last["close"] / last["high250"] - 1) * 100), 1),
                         "closes": [round(float(x), 3) for x in c.tail(120)],
                         "volume": volume_profile(d, n - 1)})

        for key in bull_keys:
            label, _, nbars, fn, fam = PATTERNS[key]
            ctx = (tp | os_) if fam == "Candlestick" else True
            hit = fn(d).fillna(False).astype(bool) & ctx & liquid
            hit.iloc[:200] = False  # need EMA200 warm-up
            for i in np.flatnonzero(hit.values):
                plan = plan_at(d, i, nbars)
                if not plan:
                    continue
                if fam == "Candlestick":
                    setup = "Trend pullback" if tp.iloc[i] else "Oversold bounce"
                else:
                    setup = fam
                res = simulate(d, i, plan)
                if res["status"] == "closed":
                    trades[key].append(res)
                    trades_by_setup.setdefault(setup, []).append(res)
                    trades_by_stock.setdefault((t, key), []).append(res)
                # live signals: today's bar, or yesterday's bar still inside the entry window
                if i >= n - S["entry_window_days"] and not stale:
                    status = {"pending": "new" if i == n - 1 else "pending",
                              "open": "triggered", "closed": "closed"}.get(res["status"], res["status"])
                    if status in ("new", "pending", "triggered"):
                        candidates.append((t, key, i, plan, setup, status, d))

        # bearish patterns: measure what usually happens next (10-day return)
        for key in bear_keys:
            fn = PATTERNS[key][3]
            up = (d["close"] > d["ema50"]) & liquid
            hit = fn(d).fillna(False).astype(bool) & up
            hit.iloc[:200] = False
            fwd = d["close"].shift(-10) / d["close"] - 1
            for i in np.flatnonzero(hit.values):
                if not np.isnan(fwd.iloc[i]):
                    bear_fwd[key].append(float(fwd.iloc[i]))
            if not stale and hit.iloc[-1]:
                warnings.append({"ticker": t, "code": t.replace(".KL", ""), "name": names.get(t, t),
                                 "pattern": PATTERNS[key][0], "close": round(float(d["close"].iloc[-1]), 3),
                                 "held": t in holdings})

        # holdings: exit management
        if t in holdings:
            h = holdings[t]
            row = d.iloc[-1]
            chandelier = round_tick(d["high"].tail(22).max() - 3 * row["atr"], "down")
            entry, stop = float(h["entry"]), float(h["stop"])
            r_now = (row["close"] - entry) / (entry - stop) if entry > stop else 0
            vp = volume_profile(d, n - 1)
            msg, action = None, "hold"
            if row["close"] <= stop:
                action, msg = "sell", f"Closed at {row['close']:.3f}, at or below your stop {stop:.3f}. Exit tomorrow."
            elif h.get("target") and row["close"] >= float(h["target"]):
                action, msg = "take_profit", f"Target {float(h['target']):.3f} reached. Take profit or trail the stop."
            elif any(w["ticker"] == t for w in warnings):
                action, msg = "caution", "Bearish candle today. Consider taking partial profit or tightening the stop."
            elif vp["label"] == "Distribution" and row["close"] < row["ema20"]:
                action, msg = "caution", "Heavy selling volume and price below the 20-day EMA. Tighten your stop."
            elif chandelier > stop and r_now > 1:
                action, msg = "raise_stop", f"Up {r_now:.1f}R. Raise stop to {chandelier:.3f} (3-ATR trailing stop)."
            exits.append({"ticker": t, "code": t.replace(".KL", ""), "name": names.get(t, t),
                          "close": round(float(row["close"]), 3), "entry": entry, "stop": round(stop, 3),
                          "r_now": round(float(r_now), 2), "trail_stop": chandelier, "action": action,
                          "message": msg, "shares": h.get("shares"), "volume": vp})

    by_pattern = []
    pstats = {}
    for key in bull_keys:
        st = stats_from(trades[key])
        st.update({"key": key, "label": PATTERNS[key][0], "direction": "bull", "family": PATTERNS[key][4],
                   "curve": curve(trades[key])})
        st["edge"] = bool(st["n"] >= A["min_trades_for_edge"] and st.get("avg_r", -9) >= A["min_expectancy_r"])
        pstats[key] = st
        by_pattern.append(st)
    for key in bear_keys:
        f = np.array(bear_fwd[key])
        by_pattern.append({"key": key, "label": PATTERNS[key][0], "direction": "bear", "family": "Candlestick",
                           "n": int(len(f)),
                           "pct_down_10d": round(float((f < 0).mean() * 100), 1) if len(f) else None,
                           "avg_ret_10d": round(float(f.mean() * 100), 2) if len(f) else None})
    all_trades = [x for v in trades.values() for x in v]
    overall = stats_from(all_trades)
    overall["curve"] = curve(all_trades)
    by_setup = []
    for k in ["Trend pullback", "Oversold bounce", "Breakout", "Swing", "Volume"]:
        st = stats_from(trades_by_setup.get(k, []))
        st["label"] = k if k in ("Breakout", "Swing", "Volume") else f"Candlestick · {k}"
        by_setup.append(st)

    market = market_overview(rows, index_df)
    breadth = market["pct_above_ema50"]
    regime = {"breadth_above_ema50": breadth}
    if "index" in market:
        ix = market["index"]
        regime.update({"index_close": ix["close"], "index_above_ema50": ix["above_ema50"],
                       "index_chg_20d_pct": ix["chg_20d"]})
    risk_on = breadth >= 45 and regime.get("index_above_ema50", True)
    regime["state"] = "healthy" if risk_on else "weak"
    regime["advice"] = ("Market healthy: normal position size." if risk_on else
                        "Market weak: most breakouts fail in weak markets. Halve your risk or wait.")

    raw_sigs = []
    for t, key, i, plan, setup, status, d in candidates:
        row = d.iloc[i]
        ps = pstats[key]
        fam = PATTERNS[key][4]
        hist = stats_from(trades_by_stock.get((t, key), []))
        risk = plan["trigger"] - plan["stop"]
        vp = volume_profile(d, i)
        room = (row["high60_prev"] - plan["trigger"]) / risk if risk > 0 else 0
        reasons, score = [], 0.0
        score += 35 * min(max(ps.get("avg_r", 0) / 0.5, 0), 1)
        score += 10 * min(max((ps.get("win", 0) - 35) / 25, 0), 1)
        if vp["ratio"] >= 1.5:
            score += 10
            reasons.append(f"Volume {vp['ratio']:.1f}x the 20-day average")
        elif vp["ratio"] >= 1.2:
            score += 6
            reasons.append(f"Volume {vp['ratio']:.1f}x the 20-day average")
        if vp["label"] == "Accumulation":
            score += 5
            reasons.append(f"Accumulation: up-day volume {vp['updown_20d']}x down-day volume, OBV rising")
        elif vp["label"] == "Distribution":
            reasons.append("Warning: more volume on down days than up days lately (distribution)")
        if row["close"] > row["ema200"]:
            score += 15
            reasons.append("Above 200-day EMA (long-term uptrend)")
        if np.isnan(room) or room >= S["reward_r"] or row["high"] >= row["high60_prev"]:
            score += 15
            reasons.append("No nearby resistance before the target")
        else:
            reasons.append(f"Resistance at {row['high60_prev']:.3f} may cap the move")
        if setup == "Trend pullback":
            reasons.insert(0, "Uptrend, pulled back to the 20-day EMA")
        elif setup == "Oversold bounce":
            reasons.insert(0, "Oversold (RSI below 35) near 60-day low")
        else:
            reasons.insert(0, FAMILY_REASON[fam])
        if not risk_on:
            score -= 10
        raw_sigs.append({
            "ticker": t, "code": t.replace(".KL", ""), "name": names.get(t, t),
            "sector": sectors.get(t, "Other"),
            "date": d.index[i].strftime("%Y-%m-%d"), "status": status,
            "pattern": key, "pattern_label": PATTERNS[key][0], "family": fam, "setup": setup,
            "close": round(float(d["close"].iloc[-1]), 3), **plan,
            "risk_pct": round(risk / plan["trigger"] * 100, 2), "reward_r": S["reward_r"],
            "rsi": round(float(row["rsi"]), 1), "volume_ratio": vp["ratio"], "volume": vp,
            "ema20": round(float(row["ema20"]), 3), "ema50": round(float(row["ema50"]), 3),
            "ema200": round(float(row["ema200"]), 3), "atr": round(float(row["atr"]), 3),
            "_score": score, "edge_ok": ps["edge"],
            "pattern_stats": {k: ps.get(k) for k in ("n", "win", "avg_r", "pf")},
            "stock_stats": {k: hist.get(k) for k in ("n", "win", "avg_r")},
            "reasons": reasons, "candles": candles(d), "warrants": [],
        })

    # one card per stock: keep the strongest setup, count the others as confirmation
    best: dict[str, dict] = {}
    for s in sorted(raw_sigs, key=lambda s: (s["edge_ok"], s["_score"]), reverse=True):
        if s["ticker"] in best:
            b = best[s["ticker"]]
            if s["pattern_label"] not in b["confirmations"]:
                b["confirmations"].append(s["pattern_label"])
        else:
            s["confirmations"] = []
            best[s["ticker"]] = s
    signals = []
    for s in best.values():
        score = s.pop("_score")
        if s["confirmations"]:
            score += 10
            s["reasons"].insert(1, "Also confirmed by: " + ", ".join(s["confirmations"]))
        score = int(round(max(0, min(100, score))))
        s["score"] = score
        s["grade"] = "A" if score >= 75 else "B" if score >= 60 else "C"
        s["alert"] = bool(s["edge_ok"] and score >= A["min_score"])
        s["position"] = position_size(s["trigger"], s["stop"], ACC["capital"],
                                      ACC["risk_pct"] * (1 if risk_on else 0.5), ACC["max_position_pct"])
        for w in warrants_by_mother.get(s["ticker"], []):
            try:
                plan = {k: s[k] for k in ("trigger", "stop", "target")}
                s["warrants"].append(warrant_plan(w, add_indicators(data[s["ticker"]]), plan, w.get("_price")))
            except Exception as e:
                print(f"warrant {w.get('code')}: {e}", file=sys.stderr)
        signals.append(s)
    signals.sort(key=lambda s: (s["alert"], s["score"]), reverse=True)

    watch_warrants = []
    for mother, ws in warrants_by_mother.items():
        if mother in data:
            for w in ws:
                try:
                    watch_warrants.append({"mother": mother, "mother_name": names.get(mother, mother),
                                           **warrant_plan(w, add_indicators(data[mother]), None, w.get("_price"))})
                except Exception as e:
                    print(f"warrant {w.get('code')}: {e}", file=sys.stderr)

    return {
        "generated_at": dt.datetime.now(MYT).strftime("%Y-%m-%d %H:%M MYT"),
        "data_date": last_date.strftime("%Y-%m-%d"),
        "universe_size": len(CFG["universe"]), "scanned": len(data),
        "missing": sorted(set(CFG["universe"]) - set(data)),
        "account": ACC, "strategy": S, "alerts_cfg": A,
        "regime": regime, "market": market, "stocks": rows,
        "signals": signals, "warnings": warnings, "exits": exits,
        "warrants": watch_warrants,
        "backtest": {"overall": overall, "by_pattern": by_pattern, "by_setup": by_setup,
                     "period": S["history_period"], "stocks": len(data)},
    }


# --------------------------------------------------------------------------
# Telegram
# --------------------------------------------------------------------------

def telegram_text(res: dict, app_url: str | None) -> list[str]:
    e = html.escape
    msgs, lines = [], []
    rg, mk = res["regime"], res.get("market", {})
    lines.append(f"<b>Bebas scan · {res['data_date']}</b>")
    se = mk.get("sentiment")
    if se:
        ix = mk.get("index", {})
        lines.append(f"<b>Sentiment: {e(se['label'])} ({se['score']}/100)</b>" +
                     (f" · KLCI {ix['close']:,.2f} ({ix['chg_1d']:+.2f}%)" if ix.get("chg_1d") is not None else ""))
        lines.append(f"{mk['advancers']} up / {mk['decliners']} down · {mk['new_52w_highs']} new highs / "
                     f"{mk['new_52w_lows']} lows")
        if mk.get("sectors"):
            lines.append(f"Leading: {e(mk['sectors'][0]['sector'])} ({mk['sectors'][0]['r20']:+.1f}% 20d) · "
                         f"Lagging: {e(mk['sectors'][-1]['sector'])} ({mk['sectors'][-1]['r20']:+.1f}%)")
    lines.append(("🟢 " if rg["state"] == "healthy" else "🟠 ") + e(rg["advice"]))
    th = (res.get("themes") or {}).get("ranking", [])
    hot = [x for x in th if x["status"] in ("Hot", "Heating up")][:3]
    if hot:
        lines.append("\n<b>Where the money is going</b>")
        for x in hot:
            lines.append(f"🔥 {e(x['name'])} · {x['status']} · {x['r20']:+.1f}% 20d · vol {x['volx']}x · "
                         f"leaders {e(', '.join(x['leaders']))}")
    pbs = res.get("playbook") or []
    if pbs:
        lines.append("\n<b>Top pick per horizon</b>")
        for pb in pbs:
            if pb["picks"]:
                p = pb["picks"][0]
                tgt = f" · T {p['target']:.3f}" if p.get("target") else ""
                lines.append(f"• {e(pb['title'])}: <b>{e(p['name'])} ({p['code']})</b> · E {p['entry']:.3f} · "
                             f"S {p['stop']:.3f}{tgt}")
            else:
                lines.append(f"• {e(pb['title'])}: no qualifying stock today")
    news = (res.get("news") or {}).get("items", [])
    if news:
        lines.append("\n<b>Headlines</b>")
        for n in news[:3]:
            dot = {"positive": "▲", "negative": "▼"}.get(n["tone"], "•")
            lines.append(f'{dot} <a href="{e(n["url"] or "")}">{e(n["title"][:110])}</a> ({e(n["source"])})')
    alerts = [s for s in res["signals"] if s["alert"]]
    if alerts:
        lines.append("")
        for s in alerts[:8]:
            tag = {"new": "NEW", "pending": "STILL VALID", "triggered": "TRIGGERED TODAY"}[s["status"]]
            p = s["position"]
            ps = s["pattern_stats"]
            v = s.get("volume", {})
            block = [
                f"<b>{e(s['name'])} ({s['code']})</b> · Grade {s['grade']} ({s['score']}) · {tag}",
                f"{e(s['family'])}: {e(s['pattern_label'])}" +
                (f" + {e(', '.join(s['confirmations']))}" if s.get("confirmations") else "") +
                (f" · {e(s['setup'])}" if s["family"] == "Candlestick" else ""),
                f"Buy-stop above <b>{s['trigger']:.3f}</b> · Stop <b>{s['stop']:.3f}</b> · Target <b>{s['target']:.3f}</b>",
                f"Risk {s['risk_pct']}% · R:R 1:{s['reward_r']:g} · " + (
                    f"Size {p['lots']} lots (RM{p['cost_rm']:,.0f}, risk RM{p['risk_rm']:,.0f})" if p["lots"]
                    else "Stop too wide for 1 lot at your risk %: skip"),
                f"Volume {v.get('ratio', 0)}x avg · {e(v.get('label', ''))} · "
                f"Backtest {ps['n']} trades, {ps['win']}% win, {ps['avg_r']:+.2f}R",
            ]
            f = s.get("fundamentals")
            if f and f.get("score") is not None:
                bits = [f"Fundamentals {f['grade']} ({f['score']})"]
                if f.get("pe"):
                    bits.append(f"P/E {f['pe']:.1f}")
                if f.get("dividend_yield"):
                    bits.append(f"DY {f['dividend_yield']}%")
                if f.get("ni_yoy_ttm") is not None:
                    bits.append(f"Profit {f['ni_yoy_ttm']:+.0f}% y/y")
                block.append(" · ".join(bits))
            if s.get("earnings_soon"):
                block.append(f"⚠️ Results due {f['earnings_date']}")
            if s.get("ai_verdict"):
                block.append(f"🧠 {e(s['ai_verdict'])}")
            neg = [n for n in s.get("news", []) if n["tone"] == "negative"][:1]
            if neg:
                block.append(f'📰 ▼ <a href="{e(neg[0]["url"] or "")}">{e(neg[0]["title"][:100])}</a>')
            for w in s["warrants"][:2]:
                block.append(f"Warrant {e(w['code'])}: ~{w['price']:.3f} → stop {w.get('at_stock_stop', 0):.3f} / "
                             f"target {w.get('at_stock_target', 0):.3f} · eff. gearing {w['effective_gearing']}x · "
                             f"{w['days_left']}d left")
            lines.extend(block + [""])
    elif A.get("send_daily_summary", True):
        lines.append("\nNo setups passed the filters today. No trade is also a position.")
    for x in res["exits"]:
        if x["action"] != "hold":
            icon = {"sell": "🔴", "take_profit": "💰", "caution": "⚠️", "raise_stop": "⬆️"}[x["action"]]
            lines.append(f"{icon} <b>{e(x['name'])}</b> ({x['code']}) {e(x['message'])}")
    for t in res.get("on_demand", []):
        lines.append(f"🔎 Research on {e(t)} is ready in the app's Research tab.")
    if app_url:
        lines.append(f'\n<a href="{e(app_url)}">Open the app</a>')
    lines.append("<i>Educational tool, not financial advice.</i>")
    buf = ""
    for ln in lines:
        if len(buf) + len(ln) > 3800:
            msgs.append(buf)
            buf = ""
        buf += ln + "\n"
    if buf.strip():
        msgs.append(buf)
    if not alerts and not any(x["action"] != "hold" for x in res["exits"]) and not A.get("send_daily_summary", True) \
            and not res.get("on_demand"):
        return []
    return msgs


def send_telegram(msgs: list[str]) -> None:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID); skipping send.")
        return
    import requests
    for m in msgs:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": m, "parse_mode": "HTML",
                                "disable_web_page_preview": True}, timeout=20)
        if not r.ok:
            print("Telegram error:", r.status_code, r.text[:300], file=sys.stderr)


def _fund_brief(f: dict | None) -> dict | None:
    if not f:
        return None
    sc = f.get("score") or {}
    keep = ("pe", "sector_pe", "pb", "dividend_yield", "roe", "ni_yoy_ttm", "rev_yoy_ttm",
            "debt_to_equity", "earnings_date", "target_upside", "analysts", "market_cap")
    out = {k: f.get(k) for k in keep}
    out.update({"score": sc.get("score"), "grade": sc.get("grade"),
                "strengths": sc.get("strengths", [])[:3], "concerns": sc.get("concerns", [])[:3]})
    return out


def _norm_ticker(t: str) -> str:
    t = t.strip().upper()
    return t if "." in t or t.startswith("^") else t + ".KL"


def enrich(res: dict, data: dict[str, pd.DataFrame]) -> None:
    """Fundamentals, news and AI notes. Every step is optional and fails soft."""
    import research as R
    rcfg = CFG.get("research", {})
    names, sectors = CFG["universe"], CFG.get("sectors", {})
    holdings = [h["ticker"] for h in CFG.get("holdings", [])]
    ai_cfg = rcfg.get("ai", {})
    on_demand = [_norm_ticker(t) for t in os.environ.get("RESEARCH_TICKER", "").split(",") if t.strip().isdigit()]
    always = [_norm_ticker(t) for t in ai_cfg.get("always", [])]
    sig_t = [s["ticker"] for s in res["signals"]]
    priority = list(dict.fromkeys(on_demand + [s["ticker"] for s in res["signals"] if s["alert"]] +
                                  holdings + always + sig_t))

    fpath = ROOT / "fundamentals.json"
    try:
        fc = R.refresh_fundamentals(list(dict.fromkeys(list(names) + holdings + on_demand)), priority, fpath, rcfg)
        fc = R.score_all(fc, sectors)
        fpath.write_text(json.dumps(fc, separators=(",", ":"), default=str), encoding="utf-8")
    except Exception as e:
        print(f"fundamentals step failed: {e}", file=sys.stderr)
        fc = R.load_json(fpath, {"stocks": {}})
    fs = fc.get("stocks", {})
    min_f = A.get("min_fundamental_score", 0) or 0
    for s in res["signals"]:
        fb = _fund_brief(fs.get(s["ticker"]))
        s["fundamentals"] = fb
        if fb and fb.get("earnings_date"):
            try:
                days = (dt.date.fromisoformat(fb["earnings_date"]) - dt.datetime.now(MYT).date()).days
                if 0 <= days <= 10:
                    s["reasons"].append(f"Results due {fb['earnings_date']} ({days} days): gap risk, consider waiting")
                    s["earnings_soon"] = True
            except ValueError:
                pass
        if min_f and fb and fb.get("score") is not None and fb["score"] < min_f:
            s["alert"] = False
            s["reasons"].append(f"Not alerted: fundamental score {fb['score']} is below your minimum {min_f}")
    for x in res["exits"]:
        x["fundamentals"] = _fund_brief(fs.get(x["ticker"]))

    res["stock_news"] = {}
    if rcfg.get("news", True):
        try:
            res["news"] = R.market_news()
            res["market"]["news_tone"] = {k: res["news"][k] for k in ("tone", "positive", "negative")}
        except Exception as e:
            print(f"market news failed: {e}", file=sys.stderr)
        for t in list(dict.fromkeys(on_demand + [s["ticker"] for s in res["signals"]][:10] + holdings)):
            f = fs.get(t, {})
            res["stock_news"][t] = R.stock_news(t, names.get(t, t.replace(".KL", "")), f.get("long_name"))
        for s in res["signals"]:
            s["news"] = res["stock_news"].get(s["ticker"], [])[:4]

    # themes, money flow and the four horizon playbooks
    pick_t = []
    try:
        import playbook as PB
        PB.build(res, data, fc, sys.modules[__name__])
        pick_t = list(dict.fromkeys(p["ticker"] for pb in res["playbook"] for p in pb["picks"]))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"playbook step failed: {e}", file=sys.stderr)
    if rcfg.get("news", True):
        try:
            res["flow_news"] = R.flow_news()
        except Exception as e:
            print(f"flow news failed: {e}", file=sys.stderr)
        res["insider_news"] = {}
        for t in list(dict.fromkeys(pick_t + holdings + on_demand))[:16]:
            f = fs.get(t, {})
            res["insider_news"][t] = R.insider_news(t, names.get(t, t.replace(".KL", "")), f.get("long_name"))
            if t not in res["stock_news"]:
                res["stock_news"][t] = R.stock_news(t, names.get(t, t.replace(".KL", "")), f.get("long_name"))
        for pb in res.get("playbook", []):
            for p in pb["picks"]:
                p["insider"] = res["insider_news"].get(p["ticker"], [])
                p["news"] = res["stock_news"].get(p["ticker"], [])[:4]

    rpath = ROOT / "research.json"
    notes = R.load_json(rpath, {})
    today = dt.datetime.now(MYT).date()
    refresh = ai_cfg.get("refresh_days", 7)

    def stale(t):
        try:
            return (today - dt.date.fromisoformat(notes[t]["date"])).days >= refresh
        except Exception:
            return True

    queue = [t for t in on_demand] + [t for t in priority if t not in on_demand and stale(t) and
                                       (t in always or t in holdings or
                                        any(s["ticker"] == t and s["alert"] for s in res["signals"]))
                                       ]
    # first pick of each horizon also gets a note
    queue += [pb["picks"][0]["ticker"] for pb in res.get("playbook", []) if pb["picks"] and stale(pb["picks"][0]["ticker"])]
    queue = on_demand + [t for t in dict.fromkeys(queue) if t not in on_demand][: ai_cfg.get("max_stocks_per_run", 5)]
    if os.environ.get("ANTHROPIC_API_KEY") and ai_cfg.get("enabled", True):
        for t in dict.fromkeys(queue):
            sig = next((s for s in res["signals"] if s["ticker"] == t), None)
            tech = None
            if sig:
                tech = {k: sig[k] for k in ("pattern_label", "family", "setup", "trigger", "stop", "target",
                                            "risk_pct", "score", "grade", "rsi")}
                tech["volume"] = sig["volume"]
                tech["backtest"] = sig["pattern_stats"]
            news = (res["stock_news"].get(t) or R.stock_news(t, names.get(t, t), fs.get(t, {}).get("long_name"))) + \
                res.get("insider_news", {}).get(t, [])
            note = R.ai_note(t, names.get(t, fs.get(t, {}).get("long_name") or t), fs.get(t), tech, news, ai_cfg)
            if note:
                note["name"] = names.get(t, fs.get(t, {}).get("long_name") or t)
                notes[t] = note
                print(f"AI note: {t}")
        mb = notes.get("_market", {})
        try:
            age = (today - dt.date.fromisoformat(mb.get("date", "2000-01-01"))).days
        except ValueError:
            age = 99
        if age >= ai_cfg.get("brief_every_days", 3) and res.get("themes"):
            brief = R.ai_market_brief(res["themes"]["ranking"], ai_cfg)
            if brief:
                notes["_market"] = brief
                print("AI market brief written")
        rpath.write_text(json.dumps(notes, ensure_ascii=False, indent=1), encoding="utf-8")
    elif queue:
        print("AI analysis skipped: add an ANTHROPIC_API_KEY secret to enable it.")
    for s in res["signals"]:
        if s["ticker"] in notes:
            s["ai_verdict"] = notes[s["ticker"]].get("verdict")
            s["ai_date"] = notes[s["ticker"]].get("date")
    for pb in res.get("playbook", []):
        for p in pb["picks"]:
            if p["ticker"] in notes:
                p["ai_verdict"] = notes[p["ticker"]].get("verdict")
                p["ai_date"] = notes[p["ticker"]].get("date")
    res["market_brief"] = notes.get("_market")
    res["on_demand"] = on_demand
    res["ai_enabled"] = bool(os.environ.get("ANTHROPIC_API_KEY")) and ai_cfg.get("enabled", True)


def main():
    tickers = list(CFG["universe"].keys())
    extra = [h["ticker"] for h in CFG.get("holdings", [])] + \
        [_norm_ticker(t) for t in os.environ.get("RESEARCH_TICKER", "").split(",") if t.strip().isdigit()]
    for t in extra:
        if t not in tickers:
            tickers.append(t)
    print(f"Downloading {len(tickers)} stocks...")
    data = fetch_data(tickers, S["history_period"])
    if not data:
        sys.exit("No data downloaded. Yahoo Finance may be down; try again later.")
    index_df = None
    if CFG.get("index"):
        idx = fetch_data([CFG["index"]], "1y", 60)
        index_df = idx.get(CFG["index"])
    # optional live warrant prices
    ws = CFG.get("warrants", [])
    if ws:
        wd = fetch_data([w.get("yahoo", w["code"] + ".KL") for w in ws if not w.get("price")], "1mo", 1)
        for w in ws:
            if w.get("price"):
                w["_price"] = float(w["price"])
            else:
                df = wd.get(w.get("yahoo", w["code"] + ".KL"))
                w["_price"] = float(df["close"].iloc[-1]) if df is not None and len(df) else None
    res = run(data, index_df)
    if "--no-research" not in sys.argv:
        enrich(res, data)
    else:
        import playbook as PB
        PB.build(res, data, {"stocks": {}}, sys.modules[__name__])
    (ROOT / "signals.json").write_text(json.dumps(res, separators=(",", ":"), default=str), encoding="utf-8")
    n_alert = sum(s["alert"] for s in res["signals"])
    print(f"Scanned {res['scanned']} stocks · {len(res['signals'])} setups · {n_alert} alerts · data {res['data_date']}")
    if "--no-send" not in sys.argv:
        send_telegram(telegram_text(res, os.environ.get("APP_URL")))


if __name__ == "__main__":
    main()

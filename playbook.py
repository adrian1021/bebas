"""
Bebas playbook: theme heat map, money flow, and four holding-horizon strategies.

  Contra (T+2)      momentum breakout, sell by the 2nd market day after buying
  Short term (<=1m) the swing / breakout / candlestick engine in scan.py
  Mid term (1-3m)   stage-2 trend leaders: buy strength, trail with the 50-day EMA
  Long term (6m+)   quality + moat + cycle: accumulate good companies near their long-term average

Each strategy is backtested on the same 3 years of prices with the exact entry/exit rules shown
in the app. Long-term picks also use fundamentals, which cannot be backtested point-in-time,
so their test is price-only and labelled that way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Money flow
# --------------------------------------------------------------------------


def add_flow(d: pd.DataFrame) -> pd.DataFrame:
    h, l, c, v = d["high"], d["low"], d["close"], d["volume"]
    rng = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / rng
    d["cmf20"] = (mfm.fillna(0) * v).rolling(20).sum() / v.rolling(20).sum()
    tp = (h + l + c) / 3
    rmf = tp * v
    pos = rmf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg = rmf.where(tp < tp.shift(1), 0).rolling(14).sum()
    d["mfi14"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    d["vol50"] = v.rolling(50).mean()
    d["vol5"] = v.rolling(5).mean()
    d["value5"] = (c * v).rolling(5).mean()
    d["value50"] = (c * v).rolling(50).mean()
    up, dn = c > c.shift(1), c < c.shift(1)
    big = (v > v.shift(1)) & (v > d["vol50"])
    d["acc_day"] = (up & big).astype(int)
    d["dist_day"] = (dn & big).astype(int)
    d["r60"] = c / c.shift(60) - 1
    return d


def flow_snapshot(d: pd.DataFrame, mcap: float | None = None) -> dict:
    last = d.iloc[-1]
    acc = int(d["acc_day"].tail(25).sum())
    dist = int(d["dist_day"].tail(25).sum())
    cmf = float(last["cmf20"]) if not np.isnan(last["cmf20"]) else 0.0
    mfi = float(last["mfi14"]) if not np.isnan(last["mfi14"]) else 50.0
    turnover_surge = float(last["value5"] / last["value50"]) if last["value50"] else 1.0
    if cmf > 0.15 and acc - dist >= 3:
        label = "Strong inflow"
    elif cmf > 0.05 and acc >= dist:
        label = "Inflow"
    elif cmf < -0.15 and dist - acc >= 3:
        label = "Strong outflow"
    elif cmf < -0.05 and dist >= acc:
        label = "Outflow"
    else:
        label = "Neutral"
    # Accumulation score 0-100: 50 = no clear bias. Each term is capped so no single input dominates.
    c = d["close"]
    r5 = float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) > 6 and c.iloc[-6] else 0.0
    clip = lambda x: max(-1.0, min(1.0, x))
    accum = 50 + 40 * clip(cmf / 0.25) + 30 * clip((acc - dist) / 8) \
        + 15 * clip(turnover_surge - 1) * (1 if r5 >= 0 else -1) + 15 * clip(r5 / 0.05)
    accum = int(round(max(0, min(100, accum))))
    accum_label = ("Accumulating" if accum >= 70 else "Leaning in" if accum >= 58
                   else "Neutral" if accum > 45 else "Leaning out" if accum > 30 else "Distributing")
    out = {"cmf20": round(cmf, 3), "mfi14": round(mfi, 1), "acc_days_25": acc, "dist_days_25": dist,
           "turnover_surge": round(turnover_surge, 2), "label": label,
           "accum_score": accum, "accum_label": accum_label, "r5": round(r5 * 100, 2)}
    if mcap:
        out["daily_turnover_pct_mcap"] = round(float(last["value5"]) / mcap * 100, 3)
    hot = turnover_surge >= 2 and (out.get("daily_turnover_pct_mcap", 0) >= 0.8 or turnover_surge >= 3)
    out["hot_money"] = bool(hot)
    if mfi >= 80:
        out["note"] = "Overbought on money flow (MFI 80+): late buyers are piling in."
    elif mfi <= 20:
        out["note"] = "Oversold on money flow (MFI 20 or less)."
    return out


# --------------------------------------------------------------------------
# Theme heat map
# --------------------------------------------------------------------------

def _rank(vals: list[float]) -> list[float]:
    s = pd.Series(vals, dtype=float)
    return list((s.rank(pct=True) * 100).fillna(50))


def _status(score, rs20_rank, rs60_rank, volx, r20=0.0, rs20=0.0):
    if score >= 72 and volx >= 1.25 and r20 > 0 and rs20 > 0:
        return "Hot"
    if rs20_rank >= 65 and rs20_rank - rs60_rank >= 15 and r20 > 0:
        return "Heating up"
    if rs60_rank >= 60 and rs60_rank - rs20_rank >= 20:
        return "Cooling"
    if score <= 30:
        return "Cold"
    return "Neutral"


def theme_heat(themes: list[dict], frames: dict, idx_close: pd.Series | None, names: dict, flows: dict) -> dict:
    def ret(c, k):
        return float(c.iloc[-1] / c.iloc[-1 - k] - 1) * 100 if len(c) > k else 0.0

    ir20 = ret(idx_close, 20) if idx_close is not None and len(idx_close) > 21 else 0.0
    ir60 = ret(idx_close, 60) if idx_close is not None and len(idx_close) > 61 else 0.0
    m = {}
    for t, d in frames.items():
        c = d["close"]
        last = d.iloc[-1]
        m[t] = {"ticker": t, "code": t.replace(".KL", ""), "name": names.get(t, t),
                "close": round(float(last["close"]), 3),
                "r5": round(ret(c, 5), 2), "r20": round(ret(c, 20), 2), "r60": round(ret(c, 60), 2),
                "rs20": round(ret(c, 20) - ir20, 2), "rs60": round(ret(c, 60) - ir60, 2),
                "volx": round(float(last["vol5"] / last["vol50"]), 2) if last["vol50"] else 1.0,
                "above20": bool(last["close"] > last["ema20"]),
                "near_high": bool(last["close"] >= 0.9 * last["high250"]),
                "flow": flows.get(t, {}).get("label", "Neutral"),
                "cmf20": flows.get(t, {}).get("cmf20", 0.0)}

    def agg(members):
        ms = [m[t] for t in members if t in m]
        if not ms:
            return None
        mean = lambda k: round(float(np.mean([x[k] for x in ms])), 2)
        return {"n": len(ms), "r5": mean("r5"), "r20": mean("r20"), "r60": mean("r60"),
                "rs20": mean("rs20"), "rs60": mean("rs60"), "volx": mean("volx"), "cmf20": mean("cmf20"),
                "breadth": round(sum(x["above20"] for x in ms) / len(ms) * 100),
                "near_high": round(sum(x["near_high"] for x in ms) / len(ms) * 100),
                "leaders": [x["code"] for x in sorted(ms, key=lambda x: x["rs20"], reverse=True)[:3]],
                "stocks": sorted(ms, key=lambda x: x["rs20"], reverse=True)}

    def score_level(nodes):
        live = [n for n in nodes if n.get("stats")]
        if not live:
            return
        r20 = _rank([n["stats"]["rs20"] for n in live])
        r60 = _rank([n["stats"]["rs60"] for n in live])
        vx = _rank([n["stats"]["volx"] for n in live])
        br = [n["stats"]["breadth"] for n in live]
        fl = _rank([n["stats"]["cmf20"] for n in live])
        for n, a, b, c_, d_, e in zip(live, r20, r60, vx, br, fl):
            sc = round(0.30 * a + 0.15 * b + 0.25 * c_ + 0.15 * d_ + 0.15 * e)
            n["score"] = sc
            n["status"] = _status(sc, a, b, n["stats"]["volx"], n["stats"]["r20"], n["stats"]["rs20"])

    tree, subs_all, themes_all = [], [], []
    for sec in themes:
        sec_members, tnodes = [], []
        for th in sec.get("themes", []):
            th_members, snodes = [], []
            for sb in th.get("subthemes", []):
                mem = [x if "." in x else x + ".KL" for x in sb["stocks"]]
                th_members += mem
                node = {"name": sb["name"], "why": sb.get("why", ""), "stats": agg(mem)}
                snodes.append(node)
                subs_all.append(node)
            sec_members += th_members
            tnode = {"name": th["name"], "why": th.get("why", ""), "stats": agg(list(dict.fromkeys(th_members))),
                     "subthemes": snodes}
            tnodes.append(tnode)
            themes_all.append(tnode)
        tree.append({"name": sec["sector"], "stats": agg(list(dict.fromkeys(sec_members))), "themes": tnodes})
    score_level(subs_all)
    score_level(themes_all)
    score_level(tree)
    for lvl in (tree,):
        lvl.sort(key=lambda n: n.get("score", -1), reverse=True)
    for s in tree:
        s["themes"].sort(key=lambda n: n.get("score", -1), reverse=True)
        for t in s["themes"]:
            t["subthemes"].sort(key=lambda n: n.get("score", -1), reverse=True)
    hot = sorted([dict(name=f"{s['name']} > {t['name']} > {sb['name']}", score=sb.get("score"),
                       status=sb.get("status"), leaders=sb["stats"]["leaders"], r20=sb["stats"]["r20"],
                       volx=sb["stats"]["volx"])
                  for s in tree for t in s["themes"] for sb in t["subthemes"] if sb.get("stats")],
                 key=lambda x: x["score"] or 0, reverse=True)
    # where each stock sits, for tagging picks
    where = {}
    for s in tree:
        for t in s["themes"]:
            for sb in t["subthemes"]:
                for x in (sb["stats"] or {}).get("stocks", []):
                    where.setdefault(x["ticker"], []).append(
                        {"path": f"{s['name']} > {t['name']} > {sb['name']}", "status": sb.get("status"),
                         "score": sb.get("score")})
    return {"tree": tree, "ranking": hot, "where": where, "index_r20": round(ir20, 2)}


# --------------------------------------------------------------------------
# Generic trade simulator (next-open entry, one trade per stock at a time)
# --------------------------------------------------------------------------

def _sim(d, i, stop, target, max_hold, cost_pct, trail_col=None, gap_max=None, check_target_on_entry=False,
         hard_exit_col=None, hard_exit_mult=None):
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    n = len(d)
    j = i + 1
    if j >= n:
        return None
    entry = o[j]
    if gap_max is not None and entry > c[i] * (1 + gap_max):
        return {"status": "skipped_gap"}
    if stop is not None and entry <= stop:
        return {"status": "skipped_gap"}
    risk = entry - stop if stop is not None else None
    trail = d[trail_col].values if trail_col else None
    hard = d[hard_exit_col].values if hard_exit_col else None
    exit_px = exit_k = None
    for k in range(j, min(j + max_hold, n)):
        if stop is not None and k > j and o[k] <= stop:
            exit_px, exit_k = o[k], k
        elif stop is not None and l[k] <= stop:
            exit_px, exit_k = stop, k
        elif target is not None and (k > j or check_target_on_entry) and h[k] >= target:
            exit_px, exit_k = max(target, o[k]) if k > j else target, k
        elif trail is not None and k > j and c[k] < trail[k]:
            exit_px, exit_k = c[k], k
        elif hard is not None and c[k] < hard[k] * hard_exit_mult:
            exit_px, exit_k = c[k], k
        if exit_px is not None:
            break
    if exit_px is None:
        if j + max_hold - 1 < n:
            exit_k = j + max_hold - 1
            exit_px = c[exit_k]
        else:
            return {"status": "open", "exit_i": n}
    ret = (exit_px / entry - 1) * 100 - cost_pct
    r = (exit_px - entry - cost_pct / 100 * entry) / risk if risk else None
    return {"status": "closed", "ret": float(ret), "r": None if r is None else float(r), "hold": int(exit_k - j + 1),
            "entry_date": d.index[j].strftime("%Y-%m-%d"), "exit_i": exit_k}


def _stats(tr: list[dict]) -> dict:
    if not tr:
        return {"n": 0}
    ret = np.array([t["ret"] for t in tr])
    rs = np.array([t["r"] for t in tr if t.get("r") is not None])
    wins, losses = ret[ret > 0], ret[ret <= 0]
    ordered = [t["ret"] for t in sorted(tr, key=lambda t: t["entry_date"])]
    cum = np.cumsum(ordered)
    streak = worst = 0
    for x in ordered:
        streak = streak + 1 if x <= 0 else 0
        worst = max(worst, streak)
    idx = np.linspace(0, len(cum) - 1, min(80, len(cum))).astype(int)
    out = {"n": int(len(ret)), "win": round(float((ret > 0).mean() * 100), 1),
           "avg_ret": round(float(ret.mean()), 2), "median_ret": round(float(np.median(ret)), 2),
           "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
           "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
           "best": round(float(ret.max()), 1), "worst": round(float(ret.min()), 1),
           "pf": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
           "avg_hold": round(float(np.mean([t["hold"] for t in tr])), 1),
           "max_losing_streak": int(worst),
           "curve": [round(float(x), 1) for x in cum[idx]]}
    if len(rs):
        out["avg_r"] = round(float(rs.mean()), 2)
    return out


# --------------------------------------------------------------------------
# Strategy rules
# --------------------------------------------------------------------------

def contra_signal(d, min_value):
    return (d["close"] > d["high20_prev"]) & (d["vr"] >= 2.0) & (d["close_pos"] >= 0.75) & \
        (d["close"] >= d["close"].shift(1) * 1.02) & (d["close"] > d["ema20"]) & (d["value20"] >= min_value)


def mid_stage2(d):
    return (d["close"] > d["ema50"]) & (d["ema50"] > d["ema200"]) & (d["ema200"] > d["ema200"].shift(20)) & \
        (d["close"] >= 0.85 * d["high250"]) & (d["r60"] >= 0.10)


def mid_signal(d, min_value):
    brk = (d["close"] > d["high20_prev"]) & (d["vr"] >= 1.2)
    pull = (d["low"] <= d["ema20"] * 1.01) & d["bull"] & (d["close"] > d["ema20"])
    return mid_stage2(d) & (brk | pull) & (d["value20"] >= min_value)


def long_zone(d):
    """Long uptrend (rising 200-day EMA) pulled back to within -5%..+8% of that average."""
    rising = d["ema200"] > d["ema200"].shift(60)
    near = (d["close"] >= d["ema200"] * 0.95) & (d["close"] <= d["ema200"] * 1.08)
    turned = d["close"] > d["close"].shift(5)
    return rising & near & turned


def backtest_all(frames: dict, SC, cfg: dict) -> dict:
    S = SC.S
    cost = S["cost_pct_round_trip"]
    minv = S["min_avg_value_rm"]
    out = {"contra": [], "mid": [], "long": []}
    for t, d in frames.items():
        n = len(d)
        # contra: entry next open, stop below signal low (max 4%), target 1.5R, out by close of T+2
        sig = contra_signal(d, 2 * minv).fillna(False).values
        nxt = -1
        for i in np.flatnonzero(sig):
            if i < 200 or i <= nxt or i + 1 >= n:
                continue
            e = d["open"].iat[i + 1]
            stop = max(d["low"].iat[i] - SC.tick_size(d["low"].iat[i]), e * 0.96)
            res = _sim(d, i, stop, e + 1.5 * (e - stop), 3, cost * 0.8, gap_max=0.02)
            if res and res["status"] == "closed":
                out["contra"].append(res)
                nxt = res["exit_i"]
        # mid: entry next open, stop = max(EMA50*0.97, close-3ATR), trail exit on close < EMA50, max 90 days
        sig = mid_signal(d, minv).fillna(False).values
        nxt = -1
        for i in np.flatnonzero(sig):
            if i < 200 or i <= nxt or i + 1 >= n:
                continue
            row = d.iloc[i]
            stop = max(row["ema50"] * 0.97, row["close"] - 3 * row["atr"])
            if (row["close"] - stop) / row["close"] > 0.15:
                continue
            res = _sim(d, i, stop, None, 90, cost, trail_col="ema50")
            if res and res["status"] == "closed":
                out["mid"].append(res)
                nxt = res["exit_i"]
        # long: buy the pullback to a rising 200-day EMA, hold up to 250 days, exit on a 15% break below it
        sig = long_zone(d).fillna(False).values
        nxt = -1
        for i in np.flatnonzero(sig):
            if i < 200 or i <= nxt or i + 1 >= n:
                continue
            res = _sim(d, i, None, None, 250, cost, hard_exit_col="ema200", hard_exit_mult=0.85)
            if res and res["status"] == "closed":
                out["long"].append(res)
                nxt = res["exit_i"]
    return {k: _stats(v) for k, v in out.items()}


# --------------------------------------------------------------------------
# Picks
# --------------------------------------------------------------------------

WARRANT_GUIDE = {
    "contra": {"use": "Good fit if liquid", "rules": [
        "Effective gearing 4-8x, delta 0.5 or higher (at or in the money)",
        "Tight bid-ask spread (1 tick) and a market maker quoting size",
        "Any expiry over 2 months is fine for a 2-day hold"]},
    "short": {"use": "Good fit", "rules": [
        "Expiry at least 3 months away (3x your holding period)",
        "Delta 0.4-0.7, effective gearing 3-7x, premium under 15%",
        "Size by the warrant's price at the share's stop-loss, not by the warrant's own chart"]},
    "mid": {"use": "Only long-dated, in-the-money", "rules": [
        "Expiry at least 6-9 months away, delta 0.6 or higher",
        "Premium under 10%; avoid anything with less than 4 months left when you plan to exit",
        "Consider rolling to a longer expiry instead of holding into the last 3 months"]},
    "long": {"use": "Not recommended", "rules": [
        "Structured warrants decay and expire: they are trading tools, not investments",
        "Use the mother share (and dividends) for long-term holding",
        "Company-issued warrants (e.g. -WA/-WB) with years to expiry are a different product: check premium and expiry"]},
}


def _theme_tag(where, t):
    tags = where.get(t, [])
    return sorted(tags, key=lambda x: x.get("score") or 0, reverse=True)[:2]


def _pos(SC, entry, stop, risk_pct):
    return SC.position_size(entry, stop, SC.ACC["capital"], risk_pct, SC.ACC["max_position_pct"])


def build(res: dict, data: dict, fund: dict, SC) -> dict:
    """Adds res['themes'] and res['playbook'] and per-stock money flow to res['stocks']."""
    names = SC.CFG["universe"]
    S, minv = SC.S, SC.S["min_avg_value_rm"]
    frames = {}
    for t, raw in data.items():
        if len(raw) < 220:
            continue
        frames[t] = add_flow(SC.add_indicators(raw))
    last_date = max(d.index[-1] for d in frames.values())
    frames = {t: d for t, d in frames.items() if d.index[-1] == last_date}
    fs = fund.get("stocks", {})
    flows = {t: flow_snapshot(d, (fs.get(t) or {}).get("market_cap")) for t, d in frames.items()}
    for r in res.get("stocks", []):
        if r["ticker"] in flows:
            r["flow"] = flows[r["ticker"]]

    idx_close = None
    ix = res.get("market", {}).get("index")
    if ix and ix.get("closes"):
        idx_close = pd.Series(ix["closes"])
    th = theme_heat(SC.CFG.get("themes", []), frames, idx_close, names, flows)
    res["themes"] = th
    where = th["where"]
    bt = backtest_all(frames, SC, SC.CFG)
    risk_on = res["regime"]["state"] == "healthy"
    rk = (1 if risk_on else 0.5)
    r60_all = pd.Series({t: d["r60"].iat[-1] for t, d in frames.items()}).rank(pct=True)

    def theme_bonus(t):
        tags = _theme_tag(where, t)
        st = tags[0]["status"] if tags else None
        return {"Hot": 15, "Heating up": 10, "Neutral": 3, "Cooling": -5, "Cold": -10}.get(st, 0), tags

    def fbrief(t):
        f = fs.get(t) or {}
        sc = f.get("score") or {}
        return {"grade": sc.get("grade"), "score": sc.get("score"), "pe": f.get("pe"),
                "dividend_yield": f.get("dividend_yield"), "roe": f.get("roe"),
                "ni_yoy_ttm": f.get("ni_yoy_ttm"), "earnings_date": f.get("earnings_date"),
                "debt_to_equity": f.get("debt_to_equity"), "strengths": sc.get("strengths", [])[:3],
                "concerns": sc.get("concerns", [])[:3]}

    # ---- contra
    contra = []
    for t, d in frames.items():
        sig = contra_signal(d, 2 * minv).fillna(False)
        if not sig.iat[-1]:
            continue
        row = d.iloc[-1]
        e = SC.round_tick(row["close"] * 1.0, "up")
        stop = SC.round_tick(max(row["low"] - SC.tick_size(row["low"]), e * 0.96), "down")
        tgt = SC.round_tick(e + 1.5 * (e - stop), "up")
        tb, tags = theme_bonus(t)
        fl = flows[t]
        score = 40 + tb + min(row["vr"], 5) * 6 + (10 if fl["label"] in ("Inflow", "Strong inflow") else 0) + \
            (10 if row["close"] > row["ema200"] else 0) - (10 if fl["mfi14"] >= 85 else 0)
        contra.append({"ticker": t, "code": t.replace(".KL", ""), "name": names.get(t, t), "score": round(score),
                       "entry": e, "entry_note": f"Buy at the open only if it opens at or below {SC.round_tick(row['close'] * 1.02, 'down'):.3f} (no chasing gaps over 2%)",
                       "stop": stop, "target": tgt, "exit": "Sell by the close of the 2nd market day after buying (T+2), win or lose",
                       "hold": "1-3 days", "position": _pos(SC, e, stop, 0.5 * rk),
                       "reasons": [f"Broke its 20-day high on {row['vr']:.1f}x volume, +{(row['close'] / d['close'].iat[-2] - 1) * 100:.1f}% today",
                                   f"Closed in the top {100 - row['close_pos'] * 100:.0f}% of the day's range"],
                       "themes": tags, "flow": fl, "fundamentals": fbrief(t), "candles": SC.candles(d, 50)})
    contra.sort(key=lambda x: x["score"], reverse=True)

    # ---- short term (from the main engine)
    short = []
    for s in [x for x in res["signals"] if x.get("alert")] + [x for x in res["signals"] if not x.get("alert")]:
        if len(short) >= 6:
            break
        tb, tags = theme_bonus(s["ticker"])
        short.append({"ticker": s["ticker"], "code": s["code"], "name": s["name"], "score": s["score"] + tb,
                      "entry": s["trigger"], "entry_note": "Buy-stop order; cancel if not filled within 2 days",
                      "stop": s["stop"], "target": s["target"],
                      "exit": f"Stop or target, otherwise sell after {S['max_hold_days']} trading days",
                      "hold": "1-4 weeks", "position": s["position"], "strategy": s["pattern_label"],
                      "family": s["family"], "alert": s["alert"], "grade": s["grade"],
                      "reasons": s["reasons"][:4], "backtest": s["pattern_stats"], "themes": tags,
                      "flow": flows.get(s["ticker"]), "fundamentals": fbrief(s["ticker"]), "candles": s["candles"][-50:]})
    short.sort(key=lambda x: (x.get("alert", False), x["score"]), reverse=True)

    # ---- mid term
    mid = []
    for t, d in frames.items():
        row = d.iloc[-1]
        if not bool(mid_stage2(d).iat[-1]) or row["value20"] < minv:
            continue
        trig_today = bool(mid_signal(d, minv).iat[-1])
        e = SC.round_tick(row["high"] + SC.tick_size(row["high"]), "up")
        stop = SC.round_tick(max(row["ema50"] * 0.97, e - 3 * row["atr"]), "down")
        if (e - stop) / e > 0.15:
            continue
        tb, tags = theme_bonus(t)
        fb = fbrief(t)
        fl = flows[t]
        ext = (row["close"] / row["ema20"] - 1) * 100
        score = 30 * float(r60_all.get(t, 0.5)) + tb + {"A": 15, "B": 10, "C": 0, "D": -10}.get(fb["grade"], 0) + \
            (10 if fl["label"] in ("Inflow", "Strong inflow") else -5 if "Outflow" in fl["label"] else 0) + \
            (10 if trig_today else 0) + (10 if row["close"] >= 0.97 * row["high250"] else 0) - (10 if ext > 10 else 0)
        mid.append({"ticker": t, "code": t.replace(".KL", ""), "name": names.get(t, t), "score": round(score),
                    "entry": e, "entry_note": ("Triggered today: buy-stop above today's high" if trig_today else
                                               f"Leader in a stage-2 uptrend. Buy-stop above {e:.3f}, or a limit near the 20-day EMA ({row['ema20']:.3f}) on a pullback")
                    + (". Extended >10% above the 20-day EMA: wait for a pullback" if ext > 10 else ""),
                    "stop": stop, "target": SC.round_tick(e + 2 * (e - stop), "up"),
                    "exit": "Take 1/3 at 2R; sell the rest on a daily close below the 50-day EMA (trailing)",
                    "hold": "1-3 months", "position": _pos(SC, e, stop, 1.0 * rk), "triggered": trig_today,
                    "reasons": [f"Up {row['r60'] * 100:.0f}% in 60 days (top {100 - float(r60_all.get(t, 0.5)) * 100:.0f}% of stocks)",
                                "Price > 50-day EMA > rising 200-day EMA (stage-2 uptrend)",
                                f"{(row['close'] / row['high250'] - 1) * 100:.1f}% from its 52-week high"],
                    "themes": tags, "flow": fl, "fundamentals": fb, "candles": SC.candles(d, 50)})
    mid.sort(key=lambda x: x["score"], reverse=True)

    # ---- long term
    moats = SC.CFG.get("moats", {})
    longs = []
    for t, d in frames.items():
        f = fs.get(t) or {}
        sc = f.get("score") or {}
        if sc.get("score") is None or sc["score"] < 55:
            continue
        row = d.iloc[-1]
        if row["close"] < row["ema200"] * 0.9:
            continue
        dist = (row["close"] / row["ema200"] - 1) * 100
        rising = bool(row["ema200"] > d["ema200"].iat[-61])
        c = d["close"]
        yrs = len(c) / 250
        cagr = ((c.iat[-1] / c.iat[0]) ** (1 / yrs) - 1) * 100 if yrs > 0.9 else None
        dd = float(((c / c.cummax()) - 1).min() * 100)
        moat = moats.get(t)
        cycle = ("Accumulation zone: near its rising long-term average" if -5 <= dist <= 8 and rising else
                 "Extended: well above its long-term average, buy in smaller tranches" if dist > 20 else
                 "Uptrend" if rising else "Long-term average flat/falling: wait for the cycle to turn")
        score = sc["score"] * 0.6 + (15 if moat else 0) + (10 if rising else -5) + \
            (15 if -5 <= dist <= 8 else 5 if dist <= 20 else -10) + min((f.get("dividend_yield") or 0) * 2, 10)
        tb, tags = theme_bonus(t)
        lot_cap = SC.ACC["capital"] * 0.10
        sh = int(lot_cap / row["close"] // 100 * 100)
        longs.append({"ticker": t, "code": t.replace(".KL", ""), "name": names.get(t, t), "score": round(score),
                      "entry": SC.round_tick(row["close"], "near"),
                      "entry_note": "Accumulate in 3 tranches: 1/3 now if in the zone, 1/3 near the 200-day EMA "
                                    f"({row['ema200']:.3f}), 1/3 after the next results confirm the trend",
                      "stop": SC.round_tick(row["ema200"] * 0.85, "down"),
                      "stop_note": "Review, don't auto-sell: exit if price closes 15% below the 200-day EMA or profit falls 2 quarters in a row",
                      "target": None, "exit": "Hold while the business case holds; trim if valuation gets stretched",
                      "hold": "6 months to years",
                      "position": {"shares": sh, "lots": sh // 100, "cost_rm": round(sh * row["close"], 2),
                                   "note": "Sized as 10% of capital, not by stop distance"},
                      "moat": moat, "cycle": cycle, "dist_ema200_pct": round(dist, 1),
                      "history": {"cagr_pct": round(cagr, 1) if cagr is not None else None,
                                  "max_drawdown_pct": round(dd, 1), "years": round(yrs, 1)},
                      "reasons": [f"Fundamental score {sc['score']} ({sc.get('grade')})"] + sc.get("strengths", [])[:3],
                      "themes": tags, "flow": flows[t], "fundamentals": fbrief(t), "candles": SC.candles(d, 50)})
    longs.sort(key=lambda x: x["score"], reverse=True)

    def pack(key, title, desc, rules, picks, stats, extra=None):
        for p in picks[:3]:
            p["warrants"] = []
            for w in SC.CFG.get("warrants", []):
                if w["mother"] == p["ticker"] and key != "long":
                    try:
                        plan = {"stop": p["stop"], "target": p["target"] or p["entry"] * 1.2}
                        p["warrants"].append(SC.warrant_plan(w, frames[p["ticker"]], plan, w.get("_price")))
                    except Exception:
                        pass
        return {"key": key, "title": title, "desc": desc, "rules": rules, "backtest": stats,
                "picks": picks[:3], "watch": [{k: p[k] for k in ("ticker", "code", "name", "score")} for p in picks[3:8]],
                "warrant_guide": WARRANT_GUIDE[key], **(extra or {})}

    res["playbook"] = [
        pack("contra", "Contra (T+2)",
             "Ride a 1-3 day momentum burst and sell before payment is due. Highest risk, smallest size.",
             ["Signal: close above the 20-day high, volume 2x+, close in the top 25% of the day's range, up 2%+",
              "Average daily value at least 2x the normal liquidity filter",
              "Buy at the next open only if the gap is 2% or less",
              "Stop: below the signal day's low (max 4%). Target: 1.5x the risk",
              "Sell by the close of T+2 no matter what. Risk 0.5% of capital"], contra, bt["contra"]),
        pack("short", "Short term (up to 1 month)",
             "Swing, breakout, volume and candlestick setups from the main scanner, ranked with theme strength.",
             ["Entry: buy-stop above the setup high, valid 2 days",
              f"Stop: below the setup low; target {S['reward_r']}R; time exit after {S['max_hold_days']} days",
              "Risk 1% of capital (half in weak markets)"], short, res["backtest"]["overall"]),
        pack("mid", "Mid term (1-3 months)",
             "Buy leaders in a stage-2 uptrend and let the trend run, trailing with the 50-day EMA.",
             ["Trend: price > 50-day EMA > rising 200-day EMA, within 15% of the 52-week high, up 10%+ in 60 days",
              "Trigger: breakout above the 20-day high on 1.2x volume, or a bounce off the 20-day EMA",
              "Stop: the higher of 3% under the 50-day EMA or 3 ATR (max 15%)",
              "Exit: take 1/3 at 2R, the rest on a close below the 50-day EMA; max 90 days in the test"],
             mid, bt["mid"]),
        pack("long", "Long term (fundamentals, moat, cycle)",
             "Own good businesses bought near their long-term average. Fundamentals first, price for timing.",
             ["Fundamental score 55+ (profitability, growth, valuation, debt, dividend)",
              "Moat: a durable edge (monopoly, network, scale, licence, brand). Listed where known",
              "Cycle: rising 200-day EMA; best entries within -5% to +8% of it",
              "Size by allocation (about 10% of capital per stock), not by stop distance"],
             longs, bt["long"],
             {"backtest_note": "Price-only test of the timing rule (buy near a rising 200-day EMA, hold up to a year). "
                               "Fundamental filters can't be tested on past data because past fundamentals aren't stored."}),
    ]
    return res

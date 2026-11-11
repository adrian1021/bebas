#!/usr/bin/env python3
"""On-demand analysis of any Bursa stock or call warrant (writes lookup.json).

Run from the GitHub "Daily scan" workflow (type codes in "Research a stock") or by hand:
    python lookup.py 1155 YTLPOWR-C2C

For each code it gives: technical state, the four horizon setups with a 0-100
confidence score, a plan (entry, stop, target, exit, size), a per-stock backtest
of each strategy, money flow, fundamentals, news, insider news, theme membership
and the call warrants that reference the stock.

Every component is shown, so the score is explainable, not a black box.
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

import playbook as PB
import research as R
import scan as SC

ROOT = SC.ROOT
MYT = SC.MYT

# Weights per horizon. Each row sums to 100.
WEIGHTS = {
    "contra": dict(tech=30, flow=25, fund=5, moat=0, bt=15, news=10, regime=15),
    "short": dict(tech=35, flow=15, fund=5, moat=0, bt=20, news=10, regime=15),
    "mid": dict(tech=30, flow=15, fund=15, moat=0, bt=15, news=10, regime=15),
    "long": dict(tech=15, flow=10, fund=40, moat=10, bt=10, news=5, regime=10),
}
HORIZON_NAMES = {"contra": "Contra (T+2)", "short": "Short term (up to 1 month)",
                 "mid": "Mid term (1-3 months)", "long": "Long term"}
MIN_TRADES = 8


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _frac(conds):
    return sum(1 for _, m in conds if m) / len(conds) if conds else 0.0


def _bt_frac(st, key):
    """Own-history expectancy. Too few past trades = neutral, not a guess."""
    if not st or st.get("n", 0) < MIN_TRADES or st.get(key) is None:
        return 0.5
    scale = 4.0 if key == "avg_ret" else 2.0  # +1% per trade or +1R both give a full bump
    return _clip(0.5 + st[key] / scale)


def _verdict(score):
    if score >= 70:
        return "Strong: high-conviction setup"
    if score >= 55:
        return "Reasonable: act on the trigger"
    if score >= 40:
        return "Weak: watch only"
    return "Avoid for now"


def _conds_out(conds):
    return [{"name": n, "met": bool(m)} for n, m in conds]


# --------------------------------------------------------------------------
# Live candlestick / breakout / swing / volume setups for ONE stock
# --------------------------------------------------------------------------

def live_patterns(d: pd.DataFrame) -> list[dict]:
    tp, os_, liquid = SC.setups(d)
    n = len(d)
    out = []
    for key, (label, direction, nbars, fn, fam) in SC.PATTERNS.items():
        if direction != "bull":
            continue
        ctx = (tp | os_) if fam == "Candlestick" else True
        hit = fn(d).fillna(False).astype(bool) & ctx & liquid
        hit.iloc[:200] = False
        idx = np.flatnonzero(hit.values)
        trades = []
        for i in idx:
            plan = SC.plan_at(d, int(i), nbars)
            if plan:
                r = SC.simulate(d, int(i), plan)
                if r["status"] == "closed":
                    trades.append(r)
        recent = [int(i) for i in idx if i >= n - 2]
        if not recent:
            continue
        i = recent[-1]
        plan = SC.plan_at(d, i, nbars)
        if not plan:
            continue
        setup = ("Trend pullback" if tp.iloc[i] else "Oversold bounce") if fam == "Candlestick" else fam
        out.append({"key": key, "label": label, "family": fam, "setup": setup, "plan": plan,
                    "status": "new" if i == n - 1 else "pending",
                    "date": d.index[i].strftime("%Y-%m-%d"), "stats": SC.stats_from(trades)})
    out.sort(key=lambda x: (x["stats"].get("avg_r") or -9), reverse=True)
    return out


# --------------------------------------------------------------------------
# Horizon views: conditions, plan, trigger state
# --------------------------------------------------------------------------

def contra_view(d, row, minv):
    prev_close = d["close"].iat[-2]
    conds = [
        ("Closes above its 20-day high", row["close"] > row["high20_prev"]),
        ("Volume at least 2x the 20-day average", row["vr"] >= 2.0),
        ("Closes in the top 25% of the day's range", row["close_pos"] >= 0.75),
        ("Up 2% or more today", row["close"] >= prev_close * 1.02),
        ("Above the 20-day EMA", row["close"] > row["ema20"]),
        ("Trades at least RM2m a day", row["value20"] >= 2 * minv),
    ]
    e = SC.round_tick(row["close"], "up")
    stop = SC.round_tick(max(row["low"] - SC.tick_size(row["low"]), e * 0.96), "down")
    plan = {"entry": e, "stop": stop, "target": SC.round_tick(e + 1.5 * (e - stop), "up"),
            "entry_note": f"Buy at the open only if it opens at or below {SC.round_tick(row['close'] * 1.02, 'down'):.3f}",
            "exit": "Sell by the close of T+2 (second market day after buying), win or lose",
            "hold": "1-3 days", "risk_pct": 0.5, "reward_r": 1.5}
    return conds, plan, _frac(conds) == 1.0


def short_view(d, row, live):
    if live:
        p = live[0]["plan"]
        conds = [("Live pattern: " + live[0]["label"], True),
                 ("Above the 50-day EMA", row["close"] > row["ema50"]),
                 ("Above the 200-day EMA", row["close"] > row["ema200"]),
                 ("Volume at least 1.2x the 20-day average", row["vr"] >= 1.2)]
        plan = {"entry": p["trigger"], "stop": p["stop"], "target": p["target"],
                "entry_note": "Buy-stop order; cancel if not filled within 2 days",
                "exit": f"Stop or target, otherwise sell after {SC.S['max_hold_days']} trading days",
                "hold": "1-4 weeks", "risk_pct": 1.0, "reward_r": SC.S["reward_r"]}
        return conds, plan, True
    conds = [("Above the 20-day EMA", row["close"] > row["ema20"]),
             ("Above the 50-day EMA", row["close"] > row["ema50"]),
             ("20-day EMA above 50-day EMA", row["ema20"] > row["ema50"]),
             ("Volume at least 1.2x the 20-day average", row["vr"] >= 1.2),
             ("RSI between 40 and 70", 40 <= row["rsi"] <= 70)]
    e = SC.round_tick(row["high"] + SC.tick_size(row["high"]), "up")
    stop = SC.round_tick(row["low"] - 0.2 * float(row["atr"]), "down")
    plan = {"entry": e, "stop": stop, "target": SC.round_tick(e + SC.S["reward_r"] * (e - stop), "up"),
            "entry_note": "No live setup today. These are the levels to watch: buy-stop only if a pattern forms",
            "exit": f"Stop or target, otherwise sell after {SC.S['max_hold_days']} trading days",
            "hold": "1-4 weeks", "risk_pct": 1.0, "reward_r": SC.S["reward_r"], "hypothetical": True}
    return conds, plan, False


def mid_view(d, row, minv):
    trig = bool(PB.mid_signal(d, minv).iat[-1])
    conds = [("Price above the 50-day EMA", row["close"] > row["ema50"]),
             ("50-day EMA above a rising 200-day EMA", row["ema50"] > row["ema200"] and row["ema200"] > d["ema200"].iat[-21]),
             ("Within 15% of its 52-week high", row["close"] >= 0.85 * row["high250"]),
             ("Up 10%+ in 60 days", row["r60"] >= 0.10),
             ("Trades at least RM2m a day", row["value20"] >= minv),
             ("Breakout or pullback trigger today", trig)]
    e = SC.round_tick(row["high"] + SC.tick_size(row["high"]), "up")
    stop = SC.round_tick(max(row["ema50"] * 0.97, e - 3 * row["atr"]), "down")
    plan = {"entry": e, "stop": stop, "target": SC.round_tick(e + 2 * (e - stop), "up"),
            "entry_note": f"Buy-stop above {e:.3f}, or a limit near the 20-day EMA ({row['ema20']:.3f}) on a pullback",
            "exit": "Take 1/3 at 2R; sell the rest on a daily close below the 50-day EMA",
            "hold": "1-3 months", "risk_pct": 1.0, "reward_r": 2}
    if stop >= e:
        plan["invalid"] = "Not a valid set-up: the stop sits above the entry, so the price is below its 50-day average"
        trig = False
    elif (e - stop) / e > 0.15:
        plan["entry_note"] += ". Stop is more than 15% away: the mid-term rule skips this set-up until the base tightens"
    return conds, plan, trig


def long_view(d, row, fscore, moat):
    dist = (row["close"] / row["ema200"] - 1) * 100
    rising = bool(row["ema200"] > d["ema200"].iat[-61])
    conds = [("Fundamental score 55 or more", fscore is not None and fscore >= 55),
             ("Long-term average rising", rising),
             ("Within -5% to +8% of the 200-day average", -5 <= dist <= 8),
             ("Named moat (durable edge)", bool(moat)),
             ("Not more than 10% below the 200-day average", row["close"] >= row["ema200"] * 0.9)]
    e = SC.round_tick(row["close"], "near")
    stop = SC.round_tick(row["ema200"] * 0.85, "down")
    sh = int(SC.ACC["capital"] * 0.10 / row["close"] // 100 * 100)
    plan = {"entry": e, "stop": stop, "target": None,
            "entry_note": "Accumulate in 3 tranches: 1/3 now if in the zone, 1/3 near the 200-day average, 1/3 after the next results",
            "exit": "Hold while the business case holds; review if price closes 15% below the 200-day average",
            "hold": "6 months to years", "risk_pct": None,
            "position": {"shares": sh, "lots": sh // 100, "cost_rm": round(sh * row["close"], 2),
                         "note": "Sized as 10% of capital, not by stop distance"},
            "dist_ema200_pct": round(float(dist), 1)}
    triggered = bool(fscore is not None and fscore >= 55 and rising and -5 <= dist <= 8)
    return conds, plan, triggered


# --------------------------------------------------------------------------
# Analysis of one stock
# --------------------------------------------------------------------------

def analyse(t: str, raw: pd.DataFrame, name: str, fund: dict | None, news: list, insider: list,
            tags: list, regime_ok: bool, bt_all: dict | None = None) -> tuple[dict, pd.DataFrame]:
    d = PB.add_flow(SC.add_indicators(raw))
    row = d.iloc[-1]
    if len(d) < 220 or pd.isna(row["ema200"]):
        raise ValueError("not enough trading history (needs about a year) for the 200-day average")
    minv = SC.S["min_avg_value_rm"]
    fb = SC._fund_brief(fund)
    fscore = (fb or {}).get("score")
    moat = SC.CFG.get("moats", {}).get(t)
    flow = PB.flow_snapshot(d, (fund or {}).get("market_cap"))
    bt = bt_all if bt_all is not None else PB.backtest_all({t: d}, SC, SC.CFG)

    items = (news or []) + (insider or [])
    pos = sum(R.tone(x.get("title") or "") == "positive" for x in items)
    neg = sum(R.tone(x.get("title") or "") == "negative" for x in items)
    news_frac = 0.5 + 0.5 * (pos - neg) / (pos + neg) if pos + neg else 0.5
    flow_frac = _clip((flow["accum_score"] - 30) / 50)
    fund_frac = fscore / 100 if fscore is not None else 0.5
    regime_frac = 1.0 if regime_ok else 0.4

    live = live_patterns(d)
    horizons = {}

    c_conds, c_plan, c_trig = contra_view(d, row, minv)
    s_conds, s_plan, s_trig = short_view(d, row, live)
    m_conds, m_plan, m_trig = mid_view(d, row, minv)
    l_conds, l_plan, l_trig = long_view(d, row, fscore, moat)

    specs = {
        "contra": (c_conds, c_plan, c_trig, bt.get("contra", {}), "avg_ret"),
        "short": (s_conds, s_plan, s_trig, live[0]["stats"] if live else {}, "avg_r"),
        "mid": (m_conds, m_plan, m_trig, bt.get("mid", {}), "avg_ret"),
        "long": (l_conds, l_plan, l_trig, bt.get("long", {}), "avg_ret"),
    }
    for key, (conds, plan, trig, bt_st, bt_key) in specs.items():
        tech = _frac(conds)
        bt_f = _bt_frac(bt_st, bt_key)
        comps = {"tech": tech, "flow": flow_frac, "fund": fund_frac,
                 "moat": 1.0 if moat else 0.0, "bt": bt_f, "news": news_frac, "regime": regime_frac}
        w = WEIGHTS[key]
        score = round(sum(w[k] * comps[k] for k in w))
        why = [f"Technical rules met: {sum(1 for _, m in conds if m)} of {len(conds)}",
               f"Money flow: accumulation score {flow['accum_score']} ({flow['accum_label']})",
               f"Fundamentals: {'score ' + str(fscore) if fscore is not None else 'not available (neutral)'}",
               f"Own backtest: {bt_st.get('n', 0)} past trades" + (
                   f", average {bt_st.get(bt_key)} {'%' if bt_key == 'avg_ret' else 'R'}"
                   if bt_st.get("n", 0) >= MIN_TRADES and bt_st.get(bt_key) is not None
                   else " (too few to judge: neutral)"),
               f"News and insider headlines: {pos} positive, {neg} negative"]
        if not regime_ok:
            why.append("Market is weak: sizes are halved and scores are reduced")
        if key == "long" and moat:
            why.append(f"Moat: {moat}")
        if key == "mid" and not m_trig:
            why.append("No trigger today: the set-up is in place, but waiting for the breakout or pullback")
        if key == "contra" and not c_trig:
            why.append("Contra needs a 2%+ breakout day on 2x volume: not present today")
        if key == "short" and not live:
            why.append("No candlestick, breakout, swing or volume pattern in the last 2 days")
        if key == "long" and not l_trig:
            why.append("Not in the long-term zone or fundamentals below the threshold today")

        if plan.get("invalid"):
            action = "NO SETUP: " + plan["invalid"]
        elif trig:
            if key == "contra":
                action = f"BUY at the open if it opens at or below {plan['entry'] * 1.02:.3f}; sell by T+2"
            elif key == "long":
                action = f"Accumulate in tranches from {plan['entry']:.3f}; stop-review at {plan['stop']:.3f}"
            else:
                action = f"BUY-STOP at {plan['entry']:.3f}; stop {plan['stop']:.3f}; target {plan['target']:.3f}"
        else:
            action = (f"WAIT: no trigger yet. Levels shown are for when it triggers ({plan['entry']:.3f} entry)"
                      if key != "long" else "WAIT: not in the long-term zone yet. Tranche levels shown for when it is")
        if key == "short" and plan.get("hypothetical"):
            action = f"WAIT: no live pattern. Watch {plan['entry']:.3f} as the level to act on if a pattern forms"

        verdict = _verdict(score)
        if not trig and score >= 55 and not plan.get("invalid"):
            verdict = "Good set-up, not triggered yet: wait"
        if plan.get("invalid"):
            verdict = "No set-up"
        horizons[key] = {
            "name": HORIZON_NAMES[key], "score": int(score), "verdict": verdict,
            "triggered": bool(trig), "action": action, "plan": {k: v for k, v in plan.items()},
            "conditions": _conds_out(conds), "backtest": bt_st,
            "backtest_key": bt_key, "components": {k: round(v * 100) for k, v in comps.items()},
            "weights": w, "why": why,
        }
        if plan.get("target") is not None and plan["target"] > plan["entry"] and not plan.get("invalid"):
            horizons[key]["plan"]["reward_pct"] = round((plan["target"] / plan["entry"] - 1) * 100, 1)
        if plan["entry"] > plan["stop"]:
            horizons[key]["plan"]["risk_pct_move"] = round((plan["entry"] - plan["stop"]) / plan["entry"] * 100, 1)
        if key in ("contra", "short", "mid") and plan["entry"] > plan["stop"] and not plan.get("invalid"):
            pos_size = SC.position_size(plan["entry"], plan["stop"], SC.ACC["capital"],
                                        plan["risk_pct"] * (1 if regime_ok else 0.5), SC.ACC["max_position_pct"])
            horizons[key]["plan"]["position"] = pos_size

    best = max(horizons, key=lambda k: horizons[k]["score"])
    c = d["close"]
    last_date = d.index[-1].strftime("%Y-%m-%d")
    earnings = (fb or {}).get("earnings_date")
    notes = []
    if earnings:
        try:
            days = (dt.date.fromisoformat(earnings) - dt.datetime.now(MYT).date()).days
            if 0 <= days <= 14:
                notes.append(f"Results due {earnings} ({days} days): gap risk, consider waiting for the result")
        except ValueError:
            pass
    if not regime_ok:
        notes.append("Market is weak right now: sizes are halved")
    if not news_items_present(news, insider):
        notes.append("No recent headlines found for this stock")

    trend = ("Uptrend" if row["close"] > row["ema50"] > row["ema200"] else
             "Downtrend" if row["close"] < row["ema50"] < row["ema200"] else "Mixed / sideways")
    item = {
        "ticker": t, "code": t.replace(".KL", ""), "name": name, "type": "stock",
        "date": last_date, "price": round(float(row["close"]), 3),
        "change": {"1d": round(float((c.iat[-1] / c.iat[-2] - 1) * 100), 2),
                   "5d": round(float((c.iat[-1] / c.iat[-6] - 1) * 100), 2),
                   "20d": round(float((c.iat[-1] / c.iat[-21] - 1) * 100), 2),
                   "60d": round(float(row["r60"] * 100), 2)},
        "technical": {"trend": trend, "rsi": round(float(row["rsi"]), 1),
                      "ema20": round(float(row["ema20"]), 3), "ema50": round(float(row["ema50"]), 3),
                      "ema200": round(float(row["ema200"]), 3), "atr": round(float(row["atr"]), 3),
                      "volume_ratio": round(float(row["vr"]), 2),
                      "vs_52w_high_pct": round(float((row["close"] / row["high250"] - 1) * 100), 1),
                      "value20_rm_m": round(float(row["value20"]) / 1e6, 2)},
        "flow": flow, "fundamentals": fb, "news": (news or [])[:6], "insider": (insider or [])[:4],
        "themes": tags, "horizons": horizons, "best": best,
        "live_patterns": [{k: v for k, v in p.items() if k != "plan"} | {"plan": p["plan"]} for p in live[:5]],
        "warrant_guide": PB.WARRANT_GUIDE[best], "notes": notes, "moat": moat,
    }
    return item, d


def news_items_present(news, insider):
    return bool((news or []) or (insider or []))


def warrant_detail(w: dict, d: pd.DataFrame, item: dict) -> dict:
    """Warrant economics at the best horizon's entry, stop and target."""
    best = item["horizons"][item["best"]]
    p = best["plan"]
    plan = {"trigger": p["entry"], "stop": p["stop"],
            "target": p["target"] if p.get("target") is not None else p["entry"] * 1.2}
    out = SC.warrant_plan(w, d, plan, w.get("_price"))
    out["horizon"] = best["name"]
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def _code(s: str) -> str:
    return s.strip().upper().replace(".KL", "")


def _j(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def main(argv: list[str]) -> None:
    raw_codes = argv or [c for c in os.environ.get("RESEARCH_TICKER", "").split(",") if c.strip()]
    codes = list(dict.fromkeys(_code(c) for c in raw_codes if c.strip()))
    if not codes:
        print("Nothing to look up.")
        return
    ws = SC.CFG.get("warrants", [])
    by_code = {}
    for w in ws:
        by_code[w["code"].upper()] = w
        by_code[w.get("yahoo", "").replace(".KL", "").upper()] = w

    errors, targets = {}, []
    for c in codes:
        if c in by_code:
            targets.append((c, "warrant"))
        elif c.isdigit():
            targets.append((c, "stock"))
        else:
            errors[c] = ("Not recognised. Use a Bursa stock code (e.g. 1155) or a warrant code that is in "
                         "config.json (warrants list, with mother, exercise, ratio and expiry).")
    if not targets:
        write_out({}, errors)
        return

    tickers = list(dict.fromkeys(
        [f"{c}.KL" for c, k in targets if k == "stock"] +
        [by_code[c]["mother"] for c, k in targets if k == "warrant"]))
    print(f"Looking up {len(targets)} code(s): {', '.join(c for c, _ in targets)}")
    data = SC.fetch_data(tickers, SC.S["history_period"], 220)
    index_df = SC.fetch_data([SC.CFG["index"]], "1y", 60).get(SC.CFG["index"])

    # warrant prices (live where Yahoo has them)
    need = [w for w in ws if w.get("price") is None]
    if need:
        wd = SC.fetch_data([w.get("yahoo", w["code"] + ".KL") for w in need], "1mo", 1)
        for w in ws:
            if w.get("price"):
                w["_price"] = float(w["price"])
            else:
                df = wd.get(w.get("yahoo", w["code"] + ".KL"))
                w["_price"] = float(df["close"].iloc[-1]) if df is not None and len(df) else None

    sig = R.load_json(ROOT / "signals.json", {})
    regime_ok = (sig.get("regime") or {}).get("state", "healthy") == "healthy"
    where = ((sig.get("themes") or {}).get("where")) or {}

    rcfg = SC.CFG.get("research", {})
    fpath = ROOT / "fundamentals.json"
    try:
        fc = R.refresh_fundamentals(list(dict.fromkeys(list(SC.CFG["universe"]) + tickers)), tickers, fpath, rcfg)
        fc = R.score_all(fc, SC.CFG.get("sectors", {}))
        fpath.write_text(json.dumps(fc, separators=(",", ":"), default=str), encoding="utf-8")
    except Exception as e:
        print(f"fundamentals step failed: {e}", file=sys.stderr)
        fc = R.load_json(fpath, {"stocks": {}})
    fs = fc.get("stocks", {})

    out = {}
    for code, kind in targets:
        t = f"{code}.KL" if kind == "stock" else by_code[code]["mother"]
        try:
            if t not in data:
                raise ValueError("no usable price history from Yahoo Finance (check the code, or try again later)")
            name = SC.CFG["universe"].get(t) or (fs.get(t) or {}).get("long_name") or code
            fund = fs.get(t)
            nm = name
            news = R.stock_news(t, nm, (fund or {}).get("long_name"))
            insider = R.insider_news(t, nm, (fund or {}).get("long_name"))
            tags = [{"path": x.get("path"), "status": x.get("status")} for x in where.get(t, [])[:3]]
            item, d = analyse(t, data[t], name, fund, news, insider, tags, regime_ok)
            item["type"] = kind
            item["warrants"] = [warrant_detail(w, d, item) for w in ws if w["mother"] == t]
            if kind == "warrant":
                w = by_code[code]
                item["requested_warrant"] = warrant_detail(w, d, item)
                item["requested_warrant"]["code"] = w["code"]
                item["code"] = w["code"]
                item["name"] = f"{w['code']} (call warrant on {item['name']})"
                out[w["code"]] = item
            else:
                out[code] = item
            print(f"ok: {code} best={item['best']} score={item['horizons'][item['best']]['score']}")
        except Exception as e:
            errors[code] = str(e)
            print(f"error: {code}: {e}", file=sys.stderr)
    write_out(out, errors)


def write_out(items: dict, errors: dict) -> None:
    path = ROOT / "lookup.json"
    prev = R.load_json(path, {"items": {}})
    stamp = dt.datetime.now(MYT).strftime("%Y-%m-%d %H:%M MYT")
    for k, v in items.items():
        v["generated_at"] = stamp
    merged = dict(prev.get("items", {}))
    merged.update(items)
    doc = {"generated_at": stamp, "items": merged, "errors": errors}
    path.write_text(json.dumps(doc, separators=(",", ":"), default=_j, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path.name}: {len(items)} new, {len(merged)} stored, {len(errors)} errors")


if __name__ == "__main__":
    main(sys.argv[1:])

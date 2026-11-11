"""
Company research for Bebas: fundamentals, news and (optional) AI deep analysis.

- Fundamentals come from Yahoo Finance via yfinance and are cached in fundamentals.json.
  Coverage for Bursa stocks is good for large caps and patchy for small caps; missing
  fields are shown as missing, never guessed.
- News comes from Google News RSS (free, no key) and Yahoo Finance.
- AI analysis is optional: add an ANTHROPIC_API_KEY secret and the scanner asks Claude
  (with web search) for a short research note on each alerted stock. Saved in research.json.
"""
from __future__ import annotations

import datetime as dt
import email.utils
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

MYT = dt.timezone(dt.timedelta(hours=8))
UA = {"User-Agent": "Mozilla/5.0 (compatible; bebas-scanner/1.0)"}


def _now():
    return dt.datetime.now(MYT)


def _num(x):
    try:
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


# --------------------------------------------------------------------------
# Fundamentals
# --------------------------------------------------------------------------

def _stmt_row(df: pd.DataFrame, names: list[str], col):
    for n in names:
        if n in df.index:
            return _num(df.at[n, col])
    return None


REV = ["Total Revenue", "Operating Revenue"]
NI = ["Net Income Common Stockholders", "Net Income",
      "Net Income From Continuing Operation Net Minority Interest"]


def _statement(df) -> list[dict]:
    out = []
    if df is None or getattr(df, "empty", True):
        return out
    for col in list(df.columns):
        out.append({"period": pd.Timestamp(col).strftime("%Y-%m-%d"),
                    "revenue": _stmt_row(df, REV, col), "net_income": _stmt_row(df, NI, col)})
    return sorted(out, key=lambda x: x["period"])


def _yoy(q: list[dict], k: str):
    """Trailing 4 quarters vs the 4 before. Needs 8 quarters."""
    vals = [x[k] for x in q if x[k] is not None]
    if len(vals) < 8:
        return None
    a, b = sum(vals[-4:]), sum(vals[-8:-4])
    return round((a - b) / abs(b) * 100, 1) if b else None


def fetch_fundamentals(ticker: str) -> dict:
    import yfinance as yf
    tk = yf.Ticker(ticker)
    info = {}
    try:
        info = tk.info or {}
    except Exception as e:
        print(f"info {ticker}: {e}", file=sys.stderr)
    g = lambda k: _num(info.get(k))
    dy = g("trailingAnnualDividendYield")
    if dy is None and g("dividendYield") is not None:
        dy = g("dividendYield")
        dy = dy / 100 if dy > 0.3 else dy  # newer yfinance returns percent
    de = g("debtToEquity")
    f = {
        "ticker": ticker,
        "fetched": _now().strftime("%Y-%m-%d"),
        "long_name": info.get("longName") or info.get("shortName"),
        "yf_sector": info.get("sector"), "industry": info.get("industry"),
        "summary": (info.get("longBusinessSummary") or "")[:700],
        "website": info.get("website"),
        "market_cap": g("marketCap"), "price": g("currentPrice") or g("regularMarketPrice"),
        "pe": g("trailingPE"), "forward_pe": g("forwardPE"), "pb": g("priceToBook"),
        "eps": g("trailingEps"), "book_value": g("bookValue"),
        "dividend_yield": round(dy * 100, 2) if dy is not None else None,
        "payout_ratio": round(g("payoutRatio") * 100, 1) if g("payoutRatio") is not None else None,
        "roe": round(g("returnOnEquity") * 100, 1) if g("returnOnEquity") is not None else None,
        "roa": round(g("returnOnAssets") * 100, 1) if g("returnOnAssets") is not None else None,
        "net_margin": round(g("profitMargins") * 100, 1) if g("profitMargins") is not None else None,
        "op_margin": round(g("operatingMargins") * 100, 1) if g("operatingMargins") is not None else None,
        "rev_growth_q": round(g("revenueGrowth") * 100, 1) if g("revenueGrowth") is not None else None,
        "earn_growth_q": round(g("earningsGrowth") * 100, 1) if g("earningsGrowth") is not None else None,
        "debt_to_equity": round(de / 100, 2) if de is not None else None,
        "current_ratio": g("currentRatio"),
        "total_cash": g("totalCash"), "total_debt": g("totalDebt"), "fcf": g("freeCashflow"),
        "beta": g("beta"),
        "high_52w": g("fiftyTwoWeekHigh"), "low_52w": g("fiftyTwoWeekLow"),
        "target_mean": g("targetMeanPrice"), "target_high": g("targetHighPrice"), "target_low": g("targetLowPrice"),
        "analysts": int(g("numberOfAnalystOpinions") or 0), "recommendation": info.get("recommendationKey"),
    }
    try:
        f["quarterly"] = _statement(tk.quarterly_income_stmt)[-8:]
    except Exception as e:
        f["quarterly"] = []
        print(f"quarterly {ticker}: {e}", file=sys.stderr)
    try:
        f["annual"] = _statement(tk.income_stmt)[-5:]
    except Exception:
        f["annual"] = []
    f["rev_yoy_ttm"] = _yoy(f["quarterly"], "revenue")
    f["ni_yoy_ttm"] = _yoy(f["quarterly"], "net_income")
    if f["rev_yoy_ttm"] is None and len(f["annual"]) >= 2:
        a, b = f["annual"][-1]["revenue"], f["annual"][-2]["revenue"]
        f["rev_yoy_ttm"] = round((a - b) / abs(b) * 100, 1) if a is not None and b else None
    if f["ni_yoy_ttm"] is None and len(f["annual"]) >= 2:
        a, b = f["annual"][-1]["net_income"], f["annual"][-2]["net_income"]
        f["ni_yoy_ttm"] = round((a - b) / abs(b) * 100, 1) if a is not None and b else None
    try:
        cal = tk.calendar
        ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if ed:
            ed = ed[0] if isinstance(ed, (list, tuple)) else ed
            f["earnings_date"] = pd.Timestamp(ed).strftime("%Y-%m-%d")
    except Exception:
        pass
    return f


def refresh_fundamentals(tickers_all: list[str], priority: list[str], cache_path: Path, cfg: dict) -> dict:
    cache = load_json(cache_path, {"stocks": {}})
    stocks = cache.setdefault("stocks", {})
    max_age = cfg.get("fundamentals_max_age_days", 7)
    today = _now().date()

    def age(t):
        try:
            return (today - dt.date.fromisoformat(stocks[t]["fetched"])).days
        except Exception:
            return 9999

    stale = sorted([t for t in tickers_all if age(t) >= max_age], key=age, reverse=True)
    budget = len(stale) if len(stocks) < len(tickers_all) / 2 else cfg.get("fundamentals_per_run", 30)
    todo = [t for t in priority if age(t) >= 1] + [t for t in stale if t not in priority][:budget]
    for t in dict.fromkeys(todo):
        try:
            stocks[t] = fetch_fundamentals(t)
        except Exception as e:
            print(f"fundamentals {t}: {e}", file=sys.stderr)
        time.sleep(0 if os.environ.get("BEBAS_TEST") else 0.4)
    print(f"Fundamentals refreshed: {len(todo)} · cached: {len(stocks)}")
    return cache


def _band(v, steps):
    """steps: list of (threshold, points) checked in order with v >= threshold."""
    for th, pts in steps:
        if v >= th:
            return pts
    return 0


def score_fundamentals(f: dict, sector: str, sector_pe: float | None) -> dict:
    parts, strengths, concerns = {}, [], []
    fin = sector == "Financials"
    roe = f.get("roe") if f.get("roe") is not None else f.get("net_margin")
    if roe is not None:
        parts["Profitability"] = (_band(roe, [(15, 25), (10, 18), (5, 10), (0.01, 5)]), 25)
        if f.get("roe") is not None and f["roe"] >= 15:
            strengths.append(f"High return on equity ({f['roe']}%)")
        elif f.get("roe") is not None and f["roe"] < 5:
            concerns.append(f"Low return on equity ({f['roe']}%)")
    growth = [x for x in (f.get("ni_yoy_ttm") if f.get("ni_yoy_ttm") is not None else f.get("earn_growth_q"),
                          f.get("rev_yoy_ttm") if f.get("rev_yoy_ttm") is not None else f.get("rev_growth_q"))
              if x is not None]
    if growth:
        pts = sum(_band(g, [(20, 25), (10, 18), (0, 12), (-10, 6)]) for g in growth) / len(growth)
        parts["Growth"] = (pts, 25)
        ni = f.get("ni_yoy_ttm")
        if ni is not None and ni >= 15:
            strengths.append(f"Profit up {ni}% over the last 12 months")
        elif ni is not None and ni <= -10:
            concerns.append(f"Profit down {abs(ni)}% over the last 12 months")
    q = [x["net_income"] for x in f.get("quarterly", []) if x.get("net_income") is not None]
    if q and q[-1] < 0:
        concerns.append("Made a loss in the latest quarter")
    pe = f.get("pe")
    if pe is not None and pe > 0:
        rel = pe / sector_pe if sector_pe else None
        pts = _band(-(rel or 1.0), [(-0.8, 20), (-1.0, 15), (-1.3, 9)]) or 4
        parts["Valuation"] = (pts, 20)
        if rel and rel <= 0.8:
            strengths.append(f"Cheaper than sector peers (P/E {pe:.1f} vs {sector_pe:.1f})")
        elif rel and rel >= 1.4:
            concerns.append(f"Expensive vs sector peers (P/E {pe:.1f} vs {sector_pe:.1f})")
    elif pe is None and f.get("eps") is not None and f["eps"] < 0:
        parts["Valuation"] = (0, 20)
        concerns.append("Loss-making over the last 12 months (no P/E)")
    de = f.get("debt_to_equity")
    if fin:
        parts["Balance sheet"] = (10, 15)
    elif de is not None:
        parts["Balance sheet"] = (_band(-de, [(-0.5, 15), (-1.0, 10), (-2.0, 5)]), 15)
        if de <= 0.3:
            strengths.append(f"Low debt (debt/equity {de})")
        elif de > 1.5:
            concerns.append(f"High debt (debt/equity {de})")
    dy = f.get("dividend_yield")
    if dy is not None:
        parts["Dividend"] = (_band(dy, [(5, 10), (3, 7), (1.5, 4)]) or 1, 10)
        if dy >= 4:
            strengths.append(f"Dividend yield {dy}%")
    if f.get("target_mean") and f.get("price") and f.get("analysts", 0) >= 3:
        up = (f["target_mean"] / f["price"] - 1) * 100
        f["target_upside"] = round(up, 1)
        parts["Analysts"] = (_band(up, [(15, 5), (5, 3), (0, 1)]), 5)
        if up >= 15:
            strengths.append(f"{f['analysts']} analysts' average target is {up:.0f}% above the price")
        elif up < 0:
            concerns.append(f"Price is above the analysts' average target ({f['target_mean']:.2f})")
    if f.get("earnings_date"):
        try:
            days = (dt.date.fromisoformat(f["earnings_date"]) - _now().date()).days
            if 0 <= days <= 10:
                concerns.append(f"Quarterly results due {f['earnings_date']}: a gap risk for new trades")
        except ValueError:
            pass
    got = sum(p[1] for p in parts.values())
    score = round(sum(p[0] for p in parts.values()) / got * 100) if got >= 45 else None
    grade = None if score is None else "A" if score >= 75 else "B" if score >= 55 else "C" if score >= 35 else "D"
    return {"score": score, "grade": grade, "coverage": round(got),
            "parts": {k: [round(v[0], 1), v[1]] for k, v in parts.items()},
            "strengths": strengths, "concerns": concerns}


def score_all(cache: dict, sectors: dict) -> dict:
    pes: dict[str, list] = {}
    for t, f in cache.get("stocks", {}).items():
        if f.get("pe") and 0 < f["pe"] < 80:
            pes.setdefault(sectors.get(t, "Other"), []).append(f["pe"])
    med = {s: float(pd.Series(v).median()) for s, v in pes.items() if len(v) >= 2}
    for t, f in cache.get("stocks", {}).items():
        sec = sectors.get(t, "Other")
        f["sector"] = sec
        f["sector_pe"] = round(med[sec], 1) if sec in med else None
        f["score"] = score_fundamentals(f, sec, med.get(sec))
    cache["sector_pe"] = {k: round(v, 1) for k, v in med.items()}
    cache["updated"] = _now().strftime("%Y-%m-%d %H:%M MYT")
    return cache


# --------------------------------------------------------------------------
# News
# --------------------------------------------------------------------------

POS = r"\b(record|surge|soar|jump|rise|rises|rose|gain|gains|rally|win|wins|won|secure|secures|bags|contract|award|" \
      r"profit up|higher profit|beat|beats|upgrade|upgraded|buy call|outperform|expand|expansion|dividend|bonus|" \
      r"strong|robust|growth|boost|acquire|acquisition|order book|recovery|rebound)\b"
NEG = r"\b(plunge|plunges|slump|tumble|fall|falls|fell|drop|drops|decline|loss|losses|lower profit|miss|misses|" \
      r"downgrade|downgraded|sell call|underperform|probe|investigation|macc|lawsuit|sued|default|pn17|" \
      r"suspend|suspension|fraud|weak|warning|cut|cuts|delay|delays|terminate|terminated|fine|fined|resign)\b"


def tone(title: str) -> str:
    t = title.lower()
    p, n = len(re.findall(POS, t)), len(re.findall(NEG, t))
    return "positive" if p > n else "negative" if n > p else "neutral"


def google_news(query: str, days: int = 2, limit: int = 10) -> list[dict]:
    import requests
    try:
        r = requests.get("https://news.google.com/rss/search",
                         params={"q": f"{query} when:{days}d", "hl": "en-MY", "gl": "MY", "ceid": "MY:en"},
                         headers=UA, timeout=15)
        r.raise_for_status()
        return parse_rss(r.content, limit)
    except Exception as e:
        print(f"news '{query}': {e}", file=sys.stderr)
        return []


def parse_rss(xml_bytes: bytes, limit: int = 10) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    out = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        src_el = it.find("source")
        source = src_el.text.strip() if src_el is not None and src_el.text else ""
        if source and title.endswith(" - " + source):
            title = title[: -len(source) - 3]
        pub = it.findtext("pubDate")
        try:
            when = email.utils.parsedate_to_datetime(pub).astimezone(MYT).strftime("%Y-%m-%d %H:%M")
        except Exception:
            when = ""
        out.append({"title": title, "url": it.findtext("link"), "source": source,
                    "published": when, "tone": tone(title)})
        if len(out) >= limit:
            break
    return out


def yahoo_news(ticker: str, limit: int = 6) -> list[dict]:
    try:
        import yfinance as yf
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out = []
    for it in items[:limit]:
        c = it.get("content", it)
        title = c.get("title")
        if not title:
            continue
        url = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link")
        src = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else c.get("publisher")
        pub = c.get("pubDate") or c.get("providerPublishTime")
        try:
            when = (pd.Timestamp(pub, unit="s") if isinstance(pub, (int, float)) else pd.Timestamp(pub))
            when = when.tz_localize("UTC") if when.tzinfo is None else when
            when = when.tz_convert("Asia/Kuala_Lumpur").strftime("%Y-%m-%d %H:%M")
        except Exception:
            when = ""
        out.append({"title": title, "url": url, "source": src or "Yahoo Finance", "published": when,
                    "tone": tone(title)})
    return out


def _dedupe(items: list[dict], limit: int) -> list[dict]:
    seen, out = set(), []
    for it in sorted(items, key=lambda x: x.get("published") or "", reverse=True):
        key = re.sub(r"\W+", "", it["title"].lower())[:60]
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out[:limit]


def market_news() -> dict:
    items = []
    for q in ("Bursa Malaysia", "FBM KLCI", "Malaysia economy ringgit OPR"):
        items += google_news(q, days=2, limit=12)
    items = _dedupe(items, 25)
    pos = sum(i["tone"] == "positive" for i in items)
    neg = sum(i["tone"] == "negative" for i in items)
    return {"items": items, "positive": pos, "negative": neg,
            "tone": "positive" if pos > neg * 1.3 else "negative" if neg > pos * 1.3 else "mixed"}


INSIDER_RX = re.compile(r"(director|substantial shareholder|ceases|emerges|acquires|disposes|buy ?back|"
                        r"share buy|stake|private placement|rights issue|epf|kwap|pnb|khazanah|"
                        r"block trade|direct business|takeover|privati[sz]|uma|unusual market)", re.I)


def insider_news(ticker: str, name: str, long_name: str | None) -> list[dict]:
    """Director dealings, substantial shareholder changes, buybacks, placements, UMA queries."""
    base = f'"{long_name}"' if long_name else f'"{name}" Bursa'
    q = base + " (director OR \"substantial shareholder\" OR buyback OR stake OR placement OR EPF OR KWAP)"
    items = [i for i in google_news(q, days=45, limit=15) if INSIDER_RX.search(i["title"])]
    for i in items:
        t = i["title"].lower()
        i["kind"] = ("insider buy" if re.search(r"(acquires|buys|raises|ups|increase|emerges)", t) else
                     "insider sell" if re.search(r"(disposes|sells|cuts|trims|ceases|pares|reduce)", t) else "corporate")
    return _dedupe(items, 6)


def flow_news() -> list[dict]:
    items = []
    for q in ('Bursa "foreign investors" net', "MIDF fund flow Bursa", "Bursa local institutions net buy"):
        items += google_news(q, days=10, limit=8)
    return _dedupe(items, 10)


def stock_news(ticker: str, name: str, long_name: str | None) -> list[dict]:
    q = f'"{long_name}"' if long_name else f'"{name}" Bursa'
    return _dedupe(google_news(q, days=14, limit=8) + yahoo_news(ticker), 8)


# --------------------------------------------------------------------------
# AI deep analysis (optional, needs ANTHROPIC_API_KEY)
# --------------------------------------------------------------------------

PROMPT = """You are an equity research analyst covering Bursa Malaysia. Write a short research note on {name} ({code}.KL) for a retail swing trader.

Today is {today}. Use the data below. {search_line}
Never invent numbers. If something is not in the data or your search results, say it is not available.
Cite sources inline in brackets, e.g. (The Edge, Oct 2026). Plain English, under 600 words.

Use exactly these markdown headings:
## Verdict
One line: fundamentally Strong / Fair / Weak, and whether the fundamentals support or contradict the technical setup below.
## Business
## Latest results
## Valuation
## Prospects and catalysts
## Money flow and insider activity
Search for director dealings, substantial shareholder changes (EPF, KWAP, PNB, foreign funds), share buybacks,
private placements, block/direct business trades and Bursa UMA queries in the last 3 months. Say what they suggest.
## Moat and cycle
Does the business have a durable edge (licence, monopoly, scale, network, brand, cost)? Where is its industry
in the cycle (early recovery, expansion, peak, downturn)? Be specific and balanced.
## Key risks
## For this trade
Two or three bullets: what could make the trade fail (e.g. results date, sector news), and what to watch.

FUNDAMENTAL DATA (Yahoo Finance, may be incomplete):
{fund}

TECHNICAL SETUP FROM THE SCANNER:
{tech}

RECENT HEADLINES:
{news}
"""


def ai_note(ticker: str, name: str, fund: dict | None, tech: dict | None, news: list[dict], cfg: dict) -> dict | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or not cfg.get("enabled", True):
        return None
    import requests
    slim = {k: v for k, v in (fund or {}).items() if k not in ("summary", "score") and v not in (None, [], "")}
    if fund and fund.get("summary"):
        slim["business_summary"] = fund["summary"]
    use_search = cfg.get("web_search", True)
    prompt = PROMPT.format(
        name=name, code=ticker.replace(".KL", ""), today=_now().strftime("%d %b %Y"),
        search_line=("Use web search to check the latest quarterly results, Bursa announcements, contracts, "
                     "guidance and analyst views from the last 6 months." if use_search else ""),
        fund=json.dumps(slim, default=str)[:6000],
        tech=json.dumps(tech or {"note": "no active setup; general review"}, default=str),
        news="\n".join(f"- {n['published'][:10]} {n['source']}: {n['title']}" for n in news[:8]) or "none found")
    body = {"model": cfg.get("model", "claude-sonnet-5-5"), "max_tokens": cfg.get("max_tokens", 2000),
            "messages": [{"role": "user", "content": prompt}]}
    if use_search:
        body["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": cfg.get("max_searches", 4)}]
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=240)
        if not r.ok and "tools" in body:
            print(f"AI with search failed ({r.status_code}); retrying without search", file=sys.stderr)
            body.pop("tools")
            use_search = False
            r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=240)
        r.raise_for_status()
        blocks = r.json().get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        i = text.find("## Verdict")
        text = text[i:] if i > 0 else text
    except Exception as e:
        print(f"AI note {ticker}: {e}", file=sys.stderr)
        return None
    m = re.search(r"## Verdict\s*\n+(.+)", text)
    return {"date": _now().strftime("%Y-%m-%d"), "model": body["model"], "web_search": use_search,
            "verdict": m.group(1).strip()[:300] if m else "", "markdown": text}


BRIEF_PROMPT = """You are a Bursa Malaysia market strategist. Today is {today}. Use web search (sources from the
last 2 weeks; The Edge, The Star, NST, Bernama, MIDF/Kenanga/CGS/RHB/HLIB research) to write this week's brief
for a retail trader. Never invent numbers; cite sources inline as (Source, date).

Our scanner's own price-based theme ranking this week (leaders = stock codes):
{themes}

Reply with ONLY a JSON object, no markdown fences:
{{"headline": "one sentence on the market mood",
  "hot_themes": [{{"path": "Sector > Theme > Sub-theme", "why": "catalyst", "stocks": ["code Name", ...], "money": "who is buying: foreign / local institutions / retail"}}],
  "cooling_themes": [{{"path": "...", "why": "..."}}],
  "fund_flow": "latest weekly foreign / local institution / retail net flows with RM figures and the trend",
  "insider_moves": [{{"stock": "code Name", "what": "who bought/sold how much, date", "read": "what it may signal"}}],
  "catalysts": [{{"date": "YYYY-MM-DD or week", "event": "..."}}],
  "risks": ["..."],
  "sources": [{{"title": "...", "url": "..."}}]}}
"""


def ai_market_brief(theme_ranking: list[dict], cfg: dict) -> dict | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or not cfg.get("enabled", True):
        return None
    import requests
    top = "\n".join(f"- {t['name']} | {t['status']} | score {t['score']} | 20d {t['r20']}% | leaders {', '.join(t['leaders'])}"
                    for t in theme_ranking[:12])
    body = {"model": cfg.get("model", "claude-sonnet-5-5"), "max_tokens": 3000,
            "messages": [{"role": "user", "content": BRIEF_PROMPT.format(today=_now().strftime("%d %b %Y"), themes=top)}],
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": cfg.get("brief_searches", 8)}]}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=300)
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
        a, b = text.find("{"), text.rfind("}")
        data = json.loads(text[a:b + 1])
        data["date"] = _now().strftime("%Y-%m-%d")
        return data
    except Exception as e:
        print(f"AI market brief: {e}", file=sys.stderr)
        return None

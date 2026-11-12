"""Whole-market stock list for Bebas.

Instead of a hand-typed list, this finds every ordinary share listed on Bursa Malaysia (main, ACE
and LEAP markets) using Yahoo Finance's screener, drops warrants, preference shares and
illiquid names, and tags each stock with a sector.

It runs inside the daily scan on GitHub Actions. The result is cached in universe_all.json, so if
Yahoo's screener fails one day the last good list is reused (up to 30 days old).

Turn it on or off in config.json:  "wide": {"enabled": true, ...}
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "universe_all.json"

# Yahoo sector -> the sector names used in the app
SECTOR_MAP = {
    "Financial Services": "Financials",
    "Utilities": "Utilities",
    "Energy": "Energy",
    "Technology": "Technology",
    "Healthcare": "Healthcare",
    "Real Estate": "Property",
    "Industrials": "Industrial",
    "Basic Materials": "Industrial",
    "Consumer Cyclical": "Consumer",
    "Consumer Defensive": "Consumer",
    "Communication Services": "Telco & Media",
}
# Yahoo industries that are better shown as one of our narrower sectors
INDUSTRY_MAP = {
    "Engineering & Construction": "Construction",
    "Infrastructure Operations": "Construction",
    "Building Products & Equipment": "Construction",
    "Farm Products": "Plantation",
    "Agricultural Inputs": "Plantation",
}

SYMBOL_RX = re.compile(r"^\d{4}[A-Z]?\.KL$")                 # 1155.KL, 5227.KL; not 5099WA.KL (warrants)
NAME_SKIP_RX = re.compile(r"(-[A-Z]{1,2}\d?\s*$)|WARRANT|ICULS|\bETF\b|\bRCPS\b|\bLOAN STOCK\b", re.I)


def _query(*conds):
    from yfinance import EquityQuery
    return EquityQuery("and", [EquityQuery("eq", ["exchange", "KLS"])] + [EquityQuery(*c) for c in conds])


def _pages(query, max_pages: int = 8) -> list[dict]:
    """All results of a Yahoo screener query (250 per page)."""
    import yfinance as yf
    out: list[dict] = []
    for page in range(max_pages):
        for attempt in range(3):
            try:
                r = yf.screen(query, offset=page * 250, size=250, sortField="intradaymarketcap", sortAsc=False)
                break
            except Exception as e:
                print(f"universe: screener retry {attempt + 1}: {e}", file=sys.stderr)
                time.sleep(3 * (attempt + 1))
        else:
            break
        quotes = (r or {}).get("quotes") or []
        out += quotes
        if len(quotes) < 250:
            break
        time.sleep(1)
    return out


def parse_quotes(quotes: list[dict], sector_of: dict[str, str] | None = None) -> dict[str, dict]:
    """Turn raw screener rows into {ticker: {name, price, value_m, sector}}; drops non-ordinary shares."""
    out: dict[str, dict] = {}
    for q in quotes:
        sym = str(q.get("symbol", "")).upper()
        if not SYMBOL_RX.match(sym):
            continue
        name = (q.get("shortName") or q.get("longName") or sym).strip()
        if NAME_SKIP_RX.search(name):
            continue
        price = q.get("regularMarketPrice")
        vol = q.get("averageDailyVolume3Month") or q.get("averageDailyVolume10Day") or 0
        if not price or price <= 0:
            continue
        out[sym] = {"name": name, "price": float(price), "value_m": round(float(price) * float(vol) / 1e6, 2),
                    "sector": (sector_of or {}).get(sym, "Other")}
    return out


def _discover_live() -> dict[str, dict]:
    base = parse_quotes(_pages(_query()))
    if not base:
        return {}
    sector_of: dict[str, str] = {}
    for ysec, ours in SECTOR_MAP.items():                      # tag every stock with a sector
        for q in _pages(_query(("eq", ["sector", ysec]))):
            sector_of[str(q.get("symbol", "")).upper()] = ours
    for ind, ours in INDUSTRY_MAP.items():                      # narrower industries override
        for q in _pages(_query(("eq", ["industry", ind])), max_pages=2):
            sector_of[str(q.get("symbol", "")).upper()] = ours
    for sym, v in base.items():
        v["sector"] = sector_of.get(sym, "Other")
    return base


def _load_cache(max_age_days: int = 30) -> dict[str, dict]:
    try:
        c = json.loads(CACHE.read_text(encoding="utf-8"))
        age = (dt.date.today() - dt.date.fromisoformat(c["updated"])).days
        return c["stocks"] if age <= max_age_days else {}
    except Exception:
        return {}


def discover(wcfg: dict, min_value_rm: float) -> dict[str, dict]:
    """The whole-market list after filters, newest data first, cache as the fallback."""
    stocks = _discover_live()
    if len(stocks) >= wcfg.get("min_expected", 300):
        CACHE.write_text(json.dumps({"updated": dt.date.today().isoformat(), "stocks": stocks},
                                    ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"universe: Yahoo screener returned {len(stocks)} ordinary shares")
    else:
        print(f"universe: screener gave only {len(stocks)}; using the cached list", file=sys.stderr)
        stocks = _load_cache()
    min_price = float(wcfg.get("min_price", 0.05))
    min_val = float(wcfg.get("min_avg_value_rm", min_value_rm)) / 1e6
    keep = {t: v for t, v in stocks.items() if v["price"] >= min_price and v["value_m"] >= min_val}
    ranked = sorted(keep.items(), key=lambda kv: kv[1]["value_m"], reverse=True)[: int(wcfg.get("max_stocks", 800))]
    print(f"universe: {len(ranked)} stocks pass price >= RM{min_price} and value >= RM{min_val}m a day")
    return dict(ranked)


def apply(cfg: dict) -> int:
    """Add the whole-market list to cfg['universe'] / cfg['sectors']. Returns how many stocks were added."""
    wcfg = cfg.get("wide") or {}
    cfg["_core"] = list(cfg["universe"])                        # the hand-picked stocks keep their names, sectors, themes
    if not wcfg.get("enabled"):
        return 0
    try:
        found = discover(wcfg, float(cfg["strategy"].get("min_avg_value_rm", 1_000_000)))
    except Exception as e:                                       # never break the daily scan over this
        print(f"universe: whole-market list failed ({e}); scanning your own list only", file=sys.stderr)
        return 0
    added = 0
    for t, v in found.items():
        if t not in cfg["universe"]:
            cfg["universe"][t] = v["name"].title() if v["name"].isupper() else v["name"]
            added += 1
        cfg.setdefault("sectors", {}).setdefault(t, v["sector"])
    return added

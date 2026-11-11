"""Live lookup API for Bebas.

Runs the same analysis as lookup.py, on demand, and returns it as JSON.
Deploy it on Render (see render.yaml) or run it on your own computer:

    pip install -r requirements-server.txt
    uvicorn server:app --port 8000

GET /lookup?code=1155   a stock code, or a warrant code from config.json
GET /health             the app pings this to wake the server up

Environment variables (all optional):
    SITE_URL       your GitHub Pages address, e.g. https://you.github.io/bebas
                   The server reads the daily scan's signals.json, fundamentals.json and
                   config.json from there, so it always uses the latest data and your latest
                   config.json edits, with no redeploy.
    LOOKUP_KEY     if set, requests must send this key (header X-Key, or ?key=)
    ALLOW_ORIGIN   comma-separated sites allowed to call the API (default: any)
    CACHE_MINUTES  how long a result is reused (default 15)
"""
from __future__ import annotations

import copy
import datetime as dt
import hmac
import json
import os
import re
import threading
import time

import requests
from fastapi import FastAPI, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import lookup as LK
import research as R
import scan as SC

SITE_URL = os.environ.get("SITE_URL", "").rstrip("/")
LOOKUP_KEY = os.environ.get("LOOKUP_KEY", "")
ALLOW_ORIGIN = [o.strip() for o in os.environ.get("ALLOW_ORIGIN", "*").split(",") if o.strip()]
CACHE_SECONDS = int(float(os.environ.get("CACHE_MINUTES", "15")) * 60)
CODE_RX = re.compile(r"^[A-Z0-9][A-Z0-9-]{2,15}$")

_lock = threading.Lock()                       # one analysis at a time: it edits shared config
_remote_cache: dict[str, tuple[float, object]] = {}
_results: dict[str, tuple[float, dict]] = {}
_fund_extra: dict[str, dict] = {}              # fundamentals fetched live for stocks outside the daily scan


def _remote(name: str, ttl: int, default):
    """A JSON file from your GitHub Pages site (cached), falling back to the copy deployed with the server."""
    now = time.time()
    hit = _remote_cache.get(name)
    if hit and now - hit[0] < ttl:
        return hit[1]
    data, ok = None, False
    if SITE_URL:
        try:
            r = requests.get(f"{SITE_URL}/{name}", params={"t": int(now)}, timeout=20)
            if r.ok:
                data, ok = r.json(), True
        except Exception as e:
            print(f"could not fetch {name} from {SITE_URL}: {e}")
    if data is None:
        data = R.load_json(LK.ROOT / name, default)
    # a fallback is only remembered for a minute, so recovery is quick
    _remote_cache[name] = (now if ok else now - ttl + 60, data)
    return data


def _sync_config() -> None:
    """Apply your latest config.json (warrants, prices, moats, account) without redeploying."""
    remote = _remote("config.json", 600, None)
    if not isinstance(remote, dict):
        return
    for key in ("strategy", "account"):          # S and ACC in scan.py point into these dicts
        if isinstance(remote.get(key), dict):
            SC.CFG[key].clear()
            SC.CFG[key].update(remote[key])
    for key in ("universe", "sectors", "warrants", "moats", "research", "alerts", "themes"):
        if key in remote:
            SC.CFG[key] = remote[key]


def _age_days(f: dict) -> int:
    try:
        return (dt.date.today() - dt.date.fromisoformat(f["fetched"])).days
    except Exception:
        return 9999


def _fund_provider(tickers: list[str]) -> dict:
    fc = copy.deepcopy(_remote("fundamentals.json", 6 * 3600, {"stocks": {}}))
    stocks = fc.setdefault("stocks", {})
    for t in tickers:
        if t in stocks and _age_days(stocks[t]) < 7:
            continue
        extra = _fund_extra.get(t)
        if extra and _age_days(extra) < 1:
            stocks[t] = copy.deepcopy(extra)
            continue
        try:
            stocks[t] = R.fetch_fundamentals(t)
            _fund_extra[t] = stocks[t]
        except Exception as e:
            print(f"fundamentals {t}: {e}")
    return R.score_all(fc, SC.CFG.get("sectors", {})).get("stocks", {})


def _analyse(code: str, refresh: bool) -> tuple[dict, bool]:
    now = time.time()
    hit = _results.get(code)
    if hit and now - hit[0] < (60 if refresh else CACHE_SECONDS):   # even "refresh" waits a minute
        return hit[1], True
    with _lock:
        hit = _results.get(code)                 # another request may have just finished it
        if hit and time.time() - hit[0] < (60 if refresh else CACHE_SECONDS):
            return hit[1], True
        _sync_config()
        sig = _remote("signals.json", 1800, {})
        out, errors = LK.analyse_codes([code], sig, _fund_provider)
        if not out:
            raise LookupError(errors.get(code, "no result"))
        item = json.loads(json.dumps(next(iter(out.values())), default=LK._j))
        _results[code] = (time.time(), item)
        return item, False


app = FastAPI(title="Bebas live lookup", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origins=ALLOW_ORIGIN, allow_methods=["GET"], allow_headers=["*"])


@app.get("/")
def root():
    return {"service": "Bebas live lookup", "use": "/lookup?code=1155", "key_required": bool(LOOKUP_KEY)}


@app.get("/health")
def health():
    return {"ok": True, "key_required": bool(LOOKUP_KEY)}


@app.get("/lookup")
def lookup(code: str = Query(..., max_length=24), refresh: int = 0, key: str = "",
           x_key: str | None = Header(default=None)):
    given = x_key or key
    if LOOKUP_KEY and not (given and hmac.compare_digest(given, LOOKUP_KEY)):
        return JSONResponse({"error": "Wrong or missing access key. Enter the key from your Render environment settings."},
                            status_code=401)
    code = LK._code(code)
    if not CODE_RX.match(code):
        return JSONResponse({"error": "That is not a valid stock or warrant code."}, status_code=422)
    try:
        item, cached = _analyse(code, bool(refresh))
    except LookupError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:                       # network trouble, Yahoo changes, etc.
        print(f"lookup {code} failed: {e!r}")
        return JSONResponse({"error": f"Analysis failed: {e}"}, status_code=502)
    return {"item": item, "cached": cached}

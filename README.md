# Bebas: a Bursa trading coach

A free, self-hosted app that:

- scans ~75 liquid Bursa stocks every weekday at 6:20 pm with **11 strategies in 4 families**:
  - **Candlestick:** engulfing, hammer, morning star, piercing, three white soldiers, harami (only at a pullback or oversold level)
  - **Breakout:** 20-day volume breakout, tight-base breakout
  - **Swing:** pullback-reversal in an uptrend
  - **Volume:** volume surge, pocket pivot, plus accumulation/distribution (OBV, up/down volume) on every signal
- **backtests every strategy on 3 years of Bursa data** and only alerts you about the ones that made money after costs,
- shows a **market sentiment overview**: 0–100 fear/greed gauge, breadth, 52-week highs vs lows, KLCI trend, sector rotation, top movers and unusual volume,
- collects **breaking news** for the market and for each signal stock (Google News + Yahoo Finance),
- grades every company's **fundamentals** (profitability, growth, valuation vs sector, debt, dividend, analyst targets, quarterly revenue/profit) and warns you when results are due,
- writes an optional **AI research note** (business, latest results, valuation, prospects, risks) for alerted stocks, or any stock you ask for,
- sends you a **Telegram alert** with entry, stop, target, lot size, fundamentals, headlines and the matching call-warrant plan,
- warns you when a stock you hold should be sold, have its stop raised, or shows selling pressure,
- maps **themes and hot money**: sector › theme › sub-theme › stock (e.g. *Technology › AI & semiconductors › Data storage › JCY*), scored daily on strength vs the KLCI, volume expansion, breadth and money flow, with an optional AI market brief on fund flows, insider moves and catalysts,
- builds a **playbook for four holding periods** (contra T+2, up to 1 month, 1–3 months, long term) with rules, a 3-year backtest of each, today's top 3 picks with entry/stop/target/exit plan, money flow (CMF, MFI, accumulation days), insider and institutional news, and call-warrant guidance,
- includes stock and warrant calculators, 16 short lessons, a financial-freedom planner and a trade journal.

Everything runs on GitHub for free: GitHub Actions runs the scanner, GitHub Pages hosts the app.

> Educational tool, not financial advice. A backtest shows what *would have* happened. Markets change.

---

## Setup (about 15 minutes, works from a phone browser)

### 1. Put the files on GitHub
1. Create a new **public** repository, e.g. `bebas`.
2. Upload every file from this folder, keeping `.github/workflows/scan.yml` in that exact path.

### 2. Create your Telegram bot
1. In Telegram, open **@BotFather** → send `/newbot` → pick a name. Copy the **token** it gives you.
2. Open your new bot and send it any message (e.g. `hi`).
3. In a browser open `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` and find `"chat":{"id":123456789` . That number is your **chat ID**.

### 3. Add the secrets to GitHub
Repo → **Settings → Secrets and variables → Actions → New repository secret**:

| Name | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | the token from BotFather |
| `TELEGRAM_CHAT_ID` | your chat ID |
| `ANTHROPIC_API_KEY` | *optional*: turns on AI research notes (see below) |

### 4. Turn on the web app
Repo → **Settings → Pages** → Source: *Deploy from a branch* → Branch `main`, folder `/ (root)` → Save.
Your app will be at `https://<your-username>.github.io/bebas/`.

### 5. Run the first scan
Repo → **Actions** → *Daily scan* → **Run workflow**. The first run takes 5 to 8 minutes because it downloads fundamentals for every stock; later runs take 2 to 4. You should get a Telegram message, and the app will show real data. After that it runs automatically every weekday at 6:20 pm.

---

## Daily use

1. Read the Telegram alert after 6:20 pm.
2. Open the app, check the chart for each A/B setup, and check for upcoming results or news.
3. Place a **buy-stop** order at the trigger price for the next morning. Cancel if it doesn't fill within 2 days.
4. When filled, set the stop-loss and target. Log the trade in the Journal tab.
5. Add the position to `holdings` in `config.json` (below) so the scanner watches it for exits.

## AI research notes (optional)

1. Create an API key at <https://console.anthropic.com> (pay-as-you-go; each note with web search costs a few US cents).
2. Add it as the `ANTHROPIC_API_KEY` secret.

Notes are written automatically for alerted stocks and your holdings (max 5 a day, refreshed weekly). To research **any** stock on demand: **Actions → Daily scan → Run workflow** → type the code(s) in *Research a stock* (e.g. `1155` or `1155,5347`). The note appears in the app's Research tab and Telegram tells you when it's ready.

The AI uses web search for the latest results, announcements and analyst views and cites its sources. It can still be wrong: check key facts in the company's Bursa announcements.

## Analyse any stock or warrant (Lookup tab)

1. Repo → **Actions → Daily scan → Run workflow**, type codes in the box (e.g. `1155,5347` or `1155,YTLPOWR-C2C`), then run.
2. Open the **Lookup** tab and type the code. Each analysis includes:
   - the four holding periods (contra T+2, up to 1 month, 1–3 months, long term), each with a 0–100 confidence score, a verdict, whether it has triggered, and entry, stop, target, exit and size,
   - technicals, money flow and accumulation score, fundamentals, moat, themes, news and insider headlines,
   - the stock's own backtest of each rule, and call-warrant economics for any warrant listed in `config.json`.
3. A warrant must be in the `warrants` list in `config.json` first (mother, exercise, ratio, expiry).

The score is a weighted sum of the parts shown on each card (technical rules met, money flow, fundamentals, moat, own backtest, news and insiders, market regime). Missing data counts as neutral, never as a bonus. A verdict of "Good set-up, not triggered yet" means wait for the trigger price; it is not a buy.

The app is a static page on GitHub Pages, so it cannot fetch prices itself. Each analysis is stored in `lookup.json` and kept until you overwrite it with a new run.

## Themes, playbooks and moats

- `themes` in config.json is the theme map. Add sub-themes or stocks as new stories appear (every stock must also be in `universe`).
- `moats` holds a one-line description of each company's durable edge, used by the long-term playbook. It's editorial: review and edit it.
- The long-term playbook's backtest is price-only (buying near a rising 200-day EMA), because past fundamentals aren't stored.

## Settings: `config.json`

Edit it on GitHub (pencil icon) and commit.

- `account`: default capital, risk per trade (%), max % of capital in one stock. The app has its own ⚙︎ setting per device.
- `strategy`: reward target in R, max holding days, stop buffer, costs, liquidity filters.
- `alerts.min_trades_for_edge` / `min_expectancy_r`: how strong a pattern's backtest must be to trigger an alert. `min_score`: minimum setup score.
- `alerts.min_fundamental_score`: set e.g. `50` to only get alerts for companies with decent fundamentals (0 = off).
- `universe`: the stocks to scan (`"CODE.KL": "Name"`). Add or remove freely, and add the stock to `sectors` too.
- `research`: how often fundamentals refresh, news on/off, and AI settings (`model`, `web_search`, `max_stocks_per_run`, `always`: codes to always keep a fresh note on).

### Holdings you want exit alerts for
```json
"holdings": [
  {"ticker": "1155.KL", "entry": 9.80, "stop": 9.42, "target": 10.56, "shares": 1000}
]
```
You'll get alerts for: stop hit, target reached, bearish candle, and when to raise your stop (3-ATR trailing).

### Call warrants
Copy the terms from Bursa / the issuer's term sheet:
```json
"warrants": [
  {"code": "1155CX", "mother": "1155.KL", "exercise": 10.50, "ratio": 5, "expiry": "2027-06-30"}
]
```
- `ratio` = number of warrants for 1 share.
- `yahoo` (optional) = the Yahoo code if it differs from `code` + `.KL` (Yahoo uses the 6-character Bursa code, e.g. `53985C.KL`).
- The config ships with five examples taken from KLSE Screener on 3 Nov 2026 (DIALOG-C3D, SUNCON-C70, GAMUDA-C5C, YTLPOWR-C2C, TENAGA-C4E). Check the terms on the issuer's term sheet and replace them as warrants expire.
- The scanner tries to fetch the warrant's price from Yahoo (`1155CX.KL`). If Yahoo doesn't have it, add `"price": 0.12` and update it yourself, or the app will show a *theoretical* price from the share's volatility.
- When the mother share gives a signal, the alert includes the warrant's estimated price at your stop and target, effective gearing, premium and days left.

## How the strategy works

| Step | Rule |
|---|---|
| Candlestick context | Uptrend pullback to 20-day EMA (RSI 35–62), or RSI < 35 near the 60-day low |
| Breakout | Close above 20-day high on ≥1.5x volume in an uptrend; or above a tight 30-day base (≤15% range) |
| Swing | Uptrend, 3–4 day dip to the 20-day EMA, then a close above the prior day's high |
| Volume | ≥2.5x volume +3% day; or pocket pivot (up-day volume beats every down day of the last 10) |
| Confluence | When several strategies fire on one stock, you get one card with a higher score |
| Liquidity | ≥ RM1m average daily value, price ≥ RM0.20 |
| Entry | Buy-stop 1 tick above the pattern high, valid 2 days |
| Stop | Pattern low − 0.2 × ATR (skip if wider than 12%) |
| Target | 2R; time exit after 15 days |
| Size | 1% of capital at risk, whole 100-share lots, max 25% in one stock; halved in weak markets |
| Alert | Strategy has ≥ 30 backtested trades and ≥ +0.10R average, and setup score ≥ 60 |

**Sentiment score (0–100)** = 25% stocks above 50-day EMA + 15% above 200-day EMA + 15% new highs vs lows + 15% advancers vs decliners + 20% KLCI trend + 10% 20-day momentum.

**Fundamental score (0–100)**: profitability (ROE) 25, growth (12-month profit and revenue) 25, valuation (P/E vs sector median) 20, balance sheet (debt/equity; banks neutral) 15, dividend yield 10, analyst target upside 5. Missing data is left out rather than guessed; stocks with too little data are not graded. Source: Yahoo Finance, which covers big Bursa names well and small caps less so.

Backtest limits to keep in mind: the stock list is today's (delisted names are missing), prices are dividend-adjusted, and 3 years is a short sample. Use your own journal results as the final judge.

## Run it on your PC (optional)
```
pip install -r requirements.txt
python scan.py --no-send          # add --no-research to skip fundamentals/news/AI
```
then open `index.html` through a local server (`python -m http.server`).

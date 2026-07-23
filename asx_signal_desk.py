#!/usr/bin/env python3
"""
ASX Signal Desk — market scanner, AI analyst, and paper-trading ledger.

What it does each scan:
  1. Pulls 3 months of real price/volume data for a watchlist of ASX stocks (yfinance)
  2. Computes momentum, RSI, volume spikes; shortlists the most active setups
  3. Sends the numbers to an LLM analyst for BUY/SELL calls with reasoning + confidence
  4. Logs every recommendation to SQLite
  5. Paper-trades high-confidence calls, closes them on target/stop, tracks P/L
  6. Regenerates dashboard.html with all signals, positions and results

Setup:
    pip install yfinance pandas requests
    export GROQ_API_KEY=gsk_...              (free tier — console.groq.com)
  or
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...      (console.anthropic.com)

Usage:
    python asx_signal_desk.py scan             # one full scan (run this on a schedule)
    python asx_signal_desk.py report           # rebuild dashboard from the database only
    python asx_signal_desk.py demo             # try the pipeline with synthetic data, no API key needed
    python asx_signal_desk.py scan --loop 3600 # keep scanning every hour

Set DESK_DEBUG=1 to dump raw API responses when something goes wrong.

This is a research tool, not financial advice. Paper positions are hypothetical.
"""

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------- configuration

MODEL = "claude-sonnet-4-6"          # used if you set ANTHROPIC_API_KEY
GROQ_MODEL = "openai/gpt-oss-120b"   # used if you set GROQ_API_KEY (free tier)

# gpt-oss models are REASONING models: the token budget below is shared between
# the hidden reasoning trace and the visible answer. Too small a budget and the
# model burns the lot thinking, returning an empty `content` with
# finish_reason="length". 2000 was not enough for a 10-stock table.
GROQ_MAX_TOKENS = 8000
GROQ_REASONING_EFFORT = "low"        # low | medium | high  (gpt-oss only)

DB_PATH = "asx_desk.db"
DASHBOARD_PATH = "docs/index.html"
DEBUG = bool(os.environ.get("DESK_DEBUG"))

# GitHub runners are UTC. Stamp the dashboard in local market time instead.
LOCAL_TZ = ZoneInfo(os.environ.get("DESK_TZ", "Australia/Perth"))
NEW_TICKER_DAYS = 14   # how long a ticker wears the NEW badge after first appearing

# Liquid ASX names across sectors. Add your own — always use the .AX suffix.
# (ALU.AX and GOR.AX removed: both taken over and delisted, Yahoo returns nothing.)
WATCHLIST = [
    "BHP.AX", "RIO.AX", "FMG.AX", "S32.AX", "MIN.AX", "PLS.AX", "LYC.AX",
    "NST.AX", "EVN.AX", "WDS.AX", "STO.AX", "BPT.AX", "PDN.AX",
    "BOE.AX", "CBA.AX", "NAB.AX", "WBC.AX", "ANZ.AX", "MQG.AX", "GQG.AX",
    "CSL.AX", "RMD.AX", "PME.AX", "TLX.AX", "NEU.AX", "WTC.AX", "XRO.AX",
    "NXT.AX", "DTL.AX", "WOW.AX", "COL.AX", "JBH.AX", "LOV.AX",
    "TPW.AX", "QAN.AX", "WEB.AX", "FLT.AX", "A2M.AX",
]

SHORTLIST_SIZE = 10        # candidates sent to the AI analyst per scan
MAX_PICKS = 4              # recommendations requested per scan
PAPER_TRADE_MIN_CONF = 65  # auto-open a paper position at/above this confidence
POSITION_VALUE_AUD = 5000  # notional AUD per paper position

# ---------------------------------------------------------------- database

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    ticker TEXT NOT NULL,
    name TEXT,
    action TEXT NOT NULL,
    confidence INTEGER,
    reasoning TEXT,
    price_at_scan REAL,
    entry REAL, target REAL, stop REAL
);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY,
    signal_id INTEGER REFERENCES signals(id),
    ticker TEXT NOT NULL,
    action TEXT NOT NULL,
    units INTEGER,
    entry REAL, target REAL, stop REAL,
    opened_ts TEXT,
    last_price REAL,
    closed_ts TEXT,
    exit_price REAL,
    pl REAL,
    close_reason TEXT
);
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    universe INTEGER,
    shortlisted INTEGER,
    picks INTEGER,
    analyst_ok INTEGER
);
CREATE TABLE IF NOT EXISTS scan_metrics (
    scan_id INTEGER REFERENCES scans(id),
    ticker TEXT NOT NULL,
    rank INTEGER,
    score REAL,
    price REAL,
    chg_1d REAL, chg_5d REAL, chg_20d REAL,
    rsi14 REAL, vol_x_avg REAL, hi_60d_pct REAL,
    shortlisted INTEGER DEFAULT 0,
    picked TEXT
);
CREATE INDEX IF NOT EXISTS idx_scan_metrics_scan ON scan_metrics(scan_id);
CREATE TABLE IF NOT EXISTS universe (
    ticker TEXT PRIMARY KEY,
    first_seen TEXT,
    last_seen TEXT,
    seeded INTEGER DEFAULT 0
);
"""


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- market data


def fetch_metrics(tickers):
    """Real OHLCV from Yahoo Finance -> per-ticker metric dict."""
    import yfinance as yf

    print(f"Fetching {len(tickers)} tickers from Yahoo Finance...", flush=True)
    data = yf.download(
        tickers, period="3mo", interval="1d",
        group_by="ticker", auto_adjust=True, progress=False, threads=True,
    )
    metrics, skipped = {}, []
    for t in tickers:
        try:
            df = data[t].dropna() if len(tickers) > 1 else data.dropna()
            if len(df) < 30:
                skipped.append(f"{t}(only {len(df)} bars)")
                continue
            close, vol = df["Close"], df["Volume"]
            last = float(close.iloc[-1])
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1e-9)
            rsi = float((100 - 100 / (1 + rs)).iloc[-1])
            metrics[t] = {
                "price": round(last, 3),
                "chg_1d": round(100 * (last / float(close.iloc[-2]) - 1), 2),
                "chg_5d": round(100 * (last / float(close.iloc[-6]) - 1), 2),
                "chg_20d": round(100 * (last / float(close.iloc[-21]) - 1), 2),
                "rsi14": round(rsi, 1),
                "vol_x_avg": round(float(vol.iloc[-1]) / max(float(vol.iloc[-21:-1].mean()), 1), 2),
                "hi_60d_pct": round(100 * (last / float(close.max()) - 1), 2),
            }
        except Exception as e:
            skipped.append(f"{t}({type(e).__name__})")
            continue
    print(f"Got usable data for {len(metrics)} tickers.", flush=True)
    if skipped:
        print(f"Skipped: {', '.join(skipped)}", flush=True)
    return metrics


def fake_metrics(tickers):
    """Synthetic metrics for demo mode."""
    out = {}
    for t in tickers:
        base = random.uniform(0.5, 120)
        out[t] = {
            "price": round(base, 2),
            "chg_1d": round(random.uniform(-6, 6), 2),
            "chg_5d": round(random.uniform(-12, 12), 2),
            "chg_20d": round(random.uniform(-25, 25), 2),
            "rsi14": round(random.uniform(18, 85), 1),
            "vol_x_avg": round(random.uniform(0.4, 4.5), 2),
            "hi_60d_pct": round(random.uniform(-30, 0), 2),
        }
    return out


def score_metrics(m):
    """'Interestingness': momentum + volume spike + RSI extremes."""
    s = abs(m["chg_5d"]) * 1.5 + abs(m["chg_1d"]) * 2 + max(m["vol_x_avg"] - 1, 0) * 10
    if m["rsi14"] < 35:
        s += (35 - m["rsi14"]) * 0.5   # oversold — bounce candidate
    elif m["rsi14"] > 70:
        s += (m["rsi14"] - 70) * 0.5   # overbought — exhaustion candidate
    return round(s, 2)


def setup_tag(m):
    """Plain-English label for why a name scored where it did."""
    tags = []
    if m["rsi14"] < 30:
        tags.append("deeply oversold")
    elif m["rsi14"] < 40:
        tags.append("oversold")
    elif m["rsi14"] > 75:
        tags.append("very overbought")
    elif m["rsi14"] > 70:
        tags.append("overbought")
    if m["vol_x_avg"] >= 2.5:
        tags.append("heavy volume")
    elif m["vol_x_avg"] >= 1.6:
        tags.append("volume spike")
    if m["chg_5d"] >= 8:
        tags.append("strong 5d run")
    elif m["chg_5d"] <= -8:
        tags.append("sharp 5d fall")
    if m["hi_60d_pct"] > -2:
        tags.append("at 60d high")
    return ", ".join(tags) or "quiet"


def rank_all(metrics):
    """Every scanned ticker, best setup first."""
    return sorted(
        ((t, m, score_metrics(m)) for t, m in metrics.items()),
        key=lambda r: r[2], reverse=True,
    )


def shortlist(metrics, n):
    """Top n by score, as a ticker -> metrics dict."""
    return {t: m for t, m, _ in rank_all(metrics)[:n]}


# ---------------------------------------------------------------- AI analyst


def build_prompt(candidates):
    table = "\n".join(
        f"{t}: price ${m['price']} | 1d {m['chg_1d']}% | 5d {m['chg_5d']}% | "
        f"20d {m['chg_20d']}% | RSI14 {m['rsi14']} | vol {m['vol_x_avg']}x avg | "
        f"{m['hi_60d_pct']}% off 60d high"
        for t, m in candidates.items()
    )
    return f"""You are a speculative short-term equities trader on the ASX. Today is {datetime.now().strftime('%A %d %B %Y')}.
Below are live technical metrics for the most active names on my watchlist. Pick the {MAX_PICKS} best risk/reward trades (long or short) purely from these numbers — momentum continuation, oversold bounces, blow-off exhaustion, volume confirmation.

{table}

Respond with ONLY minified JSON, no fences, no commentary:
{{"picks":[{{"ticker":"BHP.AX","name":"BHP Group","action":"BUY","confidence":72,"reasoning":"max 25 words citing the specific numbers","entry":48.20,"target":52.10,"stop":46.25}}]}}

Rules: ticker must be one of the tickers listed above, exactly as written. action is BUY or SELL (SELL = short). confidence integer 0-100. entry = the current price shown above. target and stop must be consistent with the direction and imply a sensible risk/reward (roughly 2:1). Be selective — mediocre setups get confidence below 50. Do not think for long; the analysis is straightforward."""


def ask_analyst(candidates):
    prompt = build_prompt(candidates)

    if os.environ.get("GROQ_API_KEY"):
        text = _ask_groq(prompt)
    elif os.environ.get("ANTHROPIC_API_KEY"):
        text = _ask_anthropic(prompt)
    else:
        raise RuntimeError("Set GROQ_API_KEY or ANTHROPIC_API_KEY.")

    payload = _extract_json(text)
    if payload is None:
        raise RuntimeError(f"Analyst reply had no parseable JSON. Raw reply ({len(text)} chars): {text[:500]!r}")
    raw_picks = payload.get("picks")
    if not isinstance(raw_picks, list):
        raise RuntimeError(f"Analyst JSON had no 'picks' list: {str(payload)[:300]}")
    return _clean_picks(raw_picks, candidates)


def _extract_json(text):
    """Tolerant JSON extraction: handles fences, preamble, and trailing prose."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    # gpt-oss sometimes leaks harmony control tokens
    text = text.replace("<|return|>", "").replace("<|end|>", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Brace matching — find the first balanced {...} block that parses.
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        start = None
    return None


def _num(v):
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _clean_picks(raw_picks, candidates):
    """Validate and repair whatever the model returned.

    Never trust the model's numbers: a literal 0 for entry (copied from the
    example schema) used to blow up position sizing with ZeroDivisionError.
    """
    picks = []
    for p in raw_picks:
        if not isinstance(p, dict):
            continue
        t = str(p.get("ticker", "")).upper().strip()
        if t and not t.endswith(".AX"):
            t += ".AX"
        if t not in candidates:
            print(f"  Ignoring pick '{p.get('ticker')}' — not in this scan's shortlist.", flush=True)
            continue
        action = str(p.get("action", "")).upper().strip()
        if action not in ("BUY", "SELL"):
            print(f"  Ignoring {t} — bad action {p.get('action')!r}.", flush=True)
            continue

        price = float(candidates[t]["price"])
        entry = _num(p.get("entry")) or price
        target = _num(p.get("target"))
        stop = _num(p.get("stop"))

        # Force target/stop onto the correct side of entry.
        if action == "BUY":
            if target is None or target <= entry:
                target = round(entry * 1.08, 3)
            if stop is None or stop >= entry:
                stop = round(entry * 0.96, 3)
        else:
            if target is None or target >= entry:
                target = round(entry * 0.92, 3)
            if stop is None or stop <= entry:
                stop = round(entry * 1.04, 3)

        try:
            conf = int(float(p.get("confidence", 0)))
        except (TypeError, ValueError):
            conf = 0
        conf = max(0, min(100, conf))

        picks.append({
            "ticker": t,
            "name": str(p.get("name") or t.replace(".AX", ""))[:80],
            "action": action,
            "confidence": conf,
            "reasoning": str(p.get("reasoning") or "")[:400],
            "entry": round(entry, 3),
            "target": target,
            "stop": stop,
        })
        if len(picks) >= MAX_PICKS:
            break

    if not picks:
        raise RuntimeError("Analyst returned no usable picks after validation.")
    return picks


def _ask_groq(prompt, attempts=4):
    """Call Groq's OpenAI-compatible endpoint.

    gpt-oss-20b/120b are reasoning models. Two gotchas this handles:
      * `reasoning_format` is NOT supported on gpt-oss — use `include_reasoning`.
      * `max_completion_tokens` covers reasoning + answer. If reasoning eats the
        budget you get HTTP 200, finish_reason="length", and content="".
    """
    import requests

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system",
             "content": "You are a disciplined ASX trading analyst. You reply with minified JSON and nothing else."},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": GROQ_MAX_TOKENS,
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
        "reasoning_effort": GROQ_REASONING_EFFORT,
        "include_reasoning": False,
    }
    optional = ("response_format", "reasoning_effort", "include_reasoning")
    dropped = set()
    last_err = "no attempts made"

    for attempt in range(1, attempts + 1):
        body = {k: v for k, v in payload.items() if k not in dropped}
        try:
            r = requests.post(url, headers=headers, json=body, timeout=180)
        except requests.RequestException as e:
            last_err = f"network error: {e}"
            print(f"  Groq attempt {attempt}: {last_err}", flush=True)
            time.sleep(5 * attempt)
            continue

        if DEBUG:
            print(f"  [debug] status={r.status_code} body={r.text[:1200]}", flush=True)

        # A model that doesn't accept one of the optional knobs -> 400. Drop and retry.
        if r.status_code == 400:
            detail = r.text[:600]
            newly = {k for k in optional if k in detail and k not in dropped}
            if newly:
                dropped |= newly
                print(f"  Groq rejected {sorted(newly)} — retrying without.", flush=True)
                continue
            raise RuntimeError(f"Groq 400 (check GROQ_MODEL={GROQ_MODEL}): {detail}")

        if r.status_code in (401, 403):
            raise RuntimeError(
                f"Groq auth failed ({r.status_code}). Is GROQ_API_KEY set correctly "
                f"in the workflow env? Body: {r.text[:200]}"
            )

        if r.status_code == 429 or r.status_code >= 500:
            wait = min(5 * 2 ** (attempt - 1), 60)
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            print(f"  Groq {r.status_code} — retrying in {wait}s...", flush=True)
            time.sleep(wait)
            continue

        if r.status_code != 200:
            raise RuntimeError(f"Groq API error {r.status_code}: {r.text[:300]}")

        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        finish = choice.get("finish_reason")
        usage = data.get("usage", {})
        text = (msg.get("content") or "").strip()

        if text:
            return text

        # Empty content. Reasoning models sometimes strand the answer here.
        reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
        if reasoning and _extract_json(reasoning) is not None:
            print("  Answer arrived in the reasoning field — using it.", flush=True)
            return reasoning

        last_err = f"empty content (finish_reason={finish}, usage={usage})"
        print(f"  Groq attempt {attempt}: {last_err}", flush=True)

        if finish == "length":
            payload["max_completion_tokens"] = min(payload["max_completion_tokens"] * 2, 32000)
            payload["reasoning_effort"] = "low"
            print(f"  Raising budget to {payload['max_completion_tokens']} tokens.", flush=True)
        else:
            time.sleep(3)

    raise RuntimeError(f"Groq returned nothing usable after {attempts} attempts — {last_err}")


def _ask_anthropic(prompt):
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    msg = client.messages.create(
        model=MODEL, max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def fake_analyst(candidates):
    picks = []
    for t, m in list(candidates.items())[:MAX_PICKS]:
        buy = m["chg_5d"] > 0
        picks.append({
            "ticker": t, "name": t.replace(".AX", ""),
            "action": "BUY" if buy else "SELL",
            "confidence": random.randint(40, 85),
            "reasoning": f"Demo pick: 5d {m['chg_5d']}%, RSI {m['rsi14']}, volume {m['vol_x_avg']}x.",
            "entry": m["price"],
            "target": round(m["price"] * (1.08 if buy else 0.92), 2),
            "stop": round(m["price"] * (0.96 if buy else 1.04), 2),
        })
    return picks


# ---------------------------------------------------------------- ledger logic


def track_universe(conn, tickers, ts):
    """Note first/last sighting of each ticker so new watchlist entries stand out.

    On a database that predates this table we seed everything as already-known,
    otherwise the first upgraded run would flag all 38 names as new.
    """
    seeding = conn.execute("SELECT COUNT(*) FROM universe").fetchone()[0] == 0
    prior_scans = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] > 0
    seed_flag = 1 if (seeding and prior_scans) else 0
    for t in tickers:
        conn.execute(
            "INSERT INTO universe (ticker, first_seen, last_seen, seeded) VALUES (?,?,?,?) "
            "ON CONFLICT(ticker) DO UPDATE SET last_seen=excluded.last_seen",
            (t, ts, ts, seed_flag),
        )
    conn.commit()


def new_tickers(conn):
    """Tickers that first appeared within the NEW_TICKER_DAYS window."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=NEW_TICKER_DAYS)).isoformat()
    rows = conn.execute(
        "SELECT ticker, first_seen FROM universe WHERE seeded=0 AND first_seen >= ? "
        "ORDER BY first_seen DESC, ticker", (cutoff,)
    ).fetchall()
    return {r["ticker"]: r["first_seen"] for r in rows}


def record_scan(conn, ranked, shortlisted, picks, analyst_ok, ts):
    """Persist the full ranked field for this scan, not just the picks."""
    picked_by_ticker = {p["ticker"]: p["action"] for p in picks}
    cur = conn.execute(
        "INSERT INTO scans (ts, universe, shortlisted, picks, analyst_ok) VALUES (?,?,?,?,?)",
        (ts, len(ranked), len(shortlisted), len(picks), 1 if analyst_ok else 0),
    )
    scan_id = cur.lastrowid
    for i, (t, m, score) in enumerate(ranked, start=1):
        conn.execute(
            "INSERT INTO scan_metrics (scan_id,ticker,rank,score,price,chg_1d,chg_5d,chg_20d,"
            "rsi14,vol_x_avg,hi_60d_pct,shortlisted,picked) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (scan_id, t, i, score, m["price"], m["chg_1d"], m["chg_5d"], m["chg_20d"],
             m["rsi14"], m["vol_x_avg"], m["hi_60d_pct"],
             1 if t in shortlisted else 0, picked_by_ticker.get(t)),
        )
    conn.commit()
    return scan_id


def record_signals(conn, picks, metrics):
    ids = []
    for p in picks:
        cur = conn.execute(
            "INSERT INTO signals (ts,ticker,name,action,confidence,reasoning,price_at_scan,entry,target,stop) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (now_iso(), p["ticker"], p.get("name"), p["action"], int(p.get("confidence", 0)),
             p.get("reasoning"), metrics.get(p["ticker"], {}).get("price"),
             p.get("entry"), p.get("target"), p.get("stop")),
        )
        ids.append((cur.lastrowid, p))
    conn.commit()
    return ids


def open_paper_positions(conn, signal_rows):
    for sid, p in signal_rows:
        if int(p.get("confidence", 0)) < PAPER_TRADE_MIN_CONF or p["action"] not in ("BUY", "SELL"):
            continue
        already = conn.execute(
            "SELECT 1 FROM positions WHERE ticker=? AND closed_ts IS NULL", (p["ticker"],)
        ).fetchone()
        if already:
            continue
        entry = _num(p.get("entry"))
        if not entry:
            print(f"  Skipping {p['ticker']} — no valid entry price.", flush=True)
            continue
        units = max(int(POSITION_VALUE_AUD / entry), 1)
        conn.execute(
            "INSERT INTO positions (signal_id,ticker,action,units,entry,target,stop,opened_ts,last_price) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, p["ticker"], p["action"], units, entry, p.get("target"),
             p.get("stop"), now_iso(), entry),
        )
        print(f"  Paper-opened {p['action']} {units}u {p['ticker']} @ ${entry}", flush=True)
    conn.commit()


def position_pl(row, price):
    direction = 1 if row["action"] == "BUY" else -1
    return (price - row["entry"]) * direction * row["units"]


def mark_and_close_positions(conn, metrics):
    open_rows = conn.execute("SELECT * FROM positions WHERE closed_ts IS NULL").fetchall()
    for r in open_rows:
        m = metrics.get(r["ticker"])
        if not m:
            continue
        price = m["price"]
        conn.execute("UPDATE positions SET last_price=? WHERE id=?", (price, r["id"]))
        reason = None
        if r["action"] == "BUY":
            if r["target"] and price >= r["target"]:
                reason = "target hit"
            elif r["stop"] and price <= r["stop"]:
                reason = "stopped out"
        else:
            if r["target"] and price <= r["target"]:
                reason = "target hit"
            elif r["stop"] and price >= r["stop"]:
                reason = "stopped out"
        if reason:
            pl = position_pl(r, price)
            conn.execute(
                "UPDATE positions SET closed_ts=?, exit_price=?, pl=?, close_reason=? WHERE id=?",
                (now_iso(), price, round(pl, 2), reason, r["id"]),
            )
            print(f"  Closed {r['ticker']} ({reason}) P/L ${pl:+,.2f}", flush=True)
    conn.commit()


# ---------------------------------------------------------------- dashboard

CSS = """
:root{--bg:#10151d;--panel:#171e29;--line:#26324a;--text:#e8edf4;--dim:#8b98ab;
--buy:#3fb47c;--sell:#e0705a;--amber:#f2b544}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;padding:24px}
.wrap{max-width:1000px;margin:0 auto}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--amber)}
h1{margin:2px 0 4px;font-size:28px}h2{font-size:15px;margin:28px 0 10px;color:var(--dim);
text-transform:uppercase;letter-spacing:.08em}
.meta{color:var(--dim);font-size:13px}
.stats{display:flex;gap:28px;flex-wrap:wrap;border-bottom:2px solid var(--amber);
padding:16px 0;margin-bottom:8px}
.stat b{display:block;font-size:10px;letter-spacing:.1em;color:var(--dim);font-weight:600}
.stat span{font-family:ui-monospace,Menlo,monospace;font-size:17px}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);
border-radius:10px;overflow:hidden;font-size:13.5px}
th{font-size:10.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);
text-align:left;padding:9px 11px;border-bottom:1px solid var(--line)}
td{padding:9px 11px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
.mono{font-family:ui-monospace,Menlo,monospace}
.buy{color:var(--buy);font-weight:700}.sell{color:var(--sell);font-weight:700}
.pos{color:var(--buy)}.neg{color:var(--sell)}
.conf{display:inline-block;min-width:34px;text-align:center;border:1px solid var(--amber);
color:var(--amber);border-radius:6px;padding:1px 5px;font-size:12px}
.foot{margin-top:26px;color:var(--dim);font-size:11px}
.badge{display:inline-block;font-size:9.5px;letter-spacing:.06em;font-weight:700;
border-radius:4px;padding:1px 5px;margin-left:5px;vertical-align:middle}
.b-new{background:var(--amber);color:#10151d}
.b-short{border:1px solid var(--dim);color:var(--dim)}
.b-pick{background:var(--buy);color:#10151d}
.b-pick.sellpick{background:var(--sell)}
.rank{color:var(--dim);font-size:12px;width:26px}
.bar{position:relative;min-width:64px}
.bar i{position:absolute;left:0;top:50%;transform:translateY(-50%);height:14px;
background:rgba(242,181,68,.18);border-radius:3px;z-index:0}
.bar span{position:relative;z-index:1;padding-left:4px}
.tag{color:var(--dim);font-size:12px}
.newbar{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--amber);
border-radius:8px;padding:10px 12px;margin:10px 0 0;font-size:13px}
.newbar b{color:var(--amber);font-size:10.5px;letter-spacing:.1em;display:block;margin-bottom:3px}
details{margin-top:8px}summary{cursor:pointer;color:var(--dim);font-size:12px;padding:6px 0}
@media(max-width:640px){body{padding:12px}td,th{padding:7px 6px}
table{font-size:12.5px}.hide-sm{display:none}}
"""


def aud(v):
    return "—" if v is None else f"${v:,.2f}"


def esc(s):
    return (str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_dashboard(conn):
    open_rows = conn.execute("SELECT * FROM positions WHERE closed_ts IS NULL ORDER BY opened_ts DESC").fetchall()
    closed = conn.execute("SELECT * FROM positions WHERE closed_ts IS NOT NULL ORDER BY closed_ts DESC LIMIT 60").fetchall()
    signals = conn.execute("SELECT * FROM signals ORDER BY ts DESC LIMIT 60").fetchall()

    unreal = sum(position_pl(r, r["last_price"]) for r in open_rows if r["last_price"])
    realized = sum(r["pl"] or 0 for r in closed)
    wins = sum(1 for r in closed if (r["pl"] or 0) > 0)
    winrate = f"{100 * wins / len(closed):.0f}%" if closed else "—"

    def cls(v):
        return "pos" if v > 0 else "neg" if v < 0 else ""

    sig_rows = "".join(
        f"<tr><td class=mono>{r['ts'][:16].replace('T', ' ')}</td>"
        f"<td class=mono><b>{esc(r['ticker'])}</b></td>"
        f"<td class={r['action'].lower()}>{esc(r['action'])}</td>"
        f"<td><span class=conf>{r['confidence']}</span></td>"
        f"<td>{esc(r['reasoning'])}</td>"
        f"<td class=mono>{aud(r['entry'])} / {aud(r['target'])} / {aud(r['stop'])}</td></tr>"
        for r in signals
    ) or "<tr><td colspan=6 class=meta>No scans logged yet — run a scan.</td></tr>"

    open_html = "".join(
        f"<tr><td class=mono><b>{esc(r['ticker'])}</b></td>"
        f"<td class={r['action'].lower()}>{esc(r['action'])}</td>"
        f"<td class=mono>{r['units']}</td>"
        f"<td class=mono>{aud(r['entry'])}</td>"
        f"<td class=mono>{aud(r['last_price'])}</td>"
        f"<td class='mono {cls(position_pl(r, r['last_price'] or r['entry']))}'>"
        f"{aud(position_pl(r, r['last_price'] or r['entry']))}</td>"
        f"<td class=mono>{aud(r['target'])} / {aud(r['stop'])}</td>"
        f"<td class=meta>{r['opened_ts'][:10]}</td></tr>"
        for r in open_rows
    ) or "<tr><td colspan=8 class=meta>No open paper positions.</td></tr>"

    closed_html = "".join(
        f"<tr><td class=mono><b>{esc(r['ticker'])}</b></td>"
        f"<td class={r['action'].lower()}>{esc(r['action'])}</td>"
        f"<td class=mono>{aud(r['entry'])} → {aud(r['exit_price'])}</td>"
        f"<td class='mono {cls(r['pl'] or 0)}'>{aud(r['pl'])}</td>"
        f"<td>{esc(r['close_reason'])}</td>"
        f"<td class=meta>{(r['closed_ts'] or '')[:10]}</td></tr>"
        for r in closed
    ) or "<tr><td colspan=6 class=meta>Nothing closed yet.</td></tr>"

    # ---- latest scan: full ranked field -------------------------------------
    fresh = new_tickers(conn)
    last_scan = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 1").fetchone()
    rank_rows = conn.execute(
        "SELECT * FROM scan_metrics WHERE scan_id=? ORDER BY rank", (last_scan["id"],)
    ).fetchall() if last_scan else []

    top_score = max((r["score"] for r in rank_rows), default=0) or 1

    def rank_tr(r):
        badges = ""
        if r["ticker"] in fresh:
            badges += "<span class='badge b-new'>NEW</span>"
        if r["picked"]:
            side = "sellpick" if r["picked"] == "SELL" else ""
            badges += f"<span class='badge b-pick {side}'>{esc(r['picked'])}</span>"
        elif r["shortlisted"]:
            badges += "<span class='badge b-short'>SHORT</span>"
        m = {k: r[k] for k in ("chg_1d", "chg_5d", "chg_20d", "rsi14", "vol_x_avg", "hi_60d_pct")}
        width = max(4, int(100 * r["score"] / top_score))
        return (
            f"<tr><td class=rank>{r['rank']}</td>"
            f"<td class=mono><b>{esc(r['ticker'])}</b>{badges}</td>"
            f"<td class='mono bar'><i style='width:{width}%'></i><span>{r['score']:.0f}</span></td>"
            f"<td class=mono>{aud(r['price'])}</td>"
            f"<td class='mono {cls(r['chg_1d'])}'>{r['chg_1d']:+.1f}%</td>"
            f"<td class='mono {cls(r['chg_5d'])}'>{r['chg_5d']:+.1f}%</td>"
            f"<td class='mono {cls(r['chg_20d'])} hide-sm'>{r['chg_20d']:+.1f}%</td>"
            f"<td class=mono>{r['rsi14']:.0f}</td>"
            f"<td class=mono>{r['vol_x_avg']:.1f}x</td>"
            f"<td class='tag hide-sm'>{esc(setup_tag(m))}</td></tr>"
        )

    TOP_N = 12
    rank_head = ("<tr><th></th><th>Ticker</th><th>Score</th><th>Price</th><th>1d</th><th>5d</th>"
                 "<th class=hide-sm>20d</th><th>RSI</th><th>Vol</th><th class=hide-sm>Setup</th></tr>")
    rank_html = "".join(rank_tr(r) for r in rank_rows[:TOP_N]) or \
        "<tr><td colspan=10 class=meta>No scan recorded yet.</td></tr>"
    rest_html = ""
    if len(rank_rows) > TOP_N:
        rest = "".join(rank_tr(r) for r in rank_rows[TOP_N:])
        rest_html = (f"<details><summary>Show remaining {len(rank_rows) - TOP_N} "
                     f"scanned names</summary><table>{rank_head}{rest}</table></details>")

    if fresh:
        items = ", ".join(f"<b class=mono>{esc(t)}</b>" for t in sorted(fresh))
        new_html = (f"<div class=newbar><b>NEW TO THE WATCHLIST</b>{items} "
                    f"— first scanned in the last {NEW_TICKER_DAYS} days.</div>")
    else:
        new_html = ""

    if last_scan:
        scan_meta = (f"Scan {last_scan['ts'][:16].replace('T', ' ')} UTC · "
                     f"{last_scan['universe']} names ranked · "
                     f"{last_scan['shortlisted']} shortlisted · {last_scan['picks']} picked"
                     + ("" if last_scan["analyst_ok"] else " · analyst unavailable"))
    else:
        scan_meta = "No scan recorded yet."

    html = f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>ASX Signal Desk</title><style>{CSS}</style></head><body><div class=wrap>
<div class=eyebrow>ASX · speculative desk</div><h1>Signal Desk</h1>
<div class=meta>Last updated {datetime.now(LOCAL_TZ).strftime('%A %d %B %Y, %H:%M')} {LOCAL_TZ.key.split('/')[-1]}</div>
<div class=stats>
<div class=stat><b>OPEN</b><span>{len(open_rows)}</span></div>
<div class=stat><b>UNREALISED</b><span class="{cls(unreal)}">{aud(unreal)}</span></div>
<div class=stat><b>REALISED</b><span class="{cls(realized)}">{aud(realized)}</span></div>
<div class=stat><b>WIN RATE</b><span>{winrate}</span></div>
<div class=stat><b>UNIVERSE</b><span>{conn.execute('SELECT COUNT(*) FROM universe').fetchone()[0]}</span></div>
<div class=stat><b>SIGNALS LOGGED</b><span>{conn.execute('SELECT COUNT(*) FROM signals').fetchone()[0]}</span></div>
</div>
{new_html}
<h2>Latest scan — ranked field</h2>
<div class=meta>{scan_meta}</div>
<table style="margin-top:8px">{rank_head}{rank_html}</table>
{rest_html}
<h2>Open paper positions</h2>
<table><tr><th>Ticker</th><th>Side</th><th>Units</th><th>Entry</th><th>Last</th><th>P/L</th><th>Target / Stop</th><th>Opened</th></tr>{open_html}</table>
<h2>Latest signals</h2>
<table><tr><th>Scan</th><th>Ticker</th><th>Call</th><th>Conf</th><th>Reasoning</th><th>Entry / Target / Stop</th></tr>{sig_rows}</table>
<h2>Closed positions</h2>
<table><tr><th>Ticker</th><th>Side</th><th>Entry → Exit</th><th>P/L</th><th>Reason</th><th>Closed</th></tr>{closed_html}</table>
<div class=foot>Research tool — AI-generated signals from market data. Paper positions are hypothetical. Not financial advice.</div>
</div></body></html>"""
    os.makedirs(os.path.dirname(DASHBOARD_PATH) or ".", exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        f.write(html)
    print(f"Dashboard written to {os.path.abspath(DASHBOARD_PATH)}", flush=True)


# ---------------------------------------------------------------- commands


def run_scan(demo=False):
    """Returns True if the analyst step succeeded, False if it was skipped.

    A failed analyst call must not lose the whole run: existing positions are
    still marked to market and the dashboard is still rebuilt.
    """
    conn = db()
    try:
        metrics = fake_metrics(WATCHLIST) if demo else fetch_metrics(WATCHLIST)
        if not metrics:
            print("No market data returned — check your connection and try again.", flush=True)
            return False

        ts = now_iso()
        track_universe(conn, metrics.keys(), ts)
        fresh = new_tickers(conn)
        if fresh:
            print(f"New to the watchlist: {', '.join(fresh)}", flush=True)

        mark_and_close_positions(conn, metrics)

        ranked = rank_all(metrics)
        candidates = {t: m for t, m, _ in ranked[:SHORTLIST_SIZE]}
        print(f"Shortlisted: {', '.join(candidates)}", flush=True)

        analyst_ok = True
        try:
            picks = fake_analyst(candidates) if demo else ask_analyst(candidates)
        except Exception as e:
            print(f"Analyst step failed: {e}", flush=True)
            print("Continuing without new signals — positions and dashboard still updated.", flush=True)
            picks, analyst_ok = [], False

        record_scan(conn, ranked, candidates, picks, analyst_ok, ts)

        if picks:
            for p in picks:
                print(f"  {p['action']} {p['ticker']} conf {p['confidence']} — {p['reasoning']}", flush=True)
            rows = record_signals(conn, picks, metrics)
            open_paper_positions(conn, rows)

        build_dashboard(conn)
        return analyst_ok
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="ASX Signal Desk")
    ap.add_argument("command", nargs="?", default="scan", choices=["scan", "report", "demo"])
    ap.add_argument("--loop", type=int, metavar="SECONDS",
                    help="keep scanning at this interval instead of running once")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if the AI analyst step fails (default: warn and carry on)")
    args = ap.parse_args()

    if args.command == "report":
        conn = db()
        build_dashboard(conn)
        conn.close()
        return

    demo = args.command == "demo"
    if not demo and not (os.environ.get("GROQ_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        sys.exit("Set GROQ_API_KEY or ANTHROPIC_API_KEY first (or run 'demo' mode).")

    if args.loop:
        while True:
            try:
                run_scan(demo)
            except Exception as e:
                print(f"Scan failed: {e}", flush=True)
            print(f"Sleeping {args.loop}s...\n", flush=True)
            time.sleep(args.loop)
    else:
        ok = run_scan(demo)
        if args.strict and not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ASX Signal Desk — market scanner, AI analyst, and paper-trading ledger.

What it does each scan:
  1. Pulls 3 months of real price/volume data for a watchlist of ASX stocks (yfinance)
  2. Computes momentum, RSI, volume spikes; shortlists the most active setups
  3. Sends the numbers to Claude for BUY/SELL calls with reasoning + confidence
  4. Logs every recommendation to SQLite
  5. Paper-trades high-confidence calls, closes them on target/stop, tracks P/L
  6. Regenerates dashboard.html with all signals, positions and results

Setup:
    pip install yfinance pandas anthropic
    export ANTHROPIC_API_KEY=sk-ant-...        (get one at console.anthropic.com)

Usage:
    python asx_signal_desk.py scan             # one full scan (run this on a schedule)
    python asx_signal_desk.py report           # rebuild dashboard from the database only
    python asx_signal_desk.py demo             # try the pipeline with synthetic data, no API key needed
    python asx_signal_desk.py scan --loop 3600 # keep scanning every hour

Schedule it (crontab -e), e.g. 10:30am and 3:30pm Sydney time on weekdays:
    30 10,15 * * 1-5  cd /path/to/desk && /usr/bin/python3 asx_signal_desk.py scan

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
from datetime import datetime, timezone

# ---------------------------------------------------------------- configuration

MODEL = "claude-sonnet-4-6"
DB_PATH = "asx_desk.db"
DASHBOARD_PATH = "docs/index.html"

# Liquid ASX names across sectors. Add your own — always use the .AX suffix.
WATCHLIST = [
    "BHP.AX", "RIO.AX", "FMG.AX", "S32.AX", "MIN.AX", "PLS.AX", "LYC.AX",
    "NST.AX", "EVN.AX", "GOR.AX", "WDS.AX", "STO.AX", "BPT.AX", "PDN.AX",
    "BOE.AX", "CBA.AX", "NAB.AX", "WBC.AX", "ANZ.AX", "MQG.AX", "GQG.AX",
    "CSL.AX", "RMD.AX", "PME.AX", "TLX.AX", "NEU.AX", "WTC.AX", "XRO.AX",
    "NXT.AX", "ALU.AX", "DTL.AX", "WOW.AX", "COL.AX", "JBH.AX", "LOV.AX",
    "TPW.AX", "QAN.AX", "WEB.AX", "FLT.AX", "A2M.AX",
]

SHORTLIST_SIZE = 10      # candidates sent to the AI analyst per scan
MAX_PICKS = 4            # recommendations requested per scan
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
    import pandas as pd
    import yfinance as yf

    print(f"Fetching {len(tickers)} tickers from Yahoo Finance...")
    data = yf.download(
        tickers, period="3mo", interval="1d",
        group_by="ticker", auto_adjust=True, progress=False, threads=True,
    )
    metrics = {}
    for t in tickers:
        try:
            df = data[t].dropna() if len(tickers) > 1 else data.dropna()
            if len(df) < 30:
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
        except Exception:
            continue
    print(f"Got usable data for {len(metrics)} tickers.")
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


def shortlist(metrics, n):
    """Rank by 'interestingness': momentum + volume spike + RSI extremes."""
    def score(m):
        s = abs(m["chg_5d"]) * 1.5 + abs(m["chg_1d"]) * 2 + max(m["vol_x_avg"] - 1, 0) * 10
        if m["rsi14"] < 35:
            s += (35 - m["rsi14"]) * 0.5   # oversold — bounce candidate
        elif m["rsi14"] > 70:
            s += (m["rsi14"] - 70) * 0.5   # overbought — exhaustion candidate
        return s
    ranked = sorted(metrics.items(), key=lambda kv: score(kv[1]), reverse=True)
    return dict(ranked[:n])


# ---------------------------------------------------------------- AI analyst


def ask_analyst(candidates):
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    table = "\n".join(
        f"{t}: price ${m['price']} | 1d {m['chg_1d']}% | 5d {m['chg_5d']}% | "
        f"20d {m['chg_20d']}% | RSI14 {m['rsi14']} | vol {m['vol_x_avg']}x avg | "
        f"{m['hi_60d_pct']}% off 60d high"
        for t, m in candidates.items()
    )
    prompt = f"""You are a speculative short-term equities trader on the ASX. Today is {datetime.now().strftime('%A %d %B %Y')}.
Below are live technical metrics for the most active names on my watchlist. Pick the {MAX_PICKS} best risk/reward trades (long or short) purely from these numbers — momentum continuation, oversold bounces, blow-off exhaustion, volume confirmation.

{table}

Respond with ONLY minified JSON, no fences, no commentary:
{{"picks":[{{"ticker":"BHP.AX","name":"BHP Group","action":"BUY","confidence":72,"reasoning":"max 25 words citing the specific numbers","entry":0,"target":0,"stop":0}}]}}

Rules: action is BUY or SELL (SELL = short). confidence integer 0-100. entry = current price. target and stop must be consistent with the direction and imply a sensible risk/reward (roughly 2:1). Be selective — mediocre setups get confidence below 50."""

    msg = client.messages.create(
        model=MODEL, max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"Analyst reply had no JSON: {text[:200]}")
    return json.loads(m.group(0))["picks"]


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
        if already or not p.get("entry"):
            continue
        units = max(int(POSITION_VALUE_AUD / float(p["entry"])), 1)
        conn.execute(
            "INSERT INTO positions (signal_id,ticker,action,units,entry,target,stop,opened_ts,last_price) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, p["ticker"], p["action"], units, p["entry"], p.get("target"),
             p.get("stop"), now_iso(), p["entry"]),
        )
        print(f"  Paper-opened {p['action']} {units}u {p['ticker']} @ ${p['entry']}")
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
            print(f"  Closed {r['ticker']} ({reason}) P/L ${pl:+,.2f}")
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
@media(max-width:640px){body{padding:12px}td,th{padding:7px 6px}}
"""


def aud(v):
    return "—" if v is None else f"${v:,.2f}"


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
        f"<td class=mono><b>{r['ticker']}</b></td>"
        f"<td class={r['action'].lower()}>{r['action']}</td>"
        f"<td><span class=conf>{r['confidence']}</span></td>"
        f"<td>{r['reasoning'] or ''}</td>"
        f"<td class=mono>{aud(r['entry'])} / {aud(r['target'])} / {aud(r['stop'])}</td></tr>"
        for r in signals
    ) or "<tr><td colspan=6 class=meta>No scans logged yet — run a scan.</td></tr>"

    open_html = "".join(
        f"<tr><td class=mono><b>{r['ticker']}</b></td>"
        f"<td class={r['action'].lower()}>{r['action']}</td>"
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
        f"<tr><td class=mono><b>{r['ticker']}</b></td>"
        f"<td class={r['action'].lower()}>{r['action']}</td>"
        f"<td class=mono>{aud(r['entry'])} → {aud(r['exit_price'])}</td>"
        f"<td class='mono {cls(r['pl'] or 0)}'>{aud(r['pl'])}</td>"
        f"<td>{r['close_reason']}</td>"
        f"<td class=meta>{(r['closed_ts'] or '')[:10]}</td></tr>"
        for r in closed
    ) or "<tr><td colspan=6 class=meta>Nothing closed yet.</td></tr>"

    html = f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>ASX Signal Desk</title><style>{CSS}</style></head><body><div class=wrap>
<div class=eyebrow>ASX · speculative desk</div><h1>Signal Desk</h1>
<div class=meta>Last updated {datetime.now().strftime('%A %d %B %Y, %H:%M')}</div>
<div class=stats>
<div class=stat><b>OPEN</b><span>{len(open_rows)}</span></div>
<div class=stat><b>UNREALISED</b><span class="{cls(unreal)}">{aud(unreal)}</span></div>
<div class=stat><b>REALISED</b><span class="{cls(realized)}">{aud(realized)}</span></div>
<div class=stat><b>WIN RATE</b><span>{winrate}</span></div>
<div class=stat><b>SIGNALS LOGGED</b><span>{conn.execute('SELECT COUNT(*) FROM signals').fetchone()[0]}</span></div>
</div>
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
    print(f"Dashboard written to {os.path.abspath(DASHBOARD_PATH)}")


# ---------------------------------------------------------------- commands


def run_scan(demo=False):
    conn = db()
    metrics = fake_metrics(WATCHLIST) if demo else fetch_metrics(WATCHLIST)
    if not metrics:
        print("No market data returned — check your connection and try again.")
        return
    mark_and_close_positions(conn, metrics)
    candidates = shortlist(metrics, SHORTLIST_SIZE)
    print(f"Shortlisted: {', '.join(candidates)}")
    picks = fake_analyst(candidates) if demo else ask_analyst(candidates)
    for p in picks:
        print(f"  {p['action']} {p['ticker']} conf {p['confidence']} — {p['reasoning']}")
    rows = record_signals(conn, picks, metrics)
    open_paper_positions(conn, rows)
    build_dashboard(conn)
    conn.close()


def main():
    ap = argparse.ArgumentParser(description="ASX Signal Desk")
    ap.add_argument("command", nargs="?", default="scan", choices=["scan", "report", "demo"])
    ap.add_argument("--loop", type=int, metavar="SECONDS",
                    help="keep scanning at this interval instead of running once")
    args = ap.parse_args()

    if args.command == "report":
        conn = db()
        build_dashboard(conn)
        conn.close()
        return

    demo = args.command == "demo"
    if not demo and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first (or run 'demo' mode to try the pipeline).")

    if args.loop:
        while True:
            try:
                run_scan(demo)
            except Exception as e:
                print(f"Scan failed: {e}")
            print(f"Sleeping {args.loop}s...\n")
            time.sleep(args.loop)
    else:
        run_scan(demo)


if __name__ == "__main__":
    main()

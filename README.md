# ASX Signal Desk — cloud edition

An automated ASX scanner that runs itself twice each trading day on GitHub's free
infrastructure. Each scan pulls real market data, has Claude pick trades with
reasoning and confidence scores, paper-trades the strong calls, and publishes a
dashboard web page you can bookmark on your phone.

No server, no laptop needed after setup. Setup takes ~15 minutes and is easiest
in a desktop browser.

## One-time setup

**1. Get an Anthropic API key.**
Sign in at https://console.anthropic.com → API Keys → Create Key. Copy it
(starts with `sk-ant-`). Add a few dollars of credit; each scan costs well
under one cent.

**2. Create the repository.**
Sign in at https://github.com (free account is fine) → **New repository** →
name it `asx-signal-desk` → set it to **Public** → Create.
*Public is required for the free dashboard web page. The dashboard shows only
paper trades, but anyone with the link could view it — if you'd rather keep it
private, you can still read the dashboard file inside the repo, just without a
nice URL.*

**3. Upload these files.**
In the repo: **Add file → Upload files**, then drag in `asx_signal_desk.py`,
`requirements.txt`, and `README.md`. Commit.
The workflow file must be created by hand (GitHub ignores uploaded workflow
files): **Add file → Create new file**, type the filename exactly
`.github/workflows/scan.yml`, paste the contents of that file from this
package, commit.

**4. Add your API key as a secret.**
Repo **Settings → Secrets and variables → Actions → New repository secret**.
Name: `ANTHROPIC_API_KEY` — Value: your `sk-ant-...` key.

**5. Turn on the dashboard page.**
Repo **Settings → Pages** → under "Build and deployment" choose
**Deploy from a branch** → Branch: `main`, folder: `/docs` → Save.

**6. Run your first scan.**
**Actions** tab → enable workflows if prompted → click **ASX scan** →
**Run workflow**. It takes 2–3 minutes. When it's green, your dashboard is
live at:

    https://YOUR-USERNAME.github.io/asx-signal-desk/

Bookmark that on your phone. It refreshes automatically after every scan
(give Pages a minute or two to update).

## How it runs from now on

- Scans fire automatically at 10:30am and 3:30pm Sydney time on weekdays
  (an hour later during daylight saving) — edit the cron lines in
  `.github/workflows/scan.yml` to change this.
- Every recommendation is logged permanently in `asx_desk.db` in the repo.
- Calls with confidence ≥ 65 are paper-traded at $5,000 notional; positions
  close automatically when their target or stop is hit, and realised P/L,
  win rate and history all appear on the dashboard.

## Tuning

Open `asx_signal_desk.py` in the repo (pencil icon to edit) and adjust the
configuration block near the top:

- `WATCHLIST` — add or remove stocks (always use the `.AX` suffix)
- `PAPER_TRADE_MIN_CONF` — confidence needed to open a paper position
- `POSITION_VALUE_AUD` — notional size per position
- `MAX_PICKS` / `SHORTLIST_SIZE` — how many picks per scan

Commit the edit; the next scan uses it.

## Important

This is a research and paper-trading tool, not financial advice. Let the win
rate build a track record before you consider acting on anything it says —
that's exactly what the dashboard is for.

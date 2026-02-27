# NEXUS v2 — Autonomous Trading Bot

Multi-agent AI trading bot optimized for momentum swing trading.
Runs 24/7 on the cloud — no browser needed.

## Quick Deploy to Railway (free/$5/month)

1. Go to railway.app and sign up
2. Click "New Project" → "Deploy from GitHub"
3. Upload these files to a GitHub repo first:
   - nexus.py
   - requirements.txt
   - Procfile
4. In Railway dashboard → Variables, add:
   - ALPACA_KEY
   - ALPACA_SECRET
   - ANTHROPIC_KEY
   - TELEGRAM_TOKEN
   - TELEGRAM_CHAT
5. Click Deploy — it starts automatically!

## Local Testing

```bash
pip install -r requirements.txt
cp .env.template .env
# Edit .env with your keys
python nexus.py
```

## What It Does

- Scans ALL watchlist stocks every 60 min in parallel
- Picks highest conviction BUY/SELL opportunity
- Runs 7 AI agents for deep analysis on best candidate
- Executes trades automatically on Alpaca paper account
- Monitors stop loss (4%) and take profit (10%) every minute
- Learns from outcomes — adjusts agent weights and confidence threshold
- Searches web for new strategies, tests them, keeps winners
- Sends Telegram alerts for every trade and daily status reports

## Optimized Settings (vs day trading)

| Setting | Value | Why |
|---------|-------|-----|
| Interval | 60 min | Less noise than 30 min |
| Position size | 15% | Bigger bets, fewer trades |
| Stop loss | 4% | Tight protection |
| Take profit | 10% | Realistic target |
| Confidence | 70% | Only high conviction trades |

## Files

- nexus.py — main bot
- nexus_state.json — auto-created, saves learning progress
- nexus.log — auto-created, full activity log

"""
NEXUS v3 — Autonomous Multi-Agent Trading Bot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Improvements over v2:
  • Real OHLCV bar data (daily + hourly) from Alpaca
  • Computed technical indicators: RSI, MACD, Bollinger Bands, ATR,
    Stochastic, SMA/EMA, volume ratio, VWAP, 52-week position
  • Live news headlines injected into agent prompts (Alpaca News API)
  • Market regime detection (bull/bear/neutral) via SPY analysis
  • ATR-based position sizing — smaller bets on volatile stocks
  • Trailing stop losses — locks in gains as positions profit
  • Performance metrics: Sharpe ratio, max drawdown, profit factor
  • Indicator tracking — learns which signals are actually predictive
  • UCB strategy selection — smarter exploration vs exploitation
  • Bear market filter — only very high conviction buys in downtrends
  • Model tiering — haiku for scans, sonnet for decisions
  • Fixed change=0 bug from v2
  • Agents now receive real computed data instead of guessing
"""

import os, json, time, logging, requests, math, re, concurrent.futures
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import anthropic
import pandas as pd
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────────
AK            = os.environ.get("ALPACA_KEY", "")
SK            = os.environ.get("ALPACA_SECRET", "")
ANK           = os.environ.get("ANTHROPIC_KEY", "")
TG_TOKEN      = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT       = os.environ.get("TELEGRAM_CHAT", "")

WATCHLIST     = os.environ.get("WATCHLIST", "AAPL,TSLA,NVDA,MSFT,AMZN,META,GOOGL,AMD").split(",")
INTERVAL_MIN  = int(os.environ.get("INTERVAL_MIN",    "60"))
POS_SIZE_PCT  = float(os.environ.get("POS_SIZE_PCT",  "15"))   # % of buying power per trade
STOP_LOSS     = float(os.environ.get("STOP_LOSS_PCT", "4"))    # hard stop loss %
TAKE_PROFIT   = float(os.environ.get("TAKE_PROFIT_PCT","10"))  # take profit %
CONF_THRESH   = int(os.environ.get("CONF_THRESHOLD",  "70"))   # min confidence to trade
AUTO_EXEC     = os.environ.get("AUTO_EXECUTE",  "true").lower() == "true"
USE_TRAILING  = os.environ.get("TRAILING_STOP", "true").lower() == "true"
MAX_POSITIONS = int(os.environ.get("MAX_POSITIONS", "4"))

ALPACA_BASE   = "https://paper-api.alpaca.markets"
ALPACA_DATA   = "https://data.alpaca.markets"
EST           = ZoneInfo("America/New_York")

# Model tiers
MODEL_FAST     = "claude-haiku-4-5-20251001"    # quick scans — cheap & fast
MODEL_MAIN     = "claude-haiku-4-5-20251001"    # sub-agents — real data compensates
MODEL_OVERSEER = "claude-sonnet-4-5-20250929"   # NEXUS final decision — best model

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("nexus.log")]
)
log = logging.getLogger("NEXUS")

# ─── STATE ────────────────────────────────────────────────────────────────────
state = {
    "version": 3,
    "cycle": 0,
    "wins": 0,
    "losses": 0,
    "streak": 0,
    "agent_weights": {
        "sentinel":  1.0,
        "oracle":    1.0,
        "arbiter":   1.0,
        "cassandra": 1.0,
        "herald":    1.0,
        "guardian":  1.5,   # guardian starts with higher weight (risk matters most)
    },
    "indicator_scores": {   # tracks which technical signals lead to winning trades
        "rsi_overbought": 0, "rsi_oversold": 0,
        "macd_bullish_cross": 0, "macd_bearish_cross": 0,
        "bb_squeeze": 0, "bb_lower_band": 0, "bb_upper_band": 0,
        "volume_surge": 0, "above_sma50": 0, "below_sma50": 0,
        "above_sma200": 0, "near_52w_high": 0, "near_52w_low": 0,
    },
    "strategies": [],
    "active_strategy": None,
    "pending_evals": [],
    "conf_threshold": CONF_THRESH,
    "trade_history": [],
    "trailing_stops": {},       # {symbol: stop_price}
    "market_regime": "neutral", # bull / bear / neutral
    "regime_history": [],
    "equity_curve": [],
    "performance": {
        "total_trades": 0,
        "total_pnl_pct": 0.0,
        "max_drawdown": 0.0,
        "peak_equity": 100000.0,
        "daily_returns": [],    # for Sharpe ratio
    },
}

STATE_FILE = "nexus_state.json"

def save_state():
    try:
        s = json.loads(json.dumps(state, default=str))
        s["performance"]["daily_returns"] = s["performance"]["daily_returns"][-252:]
        with open(STATE_FILE, "w") as f:
            json.dump(s, f, indent=2)
    except Exception as e:
        log.error(f"State save failed: {e}")

def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                saved = json.load(f)
            for k, v in saved.items():
                state[k] = v
            log.info(f"State loaded — {state['wins']}W/{state['losses']}L | Cycle {state['cycle']} | Regime: {state['market_regime']}")
    except Exception as e:
        log.error(f"State load failed: {e}")

# ─── ALPACA API ───────────────────────────────────────────────────────────────
def _alpaca_headers():
    return {"APCA-API-KEY-ID": AK, "APCA-API-SECRET-KEY": SK, "Content-Type": "application/json"}

def alpaca(path, method="GET", body=None):
    resp = requests.request(method, ALPACA_BASE + path,
                            headers=_alpaca_headers(), json=body, timeout=15)
    resp.raise_for_status()
    return resp.json()

def alpaca_data(path, params=None):
    headers = {"APCA-API-KEY-ID": AK, "APCA-API-SECRET-KEY": SK}
    resp = requests.get(ALPACA_DATA + path, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()

def get_account():   return alpaca("/v2/account")
def get_positions(): return alpaca("/v2/positions")

def place_buy(symbol, notional):
    return alpaca("/v2/orders", "POST", {
        "symbol": symbol,
        "notional": str(round(notional, 2)),
        "side": "buy", "type": "market", "time_in_force": "day"
    })

def close_position(symbol):
    return alpaca(f"/v2/positions/{symbol}", "DELETE")

# ─── MARKET DATA ──────────────────────────────────────────────────────────────
def get_bars(symbol, timeframe="1Day", limit=100):
    """Fetch OHLCV bars → pandas DataFrame"""
    try:
        data = alpaca_data(f"/v2/stocks/{symbol}/bars", params={
            "timeframe": timeframe, "limit": limit,
            "adjustment": "raw", "feed": "iex"
        })
        bars = data.get("bars", [])
        if not bars:
            return None
        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"])
        df.set_index("t", inplace=True)
        df.rename(columns={"o":"open","h":"high","l":"low","c":"close",
                            "v":"volume","vw":"vwap_bar","n":"trades"}, inplace=True)
        for col in ["open","high","low","close","volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        log.warning(f"Bars fetch failed {symbol}: {e}")
        return None

def get_snapshots(symbols):
    """Bulk snapshot fetch for multiple symbols"""
    try:
        data = alpaca_data("/v2/stocks/snapshots",
                           params={"symbols": ",".join(symbols), "feed": "iex"})
        return data
    except Exception as e:
        log.error(f"Snapshot fetch failed: {e}")
        return {}

def get_news(symbol, limit=8):
    """Fetch recent news headlines from Alpaca"""
    try:
        data = alpaca_data("/v2/news", params={"symbols": symbol, "limit": limit, "sort": "desc"})
        articles = data.get("news", [])
        return [{"headline": a.get("headline", ""),
                 "summary":  a.get("summary",  "")[:180],
                 "source":   a.get("source",   ""),
                 "time":     a.get("created_at","")[:10]} for a in articles]
    except Exception as e:
        log.warning(f"News fetch failed {symbol}: {e}")
        return []

# ─── TECHNICAL INDICATORS ─────────────────────────────────────────────────────
def compute_indicators(df):
    """Compute full indicator suite from OHLCV DataFrame"""
    if df is None or len(df) < 20:
        return df

    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # ── Returns ──────────────────────────────────────────────────────────────
    df["chg_1d"]  = c.pct_change(1)  * 100
    df["chg_5d"]  = c.pct_change(5)  * 100
    df["chg_20d"] = c.pct_change(20) * 100

    # ── Moving Averages ───────────────────────────────────────────────────────
    df["sma_20"]  = c.rolling(20).mean()
    df["sma_50"]  = c.rolling(min(50,  len(df))).mean()
    df["sma_200"] = c.rolling(min(200, len(df))).mean()
    df["ema_9"]   = c.ewm(span=9,  adjust=False).mean()
    df["ema_21"]  = c.ewm(span=21, adjust=False).mean()

    # ── RSI (14) ──────────────────────────────────────────────────────────────
    delta     = c.diff()
    gain      = delta.clip(lower=0)
    loss      = (-delta).clip(lower=0)
    avg_gain  = gain.ewm(com=13, adjust=False).mean()
    avg_loss  = loss.ewm(com=13, adjust=False).mean()
    rs        = avg_gain / avg_loss.replace(0, 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ── MACD (12, 26, 9) ──────────────────────────────────────────────────────
    ema12             = c.ewm(span=12, adjust=False).mean()
    ema26             = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["macd_hist_prev"] = df["macd_hist"].shift(1)

    # ── Bollinger Bands (20, 2σ) ──────────────────────────────────────────────
    bb_sma          = c.rolling(20).mean()
    bb_std          = c.rolling(20).std()
    df["bb_upper"]  = bb_sma + 2 * bb_std
    df["bb_lower"]  = bb_sma - 2 * bb_std
    bb_range        = (df["bb_upper"] - df["bb_lower"]).replace(0, 1e-9)
    df["bb_pct"]    = (c - df["bb_lower"]) / bb_range * 100
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / bb_sma * 100

    # ── ATR (14) ──────────────────────────────────────────────────────────────
    prev_c    = c.shift(1)
    tr        = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=13, adjust=False).mean()
    df["atr_pct"] = df["atr"] / c * 100

    # ── Volume ────────────────────────────────────────────────────────────────
    df["vol_sma_20"] = v.rolling(20).mean()
    df["vol_ratio"]  = v / df["vol_sma_20"].replace(0, 1e-9)

    # ── Stochastic %K/%D (14, 3) ──────────────────────────────────────────────
    lo14         = l.rolling(14).min()
    hi14         = h.rolling(14).max()
    stoch_range  = (hi14 - lo14).replace(0, 1e-9)
    df["stoch_k"]= (c - lo14) / stoch_range * 100
    df["stoch_d"]= df["stoch_k"].rolling(3).mean()

    # ── Price vs MA distances ─────────────────────────────────────────────────
    df["dist_sma20_pct"]  = (c - df["sma_20"])  / df["sma_20"].replace(0,1)  * 100
    df["dist_sma50_pct"]  = (c - df["sma_50"])  / df["sma_50"].replace(0,1)  * 100
    df["dist_sma200_pct"] = (c - df["sma_200"]) / df["sma_200"].replace(0,1) * 100

    # ── 52-week range position ────────────────────────────────────────────────
    days         = min(252, len(df))
    w52_high     = h.rolling(days).max()
    w52_low      = l.rolling(days).min()
    w52_range    = (w52_high - w52_low).replace(0, 1e-9)
    df["52w_pct"] = (c - w52_low) / w52_range * 100

    return df

def compute_vwap(hourly_df):
    """Compute intraday VWAP from hourly bars for today"""
    try:
        today = datetime.now(EST).date()
        mask  = hourly_df.index.tz_convert(EST).date == today
        today_bars = hourly_df[mask]
        if today_bars.empty:
            return None
        vwap = (today_bars["close"] * today_bars["volume"]).sum() / today_bars["volume"].sum()
        return float(vwap)
    except Exception:
        return None

def format_indicators(symbol, df, price, news=None, hourly_df=None):
    """Build rich context string for AI agents from real market data"""
    if df is None or df.empty:
        return f"=== {symbol} @ ${price:.2f} ===\n[Insufficient data — fewer than 20 bars available]"

    row = df.iloc[-1]

    def safe(val, fmt=".2f", fallback="N/A"):
        try:
            v = float(val)
            return (f"{v:{fmt}}" if not fmt.startswith("+") else f"{v:+{fmt[1:]}}") if not pd.isna(v) else fallback
        except Exception:
            return fallback

    # Trend flags
    above_20  = price > float(row["sma_20"])  if not pd.isna(row.get("sma_20",  float("nan"))) else None
    above_50  = price > float(row["sma_50"])  if not pd.isna(row.get("sma_50",  float("nan"))) else None
    above_200 = price > float(row["sma_200"]) if not pd.isna(row.get("sma_200", float("nan"))) else None
    sma_flags = (f"SMA20:{'✅' if above_20 else '❌' if above_20 is not None else '–'}  "
                 f"SMA50:{'✅' if above_50 else '❌' if above_50 is not None else '–'}  "
                 f"SMA200:{'✅' if above_200 else '❌' if above_200 is not None else '–'}")

    # MACD crossover detection
    macd_cross = ""
    try:
        h_now  = float(row["macd_hist"])
        h_prev = float(row["macd_hist_prev"])
        if h_prev < 0 and h_now > 0:
            macd_cross = "  🔔 BULLISH CROSSOVER"
        elif h_prev > 0 and h_now < 0:
            macd_cross = "  🔔 BEARISH CROSSOVER"
    except Exception:
        pass

    # RSI interpretation
    rsi = float(row["rsi"]) if not pd.isna(row.get("rsi", float("nan"))) else None
    rsi_label = ("OVERBOUGHT ⚠️" if rsi and rsi > 70
                 else "OVERSOLD 🔥" if rsi and rsi < 30
                 else "neutral")

    # BB position
    bb_pct = float(row["bb_pct"]) if not pd.isna(row.get("bb_pct", float("nan"))) else None
    bb_label = ("Near UPPER band ⚠️" if bb_pct and bb_pct > 80
                else "Near LOWER band 🔥" if bb_pct and bb_pct < 20
                else f"Mid-band ({bb_pct:.0f}%)" if bb_pct is not None else "N/A")

    # Volume
    vol_ratio = float(row["vol_ratio"]) if not pd.isna(row.get("vol_ratio", float("nan"))) else None
    vol_label = (f"{vol_ratio:.1f}x avg ({'🔥 surge' if vol_ratio > 2 else '↑ elevated' if vol_ratio > 1.3 else '→ normal' if vol_ratio > 0.7 else '↓ low'})"
                 if vol_ratio is not None else "N/A")

    # Stochastic
    sk = float(row["stoch_k"]) if not pd.isna(row.get("stoch_k", float("nan"))) else None
    sd = float(row["stoch_d"]) if not pd.isna(row.get("stoch_d", float("nan"))) else None
    stoch_label = (f"%K={sk:.1f} %D={sd:.1f} ({'OVERBOUGHT' if sk > 80 else 'OVERSOLD' if sk < 20 else 'neutral'})"
                   if sk is not None and sd is not None else "N/A")

    # VWAP (intraday)
    vwap_line = ""
    if hourly_df is not None:
        vwap = compute_vwap(hourly_df)
        if vwap:
            diff = (price - vwap) / vwap * 100
            vwap_line = f"\nVWAP (intraday):  ${vwap:.2f}  (price {diff:+.1f}% {'above ↑' if diff > 0 else 'below ↓'})"

    # News
    news_section = ""
    if news:
        lines = [f"  [{a['time']}] {a['headline']}" for a in news[:6]]
        news_section = "\nRECENT NEWS:\n" + "\n".join(lines)

    return (
        f"═══ {symbol} — REAL MARKET DATA ═══\n"
        f"Price:  ${price:.2f}\n"
        f"Changes: 1D={safe(row['chg_1d'],'+.2f')}%  5D={safe(row['chg_5d'],'+.2f')}%  20D={safe(row['chg_20d'],'+.2f')}%\n"
        f"\nTECHNICAL INDICATORS:\n"
        f"RSI(14):     {safe(row['rsi'],'.1f')} — {rsi_label}\n"
        f"MACD:        {safe(row['macd'],'+.3f')} | Signal: {safe(row['macd_signal'],'+.3f')} | Hist: {safe(row['macd_hist'],'+.3f')}{macd_cross}\n"
        f"Bollinger:   {bb_label} | Band width: {safe(row['bb_width'],'.1f')}%\n"
        f"Stochastic:  {stoch_label}\n"
        f"Moving Avgs: {sma_flags}\n"
        f"{vwap_line}\n"
        f"\nPRICE CONTEXT:\n"
        f"ATR(14):     ${safe(row['atr'],'.2f')} ({safe(row['atr_pct'],'.1f')}% daily volatility)\n"
        f"52-week pos: {safe(row['52w_pct'],'.0f')}%  (100=52w high, 0=52w low)\n"
        f"vs SMA50:    {safe(row['dist_sma50_pct'],'+.1f')}%  |  vs SMA200: {safe(row['dist_sma200_pct'],'+.1f')}%\n"
        f"Volume:      {vol_label}\n"
        f"{news_section}"
    )

# ─── MARKET REGIME DETECTION ─────────────────────────────────────────────────
def get_bars_sip(symbol, timeframe="1Day", limit=100):
    """Fetch OHLCV bars using SIP feed as fallback"""
    try:
        data = alpaca_data(f"/v2/stocks/{symbol}/bars", params={
            "timeframe": timeframe, "limit": limit,
            "adjustment": "raw", "feed": "sip"
        })
        bars = data.get("bars", [])
        if not bars:
            return None
        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"])
        df.set_index("t", inplace=True)
        df.rename(columns={"o":"open","h":"high","l":"low","c":"close",
                            "v":"volume","vw":"vwap_bar","n":"trades"}, inplace=True)
        for col in ["open","high","low","close","volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        log.warning(f"SIP bars fetch failed {symbol}: {e}")
        return None

def get_spy_regime_from_snapshot():
    """Lightweight regime detection using SPY snapshot when bar feeds fail"""
    try:
        snap = get_snapshots(["SPY"]).get("SPY", {})
        price = snap.get("latestTrade", {}).get("p")
        prev  = snap.get("prevDailyBar", {})
        curr  = snap.get("dailyBar", {})
        if not price or not prev or not curr:
            return None
        prev_close = prev.get("c", price)
        chg_1d = (float(price) - float(prev_close)) / float(prev_close) * 100
        # Simple heuristic: use 1-day change as proxy when we lack history
        if chg_1d > 0.5:
            return "bull"
        elif chg_1d < -0.5:
            return "bear"
        else:
            return "neutral"
    except Exception as e:
        log.warning(f"SPY snapshot regime fallback failed: {e}")
        return None

def update_market_regime():
    """Detect market regime from SPY technical structure"""
    try:
        df = get_bars("SPY", "1Day", 220)

        # ── Fallback chain: IEX bars → SIP bars → snapshot heuristic ─────────
        if df is None or len(df) < 20:
            log.warning("IEX feed insufficient for SPY — trying SIP feed fallback")
            df = get_bars_sip("SPY", "1Day", 220)

        if df is None or len(df) < 20:
            log.warning("SIP feed also insufficient — using snapshot heuristic")
            regime = get_spy_regime_from_snapshot()
            if regime:
                old = state["market_regime"]
                state["market_regime"] = regime
                state["conf_threshold"] = (
                    min(85, CONF_THRESH + 10) if regime == "bear"
                    else max(55, CONF_THRESH - 5) if regime == "bull"
                    else CONF_THRESH
                )
                log.info(f"Regime (snapshot): {regime.upper()} | Threshold: {state['conf_threshold']}%")
                if old != regime:
                    telegram(f"🌍 Market regime: {old} → {regime.upper()} (snapshot)\nNew threshold: {state['conf_threshold']}%")
            else:
                log.warning("Regime detection skipped: all data sources failed")
            return

        df  = compute_indicators(df)
        row = df.iloc[-1]
        price = float(row["close"])

        # ── FIX: use .get() with NaN fallbacks instead of direct key access ───
        sma_50  = row.get("sma_50",  float("nan"))
        sma_200 = row.get("sma_200", float("nan"))
        macd    = row.get("macd",    float("nan"))
        rsi_val = row.get("rsi",     float("nan"))
        chg_20d_val = row.get("chg_20d", float("nan"))

        above_50  = not pd.isna(sma_50)  and price > float(sma_50)
        above_200 = not pd.isna(sma_200) and price > float(sma_200)
        rsi       = float(rsi_val)    if not pd.isna(rsi_val)    else 50
        chg_20d   = float(chg_20d_val) if not pd.isna(chg_20d_val) else 0
        macd_bull = float(macd) > 0   if not pd.isna(macd)       else False

        if above_50 and above_200 and chg_20d > 1 and macd_bull:
            regime = "bull"
        elif not above_50 and not above_200 and chg_20d < -1:
            regime = "bear"
        else:
            regime = "neutral"

        old = state["market_regime"]
        state["market_regime"] = regime
        state["regime_history"].append({"time": datetime.now().isoformat(), "regime": regime,
                                         "spy_rsi": round(rsi, 1), "spy_20d": round(chg_20d, 2)})
        state["regime_history"] = state["regime_history"][-90:]

        # Auto-adjust confidence threshold for regime
        if regime == "bear":
            state["conf_threshold"] = min(85, CONF_THRESH + 10)
        elif regime == "bull":
            state["conf_threshold"] = max(55, CONF_THRESH - 5)
        else:
            state["conf_threshold"] = CONF_THRESH

        if old != regime:
            log.info(f"🌍 Regime: {old} → {regime.upper()} | SPY RSI:{rsi:.1f} 20d:{chg_20d:+.1f}% | New threshold:{state['conf_threshold']}%")
            telegram(f"🌍 Market regime: {old} → {regime.upper()}\nSPY RSI:{rsi:.1f} | 20D:{chg_20d:+.1f}%\nNew conf threshold: {state['conf_threshold']}%")
        else:
            log.info(f"Regime: {regime.upper()} | SPY RSI:{rsi:.1f} 20d:{chg_20d:+.1f}%")

    except Exception as e:
        log.error(f"Regime detection failed: {e}")

# ─── MARKET HOURS ─────────────────────────────────────────────────────────────
def is_market_open():
    now  = datetime.now(EST)
    if now.weekday() >= 5: return False
    mins = now.hour * 60 + now.minute
    return 570 <= mins < 960   # 9:30 – 16:00

def market_time():
    return datetime.now(EST).strftime("%H:%M EST")

# ─── TELEGRAM ─────────────────────────────────────────────────────────────────
def telegram(msg):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                      json={"chat_id": TG_CHAT, "text": f"🤖 NEXUS v3\n{msg}\n⏰ {market_time()}"},
                      timeout=10)
    except Exception as e:
        log.error(f"Telegram failed: {e}")

# ─── AI AGENTS ────────────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=ANK) if ANK else None

AGENTS = {
    "SENTINEL":  {
        "role":  "market sentiment & news analyst",
        "focus": "Focus on news headlines, recent catalysts, short-term momentum. "
                 "Positive news + price action = BUY bias. Negative news / overextended = SELL bias.",
        "model": MODEL_MAIN,
    },
    "ORACLE": {
        "role":  "technical analyst",
        "focus": "Analyze the indicators systematically: RSI level, MACD crossovers, Bollinger Band position, "
                 "price vs moving averages, Stochastic. Look for confirmed technical setups.",
        "model": MODEL_MAIN,
    },
    "ARBITER": {
        "role":  "quantitative strategist",
        "focus": "Apply the active trading strategy rules precisely using the indicator values. "
                 "Identify momentum, mean reversion, or breakout signals from the data.",
        "model": MODEL_MAIN,
    },
    "CASSANDRA": {
        "role":  "risk & event analyst",
        "focus": "Identify risks: overbought RSI/Stochastic, earnings uncertainty from news, "
                 "extended price moves, bearish divergences, potential reversals.",
        "model": MODEL_MAIN,
    },
    "HERALD": {
        "role":  "macro & sector analyst",
        "focus": "Consider the market regime context. In bull markets, be more aggressive. "
                 "In bear markets, be very selective. Assess how this stock fits the macro environment.",
        "model": MODEL_MAIN,
    },
    "GUARDIAN": {
        "role":  "risk manager",
        "focus": "Your job is capital preservation. Consider ATR volatility, position risk, "
                 "proximity to overbought signals, and whether reward/risk is attractive.",
        "model": MODEL_MAIN,
    },
    "NEXUS": {
        "role":  "chief investment officer",
        "focus": "Synthesize all agent signals. Weight agents by their historical accuracy. "
                 "Make the final risk-adjusted decision.",
        "model": MODEL_OVERSEER,
    },
}

current_votes = {}

def ask_agent(agent_name, market_context, strategy_ctx="", extra=""):
    """Call Claude with a specific agent persona, returns {signal, confidence, reason}"""
    defn   = AGENTS.get(agent_name, AGENTS["ORACLE"])
    regime = f"\nMARKET REGIME: {state['market_regime'].upper()}"

    if agent_name == "NEXUS":
        votes_str   = "\n".join(f"  {a.upper()}: {v['signal']} ({v['confidence']}%) — {v['reason']}"
                                 for a, v in current_votes.items() if a != "nexus")
        weights_str = "\n".join(f"  {a}: weight={w:.2f} ({'↑ trusted' if w > 1.2 else '↓ penalised' if w < 0.7 else ''})"
                                 for a, w in state["agent_weights"].items())
        prompt = (
            f"You are NEXUS, chief investment officer.\n"
            f"{market_context}\n{regime}\n{strategy_ctx}\n\n"
            f"AGENT VOTES:\n{votes_str}\n\n"
            f"AGENT RELIABILITY WEIGHTS (learned from past outcomes):\n{weights_str}\n\n"
            f"Synthesize these votes, weighting more reliable agents higher. "
            f"Make the final decision.\n"
            f'Respond ONLY with JSON: {{"signal":"BUY|SELL|HOLD","confidence":0-100,"reason":"<12 words"}}'
        )
    else:
        prompt = (
            f"You are {agent_name}, a {defn['role']}.\n"
            f"{market_context}\n{regime}\n{strategy_ctx}\n\n"
            f"Your analytical focus: {defn['focus']}\n"
            f"Give your trading recommendation based on the real data above.\n"
            f'Respond ONLY with JSON: {{"signal":"BUY|SELL|HOLD","confidence":0-100,"reason":"<12 words"}}'
        )

    if not client:
        return _sim_signal()

    try:
        resp = client.messages.create(
            model=defn["model"], max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        text  = resp.content[0].text.strip().replace("```json","").replace("```","").strip()
        match = re.search(r'\{[^{}]+\}', text)
        if match:
            text = match.group()
        result = json.loads(text)
        result["confidence"] = max(0, min(100, int(result.get("confidence", 50))))
        if result.get("signal") not in ("BUY","SELL","HOLD"):
            result["signal"] = "HOLD"
        return result
    except Exception as e:
        log.warning(f"{agent_name} call failed: {e}")
        return _sim_signal()

def _sim_signal():
    import random
    r = random.random()
    s = "BUY" if r < 0.42 else "SELL" if r < 0.72 else "HOLD"
    return {"signal": s, "confidence": random.randint(40,80),
            "reason": {"BUY":"Simulated bullish signal","SELL":"Simulated bearish signal","HOLD":"Simulation: no edge"}[s]}

def get_weighted_decision():
    scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
    for agent_id, vote in current_votes.items():
        w = state["agent_weights"].get(agent_id, 1.0)
        scores[vote["signal"]] += (vote["confidence"] / 100.0) * w
    total = sum(scores.values()) or 1
    best  = max(scores, key=scores.get)
    conf  = round((scores[best] / total) * 100)
    return best, conf, scores

# ─── INDICATOR SNAPSHOT (for learning) ───────────────────────────────────────
def snapshot_indicators(row):
    """Which signals were active at trade time — used to learn which indicators predict wins"""
    try:
        def g(key): return float(row.get(key, float("nan")))
        rsi    = g("rsi");     macd_h = g("macd_hist"); macd_hp = g("macd_hist_prev")
        bb_pct = g("bb_pct"); bb_w   = g("bb_width");  vol_r   = g("vol_ratio")
        d50    = g("dist_sma50_pct");  d200 = g("dist_sma200_pct")
        w52    = g("52w_pct")
        return {
            "rsi_overbought":    not pd.isna(rsi)    and rsi    > 70,
            "rsi_oversold":      not pd.isna(rsi)    and rsi    < 30,
            "macd_bullish_cross":not pd.isna(macd_h) and not pd.isna(macd_hp) and macd_hp < 0 < macd_h,
            "macd_bearish_cross":not pd.isna(macd_h) and not pd.isna(macd_hp) and macd_hp > 0 > macd_h,
            "bb_squeeze":        not pd.isna(bb_w)   and bb_w  < 5,
            "bb_lower_band":     not pd.isna(bb_pct) and bb_pct < 20,
            "bb_upper_band":     not pd.isna(bb_pct) and bb_pct > 80,
            "volume_surge":      not pd.isna(vol_r)  and vol_r > 2.0,
            "above_sma50":       not pd.isna(d50)    and d50   > 0,
            "below_sma50":       not pd.isna(d50)    and d50   < 0,
            "above_sma200":      not pd.isna(d200)   and d200  > 0,
            "near_52w_high":     not pd.isna(w52)    and w52   > 85,
            "near_52w_low":      not pd.isna(w52)    and w52   < 15,
        }
    except Exception:
        return {}

# ─── LEARNING SYSTEM ─────────────────────────────────────────────────────────
def record_outcome(symbol, entry_price, exit_price, agent_signals, indicators=None):
    pnl_pct = (exit_price - entry_price) / entry_price * 100
    won     = pnl_pct >= 0
    outcome = "WIN" if won else "LOSS"

    if won: state["wins"]   += 1; state["streak"] = max(0, state["streak"]) + 1
    else:   state["losses"] += 1; state["streak"] = min(0, state["streak"]) - 1

    # ── Update agent weights ───────────────────────────────────────────────
    nexus_signal = agent_signals.get("nexus", {}).get("signal", "HOLD")
    for agent_id, vote in agent_signals.items():
        if agent_id == "nexus":
            continue
        w       = state["agent_weights"].get(agent_id, 1.0)
        agreed  = vote["signal"] == nexus_signal
        if   agreed and won:      w = min(2.5, w + 0.06)
        elif agreed and not won:  w = max(0.3, w - 0.04)
        elif not agreed and won:  w = max(0.3, w - 0.02)
        else:                     w = min(2.5, w + 0.03)   # correct contrarian gets small bump
        state["agent_weights"][agent_id] = round(w, 3)

    # ── Update indicator scores ────────────────────────────────────────────
    if indicators:
        for ind, was_active in indicators.items():
            if was_active:
                state["indicator_scores"][ind] = state["indicator_scores"].get(ind, 0) + (1 if won else -1)

    # ── Update strategy stats ──────────────────────────────────────────────
    if state.get("active_strategy"):
        strat = next((s for s in state["strategies"]
                      if s["id"] == state["active_strategy"]["id"]), None)
        if strat:
            if won: strat["wins"]   += 1
            else:   strat["losses"] += 1
            total = strat["wins"] + strat["losses"]
            if total % 5 == 0:
                rotate_strategy()

    # ── Self-tune confidence threshold ────────────────────────────────────
    total = state["wins"] + state["losses"]
    if total >= 5:
        recent    = state["trade_history"][-15:]
        recent_wr = sum(1 for t in recent if t.get("outcome") == "WIN") / max(len(recent), 1)
        if recent_wr < 0.35:
            state["conf_threshold"] = min(85, state["conf_threshold"] + 4)
            log.info(f"📈 Threshold ↑ {state['conf_threshold']}% (recent WR: {recent_wr:.0%})")
        elif recent_wr > 0.65:
            state["conf_threshold"] = max(50, state["conf_threshold"] - 2)
            log.info(f"📉 Threshold ↓ {state['conf_threshold']}% (recent WR: {recent_wr:.0%})")

    # ── Performance tracking ───────────────────────────────────────────────
    state["performance"]["total_pnl_pct"] += pnl_pct
    state["performance"]["total_trades"]  += 1

    state["trade_history"].append({
        "time": datetime.now().isoformat(), "symbol": symbol,
        "entry": round(entry_price, 4), "exit": round(exit_price, 4),
        "pnl_pct": round(pnl_pct, 3), "outcome": outcome,
        "strategy": state.get("active_strategy", {}).get("name", "—"),
        "regime":   state["market_regime"],
    })

    log.info(f"Outcome: {outcome} {symbol} ({pnl_pct:+.2f}%) | W:{state['wins']} L:{state['losses']} | Streak:{state['streak']:+d}")
    telegram(f"📊 {outcome}: {symbol} ({pnl_pct:+.2f}%)\nW:{state['wins']} L:{state['losses']} | Streak:{state['streak']:+d}\nThreshold:{state['conf_threshold']}%\nStrategy: {state.get('active_strategy',{}).get('name','—')}")
    save_state()

# ─── STRATEGY ENGINE ─────────────────────────────────────────────────────────
BUILTIN_STRATEGIES = [
    {"id":"momentum",     "name":"Momentum",              "wins":0,"losses":0,"source":"built-in",
     "description":"Buy stocks where RSI is 50-68, MACD histogram rising and positive, price above SMA20 and SMA50, volume >1.3x average. Momentum is confirmed and not overbought yet."},
    {"id":"meanrev",      "name":"Mean Reversion",         "wins":0,"losses":0,"source":"built-in",
     "description":"Buy when RSI < 35, price near lower Bollinger Band (BB% < 25), price below SMA20 but above SMA50. Expect reversion toward the mean. High-conviction oversold bounce."},
    {"id":"breakout",     "name":"Breakout",               "wins":0,"losses":0,"source":"built-in",
     "description":"Buy when 52-week position > 90% (near highs) with volume surge >2x. MACD must be bullish. Price breaking to new highs with strong volume confirmation."},
    {"id":"vwap_revert",  "name":"VWAP Reversion",         "wins":0,"losses":0,"source":"built-in",
     "description":"Buy when price is below VWAP and RSI is recovering from oversold. Sell when significantly extended above VWAP. Works best in choppy range-bound days."},
    {"id":"trend",        "name":"Trend Following",        "wins":0,"losses":0,"source":"built-in",
     "description":"Only trade when price is above SMA20 > SMA50 > SMA200 (full alignment). RSI 45-65. MACD positive and rising. Pure trend confirmation, no counter-trend trades."},
    {"id":"orb",          "name":"Opening Range Breakout", "wins":0,"losses":0,"source":"built-in",
     "description":"Enter on volume-confirmed breaks of first-hour high. Requires overall bullish market (SPY green) and RSI not overbought. Exit at end of day or stop."},
    {"id":"oversold",     "name":"Oversold Bounce",        "wins":0,"losses":0,"source":"built-in",
     "description":"RSI < 28 AND Stochastic < 20 AND BB% < 15. Triple oversold condition. High-volume capitulation selloff. Buy the extreme fear with tight stop below low."},
    {"id":"bb_squeeze",   "name":"Bollinger Squeeze",      "wins":0,"losses":0,"source":"built-in",
     "description":"Buy when BB width < 5% (compressed volatility) with MACD building above signal. Volume declining during squeeze. Anticipate volatility expansion breakout."},
]

def init_strategies():
    if not state["strategies"]:
        state["strategies"] = [dict(s) for s in BUILTIN_STRATEGIES]
    if not state["active_strategy"]:
        rotate_strategy()

def rotate_strategy():
    """UCB-based strategy selection: balances exploitation of winners with exploration"""
    import random
    strategies = state["strategies"]
    untested   = [s for s in strategies if s["wins"] + s["losses"] == 0]

    if untested and random.random() < 0.3:
        chosen = random.choice(untested)
        log.info(f"🧪 Testing new strategy: {chosen['name']}")
    else:
        tested = [s for s in strategies if s["wins"] + s["losses"] >= 3]
        if tested:
            total_all = sum(s["wins"] + s["losses"] for s in tested) or 1
            def ucb(s):
                n  = s["wins"] + s["losses"]
                wr = s["wins"] / n
                return wr + 0.3 * math.sqrt(2 * math.log(total_all) / n)
            chosen = max(tested, key=ucb)
            wr = chosen["wins"] / (chosen["wins"] + chosen["losses"]) * 100
            log.info(f"🏆 Strategy: {chosen['name']} ({wr:.0f}% WR, UCB selected)")
        elif strategies:
            chosen = random.choice(strategies)
            log.info(f"▶ Starting: {chosen['name']}")
        else:
            return

    state["active_strategy"] = chosen
    save_state()

def search_strategies():
    """Use Claude with web search to discover new quantitative strategies"""
    if not client:
        return
    log.info("🔍 Searching for new strategies...")
    try:
        # Include which indicators we're tracking so strategies are compatible
        ind_top = sorted(state["indicator_scores"].items(), key=lambda x: x[1], reverse=True)[:3]
        best_inds = [i[0] for i in ind_top]

        resp = client.messages.create(
            model=MODEL_OVERSEER, max_tokens=2000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content":
                f"Search for the most effective quantitative stock trading strategies in 2025 that use "
                f"technical indicators like RSI, MACD, Bollinger Bands, VWAP, ATR, Stochastic. "
                f"Our best-performing indicator signals so far: {best_inds}. "
                f"Find 4 specific strategies with precise indicator-based entry/exit rules. "
                f"Return ONLY a JSON array: "
                f'[{{"id":"unique_id","name":"Strategy Name",'
                f'"description":"Precise entry/exit rules using specific indicator values in 2 sentences",'
                f'"source":"website"}}]'}]
        )
        full_text = "".join(c.text for c in resp.content if hasattr(c, "text"))
        match = re.search(r'\[[\s\S]*?\]', full_text)
        if match:
            discovered = json.loads(match.group())
            added = 0
            for s in discovered:
                if isinstance(s, dict) and "id" in s and "name" in s:
                    if not any(x["id"] == s["id"] for x in state["strategies"]):
                        state["strategies"].append({**s, "wins": 0, "losses": 0})
                        added += 1
            log.info(f"✅ Added {added} strategies. Total: {len(state['strategies'])}")
            telegram(f"🔍 Strategy discovery: +{added} new\nLibrary: {len(state['strategies'])} strategies")
            save_state()
    except Exception as e:
        log.error(f"Strategy search failed: {e}")

# ─── POSITION SIZING ─────────────────────────────────────────────────────────
def calc_position_size(buying_power, atr_pct):
    """ATR-based sizing: smaller bets on volatile stocks, adjust for regime"""
    base = POS_SIZE_PCT
    regime = state["market_regime"]

    if regime == "bear":    base *= 0.60   # very cautious in bear
    elif regime == "bull":  base *= 1.10   # slight aggression in bull

    # Volatility adjustment — high ATR = reduce size
    if   atr_pct > 4.0: base *= 0.65
    elif atr_pct > 3.0: base *= 0.80
    elif atr_pct > 2.0: base *= 0.90
    elif atr_pct < 1.5: base *= 1.10   # low vol = can size up

    notional = buying_power * (base / 100)
    return max(10, min(notional, 12000))   # hard cap at $12k per trade

# ─── STOPS & TRAILING ────────────────────────────────────────────────────────
def update_trailing_stops(positions):
    """Ratchet trailing stops upward as positions profit"""
    if not USE_TRAILING:
        return
    for pos in positions:
        sym         = pos["symbol"]
        upl_pct     = float(pos["unrealized_plpc"]) * 100
        cur_price   = float(pos["current_price"])
        entry_price = float(pos["avg_entry_price"])

        if sym not in state["trailing_stops"]:
            state["trailing_stops"][sym] = entry_price * (1 - STOP_LOSS / 100)
            continue

        current_stop = state["trailing_stops"][sym]

        if upl_pct >= 5:
            # Move to breakeven + 1%
            new_stop = max(current_stop, entry_price * 1.01)
        elif upl_pct >= TAKE_PROFIT * 0.6:
            # Trail 5% below current
            new_stop = max(current_stop, cur_price * 0.95)
        elif upl_pct >= 3:
            # Trail 6% below current
            new_stop = max(current_stop, cur_price * 0.94)
        else:
            new_stop = current_stop

        state["trailing_stops"][sym] = round(new_stop, 4)

def check_stops():
    try:
        positions = get_positions()
        if not positions:
            return

        update_trailing_stops(positions)

        snapshots = get_snapshots([p["symbol"] for p in positions])
        for pos in positions:
            sym      = pos["symbol"]
            upl_pct  = float(pos["unrealized_plpc"]) * 100
            snap     = snapshots.get(sym, {})
            cur_price = float(snap.get("latestTrade", {}).get("p", pos["current_price"]))

            trailing  = state["trailing_stops"].get(sym)
            hit_stop  = False; reason = ""

            if trailing and cur_price <= trailing:
                hit_stop = True
                reason   = f"trailing stop @ ${trailing:.2f}"
            elif upl_pct <= -STOP_LOSS:
                hit_stop = True
                reason   = f"hard stop ({upl_pct:.1f}%)"

            if hit_stop:
                log.warning(f"🛑 STOP {sym}: {reason}")
                try:
                    close_position(sym)
                    state["trailing_stops"].pop(sym, None)
                    telegram(f"🛑 STOP: {sym} — {reason}\nPrice: ${cur_price:.2f}")
                    _eval_pending(sym, cur_price)
                except Exception as e:
                    log.error(f"Stop close failed {sym}: {e}")

            elif upl_pct >= TAKE_PROFIT:
                log.info(f"🎯 TAKE PROFIT {sym} +{upl_pct:.1f}%")
                try:
                    close_position(sym)
                    state["trailing_stops"].pop(sym, None)
                    telegram(f"🎯 TAKE PROFIT: {sym} +{upl_pct:.1f}%\nPrice: ${cur_price:.2f}")
                    _eval_pending(sym, cur_price)
                except Exception as e:
                    log.error(f"Take profit close failed {sym}: {e}")

        save_state()
    except Exception as e:
        log.error(f"Stop check failed: {e}")

def _eval_pending(symbol, exit_price):
    for p in [x for x in state["pending_evals"] if x["symbol"] == symbol]:
        record_outcome(symbol, p["entry_price"], exit_price,
                       p.get("agent_signals", {}), p.get("indicators", {}))
        state["pending_evals"].remove(p)
    save_state()

# ─── PERFORMANCE METRICS ─────────────────────────────────────────────────────
def compute_metrics():
    history = state["trade_history"]
    if len(history) < 3:
        return {}

    wins   = [t for t in history if t.get("outcome") == "WIN"]
    losses = [t for t in history if t.get("outcome") == "LOSS"]
    total  = len(wins) + len(losses)
    wr     = len(wins) / total * 100 if total else 0

    avg_win  = sum(t.get("pnl_pct", 0) for t in wins)  / len(wins)   if wins   else 0
    avg_loss = sum(t.get("pnl_pct", 0) for t in losses) / len(losses) if losses else 0
    pf       = (avg_win * len(wins)) / max(abs(avg_loss * len(losses)), 0.01)

    daily   = state["performance"].get("daily_returns", [])
    sharpe  = None
    if len(daily) > 10:
        import statistics
        mu = statistics.mean(daily)
        sd = statistics.stdev(daily) or 1e-9
        sharpe = (mu / sd) * (252 ** 0.5)

    top_strats = sorted(
        [(s["name"], s["wins"] / (s["wins"]+s["losses"]) * 100, s["wins"]+s["losses"])
         for s in state["strategies"] if s["wins"]+s["losses"] >= 2],
        key=lambda x: x[1], reverse=True
    )[:3]

    top_inds = sorted(state["indicator_scores"].items(), key=lambda x: x[1], reverse=True)[:5]

    return {"win_rate": wr, "total": total, "avg_win": avg_win, "avg_loss": avg_loss,
            "profit_factor": pf, "sharpe": sharpe, "top_strategies": top_strats,
            "top_indicators": top_inds, "max_drawdown": state["performance"]["max_drawdown"]}

def snapshot_equity():
    try:
        acct   = get_account()
        equity = float(acct["equity"])
        prev   = state["equity_curve"][-1]["equity"] if state["equity_curve"] else equity
        daily_ret = (equity - prev) / prev
        state["performance"]["daily_returns"].append(daily_ret)
        state["equity_curve"].append({"time": datetime.now().isoformat(), "equity": equity})
        state["equity_curve"] = state["equity_curve"][-365:]

        peak = state["performance"]["peak_equity"]
        state["performance"]["peak_equity"]  = max(peak, equity)
        state["performance"]["max_drawdown"] = max(
            state["performance"]["max_drawdown"],
            (state["performance"]["peak_equity"] - equity) / state["performance"]["peak_equity"] * 100
        )
    except Exception as e:
        log.error(f"Equity snapshot failed: {e}")

# ─── SCAN & ANALYSIS ─────────────────────────────────────────────────────────
def run_full_scan():
    """Parallel scan of all watchlist stocks with real indicator data"""
    log.info(f"🔍 SCAN — {len(WATCHLIST)} stocks | Regime: {state['market_regime'].upper()} | Threshold: {state['conf_threshold']}%")
    snapshots = get_snapshots(WATCHLIST)
    results   = []

    def quick_scan(sym):
        try:
            snap  = snapshots.get(sym, {})
            price = snap.get("latestTrade", {}).get("p") or snap.get("minuteBar", {}).get("c")
            if not price:
                return None

            df = get_bars(sym, "1Day", 60)
            if df is not None and len(df) >= 20:
                df  = compute_indicators(df)
                row = df.iloc[-1]
                ctx = (f"{sym} @ ${price:.2f}\n"
                       f"RSI:{safe_fmt(row,'rsi','.1f')} | "
                       f"MACD hist:{safe_fmt(row,'macd_hist','+.3f')} | "
                       f"BB%:{safe_fmt(row,'bb_pct','.0f')} | "
                       f"1D:{safe_fmt(row,'chg_1d','+.1f')}% | "
                       f"Vol:{safe_fmt(row,'vol_ratio','.1f')}x avg | "
                       f"vs SMA50:{safe_fmt(row,'dist_sma50_pct','+.1f')}%")
            else:
                row = None
                ctx = f"{sym} @ ${price:.2f}"

            result  = ask_agent("ORACLE", ctx)
            w       = state["agent_weights"].get("oracle", 1.0)
            w_conf  = min(100, int(result["confidence"] * w))
            return {"sym": sym, "price": float(price),
                    "signal": result["signal"], "confidence": result["confidence"],
                    "weighted_conf": w_conf, "reason": result["reason"], "row": row}
        except Exception as e:
            log.warning(f"Quick scan failed {sym}: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(quick_scan, sym): sym for sym in WATCHLIST}
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            if r:
                results.append(r)
                icon = "✅" if r["signal"]=="BUY" else "🔴" if r["signal"]=="SELL" else "⏸"
                log.info(f"  {r['sym']:6} ${r['price']:8.2f}  {icon} {r['signal']:4}  {r['confidence']:3}%  {r['reason']}")

    buys  = sorted([r for r in results if r["signal"]=="BUY"],  key=lambda x: x["weighted_conf"], reverse=True)
    sells = sorted([r for r in results if r["signal"]=="SELL"], key=lambda x: x["weighted_conf"], reverse=True)

    top   = sorted(results, key=lambda x: x["confidence"], reverse=True)[:6]
    telegram("📊 SCAN: " + " | ".join(f"{r['sym']}:{r['signal']}({r['confidence']}%)" for r in top))

    return buys, sells

def safe_fmt(row, key, fmt):
    try:
        v = float(row[key])
        if pd.isna(v): return "N/A"
        return f"{v:{fmt}}"
    except Exception:
        return "N/A"

def run_deep_analysis(sym, quick_row=None):
    """7-agent deep analysis with full real market data"""
    global current_votes
    current_votes = {}

    log.info(f"\n{'═'*55}")
    log.info(f"DEEP ANALYSIS: {sym} | {market_time()} | Regime: {state['market_regime'].upper()}")

    # Fetch all data
    df_daily  = get_bars(sym, "1Day", 220)
    df_hourly = get_bars(sym, "1Hour", 48) if is_market_open() else None

    if df_daily is not None and len(df_daily) >= 20:
        df_daily = compute_indicators(df_daily)
        price    = float(df_daily.iloc[-1]["close"])
        ind_snap = snapshot_indicators(df_daily.iloc[-1].to_dict())
        atr_pct  = float(df_daily.iloc[-1].get("atr_pct", 2.0))
        if pd.isna(atr_pct): atr_pct = 2.0
    elif quick_row:
        price    = quick_row["price"]
        ind_snap = {}
        atr_pct  = 2.0
    else:
        log.error(f"No price data for {sym}")
        return None

    news = get_news(sym, 8)

    strategy_ctx = ""
    if state.get("active_strategy"):
        s = state["active_strategy"]
        strategy_ctx = f'\nACTIVE STRATEGY: "{s["name"]}" — {s["description"]}'

    market_context = format_indicators(sym, df_daily, price, news, df_hourly)

    state["cycle"] += 1
    log.info(f"Cycle #{state['cycle']} | Strategy: {state.get('active_strategy',{}).get('name','—')}")
    log.info(f"{'═'*55}")

    # Run all 6 sub-agents
    for agent in ["SENTINEL","ORACLE","ARBITER","CASSANDRA","HERALD","GUARDIAN"]:
        result = ask_agent(agent, market_context, strategy_ctx)
        current_votes[agent.lower()] = result
        icon = "✅" if result["signal"]=="BUY" else "🔴" if result["signal"]=="SELL" else "⏸"
        log.info(f"  [{agent:9}] {icon} {result['signal']:4} | {result['confidence']:3}% | {result['reason']}")

    # Weighted decision
    w_signal, w_conf, scores = get_weighted_decision()
    log.info(f"\n  Weighted — BUY:{scores['BUY']:.2f} SELL:{scores['SELL']:.2f} HOLD:{scores['HOLD']:.2f}")

    # NEXUS final decision
    nexus = ask_agent("NEXUS", market_context, strategy_ctx)
    current_votes["nexus"] = nexus

    # Override if learned weights strongly disagree with NEXUS
    if w_signal != nexus["signal"] and w_conf > 72:
        log.info(f"  ⚡ Weight override: {nexus['signal']} → {w_signal}")
        nexus["signal"]     = w_signal
        nexus["confidence"] = (nexus["confidence"] + w_conf) // 2

    # Apply confidence threshold
    if nexus["confidence"] < state["conf_threshold"] and nexus["signal"] != "HOLD":
        log.info(f"  ⚠ Conf {nexus['confidence']}% < threshold {state['conf_threshold']}% → HOLD")
        nexus["signal"] = "HOLD"

    # Bear market filter — only very high conviction buys
    if state["market_regime"] == "bear" and nexus["signal"] == "BUY" and nexus["confidence"] < 82:
        log.info(f"  🐻 Bear filter: BUY {nexus['confidence']}% → HOLD")
        nexus["signal"] = "HOLD"

    log.info(f"\n  NEXUS FINAL: {nexus['signal']} {sym} | {nexus['confidence']}% | {nexus['reason']}")
    log.info(f"{'═'*55}\n")

    return {"signal": nexus["signal"], "confidence": nexus["confidence"],
            "symbol": sym, "price": price, "reason": nexus["reason"],
            "agent_signals": dict(current_votes),
            "indicators": ind_snap, "atr_pct": atr_pct}

def execute_trade(decision):
    if not AUTO_EXEC:
        log.info("Auto-execute OFF — skipping")
        return

    sym    = decision["symbol"]
    signal = decision["signal"]
    price  = decision["price"]
    conf   = decision["confidence"]
    atr    = decision.get("atr_pct", 2.0)

    if signal == "HOLD":
        return

    try:
        if signal == "BUY":
            acct      = get_account()
            bp        = float(acct["buying_power"])
            notional  = calc_position_size(bp, atr)
            if notional < 10:
                log.warning("Insufficient buying power")
                return

            positions = get_positions()
            if any(p["symbol"] == sym for p in positions):
                log.info(f"Already holding {sym}")
                return
            if len(positions) >= MAX_POSITIONS:
                log.info(f"Max positions ({MAX_POSITIONS}) reached")
                return

            order      = place_buy(sym, notional)
            stop_price = price * (1 - STOP_LOSS / 100)
            state["trailing_stops"][sym] = round(stop_price, 4)

            log.info(f"✅ BUY: ${notional:.2f} of {sym} @ ~${price:.2f} | Stop: ${stop_price:.2f} | ATR:{atr:.1f}%")
            telegram(f"✅ BUY: {sym} @ ${price:.2f}\n"
                     f"${notional:.2f} | Conf:{conf}% | ATR:{atr:.1f}%\n"
                     f"Stop: ${stop_price:.2f}\n"
                     f"Strategy: {state.get('active_strategy',{}).get('name','—')}\n"
                     f"Regime: {state['market_regime'].upper()}")

            state["pending_evals"].append({
                "symbol": sym, "entry_price": price,
                "agent_signals": decision["agent_signals"],
                "indicators":    decision.get("indicators", {}),
                "time":          datetime.now().isoformat(),
            })
            save_state()

        elif signal == "SELL":
            positions = get_positions()
            pos = next((p for p in positions if p["symbol"] == sym), None)
            if not pos:
                log.info(f"No position in {sym} to sell")
                return

            upl = float(pos["unrealized_pl"])
            close_position(sym)
            state["trailing_stops"].pop(sym, None)

            log.info(f"✅ SELL: {sym} | P&L: ${upl:+.2f}")
            telegram(f"✅ SELL: {sym}\nP&L: ${upl:+.2f} | Conf:{conf}%")
            _eval_pending(sym, price)

    except Exception as e:
        log.error(f"Trade execution failed {sym}: {e}")
        telegram(f"❌ Trade failed: {sym} {signal}\n{e}")

# ─── STATUS REPORT ────────────────────────────────────────────────────────────
def send_status_report():
    try:
        acct      = get_account()
        equity    = float(acct["equity"])
        cash      = float(acct["cash"])
        pnl       = equity - 100000
        positions = get_positions()

        pos_str = "\n".join(
            f"  {p['symbol']}: {float(p['unrealized_plpc'])*100:+.1f}% (${float(p['unrealized_pl']):+.0f})"
            for p in positions
        ) or "  None"

        m        = compute_metrics()
        wr       = state["wins"] / max(state["wins"]+state["losses"], 1) * 100
        sharpe   = f"\nSharpe: {m['sharpe']:.2f}" if m.get("sharpe") else ""
        pf       = f"\nProfit factor: {m.get('profit_factor',0):.2f}" if m.get("profit_factor") else ""
        dd       = f"\nMax drawdown: {m.get('max_drawdown',0):.1f}%"

        top_strats = "\n".join(f"  {s[0]}: {s[1]:.0f}% WR ({s[2]} trades)" for s in m.get("top_strategies",[])) or "  None yet"
        top_inds   = ", ".join(f"{i[0]}({i[1]:+d})" for i in m.get("top_indicators",[])[:4]) or "—"

        msg = (f"📈 NEXUS v3 STATUS\n"
               f"━━━━━━━━━━━━━━━━━━━━\n"
               f"Equity: ${equity:,.2f}\n"
               f"P&L:    ${pnl:+,.2f} ({pnl/1000:.1f}%)\n"
               f"Cash:   ${cash:,.2f}\n"
               f"━━━━━━━━━━━━━━━━━━━━\n"
               f"WR: {wr:.1f}% ({state['wins']}W/{state['losses']}L)\n"
               f"Streak: {state['streak']:+d} | Threshold: {state['conf_threshold']}%"
               f"{sharpe}{pf}{dd}\n"
               f"Regime: {state['market_regime'].upper()}\n"
               f"━━━━━━━━━━━━━━━━━━━━\n"
               f"Top strategies:\n{top_strats}\n"
               f"Top indicators: {top_inds}\n"
               f"━━━━━━━━━━━━━━━━━━━━\n"
               f"Positions:\n{pos_str}")
        telegram(msg)
        log.info(f"Status: Equity=${equity:,.2f} | WR={wr:.1f}% | Regime={state['market_regime']} | Cycle={state['cycle']}")
    except Exception as e:
        log.error(f"Status report failed: {e}")

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def main():
    log.info("═" * 55)
    log.info("  NEXUS v3 — Autonomous Multi-Agent Trading Bot")
    log.info("  Real indicators • Regime-aware • Adaptive learning")
    log.info("═" * 55)

    if not AK or not SK:
        log.error("Missing ALPACA_KEY or ALPACA_SECRET!")
        return
    if not ANK:
        log.warning("No ANTHROPIC_KEY — running in simulation mode")

    load_state()
    init_strategies()

    try:
        acct = get_account()
        log.info(f"✅ Alpaca: {acct['account_number']} | Cash: ${float(acct['cash']):,.2f}")
        telegram(f"🚀 NEXUS v3 started!\nCash: ${float(acct['cash']):,.2f}\n"
                 f"Watchlist: {', '.join(WATCHLIST)}\nInterval: {INTERVAL_MIN}min")
    except Exception as e:
        log.error(f"Alpaca connect failed: {e}")
        return

    # Startup tasks
    update_market_regime()
    if len(state["strategies"]) <= len(BUILTIN_STRATEGIES):
        search_strategies()

    log.info(f"\nWatchlist: {', '.join(WATCHLIST)}")
    log.info(f"Regime:    {state['market_regime'].upper()} | Threshold: {state['conf_threshold']}%")
    log.info(f"Strategy:  {state.get('active_strategy',{}).get('name','—')}\n")

    last_scan         = 0
    last_regime_check = 0
    last_equity_snap  = 0
    last_stop_check   = 0
    status_counter    = 0

    while True:
        try:
            now = time.time()

            # Stop checks every 3 minutes
            if now - last_stop_check >= 180:
                check_stops()
                last_stop_check = now

            # Regime update every 2 hours
            if now - last_regime_check >= 7200:
                update_market_regime()
                last_regime_check = now

            # Equity snapshot every hour
            if now - last_equity_snap >= 3600:
                snapshot_equity()
                last_equity_snap = now

            if is_market_open():
                if now - last_scan >= INTERVAL_MIN * 60:
                    last_scan = now
                    buys, sells = run_full_scan()

                    # Act on best BUY opportunity
                    if buys and buys[0]["confidence"] >= state["conf_threshold"]:
                        decision = run_deep_analysis(buys[0]["sym"], buys[0])
                        if decision and decision["signal"] == "BUY":
                            execute_trade(decision)

                    # Act on best SELL signal
                    if sells and sells[0]["confidence"] >= state["conf_threshold"]:
                        decision = run_deep_analysis(sells[0]["sym"], sells[0])
                        if decision and decision["signal"] == "SELL":
                            execute_trade(decision)

                    status_counter += 1
                    if status_counter % 4 == 0:
                        send_status_report()

                    # Search for new strategies periodically
                    if state["cycle"] > 0 and state["cycle"] % 50 == 0:
                        search_strategies()

            else:
                if now - last_scan >= 3600:
                    last_scan = now
                    log.info(f"Market closed ({market_time()}) — waiting for 9:30AM EST")

            time.sleep(60)

        except KeyboardInterrupt:
            log.info("\nStopping NEXUS v3...")
            send_status_report()
            save_state()
            break
        except Exception as e:
            log.error(f"Main loop error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()

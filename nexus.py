"""
NEXUS v2 — Autonomous Multi-Agent Trading Bot
Optimized for momentum swing trading (not day trading)
Runs 24/7 on cloud — no browser needed
"""

import os, json, time, asyncio, logging, requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import anthropic

# ─── CONFIG (from environment variables) ────────────────────────────────────
AK           = os.environ.get("ALPACA_KEY", "")
SK           = os.environ.get("ALPACA_SECRET", "")
ANK          = os.environ.get("ANTHROPIC_KEY", "")
TG_TOKEN     = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT      = os.environ.get("TELEGRAM_CHAT", "")

WATCHLIST    = os.environ.get("WATCHLIST", "AAPL,TSLA,NVDA,MSFT,AMZN,META,GOOGL,AMD").split(",")
INTERVAL_MIN = int(os.environ.get("INTERVAL_MIN", "60"))   # analyze every 60 min
POS_SIZE_PCT = float(os.environ.get("POS_SIZE_PCT", "15"))  # 15% per position
STOP_LOSS    = float(os.environ.get("STOP_LOSS_PCT", "4"))  # 4% stop loss
TAKE_PROFIT  = float(os.environ.get("TAKE_PROFIT_PCT", "10")) # 10% take profit
CONF_THRESH  = int(os.environ.get("CONF_THRESHOLD", "70"))  # 70% min confidence
AUTO_EXEC    = os.environ.get("AUTO_EXECUTE", "true").lower() == "true"

ALPACA_BASE  = "https://paper-api.alpaca.markets"
ALPACA_DATA  = "https://data.alpaca.markets"
EST          = ZoneInfo("America/New_York")

# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("nexus.log")]
)
log = logging.getLogger("NEXUS")

# ─── STATE ────────────────────────────────────────────────────────────────────
state = {
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
        "guardian":  1.5,
    },
    "strategies": [],
    "active_strategy": None,
    "pending_evals": [],   # trades waiting for outcome check
    "conf_threshold": CONF_THRESH,
    "trade_history": [],
}

STATE_FILE = "nexus_state.json"

def save_state():
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        log.error(f"State save failed: {e}")

def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                saved = json.load(f)
            for k, v in saved.items():
                state[k] = v
            log.info(f"State loaded — {state['wins']}W/{state['losses']}L | Cycle: {state['cycle']}")
    except Exception as e:
        log.error(f"State load failed: {e}")

# ─── ALPACA HELPERS ───────────────────────────────────────────────────────────
def alpaca(path, method="GET", body=None):
    headers = {
        "APCA-API-KEY-ID": AK,
        "APCA-API-SECRET-KEY": SK,
        "Content-Type": "application/json"
    }
    url = ALPACA_BASE + path
    resp = requests.request(method, url, headers=headers, json=body, timeout=15)
    resp.raise_for_status()
    return resp.json()

def alpaca_data(path):
    headers = {"APCA-API-KEY-ID": AK, "APCA-API-SECRET-KEY": SK}
    resp = requests.get(ALPACA_DATA + path, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()

def get_price(symbol):
    try:
        data = alpaca_data(f"/v2/stocks/trades/latest?symbols={symbol}")
        return data["trades"][symbol]["p"]
    except:
        return None

def get_prices(symbols):
    try:
        syms = ",".join(symbols)
        data = alpaca_data(f"/v2/stocks/trades/latest?symbols={syms}")
        return {s: data["trades"][s]["p"] for s in symbols if s in data.get("trades", {})}
    except Exception as e:
        log.error(f"Price fetch failed: {e}")
        return {}

def get_account():
    return alpaca("/v2/account")

def get_positions():
    return alpaca("/v2/positions")

def place_buy(symbol, notional):
    return alpaca("/v2/orders", "POST", {
        "symbol": symbol,
        "notional": str(round(notional, 2)),
        "side": "buy",
        "type": "market",
        "time_in_force": "day"
    })

def close_position(symbol):
    return alpaca(f"/v2/positions/{symbol}", "DELETE")

# ─── MARKET HOURS ─────────────────────────────────────────────────────────────
def is_market_open():
    now = datetime.now(EST)
    if now.weekday() >= 5:
        return False
    h, m = now.hour, now.minute
    mins = h * 60 + m
    return 570 <= mins < 960  # 9:30 - 16:00

def market_open_time():
    now = datetime.now(EST)
    return now.strftime("%H:%M EST")

# ─── TELEGRAM ─────────────────────────────────────────────────────────────────
def telegram(msg):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": f"🤖 NEXUS\n{msg}\n⏰ {market_open_time()}"},
            timeout=10
        )
    except Exception as e:
        log.error(f"Telegram failed: {e}")

# ─── ANTHROPIC AI ─────────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=ANK) if ANK else None

def ask_agent(agent_name, symbol, price, change, extra_context=""):
    """Call Claude with agent-specific prompt, returns {signal, confidence, reason}"""
    strategy_ctx = ""
    if state.get("active_strategy"):
        s = state["active_strategy"]
        strategy_ctx = f'\n\nACTIVE STRATEGY: "{s["name"]}" — {s["description"]}. Apply this strategy\'s rules.'

    news_ctx = extra_context or ""

    prompts = {
        "SENTINEL":  f"You are SENTINEL, market sentiment analyst. {symbol} @ ${price:.2f}, {change:+.2f}% today.{strategy_ctx}{news_ctx} Give BUY/SELL/HOLD. JSON only: {{\"signal\":\"BUY|SELL|HOLD\",\"confidence\":0-100,\"reason\":\"<10 words\"}}",
        "ORACLE":    f"You are ORACLE, technical analyst. {symbol} @ ${price:.2f}, {change:+.2f}% today.{strategy_ctx} Analyze RSI, MACD, MAs. Give BUY/SELL/HOLD. JSON only: {{\"signal\":\"BUY|SELL|HOLD\",\"confidence\":0-100,\"reason\":\"<10 words\"}}",
        "ARBITER":   f"You are ARBITER, quant strategist. {symbol} @ ${price:.2f}, {change:+.2f}% today.{strategy_ctx} Apply momentum, mean reversion. Give BUY/SELL/HOLD. JSON only: {{\"signal\":\"BUY|SELL|HOLD\",\"confidence\":0-100,\"reason\":\"<10 words\"}}",
        "CASSANDRA": f"You are CASSANDRA, earnings/events analyst. {symbol} @ ${price:.2f}.{strategy_ctx} Consider upcoming catalysts. Give BUY/SELL/HOLD. JSON only: {{\"signal\":\"BUY|SELL|HOLD\",\"confidence\":0-100,\"reason\":\"<10 words\"}}",
        "HERALD":    f"You are HERALD, macro analyst. {symbol} @ ${price:.2f}, {change:+.2f}% today.{strategy_ctx} Consider Fed, rates, macro. Give BUY/SELL/HOLD. JSON only: {{\"signal\":\"BUY|SELL|HOLD\",\"confidence\":0-100,\"reason\":\"<10 words\"}}",
        "GUARDIAN":  f"You are GUARDIAN, risk manager. {symbol} @ ${price:.2f}, {change:+.2f}% today.{strategy_ctx} Be conservative. Give BUY/SELL/HOLD. JSON only: {{\"signal\":\"BUY|SELL|HOLD\",\"confidence\":0-100,\"reason\":\"<10 words\"}}",
        "NEXUS":     f"You are NEXUS, overseer AI. {symbol} @ ${price:.2f}.{strategy_ctx} Agent votes: {get_agent_votes_str()}. Make FINAL decision. JSON only: {{\"signal\":\"BUY|SELL|HOLD\",\"confidence\":0-100,\"reason\":\"<10 words\"}}",
    }

    if not client:
        return sim_signal()

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{"role": "user", "content": prompts[agent_name]}]
        )
        text = resp.content[0].text.strip().replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception as e:
        log.warning(f"{agent_name} AI call failed: {e} — using simulation")
        return sim_signal()

def sim_signal():
    import random
    r = random.random()
    signal = "BUY" if r < 0.42 else "SELL" if r < 0.72 else "HOLD"
    conf = random.randint(40, 85)
    reasons = {
        "BUY":  "Bullish momentum detected.",
        "SELL": "Bearish divergence detected.",
        "HOLD": "Mixed signals, awaiting confirmation."
    }
    return {"signal": signal, "confidence": conf, "reason": reasons[signal]}

# ─── AGENT VOTE TRACKING ──────────────────────────────────────────────────────
current_votes = {}

def get_agent_votes_str():
    return ", ".join([f"{a}:{v['signal']}({v['confidence']}%)" for a, v in current_votes.items()])

def get_weighted_decision():
    """Apply learned agent weights to votes"""
    scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
    for agent_id, vote in current_votes.items():
        weight = state["agent_weights"].get(agent_id, 1.0)
        scores[vote["signal"]] += (vote["confidence"] / 100.0) * weight
    total = sum(scores.values()) or 1
    best = max(scores, key=scores.get)
    conf = round((scores[best] / total) * 100)
    return best, conf, scores

# ─── LEARNING SYSTEM ─────────────────────────────────────────────────────────
def record_outcome(symbol, entry_price, current_price, agent_signals_snapshot):
    pnl_pct = ((current_price - entry_price) / entry_price) * 100
    won = pnl_pct >= 0
    outcome = "WIN" if won else "LOSS"

    if won:
        state["wins"] += 1
        state["streak"] = max(0, state["streak"]) + 1
    else:
        state["losses"] += 1
        state["streak"] = min(0, state["streak"]) - 1

    # Update agent weights
    nexus_signal = agent_signals_snapshot.get("nexus", {}).get("signal", "HOLD")
    for agent_id, vote in agent_signals_snapshot.items():
        if agent_id == "nexus":
            continue
        w = state["agent_weights"].get(agent_id, 1.0)
        agreed = vote["signal"] == nexus_signal
        if agreed and won:
            w = min(2.5, w + 0.05)
        elif agreed and not won:
            w = max(0.3, w - 0.03)
        elif not agreed and won:
            w = max(0.3, w - 0.02)
        else:
            w = min(2.5, w + 0.02)
        state["agent_weights"][agent_id] = round(w, 3)

    # Update active strategy
    if state.get("active_strategy"):
        strat = next((s for s in state["strategies"] if s["id"] == state["active_strategy"]["id"]), None)
        if strat:
            if won: strat["wins"] += 1
            else:   strat["losses"] += 1
            total = strat["wins"] + strat["losses"]
            wr = strat["wins"] / total * 100
            if total % 5 == 0:
                rotate_strategy()

    # Self-tune confidence threshold
    total = state["wins"] + state["losses"]
    if total >= 3:
        recent = state["trade_history"][-10:]
        recent_wr = sum(1 for t in recent if t.get("outcome") == "WIN") / max(len(recent), 1)
        if recent_wr < 0.4:
            state["conf_threshold"] = min(80, state["conf_threshold"] + 3)
            log.info(f"📈 Raising threshold to {state['conf_threshold']}% (recent WR: {recent_wr:.0%})")
        elif recent_wr > 0.65:
            state["conf_threshold"] = max(50, state["conf_threshold"] - 2)
            log.info(f"📉 Lowering threshold to {state['conf_threshold']}% (recent WR: {recent_wr:.0%})")

    log.info(f"Outcome: {outcome} {symbol} ({pnl_pct:+.2f}%) | W:{state['wins']} L:{state['losses']} | Streak:{state['streak']}")
    telegram(f"📊 Trade outcome: {outcome} {symbol} ({pnl_pct:+.2f}%)\nW:{state['wins']} L:{state['losses']} | Streak:{state['streak']}\nStrategy: {state.get('active_strategy',{}).get('name','—')}")

    save_state()

# ─── STRATEGY ENGINE ──────────────────────────────────────────────────────────
BUILTIN_STRATEGIES = [
    {"id":"momentum",    "name":"Momentum Trading",    "description":"Buy breaking out stocks with high volume and strong upward movement", "wins":0,"losses":0,"source":"built-in"},
    {"id":"meanrev",     "name":"Mean Reversion",       "description":"Buy oversold stocks expecting price to revert to average; sell overbought", "wins":0,"losses":0,"source":"built-in"},
    {"id":"breakout",    "name":"Breakout Trading",     "description":"Enter when price breaks above resistance with volume confirmation", "wins":0,"losses":0,"source":"built-in"},
    {"id":"vwap",        "name":"VWAP Strategy",        "description":"Trade relative to VWAP; buy below VWAP in uptrend, sell above in downtrend", "wins":0,"losses":0,"source":"built-in"},
    {"id":"news_play",   "name":"News Catalyst Play",   "description":"Enter after major positive news before full price discovery completes", "wins":0,"losses":0,"source":"built-in"},
    {"id":"trend_follow","name":"Trend Following",      "description":"Identify and ride established trends using moving averages and momentum", "wins":0,"losses":0,"source":"built-in"},
    {"id":"gap_fill",    "name":"Gap Fill Strategy",    "description":"Trade stocks that gap up at open expecting partial gap fill intraday", "wins":0,"losses":0,"source":"built-in"},
    {"id":"orb",         "name":"Opening Range Breakout","description":"Trade breakout of first 30min high/low range with volume confirmation", "wins":0,"losses":0,"source":"built-in"},
]

def init_strategies():
    if not state["strategies"]:
        state["strategies"] = BUILTIN_STRATEGIES.copy()
    if not state["active_strategy"]:
        rotate_strategy()

def rotate_strategy():
    import random
    strategies = state["strategies"]
    untested = [s for s in strategies if s["wins"] + s["losses"] == 0]

    if untested and random.random() < 0.4:
        chosen = random.choice(untested)
        log.info(f"🧪 Testing new strategy: {chosen['name']}")
    else:
        tested = [s for s in strategies if s["wins"] + s["losses"] >= 2]
        if tested:
            tested.sort(key=lambda s: s["wins"] / (s["wins"] + s["losses"]), reverse=True)
            # 70% chance to use best, 30% to explore second/third best
            if len(tested) > 1 and random.random() < 0.3:
                chosen = tested[1] if len(tested) > 1 else tested[0]
            else:
                chosen = tested[0]
            wr = chosen["wins"] / (chosen["wins"] + chosen["losses"]) * 100
            log.info(f"🏆 Best strategy: {chosen['name']} ({wr:.0f}% WR)")
        else:
            chosen = random.choice(strategies)
            log.info(f"▶ Starting with: {chosen['name']}")

    state["active_strategy"] = chosen
    save_state()

def search_strategies():
    """Use Claude with web search to discover new strategies"""
    if not client:
        return
    log.info("🔍 Searching web for new trading strategies...")
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": "Search the web for the most effective day trading and swing trading strategies used by professional traders in 2025. Find 5 specific named strategies with clear entry/exit rules. Return ONLY a JSON array: [{\"id\":\"unique_id\",\"name\":\"Strategy Name\",\"description\":\"Entry/exit rules in 1-2 sentences\",\"source\":\"website\"}]"}]
        )
        full_text = "".join(c.text for c in resp.content if hasattr(c, "text"))
        import re
        match = re.search(r'\[[\s\S]*?\]', full_text)
        if match:
            discovered = json.loads(match.group())
            added = 0
            for s in discovered:
                if not any(x["id"] == s["id"] for x in state["strategies"]):
                    state["strategies"].append({**s, "wins": 0, "losses": 0})
                    added += 1
            log.info(f"✅ Discovered {added} new strategies. Library: {len(state['strategies'])} total")
            telegram(f"🔍 Strategy search complete!\nDiscovered {added} new strategies\nLibrary: {len(state['strategies'])} total")
            save_state()
    except Exception as e:
        log.error(f"Strategy search failed: {e}")

# ─── STOP LOSS / TAKE PROFIT ──────────────────────────────────────────────────
def check_stops():
    try:
        positions = get_positions()
        prices = get_prices([p["symbol"] for p in positions])
        for pos in positions:
            sym = pos["symbol"]
            upl_pct = float(pos["unrealized_plpc"]) * 100
            price = prices.get(sym, float(pos["current_price"]))

            if upl_pct <= -STOP_LOSS:
                log.warning(f"🛑 STOP LOSS: {sym} down {upl_pct:.1f}% — selling")
                try:
                    close_position(sym)
                    telegram(f"🛑 STOP LOSS triggered\n{sym} down {upl_pct:.1f}%\nPosition closed")
                    # Find and evaluate pending
                    eval_pending_trade(sym, price)
                except Exception as e:
                    log.error(f"Stop loss failed for {sym}: {e}")

            elif upl_pct >= TAKE_PROFIT:
                log.info(f"🎯 TAKE PROFIT: {sym} up {upl_pct:.1f}% — selling")
                try:
                    close_position(sym)
                    telegram(f"🎯 TAKE PROFIT triggered\n{sym} up {upl_pct:.1f}%\nPosition closed")
                    eval_pending_trade(sym, price)
                except Exception as e:
                    log.error(f"Take profit failed for {sym}: {e}")
    except Exception as e:
        log.error(f"Stop check failed: {e}")

def eval_pending_trade(symbol, current_price):
    pending = [p for p in state["pending_evals"] if p["symbol"] == symbol]
    for p in pending:
        record_outcome(symbol, p["entry_price"], current_price, p["agent_signals"])
        state["pending_evals"].remove(p)
    save_state()

# ─── MAIN ANALYSIS CYCLE ──────────────────────────────────────────────────────
def run_full_scan():
    """Scan all watchlist stocks in parallel using threads, pick best opportunity"""
    import concurrent.futures
    log.info(f"🔍 FULL SCAN — {len(WATCHLIST)} stocks")

    prices = get_prices(WATCHLIST)
    results = []

    def quick_scan(sym):
        price = prices.get(sym)
        if not price:
            return None
        # Quick single-agent scan
        result = ask_agent("ORACLE", sym, price, 0)
        weight = state["agent_weights"].get("oracle", 1.0)
        weighted_conf = min(100, int(result["confidence"] * weight))
        return {"sym": sym, "price": price, "signal": result["signal"],
                "confidence": result["confidence"], "weighted_conf": weighted_conf,
                "reason": result["reason"]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(quick_scan, sym): sym for sym in WATCHLIST}
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            if r:
                results.append(r)
                col = "BUY ✅" if r["signal"]=="BUY" else "SELL 🔴" if r["signal"]=="SELL" else "HOLD ⏸"
                log.info(f"  {r['sym']:6} ${r['price']:8.2f}  {col}  {r['confidence']}%  {r['reason']}")

    # Sort by weighted confidence
    buys  = sorted([r for r in results if r["signal"]=="BUY"],  key=lambda x: x["weighted_conf"], reverse=True)
    sells = sorted([r for r in results if r["signal"]=="SELL"], key=lambda x: x["weighted_conf"], reverse=True)

    summary = "\n".join([f"{r['sym']}: {r['signal']} ({r['confidence']}%)" for r in results])
    telegram(f"📊 FULL SCAN COMPLETE\n{summary}")

    if buys:
        log.info(f"🏆 Best BUY: {buys[0]['sym']} @ ${buys[0]['price']:.2f} ({buys[0]['confidence']}%)")
    if sells:
        log.info(f"🏆 Best SELL: {sells[0]['sym']} @ ${sells[0]['price']:.2f} ({sells[0]['confidence']}%)")

    return buys, sells

def run_deep_analysis(symbol):
    """Run all 7 agents on a single symbol"""
    global current_votes
    current_votes = {}

    price = get_price(symbol)
    if not price:
        log.error(f"Could not get price for {symbol}")
        return None

    state["cycle"] += 1
    log.info(f"\n{'═'*55}")
    log.info(f"CYCLE #{state['cycle']} | {symbol} @ ${price:.2f} | {market_open_time()}")
    log.info(f"Active Strategy: {state.get('active_strategy',{}).get('name','None')}")
    log.info(f"{'═'*55}")

    agents = ["SENTINEL", "ORACLE", "ARBITER", "CASSANDRA", "HERALD", "GUARDIAN"]
    for agent in agents:
        result = ask_agent(agent, symbol, price, 0)
        current_votes[agent.lower()] = result
        sig = result["signal"]
        col = "✅" if sig=="BUY" else "🔴" if sig=="SELL" else "⏸"
        log.info(f"  [{agent:9}] {col} {sig:4} | {result['confidence']:3}% | {result['reason']}")

    # Weighted decision
    w_signal, w_conf, scores = get_weighted_decision()
    log.info(f"\n  Weighted scores — BUY:{scores['BUY']:.2f} SELL:{scores['SELL']:.2f} HOLD:{scores['HOLD']:.2f}")

    # NEXUS final decision
    nexus = ask_agent("NEXUS", symbol, price, 0)
    current_votes["nexus"] = nexus

    # If weighted vote strongly disagrees, override
    if w_signal != nexus["signal"] and w_conf > 70:
        log.info(f"  ⚡ Learning override: {nexus['signal']} → {w_signal} (weights favor {w_signal})")
        nexus["signal"] = w_signal
        nexus["confidence"] = (nexus["confidence"] + w_conf) // 2

    # Apply threshold
    if nexus["confidence"] < state["conf_threshold"] and nexus["signal"] != "HOLD":
        log.info(f"  ⚠ Confidence {nexus['confidence']}% < threshold {state['conf_threshold']}% → HOLD")
        nexus["signal"] = "HOLD"

    log.info(f"\n  {'━'*40}")
    log.info(f"  NEXUS FINAL: {nexus['signal']} {symbol} | {nexus['confidence']}% | {nexus['reason']}")
    log.info(f"  {'━'*40}\n")

    return {"signal": nexus["signal"], "confidence": nexus["confidence"],
            "symbol": symbol, "price": price, "reason": nexus["reason"],
            "agent_signals": dict(current_votes)}

def execute_trade(decision):
    """Execute trade based on NEXUS decision"""
    if not AUTO_EXEC:
        log.info("Auto-execute OFF — skipping trade")
        return

    symbol = decision["symbol"]
    signal = decision["signal"]
    price  = decision["price"]
    conf   = decision["confidence"]

    if signal == "HOLD":
        log.info(f"HOLD {symbol} — no trade")
        return

    try:
        if signal == "BUY":
            acct = get_account()
            bp = float(acct["buying_power"])
            notional = min(bp * (POS_SIZE_PCT / 100), 5000)
            if notional < 10:
                log.warning("Insufficient buying power")
                return
            # Check if already have position
            positions = get_positions()
            if any(p["symbol"] == symbol for p in positions):
                log.info(f"Already have position in {symbol} — skipping")
                return
            order = place_buy(symbol, notional)
            log.info(f"✅ BUY ORDER: ${notional:.2f} of {symbol} @ ~${price:.2f} | ID: {order['id'][:8]}")
            telegram(f"✅ BUY EXECUTED\n{symbol} @ ${price:.2f}\nNotional: ${notional:.2f}\nConfidence: {conf}%\nStrategy: {state.get('active_strategy',{}).get('name','—')}")
            # Queue for outcome evaluation
            state["pending_evals"].append({
                "symbol": symbol, "entry_price": price,
                "agent_signals": decision["agent_signals"],
                "time": datetime.now().isoformat()
            })
            state["trade_history"].append({
                "time": datetime.now().isoformat(), "side": "BUY",
                "symbol": symbol, "price": price, "notional": notional,
                "confidence": conf, "strategy": state.get("active_strategy", {}).get("name", "—")
            })
            save_state()

        elif signal == "SELL":
            positions = get_positions()
            pos = next((p for p in positions if p["symbol"] == symbol), None)
            if not pos:
                log.info(f"No position in {symbol} to sell")
                return
            upl = float(pos["unrealized_pl"])
            order = close_position(symbol)
            log.info(f"✅ SELL ORDER: {symbol} x{pos['qty']} | P&L: ${upl:+.2f}")
            telegram(f"✅ SELL EXECUTED\n{symbol} x{pos['qty']}\nP&L: ${upl:+.2f}\nConfidence: {conf}%")
            eval_pending_trade(symbol, price)
            state["trade_history"].append({
                "time": datetime.now().isoformat(), "side": "SELL",
                "symbol": symbol, "price": price, "pnl": upl,
                "strategy": state.get("active_strategy", {}).get("name", "—")
            })
            save_state()

    except Exception as e:
        log.error(f"Trade execution failed: {e}")
        telegram(f"❌ Trade failed: {symbol} {signal}\n{str(e)}")

# ─── STATUS REPORT ────────────────────────────────────────────────────────────
def send_status_report():
    try:
        acct = get_account()
        equity = float(acct["equity"])
        cash   = float(acct["cash"])
        pnl    = equity - 100000
        positions = get_positions()
        total = state["wins"] + state["losses"]
        wr = state["wins"] / total * 100 if total > 0 else 0

        pos_str = "\n".join([f"  {p['symbol']}: {float(p['unrealized_plpc'])*100:+.1f}%" for p in positions]) or "  None"
        strat = state.get("active_strategy", {})

        report = (
            f"📈 NEXUS STATUS REPORT\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Equity: ${equity:,.2f}\n"
            f"Cash:   ${cash:,.2f}\n"
            f"P&L:    ${pnl:+,.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Win Rate: {wr:.1f}% ({state['wins']}W/{state['losses']}L)\n"
            f"Streak:   {state['streak']:+d}\n"
            f"Cycles:   {state['cycle']}\n"
            f"Threshold: {state['conf_threshold']}%\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Strategy: {strat.get('name','None')}\n"
            f"Open positions:\n{pos_str}"
        )
        telegram(report)
        log.info(f"Status: Equity=${equity:,.2f} | WR={wr:.1f}% | Cycle={state['cycle']}")
    except Exception as e:
        log.error(f"Status report failed: {e}")

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def main():
    log.info("═" * 55)
    log.info("  NEXUS v2 — Autonomous Trading Bot")
    log.info("  Optimized for momentum swing trading")
    log.info("═" * 55)

    if not AK or not SK:
        log.error("Missing ALPACA_KEY or ALPACA_SECRET env vars!")
        return
    if not ANK:
        log.warning("No ANTHROPIC_KEY — running in simulation mode")

    # Load saved state
    load_state()
    init_strategies()

    # Verify Alpaca connection
    try:
        acct = get_account()
        log.info(f"✅ Alpaca connected — Account: {acct['account_number']} | Cash: ${float(acct['cash']):,.2f}")
        telegram(f"🚀 NEXUS v2 started!\nAccount: {acct['account_number']}\nCash: ${float(acct['cash']):,.2f}\nWatchlist: {', '.join(WATCHLIST)}\nInterval: {INTERVAL_MIN}min")
    except Exception as e:
        log.error(f"Alpaca connection failed: {e}")
        return

    # Search for strategies on startup (once)
    if len(state["strategies"]) <= len(BUILTIN_STRATEGIES):
        search_strategies()

    status_counter = 0
    last_scan = 0

    log.info(f"\nWatchlist: {', '.join(WATCHLIST)}")
    log.info(f"Interval:  {INTERVAL_MIN} min")
    log.info(f"Auto exec: {AUTO_EXEC}")
    log.info(f"Strategy:  {state.get('active_strategy',{}).get('name','None')}")
    log.info(f"\nStarting main loop...\n")

    while True:
        try:
            now = time.time()

            # Check stops every 5 minutes regardless of market
            check_stops()

            if is_market_open():
                # Run full scan every INTERVAL_MIN minutes
                if now - last_scan >= INTERVAL_MIN * 60:
                    last_scan = now
                    buys, sells = run_full_scan()

                    # Deep analysis on best opportunity
                    if buys and buys[0]["confidence"] >= state["conf_threshold"]:
                        decision = run_deep_analysis(buys[0]["sym"])
                        if decision and decision["signal"] == "BUY":
                            execute_trade(decision)

                    # Auto-sell best sell signal
                    if sells and sells[0]["confidence"] >= state["conf_threshold"]:
                        decision = run_deep_analysis(sells[0]["sym"])
                        if decision and decision["signal"] == "SELL":
                            execute_trade(decision)

                    status_counter += 1
                    # Send status report every 4 cycles
                    if status_counter % 4 == 0:
                        send_status_report()

                    # Re-search strategies every 50 cycles
                    if state["cycle"] % 50 == 0 and state["cycle"] > 0:
                        search_strategies()

            else:
                # Outside market hours — log once per hour
                if now - last_scan >= 3600:
                    last_scan = now
                    log.info(f"Market closed ({market_open_time()}) — waiting for open (9:30AM EST)")

            time.sleep(60)  # check every minute

        except KeyboardInterrupt:
            log.info("\nStopping NEXUS...")
            send_status_report()
            save_state()
            break
        except Exception as e:
            log.error(f"Main loop error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()

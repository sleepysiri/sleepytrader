"""
FastAPI server + simulation loop.
Every round: 200 AI agents decide (4 parallel Groq calls x 50 agents each),
then the market moves based on their aggregate buy/sell pressure.
"""
import asyncio
import json
import time
import os
import sys
from typing import Set, List, Dict
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

sys.path.insert(0, os.path.dirname(__file__))

from market import StockMarket
from agents import Agent, create_agents, execute_decisions
from indicators import get_all_indicators
from ai_client import AIClient

app = FastAPI()

# ── Globals ────────────────────────────────────────────────────────────────────
INITIAL_PRICE  = 100.0
N_AGENTS       = 200
BATCH_SIZE     = 50          # 50 agents per round (1 API call), rotates through 200
ROUND_INTERVAL = 35.0        # seconds between rounds — keeps Groq free-tier < 6000 TPM

market: StockMarket          = StockMarket(initial_price=INITIAL_PRICE)
all_agents: List[Agent]      = create_agents(N_AGENTS, initial_price=INITIAL_PRICE)
agents_map: Dict[int, Agent] = {a.id: a for a in all_agents}
ai_client: AIClient          = AIClient()

clients: Set[WebSocket] = set()
sim_running: bool       = False
round_number: int       = 0


# ── WebSocket ──────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)

    await websocket.send_json({
        "type":     "init",
        "price":    round(market.price, 4),
        "candles":  market.candles_as_dicts(100),
        "agents":   [a.to_dict(market.price) for a in all_agents],
        "round":    round_number,
        "n_agents": N_AGENTS,
    })

    try:
        while True:
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except (WebSocketDisconnect, Exception):
        clients.discard(websocket)


@app.get("/")
async def index():
    html = (Path(__file__).parent.parent / "frontend" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


# ── Broadcast ──────────────────────────────────────────────────────────────────
async def broadcast(data: dict):
    global clients
    if not clients:
        return
    msg = json.dumps(data)
    dead: Set[WebSocket] = set()
    for ws in clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    clients -= dead


# ── Simulation loop ────────────────────────────────────────────────────────────
async def simulation_loop():
    global sim_running, round_number
    sim_running = True

    while sim_running:
        round_number += 1
        t0 = time.time()

        # 1. Snapshot market indicators ───────────────────────────────────────
        ph         = list(market.price_history)
        indicators = get_all_indicators(ph)
        macd_d     = indicators.get("macd", {})
        rsi_d      = indicators.get("rsi",  [50])
        bb_d       = indicators.get("bollinger", {})
        macd_hist  = indicators.get("macd_hist", [0])

        rsi_val  = float(rsi_d[-1])     if rsi_d     else 50.0
        macd_val = float(macd_hist[-1]) if macd_hist else 0.0

        # Bollinger position label
        bb_upper  = bb_d.get("upper",  [])
        bb_lower  = bb_d.get("lower",  [])
        if bb_upper and bb_lower:
            u, l = bb_upper[-1], bb_lower[-1]
            if market.price >= u * 0.998:
                bb_pos = "upper"
            elif market.price <= l * 1.002:
                bb_pos = "lower"
            else:
                bb_pos = "middle"
        else:
            bb_pos = "middle"

        market_data = {
            "price":         round(market.price, 4),
            "change_pct":    round(market.price_change_24h, 2),
            "rsi":           round(rsi_val, 1),
            "macd_hist":     round(macd_val, 4),
            "recent_prices": [round(p, 2) for p in ph[-8:]],
            "bb_position":   bb_pos,
        }

        # 2. Notify frontend: agents are deliberating ─────────────────────────
        await broadcast({
            "type":   "agents_deciding",
            "round":  round_number,
            "market": market_data,
        })

        # 3. Rotate through agents — 50 per round (1 API call stays within free TPM limit)
        n_batches  = N_AGENTS // BATCH_SIZE
        batch_idx  = (round_number - 1) % n_batches
        start      = batch_idx * BATCH_SIZE
        active_batch = all_agents[start : start + BATCH_SIZE]
        batch_prompt = [a.prompt_repr() for a in active_batch]

        print(f"[Round {round_number}] Calling LLM for agents {start}–{start+BATCH_SIZE-1}...")

        # 4. Single sequential LLM call (1 call = ~2500 tokens, safe for free tier) ──
        result = await ai_client.batch_decisions(batch_prompt, market_data)

        # 5. Execute decisions on those 50 agents' portfolios ──────────────────
        total_buy:  float         = 0.0
        total_sell: float         = 0.0
        all_decisions: List[dict] = []

        if isinstance(result, list):
            b, s = execute_decisions(result, agents_map, market.price)
            total_buy  += b
            total_sell += s
            all_decisions.extend(result)

        # 6. Move the market based on aggregate order flow ────────────────────
        market.process_orders(total_buy, total_sell, 0.0, 0.0)

        # 7. Recalculate indicators after price move ───────────────────────────
        ph2   = list(market.price_history)
        ind2  = get_all_indicators(ph2)
        macd2 = ind2.get("macd",       {})
        rsi2  = ind2.get("rsi",        [])
        bb2   = ind2.get("bollinger",  {})

        # Enrich decisions with agent names for display
        for d in all_decisions:
            ag = agents_map.get(int(d.get("id", -1)))
            if ag:
                d["name"]        = ag.name
                d["personality"] = ag.personality
                d["portfolio"]   = round(ag.portfolio_value(market.price), 2)

        # 8. Broadcast full tick to all connected browsers ────────────────────
        await broadcast({
            "type":       "tick",
            "round":      round_number,
            "price":      round(market.price, 4),
            "change_pct": round(market.price_change_24h, 2),
            "total_buy":  round(total_buy,  1),
            "total_sell": round(total_sell, 1),
            "volume":     round(total_buy + total_sell, 1),
            "candles":    market.candles_as_dicts(100),
            "indicators": {
                "macd":        (macd2.get("macd")      or [])[-60:],
                "macd_signal": (macd2.get("signal")    or [])[-60:],
                "macd_hist":   (macd2.get("histogram") or [])[-60:],
                "rsi":         (rsi2  or [])[-60:],
                "bb_upper":    (bb2.get("upper")  or [])[-60:],
                "bb_middle":   (bb2.get("middle") or [])[-60:],
                "bb_lower":    (bb2.get("lower")  or [])[-60:],
            },
            "agents":    [a.to_dict(market.price) for a in all_agents],
            "decisions": all_decisions,
            "orderbook": market.get_orderbook_snapshot(),
            "trades":    list(market.recent_trades)[:12],
        })

        elapsed = time.time() - t0
        wait    = max(2.0, ROUND_INTERVAL - elapsed)
        print(
            f"[Round {round_number:04d}]  "
            f"price=${market.price:.2f}  "
            f"buy={total_buy:.0f} sell={total_sell:.0f}  "
            f"elapsed={elapsed:.1f}s  next={wait:.1f}s"
        )
        await asyncio.sleep(wait)


@app.on_event("startup")
async def startup():
    asyncio.create_task(simulation_loop())


@app.on_event("shutdown")
async def shutdown():
    global sim_running
    sim_running = False
    await ai_client.close()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

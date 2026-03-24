"""
AI-driven agent population.
200 agents with distinct personalities — every trading decision is made by the LLM.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
import random

# ── 40 diverse personalities ─────────────────────────────────────────────────
PERSONALITIES: List[str] = [
    "aggressive momentum chaser — buys hard on breakouts, FOMO-driven, ignores fundamentals",
    "deep value investor — only buys at 20%+ discount to intrinsic value, extremely patient",
    "day-trader scalper — targets tiny 0.2% moves, very active, cuts losses immediately",
    "pure contrarian — always bets against the crowd, buys panic, sells euphoria",
    "mechanical trend follower — uses moving averages strictly, no emotional override",
    "RSI oscillator trader — buys when RSI<35 (oversold), sells when RSI>65 (overbought)",
    "mean-reversion specialist — systematically fades extreme price moves back to average",
    "crypto-bro turned stock trader — extreme risk tolerance, YOLO mentality, all-in or all-out",
    "retired conservative — capital preservation above all, terrified of any meaningful loss",
    "Warren Buffett disciple — long-term value only, ignores all short-term price noise",
    "MACD crossover trader — buys bullish cross, sells bearish cross, purely mechanical",
    "emotional retail panic trader — sells every 1%+ drop, FOMO buys every green candle",
    "overconfident speculator — always thinks correct, takes oversized concentrated positions",
    "loss-averse behavioral trader — holds losers too long hoping to break even, sells winners early",
    "quant statistical trader — buys 2 std-devs below rolling mean, sells 2 above",
    "news-flow interpreter — reads every price move as a fundamental news signal",
    "young DCA accumulator — dollar-cost-averaging mindset, buys regularly regardless of price",
    "leveraged speculator — max aggression on strong signals, would use 3x leverage if available",
    "social media herd follower — buys whatever is trending, pure crowd mentality",
    "permabull — never sells, always adds on dips, believes in perpetual growth",
    "permabear — always expects a crash, sells any strength, perpetual doom-caller",
    "anxious emotional retail investor — fear and greed whipsaw every decision",
    "global-macro thinker — inflation fears and rate expectations drive every move",
    "breakout trader — waits for price to clear resistance, then chases the breakout",
    "Bollinger Band mean-reversion — buys lower band touches, sells upper band touches",
    "impatient scalper — must be in and out fast, minimal tolerance for open positions",
    "patient swing trader — waits for major setups, takes large concentrated positions",
    "rational Bayesian — continuously updates position size as new price evidence arrives",
    "sentiment contrarian — max buy at extreme fear RSI<25, max sell at extreme greed RSI>75",
    "dividend income investor — accumulates for long-term yield, ignores short-term swings",
    "strict risk/reward calculator — only trades when R:R ratio exceeds 3:1",
    "ultra-nervous cash hoarder — extremely risk-averse, defaults to holding cash",
    "dual-signal confirmationist — needs both positive momentum AND good fundamentals to buy",
    "relative-strength rotator — always rotates into the strongest recent performer",
    "Elliott-Wave analyst — sees 5-wave impulse and 3-wave corrective structures in charts",
    "institutional-flow tracker — reads large volume spikes as smart-money signals",
    "tax-loss harvester — sells losers strategically, holds winners as long as possible",
    "short-term speculator — targets 5–10% gains per trade, exits quickly at target",
    "portfolio rebalancer — keeps strict cash/stock ratio, buys when under-weight stock",
    "fear-greed extremist — maximum buy at peak fear, maximum sell at peak greed",
]

# ── Agent dataclass ───────────────────────────────────────────────────────────
@dataclass
class Agent:
    id: int
    name: str
    personality: str
    cash: float
    holdings: float
    avg_buy_price: float = 100.0
    last_action: str = "none"
    last_thought: str = "Waiting for first signal..."
    last_decision: str = "hold"
    trade_count: int = 0

    def portfolio_value(self, price: float) -> float:
        return self.cash + self.holdings * price

    def to_dict(self, price: float) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "personality": self.personality,
            "cash": round(self.cash, 2),
            "holdings": round(self.holdings, 2),
            "portfolio_value": round(self.portfolio_value(price), 2),
            "avg_buy_price": round(self.avg_buy_price, 2),
            "last_action": self.last_action,
            "last_thought": self.last_thought,
            "last_decision": self.last_decision,
            "trade_count": self.trade_count,
        }

    def prompt_repr(self) -> dict:
        """Compact representation sent inside each batch prompt (token-efficient)."""
        # First clause of personality (before the dash) = the key trading style
        short_p = self.personality.split("—")[0].strip()
        return {
            "id":   self.id,
            "p":    short_p,                         # personality keyword
            "cash": round(self.cash, 0),
            "sh":   round(self.holdings, 0),         # shares held
            "prev": self.last_action,                # previous action
        }


# ── Factory ───────────────────────────────────────────────────────────────────
def create_agents(n_agents: int = 200, initial_price: float = 100.0) -> List[Agent]:
    """Create N agents with diverse personalities and randomised starting portfolios."""
    rng = random.Random(42)
    agents: List[Agent] = []

    for i in range(n_agents):
        personality = PERSONALITIES[i % len(PERSONALITIES)]

        # Base cash / holdings
        cash = rng.uniform(5_000, 200_000)
        holdings = rng.uniform(10, 500) if rng.random() < 0.4 else 0.0

        # Personality-based portfolio tilt
        p = personality.lower()
        if any(k in p for k in ("conservative", "retired", "cash hoarder", "nervous")):
            cash = rng.uniform(50_000, 500_000)
            holdings = 0.0
        elif any(k in p for k in ("aggressive", "leveraged", "yolo", "speculator", "crypto")):
            cash = rng.uniform(1_000, 30_000)
            holdings = rng.uniform(200, 2_000)
        elif "dca" in p or "accumulator" in p:
            cash = rng.uniform(20_000, 80_000)
            holdings = rng.uniform(100, 600)

        # Build short display name from first few words of personality
        label = personality.split("—")[0].strip().title()[:28]
        name = f"{label} #{i:04d}"

        agents.append(Agent(
            id=i,
            name=name,
            personality=personality,
            cash=round(cash, 2),
            holdings=round(holdings, 2),
            avg_buy_price=round(initial_price * rng.uniform(0.88, 1.12), 2),
        ))

    return agents


# ── Execute decisions ─────────────────────────────────────────────────────────
def execute_decisions(
    decisions: List[dict],
    agents_map: Dict[int, "Agent"],
    price: float,
) -> tuple:
    """
    Apply LLM decisions to agent portfolios.
    Returns (total_shares_bought, total_shares_sold).
    """
    total_buy = 0.0
    total_sell = 0.0

    for d in decisions:
        aid = int(d.get("id", -1))
        if aid not in agents_map:
            continue

        ag = agents_map[aid]
        action = str(d.get("action", "hold")).lower().strip()
        quantity = max(0, int(d.get("quantity", 0)))
        thought = str(d.get("thought", "..."))

        ag.last_thought = thought
        ag.last_decision = action

        if action == "buy" and quantity > 0:
            cost = quantity * price
            if cost > ag.cash:                       # clamp to available cash
                quantity = int(ag.cash / price)
                cost = quantity * price
            if quantity > 0:
                old = ag.holdings
                ag.holdings += quantity
                ag.cash -= cost
                # update weighted avg buy price
                ag.avg_buy_price = (
                    (old * ag.avg_buy_price + quantity * price) / ag.holdings
                    if ag.holdings > 0 else price
                )
                ag.last_action = f"bought {quantity} @ ${price:.2f}"
                ag.trade_count += 1
                total_buy += quantity
            else:
                ag.last_action = "buy blocked — insufficient cash"
                ag.last_decision = "hold"

        elif action == "sell" and quantity > 0:
            quantity = min(quantity, int(ag.holdings))
            if quantity > 0:
                ag.cash += quantity * price
                ag.holdings -= quantity
                ag.last_action = f"sold {quantity} @ ${price:.2f}"
                ag.trade_count += 1
                total_sell += quantity
            else:
                ag.last_action = "sell blocked — no holdings"
                ag.last_decision = "hold"

        else:
            ag.last_decision = "hold"
            ag.last_action = "held"

    return total_buy, total_sell

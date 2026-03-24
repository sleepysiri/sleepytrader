"""
Stock Market simulation using Kyle's Lambda price impact model.
"""
from dataclasses import dataclass, field
from collections import deque
from typing import List, Dict, Any
import numpy as np
import time


@dataclass
class Candle:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


class StockMarket:
    def __init__(self, initial_price: float = 100.0):
        self.price = initial_price
        self.initial_price = initial_price
        self.fundamental_value = initial_price

        self.step_count = 0
        self.candle_steps = 1

        self.price_history: deque = deque(maxlen=2000)
        self.volume_history: deque = deque(maxlen=2000)
        self.price_history.append(initial_price)
        self.volume_history.append(0.0)

        self.candles: List[Candle] = []
        self.recent_trades: deque = deque(maxlen=50)

        # Current candle state
        self._candle_open = initial_price
        self._candle_high = initial_price
        self._candle_low = initial_price
        self._candle_volume = 0.0
        self._candle_start_time = time.time()

        # 24h tracking
        self._price_24h_ago = initial_price
        self._price_24h_step = 0
        self._steps_per_day = 1000  # approximate

        self._rng = np.random.default_rng(None)

    def process_orders(
        self,
        buy_volume: float,
        sell_volume: float,
        buy_market: float,
        sell_market: float,
    ) -> Dict[str, Any]:
        """
        Process orders using Kyle's lambda price impact model.
        Returns info about the tick.
        """
        total_volume = buy_volume + sell_volume + 1e-9
        net_volume = buy_volume - sell_volume
        imbalance = net_volume / total_volume

        # Kyle's lambda price impact
        price_impact = 0.001 * imbalance * self.price

        # Random noise
        noise = self._rng.normal(0, 0.0005) * self.price

        # Mean reversion toward fundamental value
        mean_rev = 0.001 * (self.fundamental_value - self.price)

        new_price = self.price + price_impact + noise + mean_rev
        new_price = max(0.01, new_price)

        tick_volume = buy_volume + sell_volume
        direction = "buy" if net_volume > 0 else "sell"

        # Record trade
        self.recent_trades.append({
            "price": round(new_price, 4),
            "volume": round(tick_volume, 2),
            "direction": direction,
            "timestamp": time.time(),
        })

        self.price = new_price
        self.price_history.append(new_price)
        self.volume_history.append(tick_volume)

        # Update current candle
        self._candle_high = max(self._candle_high, new_price)
        self._candle_low = min(self._candle_low, new_price)
        self._candle_volume += tick_volume

        self.step_count += 1

        # Drift fundamental value every 100 steps
        if self.step_count % 100 == 0:
            drift = self._rng.normal(0, 0.002)
            self.fundamental_value *= (1 + drift)
            self.fundamental_value = float(np.clip(self.fundamental_value, 10.0, 1000.0))

        # Close candle every candle_steps
        if self.step_count % self.candle_steps == 0:
            candle = Candle(
                timestamp=self._candle_start_time,
                open=self._candle_open,
                high=self._candle_high,
                low=self._candle_low,
                close=self.price,
                volume=self._candle_volume,
            )
            self.candles.append(candle)
            if len(self.candles) > 500:
                self.candles = self.candles[-500:]

            # Reset candle state
            self._candle_open = self.price
            self._candle_high = self.price
            self._candle_low = self.price
            self._candle_volume = 0.0
            self._candle_start_time = time.time()

        # Track 24h price
        if self.step_count % self._steps_per_day == 0:
            self._price_24h_ago = self.price

        return {
            "price": new_price,
            "volume": tick_volume,
            "direction": direction,
            "price_impact": price_impact,
        }

    @property
    def price_change_24h(self) -> float:
        """Returns percentage price change over last ~24h of simulation steps."""
        if self._price_24h_ago and self._price_24h_ago > 0:
            return (self.price - self._price_24h_ago) / self._price_24h_ago * 100.0
        return 0.0

    def get_orderbook_snapshot(self) -> Dict[str, Any]:
        """
        Generate a realistic order book snapshot with bids and asks.
        Uses current price and some spread simulation.
        """
        price = self.price
        # Realistic spread: ~0.05% to 0.2%
        spread_pct = self._rng.uniform(0.0005, 0.002)
        half_spread = price * spread_pct / 2.0

        best_bid = price - half_spread
        best_ask = price + half_spread

        bids = []
        asks = []

        for i in range(10):
            # Bids decrease from best_bid
            bid_price = best_bid * (1 - i * self._rng.uniform(0.0003, 0.0008))
            bid_volume = self._rng.uniform(50, 2000) * (1 / (i + 1)) ** 0.5
            bids.append({"price": round(bid_price, 4), "volume": round(bid_volume, 2)})

            # Asks increase from best_ask
            ask_price = best_ask * (1 + i * self._rng.uniform(0.0003, 0.0008))
            ask_volume = self._rng.uniform(50, 2000) * (1 / (i + 1)) ** 0.5
            asks.append({"price": round(ask_price, 4), "volume": round(ask_volume, 2)})

        return {"bids": bids, "asks": asks}

    def get_current_candle(self) -> Dict[str, Any]:
        """Return the currently forming candle as a dict."""
        return {
            "timestamp": self._candle_start_time,
            "open": round(self._candle_open, 4),
            "high": round(self._candle_high, 4),
            "low": round(self._candle_low, 4),
            "close": round(self.price, 4),
            "volume": round(self._candle_volume, 2),
        }

    def candles_as_dicts(self, n: int = 100) -> List[Dict[str, Any]]:
        """Return last n closed candles as list of dicts."""
        result = []
        for c in self.candles[-n:]:
            result.append({
                "timestamp": c.timestamp,
                "open": round(c.open, 4),
                "high": round(c.high, 4),
                "low": round(c.low, 4),
                "close": round(c.close, 4),
                "volume": round(c.volume, 2),
            })
        # Append current forming candle
        result.append(self.get_current_candle())
        return result

"""
Technical indicators for the ABM stock market simulator.
"""
from typing import List, Dict, Any
import numpy as np


def calculate_ema(prices_array: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average.
    Returns numpy array of same length as input.
    """
    if len(prices_array) == 0:
        return np.array([])

    ema = np.zeros_like(prices_array, dtype=float)
    k = 2.0 / (period + 1)

    # Seed with first value
    ema[0] = prices_array[0]
    for i in range(1, len(prices_array)):
        ema[i] = prices_array[i] * k + ema[i - 1] * (1 - k)

    return ema


def calculate_macd(
    prices_list: List[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Dict[str, List[float]]:
    """
    Calculate MACD indicator.
    Returns dict with 'macd', 'signal', 'histogram' lists.
    """
    if len(prices_list) < slow + signal:
        empty: List[float] = []
        return {"macd": empty, "signal": empty, "histogram": empty}

    prices = np.array(prices_list, dtype=float)
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return {
        "macd": macd_line.tolist(),
        "signal": signal_line.tolist(),
        "histogram": histogram.tolist(),
    }


def calculate_rsi(prices_list: List[float], period: int = 14) -> List[float]:
    """
    Calculate RSI (Relative Strength Index).
    Returns list of RSI values (0-100).
    """
    if len(prices_list) < period + 1:
        return []

    prices = np.array(prices_list, dtype=float)
    deltas = np.diff(prices)

    rsi_values = [float("nan")] * (period)

    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100.0 - 100.0 / (1.0 + rs))

    for i in range(period, len(deltas)):
        gain = gains[i]
        loss = losses[i]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - 100.0 / (1.0 + rs))

    # Replace NaN with 50 for display purposes
    result = []
    for v in rsi_values:
        if isinstance(v, float) and (v != v):  # NaN check
            result.append(50.0)
        else:
            result.append(round(float(v), 2))

    return result


def calculate_bollinger_bands(
    prices_list: List[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> Dict[str, List[float]]:
    """
    Calculate Bollinger Bands.
    Returns dict with 'upper', 'middle', 'lower' lists.
    """
    if len(prices_list) < period:
        empty: List[float] = []
        return {"upper": empty, "middle": empty, "lower": empty}

    prices = np.array(prices_list, dtype=float)
    n = len(prices)

    upper = []
    middle = []
    lower = []

    for i in range(n):
        if i < period - 1:
            upper.append(float("nan"))
            middle.append(float("nan"))
            lower.append(float("nan"))
        else:
            window = prices[i - period + 1 : i + 1]
            sma = float(np.mean(window))
            std = float(np.std(window, ddof=1))
            middle.append(round(sma, 4))
            upper.append(round(sma + std_dev * std, 4))
            lower.append(round(sma - std_dev * std, 4))

    # Replace NaN with 0 for JSON serialization
    def clean(lst: List[float]) -> List[float]:
        return [0.0 if (isinstance(v, float) and v != v) else v for v in lst]

    return {
        "upper": clean(upper),
        "middle": clean(middle),
        "lower": clean(lower),
    }


def get_all_indicators(price_history_list: List[float]) -> Dict[str, Any]:
    """
    Calculate all indicators from price history.
    Returns dict with macd, rsi, bollinger, macd_hist.
    """
    if len(price_history_list) < 30:
        return {
            "macd": {"macd": [], "signal": [], "histogram": []},
            "rsi": [],
            "bollinger": {"upper": [], "middle": [], "lower": []},
            "macd_hist": [],
        }

    macd_data = calculate_macd(price_history_list)
    rsi_data = calculate_rsi(price_history_list)
    bb_data = calculate_bollinger_bands(price_history_list)

    return {
        "macd": macd_data,
        "rsi": rsi_data,
        "bollinger": bb_data,
        "macd_hist": macd_data.get("histogram", []),
    }

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict


def calculate_sma(values: List[float], period: int) -> List[Optional[float]]:
    result = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(None)
        else:
            window = values[i - period + 1:i + 1]
            result.append(sum(window) / period)
    return result


def calculate_ema(values: List[float], period: int) -> List[Optional[float]]:
    result = []
    multiplier = 2.0 / (period + 1)
    for i, val in enumerate(values):
        if i < period - 1:
            result.append(None)
        elif i == period - 1:
            seed = sum(values[:period]) / period
            result.append(seed)
        else:
            prev = result[-1]
            result.append((val - prev) * multiplier + prev)
    return result


def calculate_rsi(closes: List[float], period: int = 14) -> List[Optional[float]]:
    rsi_values = [None] * len(closes)
    if len(closes) < period + 1:
        return rsi_values
    
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(0, c) for c in changes]
    losses = [abs(min(0, c)) for c in changes]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi_values[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    
    if period < len(changes):
        init_loss = sum(losses[:period]) / period
        if init_loss == 0:
            rsi_values[period] = 100.0
        else:
            rs = (sum(gains[:period]) / period) / init_loss
            rsi_values[period] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi_values


def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], 
                          k_period: int = 14, d_period: int = 3) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    k_values = []
    for i in range(len(closes)):
        if i < k_period - 1:
            k_values.append(None)
        else:
            highest = max(highs[i - k_period + 1:i + 1])
            lowest = min(lows[i - k_period + 1:i + 1])
            if highest == lowest:
                k_values.append(50.0)
            else:
                k_values.append(100.0 * (closes[i] - lowest) / (highest - lowest))
    
    d_values = []
    for i in range(len(k_values)):
        if k_values[i] is None or i < k_period - 1 + d_period - 1:
            d_values.append(None)
        else:
            window = [k_values[j] for j in range(i - d_period + 1, i + 1) if k_values[j] is not None]
            if len(window) == d_period:
                d_values.append(sum(window) / d_period)
            else:
                d_values.append(None)
    
    return k_values, d_values


def calculate_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    fast_ema = calculate_ema(closes, fast)
    slow_ema = calculate_ema(closes, slow)
    
    macd_line = []
    for i in range(len(closes)):
        if fast_ema[i] is None or slow_ema[i] is None:
            macd_line.append(None)
        else:
            macd_line.append(fast_ema[i] - slow_ema[i])
    
    valid_macd = [m for m in macd_line if m is not None]
    signal_ema = calculate_ema(valid_macd, signal) if len(valid_macd) >= signal else []
    
    signal_line = [None] * len(closes)
    histogram = [None] * len(closes)
    
    valid_idx = 0
    for i in range(len(closes)):
        if macd_line[i] is not None:
            if valid_idx < len(signal_ema):
                signal_line[i] = signal_ema[valid_idx]
            valid_idx += 1
    
    for i in range(len(closes)):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(closes: List[float], period: int = 20, num_std: float = 2.0) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    lower, middle, upper = [], [], []
    
    for i in range(len(closes)):
        if i < period - 1:
            lower.append(None)
            middle.append(None)
            upper.append(None)
        else:
            window = closes[i - period + 1:i + 1]
            sma = sum(window) / period
            variance = sum((x - sma) ** 2 for x in window) / period
            std = variance ** 0.5
            middle.append(sma)
            upper.append(sma + num_std * std)
            lower.append(sma - num_std * std)
    
    return lower, middle, upper


def calculate_williams_r(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[Optional[float]]:
    result = []
    for i in range(len(closes)):
        if i < period - 1:
            result.append(None)
        else:
            highest = max(highs[i - period + 1:i + 1])
            lowest = min(lows[i - period + 1:i + 1])
            if highest == lowest:
                result.append(-50.0)
            else:
                result.append(-100.0 * (highest - closes[i]) / (highest - lowest))
    return result


def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[Optional[float]]:
    true_ranges = []
    for i in range(len(closes)):
        if i == 0:
            true_ranges.append(highs[i] - lows[i])
        else:
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            true_ranges.append(tr)
    
    atr_values = [None] * len(closes)
    if len(true_ranges) >= period:
        atr_values[period - 1] = sum(true_ranges[:period]) / period
        for i in range(period, len(closes)):
            atr_values[i] = (atr_values[i-1] * (period - 1) + true_ranges[i]) / period
    
    return atr_values


def calculate_volume_sma(volumes: List[float], period: int = 20) -> List[Optional[float]]:
    return calculate_sma(volumes, period)


def calculate_atr_percentile(atr_values: List[Optional[float]], idx: int, lookback: int = 60) -> Optional[float]:
    if idx < lookback:
        return None
    window = [atr_values[i] for i in range(idx - lookback + 1, idx + 1) if atr_values[i] is not None]
    if len(window) < lookback // 2:
        return None
    current = atr_values[idx]
    if current is None:
        return None
    count_below = sum(1 for v in window if v < current)
    return (count_below / len(window)) * 100


def detect_macd_divergence_buy(closes: List[float], histogram: List[Optional[float]], idx: int, lookback: int = 10) -> bool:
    if idx < lookback:
        return False
    
    current_close = closes[idx]
    current_hist = histogram[idx]
    if current_hist is None:
        return False
    
    min_close_idx = idx
    for i in range(idx - lookback, idx):
        if closes[i] < closes[min_close_idx]:
            min_close_idx = i
    
    if min_close_idx == idx or current_close >= closes[min_close_idx]:
        return False
    
    min_hist_at_price_low = histogram[min_close_idx]
    if min_hist_at_price_low is None:
        return False
    
    if current_hist > min_hist_at_price_low and current_close < closes[min_close_idx] * 1.01:
        return True
    return False


def detect_macd_divergence_sell(closes: List[float], histogram: List[Optional[float]], idx: int, lookback: int = 10) -> bool:
    if idx < lookback:
        return False
    
    current_close = closes[idx]
    current_hist = histogram[idx]
    if current_hist is None:
        return False
    
    max_close_idx = idx
    for i in range(idx - lookback, idx):
        if closes[i] > closes[max_close_idx]:
            max_close_idx = i
    
    if max_close_idx == idx or current_close <= closes[max_close_idx]:
        return False
    
    max_hist_at_price_high = histogram[max_close_idx]
    if max_hist_at_price_high is None:
        return False
    
    if current_hist < max_hist_at_price_high and current_close > closes[max_close_idx] * 0.99:
        return True
    return False


def calculate_scores(idx: int, closes: List[float], highs: List[float], lows: List[float],
                     rsi: List[Optional[float]], stoch_k: List[Optional[float]], stoch_d: List[Optional[float]],
                     histogram: List[Optional[float]], bb_lower: List[Optional[float]], 
                     bb_middle: List[Optional[float]], bb_upper: List[Optional[float]],
                     williams: List[Optional[float]], regime: str) -> Tuple[float, float]:
    
    buy_score = 0.0
    sell_score = 0.0
    
    # RSI scoring
    rsi_val = rsi[idx]
    if rsi_val is not None:
        if rsi_val < 15:
            buy_score += 2.5
        elif rsi_val < 25:
            buy_score += 1.5
        elif rsi_val < 35:
            buy_score += 0.5
        
        if rsi_val > 85:
            sell_score += 2.5
        elif rsi_val > 70:
            sell_score += 1.5
        elif rsi_val > 65:
            sell_score += 0.5
    
    # Stochastic scoring
    k_val = stoch_k[idx]
    d_val = stoch_d[idx]
    prev_k = stoch_k[idx - 1] if idx > 0 else None
    prev_d = stoch_d[idx - 1] if idx > 0 else None
    
    if k_val is not None and d_val is not None:
        k_cross_above_d = prev_k is not None and prev_d is not None and prev_k <= prev_d and k_val > d_val
        k_cross_below_d = prev_k is not None and prev_d is not None and prev_k >= prev_d and k_val < d_val
        
        if k_val < 10 and k_cross_above_d:
            buy_score += 2.0
        elif k_val < 20:
            buy_score += 1.0
        
        if k_val > 90 and k_cross_below_d:
            sell_score += 2.0
        elif k_val > 80:
            sell_score += 1.0
    
    # MACD histogram crossing scoring
    hist_val = histogram[idx]
    prev_hist = histogram[idx - 1] if idx > 0 else None
    
    if hist_val is not None and prev_hist is not None:
        if prev_hist < 0 and hist_val >= 0:
            buy_score += 1.5
        if prev_hist > 0 and hist_val <= 0:
            sell_score += 1.5
    
    # MACD divergence
    if detect_macd_divergence_buy(closes, histogram, idx):
        buy_score += 1.0
    if detect_macd_divergence_sell(closes, histogram, idx):
        sell_score += 1.0
    
    # Bollinger Bands scoring
    close = closes[idx]
    bb_l = bb_lower[idx]
    bb_m = bb_middle[idx]
    bb_u = bb_upper[idx]
    
    if bb_l is not None and bb_m is not None and bb_u is not None:
        band_range = bb_u - bb_l
        lower_25_threshold = bb_l + band_range * 0.25
        upper_75_threshold = bb_u - band_range * 0.25
        
        if close < bb_l:
            buy_score += 1.5
        elif close < lower_25_threshold:
            buy_score += 0.75
        
        if close > bb_u:
            sell_score += 1.5
        elif close > upper_75_threshold:
            sell_score += 0.75
        
        # Band return bonus
        if idx > 0:
            prev_close = closes[idx - 1]
            prev_bb_l = bb_lower[idx - 1]
            prev_bb_u = bb_upper[idx - 1]
            if prev_bb_l is not None and prev_bb_u is not None:
                if prev_close < prev_bb_l and close >= bb_l:
                    buy_score += 0.5
                if prev_close > prev_bb_u and close <= bb_u:
                    sell_score += 0.5
    
    # Williams %R scoring
    wr_val = williams[idx]
    if wr_val is not None:
        if wr_val < -90:
            buy_score += 1.0
        elif wr_val < -80:
            buy_score += 0.5
        
        if wr_val > -10:
            sell_score += 1.0
        elif wr_val > -20:
            sell_score += 0.5
    
    return buy_score, sell_score


def process_trading_data(df: pd.DataFrame) -> pd.DataFrame:
    closes = df['close'].tolist()
    highs = df['high'].tolist()
    lows = df['low'].tolist()
    volumes = df['volume'].tolist()
    dates = df['date'].tolist()
    
    n = len(closes)
    
    # Calculate all indicators
    sma_50 = calculate_sma(closes, 50)
    rsi = calculate_rsi(closes, 14)
    stoch_k, stoch_d = calculate_stochastic(highs, lows, closes, 14, 3)
    macd_line, signal_line, histogram = calculate_macd(closes, 12, 26, 9)
    bb_lower, bb_middle, bb_upper = calculate_bollinger_bands(closes, 20, 2.0)
    williams = calculate_williams_r(highs, lows, closes, 14)
    atr_14 = calculate_atr(highs, lows, closes, 14)
    atr_30 = calculate_atr(highs, lows, closes, 30)
    vol_sma = calculate_volume_sma(volumes, 20)
    
    warmup_period = 55
    min_spacing = 5
    base_threshold = 4.0
    
    signals = []
    last_signal_bar = -100
    consecutive_buy = 0
    consecutive_sell = 0
    cumulative_pnl = 0.0
    pending_buy_price = None
    pending_buy_size = None
    previous_regime = None
    
    for idx in range(warmup_period, n):
        if idx - last_signal_bar < min_spacing:
            continue
        
        vol = volumes[idx]
        vol_ma = vol_sma[idx]
        if vol_ma is None or vol < vol_ma:
            continue
        
        volume_ratio = vol / vol_ma
        if volume_ratio < 1.2:
            continue
        
        volume_bonus = 0.5 if volume_ratio >= 2.0 else 0.0
        
        sma50_val = sma_50[idx]
        if sma50_val is None:
            continue
        regime = "UPTREND" if closes[idx] > sma50_val else "DOWNTREND"
        
        if previous_regime is not None and regime != previous_regime:
            consecutive_buy = 0
            consecutive_sell = 0
        previous_regime = regime
        
        # Calculate scores
        buy_score, sell_score = calculate_scores(
            idx, closes, highs, lows, rsi, stoch_k, stoch_d,
            histogram, bb_lower, bb_middle, bb_upper, williams, regime
        )
        
        # Add volume bonus
        buy_score += volume_bonus
        sell_score += volume_bonus
        
        # ATR percentile bonus
        atr_pct = calculate_atr_percentile(atr_14, idx, 60)
        if atr_pct is not None and atr_pct > 80:
            buy_score += 0.25
            sell_score += 0.25
        
        # Store raw scores before adjustments
        raw_buy = round(buy_score, 2)
        raw_sell = round(sell_score, 2)
        
        # Volatility filter
        atr14 = atr_14[idx]
        atr30 = atr_30[idx]
        high_volatility = False
        if atr14 is not None and atr30 is not None and atr30 > 0:
            if atr14 > 1.5 * atr30:
                high_volatility = True
                buy_score *= 0.8
                sell_score *= 0.8
        
        # Regime threshold adjustment
        buy_threshold = base_threshold
        sell_threshold = base_threshold
        if regime == "UPTREND":
            buy_threshold -= 0.5
            sell_threshold += 0.5
        else:
            buy_threshold += 0.5
            sell_threshold -= 0.5
        
        # Consecutive signal penalty
        if consecutive_buy >= 6:
            buy_score = 0
        elif consecutive_buy >= 4:
            buy_threshold += 1.0
        elif consecutive_buy >= 2:
            buy_threshold += 0.5
        
        if consecutive_sell >= 6:
            sell_score = 0
        elif consecutive_sell >= 4:
            sell_threshold += 1.0
        elif consecutive_sell >= 2:
            sell_threshold += 0.5
        
        buy_qualifies = buy_score >= buy_threshold
        sell_qualifies = sell_score >= sell_threshold
        
        if not buy_qualifies and not sell_qualifies:
            continue
        
        if buy_qualifies and sell_qualifies:
            if buy_score > sell_score:
                signal_type = "BUY"
                composite = raw_buy
            else:
                signal_type = "SELL"
                composite = raw_sell
        elif buy_qualifies:
            signal_type = "BUY"
            composite = raw_buy
        else:
            signal_type = "SELL"
            composite = raw_sell
        
        # Position sizing
        if atr14 is None or atr14 <= 0:
            continue
        
        vol_mult = 3.0 if high_volatility else 2.0
        base_size = int(2000 / (atr14 * vol_mult))
        
        # Drawdown adjustment
        if cumulative_pnl < 0:
            base_size = int(base_size * 0.8)
        
        if base_size < 25 or base_size > 500:
            continue
        
        # Update PnL tracking
        if signal_type == "BUY":
            if pending_buy_price is None:
                pending_buy_price = closes[idx]
                pending_buy_size = base_size
        else:  # SELL
            if pending_buy_price is not None:
                pnl = (closes[idx] - pending_buy_price) * pending_buy_size
                cumulative_pnl += pnl
                pending_buy_price = None
                pending_buy_size = None
        
        # Update consecutive counters
        if signal_type == "BUY":
            consecutive_buy += 1
            consecutive_sell = 0
        else:
            consecutive_sell += 1
            consecutive_buy = 0
        
        signals.append({
            'date': dates[idx],
            'signal_type': signal_type,
            'composite_score': composite,
            'position_size': base_size,
            'regime': regime,
            'volume_ratio': round(volume_ratio, 2)
        })
        
        last_signal_bar = idx
    
    result_df = pd.DataFrame(signals)
    if len(result_df) > 0:
        result_df = result_df.sort_values('date').reset_index(drop=True)
    
    return result_df


def main():
    task_dir = Path(__file__).parent.parent / "task"
    input_file = task_dir / "ohlcv_data.csv"
    output_file = task_dir / "signal_analysis.csv"
    
    df = pd.read_csv(input_file)
    result_df = process_trading_data(df)
    result_df.to_csv(output_file, index=False)
    
    print(f"Generated {len(result_df)} trading signals")
    print(f"Output saved to {output_file}")


if __name__ == "__main__":
    main()

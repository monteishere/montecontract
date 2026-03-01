import pandas as pd
import numpy as np
from pathlib import Path


def compute_sma(series, period):
    return series.rolling(window=period, min_periods=period).mean()


def compute_ema(series, period):
    alpha = 2 / (period + 1)
    ema = pd.Series(index=series.index, dtype=float)
    ema[:] = np.nan
    
    first_valid_idx = series.first_valid_index()
    start_pos = series.index.get_loc(first_valid_idx)
    
    if start_pos + period - 1 < len(series):
        first_ema_pos = start_pos + period - 1
        first_ema = series.iloc[start_pos:first_ema_pos + 1].mean()
        ema.iloc[first_ema_pos] = first_ema
        
        for i in range(first_ema_pos + 1, len(series)):
            ema.iloc[i] = series.iloc[i] * alpha + ema.iloc[i - 1] * (1 - alpha)
    
    return ema


def compute_wilder_smooth(series, period):
    result = pd.Series(index=series.index, dtype=float)
    result[:] = np.nan
    
    valid_data = series.dropna()
    if len(valid_data) < period:
        return result
    
    first_valid_idx = valid_data.index[0]
    start_pos = series.index.get_loc(first_valid_idx)
    
    if start_pos + period - 1 < len(series):
        first_pos = start_pos + period - 1
        first_val = series.iloc[start_pos:first_pos + 1].mean()
        result.iloc[first_pos] = first_val
        
        alpha = 1 / period
        for i in range(first_pos + 1, len(series)):
            if pd.notna(series.iloc[i]):
                result.iloc[i] = result.iloc[i - 1] * (1 - alpha) + series.iloc[i] * alpha
    
    return result


def compute_rsi(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = pd.Series(index=close.index, dtype=float)
    avg_loss = pd.Series(index=close.index, dtype=float)
    avg_gain[:] = np.nan
    avg_loss[:] = np.nan
    
    if len(close) > period:
        first_avg_gain = gain.iloc[1:period + 1].mean()
        first_avg_loss = loss.iloc[1:period + 1].mean()
        avg_gain.iloc[period] = first_avg_gain
        avg_loss.iloc[period] = first_avg_loss
        
        for i in range(period + 1, len(close)):
            avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], 100)
    
    return rsi


def compute_true_range(high, low, close):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def compute_atr(high, low, close, period):
    tr = compute_true_range(high, low, close)
    return compute_wilder_smooth(tr, period)


def compute_macd(close):
    ema_12 = compute_ema(close, 12)
    ema_26 = compute_ema(close, 26)
    macd = ema_12 - ema_26
    
    macd_for_signal = macd.copy()
    signal = pd.Series(index=close.index, dtype=float)
    signal[:] = np.nan
    
    valid_macd = macd_for_signal.dropna()
    if len(valid_macd) >= 9:
        first_valid_pos = macd_for_signal.first_valid_index()
        start_pos = close.index.get_loc(first_valid_pos)
        
        alpha = 2 / (9 + 1)
        first_signal_pos = start_pos + 8
        if first_signal_pos < len(close):
            first_signal = macd.iloc[start_pos:first_signal_pos + 1].mean()
            signal.iloc[first_signal_pos] = first_signal
            
            for i in range(first_signal_pos + 1, len(close)):
                if pd.notna(macd.iloc[i]):
                    signal.iloc[i] = macd.iloc[i] * alpha + signal.iloc[i - 1] * (1 - alpha)
    
    histogram = macd - signal
    
    return ema_12, ema_26, macd, signal, histogram


def compute_bollinger_bands(close, period=20, num_std=2):
    middle = compute_sma(close, period)
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def compute_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    
    stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    stoch_k = stoch_k.replace([np.inf, -np.inf], np.nan)
    
    stoch_d = compute_sma(stoch_k, d_period)
    
    return stoch_k, stoch_d


def compute_adx(high, low, close, period=14):
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    
    plus_dm = pd.Series(index=high.index, dtype=float)
    minus_dm = pd.Series(index=high.index, dtype=float)
    
    for i in range(len(high)):
        if i == 0:
            plus_dm.iloc[i] = 0
            minus_dm.iloc[i] = 0
        else:
            up_move = high.iloc[i] - prev_high.iloc[i]
            down_move = prev_low.iloc[i] - low.iloc[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm.iloc[i] = up_move
            else:
                plus_dm.iloc[i] = 0
            
            if down_move > up_move and down_move > 0:
                minus_dm.iloc[i] = down_move
            else:
                minus_dm.iloc[i] = 0
    
    atr = compute_atr(high, low, close, period)
    
    smoothed_plus_dm = compute_wilder_smooth(plus_dm, period)
    smoothed_minus_dm = compute_wilder_smooth(minus_dm, period)
    
    plus_di = 100 * smoothed_plus_dm / atr
    minus_di = 100 * smoothed_minus_dm / atr
    
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100 * di_diff / di_sum
    dx = dx.replace([np.inf, -np.inf], np.nan)
    
    adx = compute_wilder_smooth(dx, period)
    
    return plus_di, minus_di, adx


def generate_signals(df):
    signals = pd.Series(index=df.index, dtype=str)
    signals[:] = 'HOLD'
    
    for i in range(1, len(df)):
        if pd.isna(df['MACD'].iloc[i]) or pd.isna(df['Signal'].iloc[i]):
            continue
        if pd.isna(df['MACD'].iloc[i-1]) or pd.isna(df['Signal'].iloc[i-1]):
            continue
        if pd.isna(df['RSI_14'].iloc[i]) or pd.isna(df['SMA_50'].iloc[i]):
            continue
        
        prev_diff = df['MACD'].iloc[i-1] - df['Signal'].iloc[i-1]
        curr_diff = df['MACD'].iloc[i] - df['Signal'].iloc[i]
        
        macd_cross_up = prev_diff <= 0 and curr_diff > 0
        macd_cross_down = prev_diff >= 0 and curr_diff < 0
        
        rsi = df['RSI_14'].iloc[i]
        close = df['close'].iloc[i]
        sma_50 = df['SMA_50'].iloc[i]
        
        if macd_cross_up and rsi < 70 and close > sma_50:
            signals.iloc[i] = 'BUY'
        elif macd_cross_down and rsi > 30 and close < sma_50:
            signals.iloc[i] = 'SELL'
    
    return signals


def compute_performance_metrics(df):
    signal_rows = df[df['signal'].isin(['BUY', 'SELL'])]
    buy_rows = df[df['signal'] == 'BUY']
    sell_rows = df[df['signal'] == 'SELL']
    
    metrics = []
    
    metrics.append({'metric': 'total_signals', 'value': len(signal_rows)})
    metrics.append({'metric': 'buy_signals', 'value': len(buy_rows)})
    metrics.append({'metric': 'sell_signals', 'value': len(sell_rows)})
    
    if len(buy_rows) > 0:
        avg_rsi_buy = round(buy_rows['RSI_14'].mean(), 4)
    else:
        avg_rsi_buy = np.nan
    metrics.append({'metric': 'avg_rsi_at_buy', 'value': avg_rsi_buy})
    
    if len(sell_rows) > 0:
        avg_rsi_sell = round(sell_rows['RSI_14'].mean(), 4)
    else:
        avg_rsi_sell = np.nan
    metrics.append({'metric': 'avg_rsi_at_sell', 'value': avg_rsi_sell})
    
    above_upper = signal_rows[signal_rows['close'] > signal_rows['BB_Upper']]
    metrics.append({'metric': 'signals_above_bb_upper', 'value': len(above_upper)})
    
    below_lower = signal_rows[signal_rows['close'] < signal_rows['BB_Lower']]
    metrics.append({'metric': 'signals_below_bb_lower', 'value': len(below_lower)})
    
    if len(signal_rows) > 0:
        avg_adx = round(signal_rows['ADX'].mean(), 4)
    else:
        avg_adx = np.nan
    metrics.append({'metric': 'avg_adx_at_signal', 'value': avg_adx})
    
    if len(signal_rows) > 0:
        metrics.append({'metric': 'first_signal_date', 'value': signal_rows['date'].iloc[0]})
        metrics.append({'metric': 'last_signal_date', 'value': signal_rows['date'].iloc[-1]})
    else:
        metrics.append({'metric': 'first_signal_date', 'value': np.nan})
        metrics.append({'metric': 'last_signal_date', 'value': np.nan})
    
    if len(signal_rows) > 0:
        avg_atr = round(signal_rows['ATR_14'].mean(), 4)
    else:
        avg_atr = np.nan
    metrics.append({'metric': 'avg_atr_at_signal', 'value': avg_atr})
    
    valid_hist = df['Histogram'].dropna()
    if len(valid_hist) > 0:
        metrics.append({'metric': 'max_histogram', 'value': round(valid_hist.max(), 4)})
        metrics.append({'metric': 'min_histogram', 'value': round(valid_hist.min(), 4)})
    else:
        metrics.append({'metric': 'max_histogram', 'value': np.nan})
        metrics.append({'metric': 'min_histogram', 'value': np.nan})
    
    if len(signal_rows) > 1:
        signal_indices = signal_rows.index.tolist()
        gaps = [signal_indices[i+1] - signal_indices[i] for i in range(len(signal_indices) - 1)]
        avg_gap = round(sum(gaps) / len(gaps), 2)
    else:
        avg_gap = np.nan
    metrics.append({'metric': 'days_between_signals', 'value': avg_gap})
    
    return pd.DataFrame(metrics)


def main():
    task_dir = Path(__file__).parent.parent / 'task'
    
    df = pd.read_csv(task_dir / 'prices.csv')
    
    df['SMA_20'] = compute_sma(df['close'], 20)
    df['SMA_50'] = compute_sma(df['close'], 50)
    
    df['EMA_12'], df['EMA_26'], df['MACD'], df['Signal'], df['Histogram'] = compute_macd(df['close'])
    
    df['RSI_14'] = compute_rsi(df['close'], 14)
    
    df['ATR_14'] = compute_atr(df['high'], df['low'], df['close'], 14)
    
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = compute_bollinger_bands(df['close'], 20, 2)
    
    df['Stoch_K'], df['Stoch_D'] = compute_stochastic(df['high'], df['low'], df['close'], 14, 3)
    
    df['Plus_DI'], df['Minus_DI'], df['ADX'] = compute_adx(df['high'], df['low'], df['close'], 14)
    
    df['signal'] = generate_signals(df)
    
    output_cols = ['date', 'close', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14', 'ATR_14',
                   'MACD', 'Signal', 'Histogram', 'BB_Upper', 'BB_Middle', 'BB_Lower',
                   'Stoch_K', 'Stoch_D', 'Plus_DI', 'Minus_DI', 'ADX', 'signal']
    
    output_df = df[output_cols].copy()
    
    numeric_cols = ['close', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14', 'ATR_14',
                    'MACD', 'Signal', 'Histogram', 'BB_Upper', 'BB_Middle', 'BB_Lower',
                    'Stoch_K', 'Stoch_D', 'Plus_DI', 'Minus_DI', 'ADX']
    
    for col in numeric_cols:
        output_df[col] = output_df[col].round(4)
    
    output_df.to_csv(task_dir / 'indicators.csv', index=False)
    
    perf_df = compute_performance_metrics(output_df)
    perf_df.to_csv(task_dir / 'performance.csv', index=False)
    
    print(f"Indicators saved to {task_dir / 'indicators.csv'}")
    print(f"Performance saved to {task_dir / 'performance.csv'}")
    print(f"Total rows: {len(output_df)}")
    print(f"Signals: {(output_df['signal'] != 'HOLD').sum()}")


if __name__ == '__main__':
    main()

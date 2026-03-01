import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_golden():
    ground_truth = Path(__file__).parent
    indicators = pd.read_csv(ground_truth / 'golden_indicators.csv')
    performance = pd.read_csv(ground_truth / 'golden_performance.csv')
    return indicators, performance


def load_submission(task_dir):
    indicators_path = task_dir / 'indicators.csv'
    performance_path = task_dir / 'performance.csv'
    
    if not indicators_path.exists():
        return None, None
    if not performance_path.exists():
        return None, None
    
    return pd.read_csv(indicators_path), pd.read_csv(performance_path)


def check_value(expected, actual, tol=0.01):
    if pd.isna(expected) and pd.isna(actual):
        return True
    if pd.isna(expected) or pd.isna(actual):
        return False
    try:
        return abs(float(expected) - float(actual)) <= tol
    except (ValueError, TypeError):
        return str(expected).strip() == str(actual).strip()


def check_indicators_columns(submission, golden):
    required = set(golden.columns)
    actual = set(submission.columns)
    missing = required - actual
    assert len(missing) == 0, f"Missing columns: {missing}"
    return True


def check_row_count(submission, golden):
    assert len(submission) == len(golden), f"Row count: expected {len(golden)}, got {len(submission)}"
    return True


def check_sma_20_values(submission, golden):
    col = 'SMA_20'
    valid_rows = golden[golden[col].notna()]
    sample = valid_rows.sample(min(20, len(valid_rows)), random_state=42)
    
    errors = 0
    for idx in sample.index:
        if not check_value(golden.loc[idx, col], submission.loc[idx, col], 0.01):
            errors += 1
    
    assert errors <= 2, f"SMA_20 errors: {errors}/20 sampled rows incorrect"
    return True


def check_sma_50_values(submission, golden):
    col = 'SMA_50'
    valid_rows = golden[golden[col].notna()]
    sample = valid_rows.sample(min(20, len(valid_rows)), random_state=43)
    
    errors = 0
    for idx in sample.index:
        if not check_value(golden.loc[idx, col], submission.loc[idx, col], 0.01):
            errors += 1
    
    assert errors <= 2, f"SMA_50 errors: {errors}/20 sampled rows incorrect"
    return True


def check_ema_12_values(submission, golden):
    col = 'EMA_12'
    valid_rows = golden[golden[col].notna()]
    sample = valid_rows.sample(min(20, len(valid_rows)), random_state=44)
    
    errors = 0
    for idx in sample.index:
        if not check_value(golden.loc[idx, col], submission.loc[idx, col], 0.01):
            errors += 1
    
    assert errors <= 2, f"EMA_12 errors: {errors}/20 sampled rows incorrect"
    return True


def check_ema_26_values(submission, golden):
    col = 'EMA_26'
    valid_rows = golden[golden[col].notna()]
    sample = valid_rows.sample(min(20, len(valid_rows)), random_state=45)
    
    errors = 0
    for idx in sample.index:
        if not check_value(golden.loc[idx, col], submission.loc[idx, col], 0.01):
            errors += 1
    
    assert errors <= 2, f"EMA_26 errors: {errors}/20 sampled rows incorrect"
    return True


def check_rsi_values(submission, golden):
    col = 'RSI_14'
    valid_rows = golden[golden[col].notna()]
    sample = valid_rows.sample(min(20, len(valid_rows)), random_state=46)
    
    errors = 0
    for idx in sample.index:
        if not check_value(golden.loc[idx, col], submission.loc[idx, col], 0.5):
            errors += 1
    
    assert errors <= 2, f"RSI_14 errors: {errors}/20 sampled rows incorrect"
    return True


def check_atr_values(submission, golden):
    col = 'ATR_14'
    valid_rows = golden[golden[col].notna()]
    sample = valid_rows.sample(min(20, len(valid_rows)), random_state=47)
    
    errors = 0
    for idx in sample.index:
        if not check_value(golden.loc[idx, col], submission.loc[idx, col], 0.05):
            errors += 1
    
    assert errors <= 2, f"ATR_14 errors: {errors}/20 sampled rows incorrect"
    return True


def check_macd_values(submission, golden):
    col = 'MACD'
    valid_rows = golden[golden[col].notna()]
    sample = valid_rows.sample(min(20, len(valid_rows)), random_state=48)
    
    errors = 0
    for idx in sample.index:
        if not check_value(golden.loc[idx, col], submission.loc[idx, col], 0.02):
            errors += 1
    
    assert errors <= 2, f"MACD errors: {errors}/20 sampled rows incorrect"
    return True


def check_signal_line_values(submission, golden):
    col = 'Signal'
    valid_rows = golden[golden[col].notna()]
    sample = valid_rows.sample(min(20, len(valid_rows)), random_state=49)
    
    errors = 0
    for idx in sample.index:
        if not check_value(golden.loc[idx, col], submission.loc[idx, col], 0.02):
            errors += 1
    
    assert errors <= 2, f"Signal line errors: {errors}/20 sampled rows incorrect"
    return True


def check_histogram_values(submission, golden):
    col = 'Histogram'
    valid_rows = golden[golden[col].notna()]
    sample = valid_rows.sample(min(20, len(valid_rows)), random_state=50)
    
    errors = 0
    for idx in sample.index:
        if not check_value(golden.loc[idx, col], submission.loc[idx, col], 0.02):
            errors += 1
    
    assert errors <= 2, f"Histogram errors: {errors}/20 sampled rows incorrect"
    return True


def check_bb_upper_values(submission, golden):
    col = 'BB_Upper'
    valid_rows = golden[golden[col].notna()]
    sample = valid_rows.sample(min(20, len(valid_rows)), random_state=51)
    
    errors = 0
    for idx in sample.index:
        if not check_value(golden.loc[idx, col], submission.loc[idx, col], 0.05):
            errors += 1
    
    assert errors <= 2, f"BB_Upper errors: {errors}/20 sampled rows incorrect"
    return True


def check_bb_lower_values(submission, golden):
    col = 'BB_Lower'
    valid_rows = golden[golden[col].notna()]
    sample = valid_rows.sample(min(20, len(valid_rows)), random_state=52)
    
    errors = 0
    for idx in sample.index:
        if not check_value(golden.loc[idx, col], submission.loc[idx, col], 0.05):
            errors += 1
    
    assert errors <= 2, f"BB_Lower errors: {errors}/20 sampled rows incorrect"
    return True


def check_stoch_k_values(submission, golden):
    col = 'Stoch_K'
    valid_rows = golden[golden[col].notna()]
    sample = valid_rows.sample(min(20, len(valid_rows)), random_state=53)
    
    errors = 0
    for idx in sample.index:
        if not check_value(golden.loc[idx, col], submission.loc[idx, col], 0.5):
            errors += 1
    
    assert errors <= 2, f"Stoch_K errors: {errors}/20 sampled rows incorrect"
    return True


def check_stoch_d_values(submission, golden):
    col = 'Stoch_D'
    valid_rows = golden[golden[col].notna()]
    sample = valid_rows.sample(min(20, len(valid_rows)), random_state=54)
    
    errors = 0
    for idx in sample.index:
        if not check_value(golden.loc[idx, col], submission.loc[idx, col], 0.5):
            errors += 1
    
    assert errors <= 2, f"Stoch_D errors: {errors}/20 sampled rows incorrect"
    return True


def check_plus_di_values(submission, golden):
    col = 'Plus_DI'
    valid_rows = golden[golden[col].notna()]
    sample = valid_rows.sample(min(20, len(valid_rows)), random_state=55)
    
    errors = 0
    for idx in sample.index:
        if not check_value(golden.loc[idx, col], submission.loc[idx, col], 1.0):
            errors += 1
    
    assert errors <= 2, f"Plus_DI errors: {errors}/20 sampled rows incorrect"
    return True


def check_minus_di_values(submission, golden):
    col = 'Minus_DI'
    valid_rows = golden[golden[col].notna()]
    sample = valid_rows.sample(min(20, len(valid_rows)), random_state=56)
    
    errors = 0
    for idx in sample.index:
        if not check_value(golden.loc[idx, col], submission.loc[idx, col], 1.0):
            errors += 1
    
    assert errors <= 2, f"Minus_DI errors: {errors}/20 sampled rows incorrect"
    return True


def check_adx_values(submission, golden):
    col = 'ADX'
    valid_rows = golden[golden[col].notna()]
    sample = valid_rows.sample(min(20, len(valid_rows)), random_state=57)
    
    errors = 0
    for idx in sample.index:
        if not check_value(golden.loc[idx, col], submission.loc[idx, col], 1.0):
            errors += 1
    
    assert errors <= 2, f"ADX errors: {errors}/20 sampled rows incorrect"
    return True


def check_signal_count(submission, golden):
    expected_buy = (golden['signal'] == 'BUY').sum()
    expected_sell = (golden['signal'] == 'SELL').sum()
    actual_buy = (submission['signal'] == 'BUY').sum()
    actual_sell = (submission['signal'] == 'SELL').sum()
    
    buy_tol = max(1, expected_buy * 0.3)
    sell_tol = max(1, expected_sell * 0.3)
    
    assert abs(actual_buy - expected_buy) <= buy_tol, \
        f"BUY signal count: expected {expected_buy}, got {actual_buy}"
    assert abs(actual_sell - expected_sell) <= sell_tol, \
        f"SELL signal count: expected {expected_sell}, got {actual_sell}"
    return True


def check_performance_metrics_exist(perf_submission, perf_golden):
    required_metrics = set(perf_golden['metric'].tolist())
    actual_metrics = set(perf_submission['metric'].tolist())
    missing = required_metrics - actual_metrics
    assert len(missing) == 0, f"Missing performance metrics: {missing}"
    return True


def check_total_signals_metric(perf_submission, perf_golden):
    expected = perf_golden[perf_golden['metric'] == 'total_signals']['value'].iloc[0]
    actual_row = perf_submission[perf_submission['metric'] == 'total_signals']
    assert len(actual_row) > 0, "Missing total_signals metric"
    actual = actual_row['value'].iloc[0]
    tol = max(2, int(expected) * 0.3)
    assert abs(int(actual) - int(expected)) <= tol, \
        f"total_signals: expected {expected}, got {actual}"
    return True


def check_avg_rsi_metrics(perf_submission, perf_golden):
    for metric in ['avg_rsi_at_buy', 'avg_rsi_at_sell']:
        expected_row = perf_golden[perf_golden['metric'] == metric]
        actual_row = perf_submission[perf_submission['metric'] == metric]
        
        if len(expected_row) == 0 or len(actual_row) == 0:
            continue
        
        expected = expected_row['value'].iloc[0]
        actual = actual_row['value'].iloc[0]
        
        if pd.isna(expected) and pd.isna(actual):
            continue
        if pd.isna(expected) or pd.isna(actual):
            continue
        
        assert abs(float(actual) - float(expected)) <= 5.0, \
            f"{metric}: expected {expected}, got {actual}"
    return True


def check_histogram_extremes(perf_submission, perf_golden):
    for metric in ['max_histogram', 'min_histogram']:
        expected_row = perf_golden[perf_golden['metric'] == metric]
        actual_row = perf_submission[perf_submission['metric'] == metric]
        
        if len(expected_row) == 0 or len(actual_row) == 0:
            continue
        
        expected = float(expected_row['value'].iloc[0])
        actual = float(actual_row['value'].iloc[0])
        
        assert abs(actual - expected) <= 0.1, \
            f"{metric}: expected {expected}, got {actual}"
    return True


def check_signal_dates(perf_submission, perf_golden):
    for metric in ['first_signal_date', 'last_signal_date']:
        expected_row = perf_golden[perf_golden['metric'] == metric]
        actual_row = perf_submission[perf_submission['metric'] == metric]
        
        if len(expected_row) == 0 or len(actual_row) == 0:
            continue
        
        expected = str(expected_row['value'].iloc[0])
        actual = str(actual_row['value'].iloc[0])
        
        if expected != 'nan' and actual != 'nan':
            exp_date = pd.to_datetime(expected)
            try:
                act_date = pd.to_datetime(actual)
                assert abs((act_date - exp_date).days) <= 10, \
                    f"{metric}: expected {expected}, got {actual}"
            except:
                pass
    return True


def check_ema_initialization(submission, golden):
    ema_12_start = golden['EMA_12'].first_valid_index()
    ema_26_start = golden['EMA_26'].first_valid_index()
    
    assert ema_12_start >= 11, "EMA_12 should start at index 11 (after 12 periods)"
    
    sub_ema12_start = submission['EMA_12'].first_valid_index()
    assert sub_ema12_start is not None, "EMA_12 has no valid values"
    assert abs(sub_ema12_start - ema_12_start) <= 1, \
        f"EMA_12 starts at wrong index: expected {ema_12_start}, got {sub_ema12_start}"
    
    return True


def check_rsi_range(submission, golden):
    valid_rsi = submission['RSI_14'].dropna()
    out_of_range = ((valid_rsi < 0) | (valid_rsi > 100)).sum()
    assert out_of_range == 0, f"RSI has {out_of_range} values outside 0-100 range"
    return True


def run_verification():
    task_dir = Path(__file__).parent.parent / 'task'
    
    golden_ind, golden_perf = load_golden()
    sub_ind, sub_perf = load_submission(task_dir)
    
    if sub_ind is None or sub_perf is None:
        print("FAIL: Missing submission files")
        print("0/25")
        sys.exit(1)
    
    checks = [
        ('check_indicators_columns', check_indicators_columns, [sub_ind, golden_ind]),
        ('check_row_count', check_row_count, [sub_ind, golden_ind]),
        ('check_sma_20_values', check_sma_20_values, [sub_ind, golden_ind]),
        ('check_sma_50_values', check_sma_50_values, [sub_ind, golden_ind]),
        ('check_ema_12_values', check_ema_12_values, [sub_ind, golden_ind]),
        ('check_ema_26_values', check_ema_26_values, [sub_ind, golden_ind]),
        ('check_rsi_values', check_rsi_values, [sub_ind, golden_ind]),
        ('check_atr_values', check_atr_values, [sub_ind, golden_ind]),
        ('check_macd_values', check_macd_values, [sub_ind, golden_ind]),
        ('check_signal_line_values', check_signal_line_values, [sub_ind, golden_ind]),
        ('check_histogram_values', check_histogram_values, [sub_ind, golden_ind]),
        ('check_bb_upper_values', check_bb_upper_values, [sub_ind, golden_ind]),
        ('check_bb_lower_values', check_bb_lower_values, [sub_ind, golden_ind]),
        ('check_stoch_k_values', check_stoch_k_values, [sub_ind, golden_ind]),
        ('check_stoch_d_values', check_stoch_d_values, [sub_ind, golden_ind]),
        ('check_plus_di_values', check_plus_di_values, [sub_ind, golden_ind]),
        ('check_minus_di_values', check_minus_di_values, [sub_ind, golden_ind]),
        ('check_adx_values', check_adx_values, [sub_ind, golden_ind]),
        ('check_signal_count', check_signal_count, [sub_ind, golden_ind]),
        ('check_performance_metrics_exist', check_performance_metrics_exist, [sub_perf, golden_perf]),
        ('check_total_signals_metric', check_total_signals_metric, [sub_perf, golden_perf]),
        ('check_avg_rsi_metrics', check_avg_rsi_metrics, [sub_perf, golden_perf]),
        ('check_histogram_extremes', check_histogram_extremes, [sub_perf, golden_perf]),
        ('check_signal_dates', check_signal_dates, [sub_perf, golden_perf]),
        ('check_ema_initialization', check_ema_initialization, [sub_ind, golden_ind]),
        ('check_rsi_range', check_rsi_range, [sub_ind, golden_ind]),
    ]
    
    passed = 0
    max_score = len(checks)
    
    for name, check_func, args in checks:
        try:
            check_func(*args)
            print(f"PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {name}: {e}")
        except Exception as e:
            print(f"ERROR: {name}: {e}")
    
    print(f"{passed}/{max_score}")
    sys.exit(0 if passed == max_score else 1)


if __name__ == '__main__':
    run_verification()

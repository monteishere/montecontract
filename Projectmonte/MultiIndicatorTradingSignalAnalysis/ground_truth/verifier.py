import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def load_golden_data() -> pd.DataFrame:
    golden_path = Path(__file__).parent / "golden_signal_analysis.csv"
    return pd.read_csv(golden_path)


def load_submission_data(task_dir: Path) -> Optional[pd.DataFrame]:
    submission_path = task_dir / "signal_analysis.csv"
    if not submission_path.exists():
        return None
    return pd.read_csv(submission_path)


def check_file_exists(submission: Optional[pd.DataFrame]) -> Tuple[bool, str]:
    if submission is None:
        return False, "Output file signal_analysis.csv not found"
    return True, "Output file exists"


def check_required_columns(submission: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, str]:
    missing = [c for c in required_cols if c not in submission.columns]
    if missing:
        return False, f"Missing columns: {missing}"
    return True, "All required columns present"


def check_data_types(submission: pd.DataFrame) -> Tuple[bool, str]:
    try:
        submission['composite_score'].astype(float)
        submission['position_size'].astype(int)
        submission['volume_ratio'].astype(float)
        return True, "Data types are valid"
    except (ValueError, TypeError) as e:
        return False, f"Invalid data types: {str(e)}"


def check_row_count_range(submission: pd.DataFrame, golden: pd.DataFrame, tolerance: float = 0.7) -> Tuple[bool, str]:
    expected = len(golden)
    actual = len(submission)
    min_acceptable = max(3, int(expected * (1 - tolerance)))
    max_acceptable = int(expected * (1 + tolerance))
    if min_acceptable <= actual <= max_acceptable:
        return True, f"Row count {actual} is within acceptable range [{min_acceptable}, {max_acceptable}]"
    return False, f"Row count {actual} outside acceptable range [{min_acceptable}, {max_acceptable}]"


def check_signal_types_valid(submission: pd.DataFrame) -> Tuple[bool, str]:
    valid_types = {'BUY', 'SELL'}
    actual_types = set(submission['signal_type'].unique())
    invalid = actual_types - valid_types
    if invalid:
        return False, f"Invalid signal types found: {invalid}"
    return True, "All signal types are valid (BUY/SELL)"


def check_regime_values_valid(submission: pd.DataFrame) -> Tuple[bool, str]:
    valid_regimes = {'UPTREND', 'DOWNTREND'}
    actual_regimes = set(submission['regime'].unique())
    invalid = actual_regimes - valid_regimes
    if invalid:
        return False, f"Invalid regime values found: {invalid}"
    return True, "All regime values are valid (UPTREND/DOWNTREND)"


def check_signal_type_distribution(submission: pd.DataFrame) -> Tuple[bool, str]:
    actual_buy = (submission['signal_type'] == 'BUY').sum()
    actual_sell = (submission['signal_type'] == 'SELL').sum()
    total = len(submission)
    
    if actual_buy >= 1 and actual_sell >= 1:
        return True, f"Signal distribution has both types (BUY: {actual_buy}, SELL: {actual_sell})"
    return False, f"Signal distribution missing type (BUY: {actual_buy}, SELL: {actual_sell})"


def check_composite_score_range(submission: pd.DataFrame) -> Tuple[bool, str]:
    min_score = submission['composite_score'].min()
    max_score = submission['composite_score'].max()
    
    # New scoring system has higher threshold (4.0 base) and more indicators (max ~10 points)
    if min_score >= 3.0 and max_score <= 12.0:
        return True, f"Composite scores in valid range [{min_score}, {max_score}]"
    if min_score < 3.0:
        return False, f"Some composite scores too low: minimum = {min_score}"
    return False, f"Composite scores outside expected range [{min_score}, {max_score}]"


def check_position_size_positive(submission: pd.DataFrame) -> Tuple[bool, str]:
    min_position = submission['position_size'].min()
    if min_position > 0:
        return True, f"All position sizes are positive (min: {min_position})"
    return False, f"Invalid position sizes found (min: {min_position})"


def check_position_size_range(submission: pd.DataFrame) -> Tuple[bool, str]:
    actual_min = submission['position_size'].min()
    actual_max = submission['position_size'].max()
    
    # Specified min 25, max 500
    if actual_min >= 25 and actual_max <= 500:
        return True, f"Position size range acceptable [{actual_min}, {actual_max}]"
    return False, f"Position size range [{actual_min}, {actual_max}] outside bounds [25, 500]"


def check_volume_ratio_range(submission: pd.DataFrame) -> Tuple[bool, str]:
    min_vol = submission['volume_ratio'].min()
    max_vol = submission['volume_ratio'].max()
    
    # Volume ratio must be >= 1.2 for valid signals
    if min_vol >= 1.2:
        return True, f"Volume ratios valid (min: {min_vol:.2f}, max: {max_vol:.2f})"
    return False, f"Volume ratio below 1.2 minimum: {min_vol:.2f}"


def check_date_ordering(submission: pd.DataFrame) -> Tuple[bool, str]:
    dates = submission['date'].tolist()
    is_sorted = all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
    if is_sorted:
        return True, "Dates are properly ordered"
    return False, "Dates are not in ascending order"


def check_no_duplicate_dates(submission: pd.DataFrame) -> Tuple[bool, str]:
    duplicate_count = submission['date'].duplicated().sum()
    if duplicate_count == 0:
        return True, "No duplicate dates found"
    return False, f"Found {duplicate_count} duplicate dates"


def check_signal_spacing(submission: pd.DataFrame) -> Tuple[bool, str]:
    dates = pd.to_datetime(submission['date'])
    dates_sorted = dates.sort_values()
    
    violations = 0
    for i in range(1, len(dates_sorted)):
        diff = (dates_sorted.iloc[i] - dates_sorted.iloc[i-1]).days
        if diff < 4:  # 5 bars minimum = at least 4 days apart
            violations += 1
    
    if violations <= 1:
        return True, f"Signal spacing acceptable (violations: {violations})"
    return False, f"Too many closely spaced signals (violations: {violations})"


def check_buy_signals_valid(submission: pd.DataFrame) -> Tuple[bool, str]:
    buy_signals = submission[submission['signal_type'] == 'BUY']
    if len(buy_signals) == 0:
        return False, "No BUY signals found"
    
    avg_score = buy_signals['composite_score'].mean()
    if avg_score >= 3.0:
        return True, f"BUY signals have valid average score: {avg_score:.2f}"
    return False, f"BUY signals have low average score: {avg_score:.2f}"


def check_sell_signals_valid(submission: pd.DataFrame) -> Tuple[bool, str]:
    sell_signals = submission[submission['signal_type'] == 'SELL']
    if len(sell_signals) == 0:
        return False, "No SELL signals found"
    
    avg_score = sell_signals['composite_score'].mean()
    if avg_score >= 3.0:
        return True, f"SELL signals have valid average score: {avg_score:.2f}"
    return False, f"SELL signals have low average score: {avg_score:.2f}"


def check_warmup_respected(submission: pd.DataFrame) -> Tuple[bool, str]:
    first_date = submission['date'].iloc[0]
    # 55 bars warmup = signals should not start before late February
    if first_date >= "2024-02-15":
        return True, f"Warmup period respected (first signal: {first_date})"
    return False, f"Signal too early (first signal: {first_date}), warmup not respected"


def check_score_precision(submission: pd.DataFrame) -> Tuple[bool, str]:
    scores = submission['composite_score']
    # Scores should be rounded to 2 decimal places
    valid = all(abs(score - round(score, 2)) < 0.001 for score in scores)
    if valid:
        return True, "Score precision is valid (2 decimal places)"
    return False, "Score precision invalid"


def check_volume_ratio_precision(submission: pd.DataFrame) -> Tuple[bool, str]:
    ratios = submission['volume_ratio']
    # Volume ratios should be rounded to 2 decimal places
    valid = all(abs(r - round(r, 2)) < 0.001 for r in ratios)
    if valid:
        return True, "Volume ratio precision is valid (2 decimal places)"
    return False, "Volume ratio precision invalid"


def check_minimum_signals(submission: pd.DataFrame, minimum: int = 5) -> Tuple[bool, str]:
    if len(submission) >= minimum:
        return True, f"Has at least {minimum} signals (actual: {len(submission)})"
    return False, f"Too few signals: {len(submission)} < {minimum}"


def check_maximum_signals(submission: pd.DataFrame, maximum: int = 50) -> Tuple[bool, str]:
    if len(submission) <= maximum:
        return True, f"Signal count within limit (actual: {len(submission)})"
    return False, f"Too many signals: {len(submission)} > {maximum}"


def check_both_signal_types_present(submission: pd.DataFrame) -> Tuple[bool, str]:
    types = set(submission['signal_type'].unique())
    if 'BUY' in types and 'SELL' in types:
        return True, "Both BUY and SELL signals present"
    return False, f"Missing signal type(s), only found: {types}"


def check_both_regimes_present(submission: pd.DataFrame) -> Tuple[bool, str]:
    regimes = set(submission['regime'].unique())
    if 'UPTREND' in regimes and 'DOWNTREND' in regimes:
        return True, "Both UPTREND and DOWNTREND regimes present"
    # Allow single regime if data doesn't have both
    if len(regimes) >= 1:
        return True, f"Regime(s) present: {regimes}"
    return False, "No regime values found"


def verify(task_dir: Path) -> None:
    golden = load_golden_data()
    submission = load_submission_data(task_dir)
    
    required_cols = ["date", "signal_type", "composite_score", "position_size", "regime", "volume_ratio"]
    
    test_results = []
    
    passed, msg = check_file_exists(submission)
    test_results.append(("file_exists", passed, msg))
    
    if not passed:
        total_tests = 22
        print(msg)
        print(f"0/{total_tests}")
        sys.exit(1)
    
    passed, msg = check_required_columns(submission, required_cols)
    test_results.append(("required_columns", passed, msg))
    
    if not passed:
        total_tests = 22
        print(msg)
        print(f"1/{total_tests}")
        sys.exit(1)
    
    tests = [
        ("data_types", lambda: check_data_types(submission)),
        ("row_count_range", lambda: check_row_count_range(submission, golden)),
        ("signal_types_valid", lambda: check_signal_types_valid(submission)),
        ("regime_values_valid", lambda: check_regime_values_valid(submission)),
        ("signal_distribution", lambda: check_signal_type_distribution(submission)),
        ("composite_score_range", lambda: check_composite_score_range(submission)),
        ("position_size_positive", lambda: check_position_size_positive(submission)),
        ("position_size_range", lambda: check_position_size_range(submission)),
        ("volume_ratio_range", lambda: check_volume_ratio_range(submission)),
        ("date_ordering", lambda: check_date_ordering(submission)),
        ("no_duplicate_dates", lambda: check_no_duplicate_dates(submission)),
        ("signal_spacing", lambda: check_signal_spacing(submission)),
        ("buy_signals_valid", lambda: check_buy_signals_valid(submission)),
        ("sell_signals_valid", lambda: check_sell_signals_valid(submission)),
        ("warmup_respected", lambda: check_warmup_respected(submission)),
        ("score_precision", lambda: check_score_precision(submission)),
        ("volume_ratio_precision", lambda: check_volume_ratio_precision(submission)),
        ("minimum_signals", lambda: check_minimum_signals(submission)),
        ("maximum_signals", lambda: check_maximum_signals(submission)),
        ("both_types_present", lambda: check_both_signal_types_present(submission)),
    ]
    
    for test_name, test_func in tests:
        try:
            passed, msg = test_func()
            test_results.append((test_name, passed, msg))
        except Exception as e:
            test_results.append((test_name, False, f"Error: {str(e)}"))
    
    total_tests = len(test_results)
    passed_count = sum(1 for _, passed, _ in test_results if passed)
    
    for test_name, passed, msg in test_results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {test_name}: {msg}")
    
    print(f"{passed_count}/{total_tests}")
    sys.exit(0 if passed_count == total_tests else 1)


if __name__ == "__main__":
    task_dir = Path(__file__).parent.parent / "task"
    verify(task_dir)

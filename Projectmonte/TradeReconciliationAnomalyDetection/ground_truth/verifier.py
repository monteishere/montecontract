import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_generated_files(task_dir):
    report_path = task_dir / 'anomaly_report.csv'
    summary_path = task_dir / 'anomaly_summary.csv'
    if not report_path.exists():
        raise FileNotFoundError("anomaly_report.csv not found")
    if not summary_path.exists():
        raise FileNotFoundError("anomaly_summary.csv not found")
    return pd.read_csv(report_path), pd.read_csv(summary_path)


def load_golden(ground_truth_dir):
    return (
        pd.read_csv(ground_truth_dir / 'golden_anomaly_report.csv'),
        pd.read_csv(ground_truth_dir / 'golden_anomaly_summary.csv')
    )


def check_report_columns(report, ground_truth_dir):
    required = {'anomaly_type', 'identifier', 'date', 'symbol', 'details', 'severity'}
    actual = set(report.columns)
    missing = required - actual
    assert len(missing) == 0, f"Missing columns in report: {missing}"
    return True


def check_summary_columns(summary, ground_truth_dir):
    required = {'metric', 'value'}
    actual = set(summary.columns)
    missing = required - actual
    assert len(missing) == 0, f"Missing columns in summary: {missing}"
    return True


def check_valid_anomaly_types(report, ground_truth_dir):
    valid_types = {
        'UNMATCHED_TRADE', 'DUPLICATE_TRADE', 'TIMING_VIOLATION',
        'PRICE_DEVIATION', 'POSITION_MISMATCH', 'SAME_DAY_REVERSAL'
    }
    reported_types = set(report['anomaly_type'].unique())
    invalid = reported_types - valid_types
    assert len(invalid) == 0, f"Invalid anomaly types: {invalid}"
    return True


def check_valid_severity_values(report, ground_truth_dir):
    valid_severities = {'HIGH', 'MEDIUM', 'LOW'}
    reported = set(report['severity'].unique())
    invalid = reported - valid_severities
    assert len(invalid) == 0, f"Invalid severity values: {invalid}"
    return True


def check_all_six_types_detected(report, ground_truth_dir):
    expected_types = {
        'UNMATCHED_TRADE', 'DUPLICATE_TRADE', 'TIMING_VIOLATION',
        'PRICE_DEVIATION', 'POSITION_MISMATCH', 'SAME_DAY_REVERSAL'
    }
    found_types = set(report['anomaly_type'].unique())
    missing = expected_types - found_types
    assert len(missing) == 0, f"Missing anomaly types: {missing}"
    return True


def check_total_anomaly_count(report, ground_truth_dir):
    golden, _ = load_golden(ground_truth_dir)
    expected = len(golden)
    actual = len(report)
    tolerance = expected * 0.15
    assert abs(actual - expected) <= tolerance, \
        f"Anomaly count {actual} differs from expected {expected} by more than 15%"
    return True


def check_unmatched_trades_count(report, ground_truth_dir):
    golden, _ = load_golden(ground_truth_dir)
    expected = len(golden[golden['anomaly_type'] == 'UNMATCHED_TRADE'])
    actual = len(report[report['anomaly_type'] == 'UNMATCHED_TRADE'])
    tolerance = max(3, expected * 0.2)
    assert abs(actual - expected) <= tolerance, \
        f"UNMATCHED_TRADE count {actual} differs from expected {expected} beyond tolerance"
    return True


def check_duplicate_trades_count(report, ground_truth_dir):
    golden, _ = load_golden(ground_truth_dir)
    expected = len(golden[golden['anomaly_type'] == 'DUPLICATE_TRADE'])
    actual = len(report[report['anomaly_type'] == 'DUPLICATE_TRADE'])
    tolerance = max(2, expected * 0.25)
    assert abs(actual - expected) <= tolerance, \
        f"DUPLICATE_TRADE count {actual} differs from expected {expected}"
    return True


def check_timing_violations_count(report, ground_truth_dir):
    golden, _ = load_golden(ground_truth_dir)
    expected = len(golden[golden['anomaly_type'] == 'TIMING_VIOLATION'])
    actual = len(report[report['anomaly_type'] == 'TIMING_VIOLATION'])
    tolerance = max(2, expected * 0.15)
    assert abs(actual - expected) <= tolerance, \
        f"TIMING_VIOLATION count {actual} differs from expected {expected}"
    return True


def check_price_deviations_count(report, ground_truth_dir):
    golden, _ = load_golden(ground_truth_dir)
    expected = len(golden[golden['anomaly_type'] == 'PRICE_DEVIATION'])
    actual = len(report[report['anomaly_type'] == 'PRICE_DEVIATION'])
    tolerance = max(2, expected * 0.25)
    assert abs(actual - expected) <= tolerance, \
        f"PRICE_DEVIATION count {actual} differs from expected {expected}"
    return True


def check_position_mismatches_count(report, ground_truth_dir):
    golden, _ = load_golden(ground_truth_dir)
    expected = len(golden[golden['anomaly_type'] == 'POSITION_MISMATCH'])
    actual = len(report[report['anomaly_type'] == 'POSITION_MISMATCH'])
    tolerance = max(2, expected * 0.25)
    assert abs(actual - expected) <= tolerance, \
        f"POSITION_MISMATCH count {actual} differs from expected {expected}"
    return True


def check_same_day_reversals_count(report, ground_truth_dir):
    golden, _ = load_golden(ground_truth_dir)
    expected = len(golden[golden['anomaly_type'] == 'SAME_DAY_REVERSAL'])
    actual = len(report[report['anomaly_type'] == 'SAME_DAY_REVERSAL'])
    tolerance = max(1, expected * 0.3)
    assert abs(actual - expected) <= tolerance, \
        f"SAME_DAY_REVERSAL count {actual} differs from expected {expected}"
    return True


def check_high_severity_count(report, ground_truth_dir):
    golden, _ = load_golden(ground_truth_dir)
    expected = len(golden[golden['severity'] == 'HIGH'])
    actual = len(report[report['severity'] == 'HIGH'])
    tolerance = max(2, expected * 0.3)
    assert abs(actual - expected) <= tolerance, \
        f"HIGH severity count {actual} differs from expected {expected}"
    return True


def check_medium_severity_count(report, ground_truth_dir):
    golden, _ = load_golden(ground_truth_dir)
    expected = len(golden[golden['severity'] == 'MEDIUM'])
    actual = len(report[report['severity'] == 'MEDIUM'])
    tolerance = max(3, expected * 0.25)
    assert abs(actual - expected) <= tolerance, \
        f"MEDIUM severity count {actual} differs from expected {expected}"
    return True


def check_summary_total_anomalies(summary, ground_truth_dir):
    total_row = summary[summary['metric'] == 'total_anomalies']
    assert len(total_row) > 0, "Summary missing total_anomalies metric"
    value = int(total_row['value'].iloc[0])
    assert value >= 50, f"total_anomalies {value} too low, expected at least 50"
    assert value <= 120, f"total_anomalies {value} too high, expected at most 120"
    return True


def check_summary_affected_value(summary, ground_truth_dir):
    _, golden_summary = load_golden(ground_truth_dir)
    golden_row = golden_summary[golden_summary['metric'] == 'total_affected_value']
    expected = float(golden_row['value'].iloc[0])
    value_row = summary[summary['metric'] == 'total_affected_value']
    assert len(value_row) > 0, "Summary missing total_affected_value metric"
    value = float(value_row['value'].iloc[0])
    tolerance = expected * 0.20
    assert abs(value - expected) <= tolerance, \
        f"Affected value {value} differs from expected {expected} by more than 20%"
    return True


def check_date_format(report, ground_truth_dir):
    for date in report['date'].unique():
        parts = str(date).split('-')
        assert len(parts) == 3, f"Invalid date format: {date}"
        assert len(parts[0]) == 4, f"Invalid year format in date: {date}"
        assert len(parts[1]) == 2, f"Invalid month format in date: {date}"
        assert len(parts[2]) == 2, f"Invalid day format in date: {date}"
    return True


def check_symbols_valid(report, ground_truth_dir):
    valid_symbols = {'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'}
    reported = set(report['symbol'].unique())
    invalid = reported - valid_symbols
    assert len(invalid) == 0, f"Invalid symbols detected: {invalid}"
    return True


def check_no_duplicate_identifiers_same_type(report, ground_truth_dir):
    keys = report.groupby(['anomaly_type', 'identifier']).size()
    duplicates = keys[keys > 1]
    assert len(duplicates) == 0, f"Duplicate anomaly entries found: {duplicates.index.tolist()}"
    return True


def check_details_not_empty(report, ground_truth_dir):
    empty_details = report[report['details'].isna() | (report['details'] == '')]
    assert len(empty_details) == 0, f"Found {len(empty_details)} entries with empty details"
    return True


def check_price_deviations_above_threshold(report, ground_truth_dir):
    price_devs = report[report['anomaly_type'] == 'PRICE_DEVIATION']
    for _, row in price_devs.iterrows():
        details = row['details']
        if 'dev=' in details:
            pct_str = details.split('dev=')[1].split('%')[0]
            pct = float(pct_str)
            assert pct > 3.0, f"Price deviation {pct}% below 3% threshold: {row['identifier']}"
    return True


def check_position_mismatch_above_threshold(report, ground_truth_dir):
    pos_mismatches = report[report['anomaly_type'] == 'POSITION_MISMATCH']
    for _, row in pos_mismatches.iterrows():
        details = row['details']
        if 'diff=' in details:
            diff = int(details.split('diff=')[1])
            assert diff > 100, f"Position mismatch diff={diff} below threshold 100: {row['identifier']}"
    return True


def check_timing_includes_time_info(report, ground_truth_dir):
    timing = report[report['anomaly_type'] == 'TIMING_VIOLATION']
    for _, row in timing.iterrows():
        details = row['details']
        assert ':' in details, f"Timing violation missing time info: {row['identifier']}"
    return True


def check_unmatched_includes_signal_info(report, ground_truth_dir):
    unmatched = report[report['anomaly_type'] == 'UNMATCHED_TRADE']
    for _, row in unmatched.iterrows():
        details = row['details']
        assert 'signal_id=' in details or 'signal' in details.lower(), \
            f"Unmatched trade missing signal info: {row['identifier']}"
    return True


def check_summary_metrics_count(summary, ground_truth_dir):
    assert len(summary) >= 10, f"Summary has only {len(summary)} metrics, expected at least 10"
    return True


def check_affected_symbols_metric(summary, ground_truth_dir):
    symbols_row = summary[summary['metric'] == 'affected_symbols']
    assert len(symbols_row) > 0, "Summary missing affected_symbols metric"
    value = str(symbols_row['value'].iloc[0])
    symbols_list = value.split(',')
    assert len(symbols_list) >= 3, f"Expected at least 3 affected symbols, found {len(symbols_list)}"
    return True


def check_all_five_symbols_affected(report, ground_truth_dir):
    expected_symbols = {'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'}
    found_symbols = set(report['symbol'].unique())
    missing = expected_symbols - found_symbols
    assert len(missing) == 0, f"Missing anomalies for symbols: {missing}"
    return True


def run_all_checks():
    task_dir = Path(__file__).parent.parent / 'task'
    ground_truth_dir = Path(__file__).parent

    try:
        report, summary = load_generated_files(task_dir)
    except FileNotFoundError as e:
        print(f"FAIL: {e}")
        print("0/27")
        sys.exit(1)

    checks = [
        (check_report_columns, [report, ground_truth_dir]),
        (check_summary_columns, [summary, ground_truth_dir]),
        (check_valid_anomaly_types, [report, ground_truth_dir]),
        (check_valid_severity_values, [report, ground_truth_dir]),
        (check_all_six_types_detected, [report, ground_truth_dir]),
        (check_total_anomaly_count, [report, ground_truth_dir]),
        (check_unmatched_trades_count, [report, ground_truth_dir]),
        (check_duplicate_trades_count, [report, ground_truth_dir]),
        (check_timing_violations_count, [report, ground_truth_dir]),
        (check_price_deviations_count, [report, ground_truth_dir]),
        (check_position_mismatches_count, [report, ground_truth_dir]),
        (check_same_day_reversals_count, [report, ground_truth_dir]),
        (check_high_severity_count, [report, ground_truth_dir]),
        (check_medium_severity_count, [report, ground_truth_dir]),
        (check_summary_total_anomalies, [summary, ground_truth_dir]),
        (check_summary_affected_value, [summary, ground_truth_dir]),
        (check_date_format, [report, ground_truth_dir]),
        (check_symbols_valid, [report, ground_truth_dir]),
        (check_no_duplicate_identifiers_same_type, [report, ground_truth_dir]),
        (check_details_not_empty, [report, ground_truth_dir]),
        (check_price_deviations_above_threshold, [report, ground_truth_dir]),
        (check_position_mismatch_above_threshold, [report, ground_truth_dir]),
        (check_timing_includes_time_info, [report, ground_truth_dir]),
        (check_unmatched_includes_signal_info, [report, ground_truth_dir]),
        (check_summary_metrics_count, [summary, ground_truth_dir]),
        (check_affected_symbols_metric, [summary, ground_truth_dir]),
        (check_all_five_symbols_affected, [report, ground_truth_dir]),
    ]

    passed = 0
    failed = 0
    max_score = len(checks)

    for check_func, args in checks:
        try:
            check_func(*args)
            print(f"PASS: {check_func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {check_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {check_func.__name__}: {e}")
            failed += 1

    print(f"{passed}/{max_score}")
    sys.exit(0 if passed == max_score else 1)


if __name__ == '__main__':
    run_all_checks()

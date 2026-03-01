import pandas as pd
import numpy as np
from pathlib import Path
import sys


GOLDEN_DIR = Path(__file__).parent
TASK_DIR = GOLDEN_DIR.parent / "task"

VALID_VIOLATION_CLASSES = {
    "CRITICAL_REPEAT", "CRITICAL", "MULTI_VIOLATION", "HEALTH_HAZARD",
    "SINGLE_VIOLATION", "WATCH", "COMPLIANT", "NO_DATA",
}

VALID_RISK_TRENDS = {"IMPROVING", "DECLINING", "STABLE", "NEW", "INSUFFICIENT"}

VALID_RISK_CATEGORIES = {"HIGH_RISK", "ELEVATED", "MODERATE", "LOW_RISK"}

VALID_THRESHOLD_TYPES = {"CRITICAL", "VIOLATION", "CAUTION"}

VALID_PARAMETERS = {"ph", "turbidity_ntu", "chlorine_mg_l", "lead_ppb", "coliform_count"}

PARAM_NORMALIZE = {
    "turbidity": "turbidity_ntu",
    "chlorine": "chlorine_mg_l",
    "lead": "lead_ppb",
    "coliform": "coliform_count",
}


def normalize_parameters(df):
    if "parameter" in df.columns:
        df["parameter"] = df["parameter"].map(lambda x: PARAM_NORMALIZE.get(str(x).strip(), str(x).strip()))
    return df


def load_golden(name):
    return pd.read_csv(GOLDEN_DIR / f"golden_{name}.csv")


def load_sub(name):
    path = TASK_DIR / f"{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = normalize_parameters(df)
    return df


def check_val(expected, actual, tol=0.01):
    if pd.isna(expected) and pd.isna(actual):
        return True
    if pd.isna(expected) or pd.isna(actual):
        return False
    try:
        return abs(float(expected) - float(actual)) <= tol
    except (ValueError, TypeError):
        return str(expected).strip() == str(actual).strip()


def merge_on_keys(sub, golden, keys):
    merged = golden.merge(sub, on=keys, suffixes=("_g", "_s"), how="inner")
    return merged


# ============================================================
#  station_compliance.csv checks (1-22)
# ============================================================

def check_compliance_exists(sub, golden):
    assert sub is not None, "station_compliance.csv not found"

def check_compliance_row_count(sub, golden):
    assert len(sub) == 180, f"Expected 180 rows, got {len(sub)}"

def check_compliance_columns(sub, golden):
    missing = set(golden.columns) - set(sub.columns)
    assert not missing, f"Missing columns: {missing}"

def check_compliance_station_ids(sub, golden):
    expected = set(golden["station_id"].unique())
    actual = set(sub["station_id"].unique())
    assert expected == actual, f"Station IDs mismatch: missing={expected-actual}, extra={actual-expected}"

def check_compliance_months(sub, golden):
    expected = set(golden["month"].unique())
    actual = set(sub["month"].unique())
    assert expected == actual, f"Months mismatch: expected {expected}, got {actual}"

def check_readings_ge_valid(sub, golden):
    bad = sub[sub["readings_count"] < sub["valid_readings_count"]]
    assert len(bad) == 0, f"{len(bad)} rows have readings_count < valid_readings_count"

def check_cqi_range(sub, golden):
    cqi = sub["cqi"].dropna()
    bad = cqi[(cqi < 0) | (cqi > 100)]
    assert len(bad) == 0, f"{len(bad)} CQI values out of [0,100] range"

def check_cqi_rounding(sub, golden):
    cqi = sub["cqi"].dropna()
    rounded = cqi.round(2)
    mismatches = (cqi - rounded).abs() > 1e-9
    assert mismatches.sum() == 0, f"{mismatches.sum()} CQI values not rounded to 2 decimals"

def check_rcs_rounding(sub, golden):
    rcs = sub["rcs"].dropna()
    rounded = rcs.round(3)
    mismatches = (rcs - rounded).abs() > 1e-9
    assert mismatches.sum() == 0, f"{mismatches.sum()} RCS values not rounded to 3 decimals"

def check_violation_class_values(sub, golden):
    invalid = set(sub["violation_class"].unique()) - VALID_VIOLATION_CLASSES
    assert not invalid, f"Invalid violation classes: {invalid}"

def check_risk_trend_values(sub, golden):
    invalid = set(sub["risk_trend"].unique()) - VALID_RISK_TRENDS
    assert not invalid, f"Invalid risk_trend values: {invalid}"

def check_consec_nonneg(sub, golden):
    bad = sub[sub["consecutive_violations"] < 0]
    assert len(bad) == 0, f"{len(bad)} rows have negative consecutive_violations"

def check_penalty_nonneg(sub, golden):
    bad = sub[sub["penalty_amount"] < -0.001]
    assert len(bad) == 0, f"{len(bad)} rows have negative penalty_amount"

def check_net_le_penalty(sub, golden):
    bad = sub[sub["net_penalty"] > sub["penalty_amount"] + 0.01]
    assert len(bad) == 0, f"{len(bad)} rows have net_penalty > penalty_amount"

def check_penalty_zero_compliant(sub, golden):
    no_pen = sub[sub["violation_class"].isin({"COMPLIANT", "NO_DATA"})]
    bad = no_pen[no_pen["penalty_amount"] > 0.001]
    assert len(bad) == 0, f"{len(bad)} COMPLIANT/NO_DATA rows have nonzero penalty"

def check_cqi_match(sub, golden):
    m = merge_on_keys(sub, golden, ["station_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["cqi_g"], row["cqi_s"], tol=0.05):
            mismatches += 1
    assert mismatches <= 3, f"{mismatches} CQI mismatches (tolerance 0.05)"

def check_rcs_match(sub, golden):
    m = merge_on_keys(sub, golden, ["station_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["rcs_g"], row["rcs_s"], tol=0.1):
            mismatches += 1
    assert mismatches <= 3, f"{mismatches} RCS mismatches (tolerance 0.1)"

def check_violation_class_match(sub, golden):
    m = merge_on_keys(sub, golden, ["station_id", "month"])
    mismatches = (m["violation_class_g"] != m["violation_class_s"]).sum()
    assert mismatches <= 2, f"{mismatches} violation_class mismatches"

def check_risk_trend_match(sub, golden):
    m = merge_on_keys(sub, golden, ["station_id", "month"])
    mismatches = (m["risk_trend_g"] != m["risk_trend_s"]).sum()
    assert mismatches <= 3, f"{mismatches} risk_trend mismatches"

def check_penalty_match(sub, golden):
    m = merge_on_keys(sub, golden, ["station_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["penalty_amount_g"], row["penalty_amount_s"], tol=1.0):
            mismatches += 1
    assert mismatches <= 3, f"{mismatches} penalty_amount mismatches (tolerance 1.0)"

def check_maint_credit_match(sub, golden):
    m = merge_on_keys(sub, golden, ["station_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["maintenance_credit_g"], row["maintenance_credit_s"], tol=0.5):
            mismatches += 1
    assert mismatches <= 5, f"{mismatches} maintenance_credit mismatches"

def check_net_penalty_match(sub, golden):
    m = merge_on_keys(sub, golden, ["station_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["net_penalty_g"], row["net_penalty_s"], tol=1.0):
            mismatches += 1
    assert mismatches <= 3, f"{mismatches} net_penalty mismatches (tolerance 1.0)"


# ============================================================
#  regional_summary.csv checks (23-30)
# ============================================================

def check_regional_exists(sub, golden):
    assert sub is not None, "regional_summary.csv not found"

def check_regional_row_count(sub, golden):
    assert len(sub) == len(golden), f"Expected {len(golden)} rows, got {len(sub)}"

def check_regional_columns(sub, golden):
    missing = set(golden.columns) - set(sub.columns)
    assert not missing, f"Missing columns: {missing}"

def check_regional_pairs(sub, golden):
    expected = set(zip(golden["region"], golden["quarter"]))
    actual = set(zip(sub["region"], sub["quarter"]))
    assert expected == actual, f"Region-quarter pairs mismatch"

def check_compliance_rate_range(sub, golden):
    cr = sub["compliance_rate"].dropna()
    bad = cr[(cr < 0) | (cr > 1)]
    assert len(bad) == 0, f"{len(bad)} compliance_rate values out of [0,1]"

def check_regional_avg_cqi_match(sub, golden):
    m = merge_on_keys(sub, golden, ["region", "quarter"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["avg_cqi_g"], row["avg_cqi_s"], tol=0.5):
            mismatches += 1
    assert mismatches == 0, f"{mismatches} regional avg_cqi mismatches"

def check_regional_penalties_match(sub, golden):
    m = merge_on_keys(sub, golden, ["region", "quarter"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["total_net_penalties_g"], row["total_net_penalties_s"], tol=5.0):
            mismatches += 1
    assert mismatches <= 1, f"{mismatches} regional total_net_penalties mismatches"

def check_regional_compliance_match(sub, golden):
    m = merge_on_keys(sub, golden, ["region", "quarter"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["compliance_rate_g"], row["compliance_rate_s"], tol=0.01):
            mismatches += 1
    assert mismatches == 0, f"{mismatches} compliance_rate mismatches"


# ============================================================
#  anomaly_report.csv checks (31-37)
# ============================================================

def check_anomaly_exists(sub, golden):
    assert sub is not None, "anomaly_report.csv not found"

def check_anomaly_row_count(sub, golden):
    diff = abs(len(sub) - len(golden))
    assert diff <= 20, f"Row count diff: expected ~{len(golden)}, got {len(sub)} (diff {diff})"

def check_anomaly_columns(sub, golden):
    missing = set(golden.columns) - set(sub.columns)
    assert not missing, f"Missing columns: {missing}"

def check_anomaly_pdi_above_05(sub, golden):
    bad = sub[sub["pdi_value"] <= 0.5]
    assert len(bad) == 0, f"{len(bad)} anomaly rows with PDI <= 0.5"

def check_anomaly_threshold_types(sub, golden):
    invalid = set(sub["threshold_type"].unique()) - VALID_THRESHOLD_TYPES
    assert not invalid, f"Invalid threshold types: {invalid}"

def check_anomaly_parameters(sub, golden):
    invalid = set(sub["parameter"].unique()) - VALID_PARAMETERS
    assert not invalid, f"Invalid parameters: {invalid}"

def check_anomaly_critical_match(sub, golden):
    g_critical = golden[golden["threshold_type"] == "CRITICAL"]
    s_critical = sub[sub["threshold_type"] == "CRITICAL"]
    diff = abs(len(g_critical) - len(s_critical))
    assert diff <= 10, f"CRITICAL anomaly count: expected ~{len(g_critical)}, got {len(s_critical)}"


# ============================================================
#  penalty_ledger.csv checks (38-44)
# ============================================================

def check_ledger_exists(sub, golden):
    assert sub is not None, "penalty_ledger.csv not found"

def check_ledger_row_count(sub, golden):
    diff = abs(len(sub) - len(golden))
    assert diff <= 5, f"Row count: expected ~{len(golden)}, got {len(sub)} (diff {diff})"

def check_ledger_columns(sub, golden):
    missing = set(golden.columns) - set(sub.columns)
    assert not missing, f"Missing columns: {missing}"

def check_ledger_tier_positive(sub, golden):
    bad = sub[sub["violation_tier"] <= 0]
    assert len(bad) == 0, f"{len(bad)} ledger rows with tier <= 0"

def check_ledger_workday_fraction(sub, golden):
    m = merge_on_keys(sub, golden, ["station_id", "month"])
    if len(m) == 0:
        assert False, "No matching station-month rows in penalty_ledger"
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["workday_fraction_g"], row["workday_fraction_s"], tol=0.001):
            mismatches += 1
    assert mismatches <= 3, f"{mismatches} workday_fraction mismatches"

def check_ledger_maint_cost(sub, golden):
    m = merge_on_keys(sub, golden, ["station_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["maintenance_cost_g"], row["maintenance_cost_s"], tol=1.0):
            mismatches += 1
    assert mismatches <= 5, f"{mismatches} maintenance_cost mismatches"

def check_ledger_net_penalty(sub, golden):
    m = merge_on_keys(sub, golden, ["station_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["net_penalty_g"], row["net_penalty_s"], tol=1.0):
            mismatches += 1
    assert mismatches <= 5, f"{mismatches} ledger net_penalty mismatches"


# ============================================================
#  station_risk_profile.csv checks (45-52)
# ============================================================

def check_profile_exists(sub, golden):
    assert sub is not None, "station_risk_profile.csv not found"

def check_profile_row_count(sub, golden):
    assert len(sub) == 30, f"Expected 30 rows, got {len(sub)}"

def check_profile_columns(sub, golden):
    missing = set(golden.columns) - set(sub.columns)
    assert not missing, f"Missing columns: {missing}"

def check_profile_risk_categories(sub, golden):
    invalid = set(sub["risk_category"].unique()) - VALID_RISK_CATEGORIES
    assert not invalid, f"Invalid risk categories: {invalid}"

def check_profile_category_match(sub, golden):
    m = merge_on_keys(sub, golden, ["station_id"])
    mismatches = (m["risk_category_g"] != m["risk_category_s"]).sum()
    assert mismatches <= 2, f"{mismatches} risk_category mismatches"

def check_profile_violations_match(sub, golden):
    m = merge_on_keys(sub, golden, ["station_id"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["total_violations_g"], row["total_violations_s"], tol=0):
            mismatches += 1
    assert mismatches <= 2, f"{mismatches} total_violations mismatches"

def check_profile_worst_violation(sub, golden):
    m = merge_on_keys(sub, golden, ["station_id"])
    mismatches = (m["worst_violation_g"] != m["worst_violation_s"]).sum()
    assert mismatches <= 2, f"{mismatches} worst_violation mismatches"

def check_profile_avg_cqi(sub, golden):
    m = merge_on_keys(sub, golden, ["station_id"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["avg_cqi_g"], row["avg_cqi_s"], tol=0.5):
            mismatches += 1
    assert mismatches <= 2, f"{mismatches} profile avg_cqi mismatches"


# ============================================================
#  Cross-file consistency checks (53-55)
# ============================================================

def check_cross_penalty_consistency(sub_compliance, sub_ledger):
    if sub_compliance is None or sub_ledger is None:
        assert False, "Missing files for cross-check"
    comp_penalized = sub_compliance[sub_compliance["penalty_amount"] > 0.001]
    diff = abs(len(comp_penalized) - len(sub_ledger))
    assert diff <= 5, f"Penalized rows in compliance ({len(comp_penalized)}) vs ledger rows ({len(sub_ledger)}) differ by {diff}"

def check_cross_penalty_sums(sub_compliance, sub_ledger):
    if sub_compliance is None or sub_ledger is None:
        assert False, "Missing files for cross-check"
    comp_sum = sub_compliance["net_penalty"].sum()
    ledger_sum = sub_ledger["net_penalty"].sum()
    assert abs(comp_sum - ledger_sum) < 10.0, f"net_penalty sums differ: compliance={comp_sum:.2f}, ledger={ledger_sum:.2f}"

def check_cross_profile_total_penalties(sub_compliance, sub_profile):
    if sub_compliance is None or sub_profile is None:
        assert False, "Missing files for cross-check"
    for _, prow in sub_profile.iterrows():
        sid = prow["station_id"]
        comp_sum = sub_compliance[sub_compliance["station_id"] == sid]["net_penalty"].sum()
        if not check_val(comp_sum, prow["total_net_penalties"], tol=1.0):
            assert False, f"Station {sid}: compliance sum={comp_sum:.2f} vs profile total={prow['total_net_penalties']:.2f}"


# ============================================================
#  Main verification runner
# ============================================================

def run_verification():
    golden_compliance = load_golden("station_compliance")
    golden_regional = load_golden("regional_summary")
    golden_anomalies = load_golden("anomaly_report")
    golden_ledger = load_golden("penalty_ledger")
    golden_profile = load_golden("station_risk_profile")

    sub_compliance = load_sub("station_compliance")
    sub_regional = load_sub("regional_summary")
    sub_anomalies = load_sub("anomaly_report")
    sub_ledger = load_sub("penalty_ledger")
    sub_profile = load_sub("station_risk_profile")

    checks = []

    # station_compliance checks (1-22)
    checks.append(("check_compliance_exists", check_compliance_exists, [sub_compliance, golden_compliance]))
    if sub_compliance is not None:
        checks.extend([
            ("check_compliance_row_count", check_compliance_row_count, [sub_compliance, golden_compliance]),
            ("check_compliance_columns", check_compliance_columns, [sub_compliance, golden_compliance]),
            ("check_compliance_station_ids", check_compliance_station_ids, [sub_compliance, golden_compliance]),
            ("check_compliance_months", check_compliance_months, [sub_compliance, golden_compliance]),
            ("check_readings_ge_valid", check_readings_ge_valid, [sub_compliance, golden_compliance]),
            ("check_cqi_range", check_cqi_range, [sub_compliance, golden_compliance]),
            ("check_cqi_rounding", check_cqi_rounding, [sub_compliance, golden_compliance]),
            ("check_rcs_rounding", check_rcs_rounding, [sub_compliance, golden_compliance]),
            ("check_violation_class_values", check_violation_class_values, [sub_compliance, golden_compliance]),
            ("check_risk_trend_values", check_risk_trend_values, [sub_compliance, golden_compliance]),
            ("check_consec_nonneg", check_consec_nonneg, [sub_compliance, golden_compliance]),
            ("check_penalty_nonneg", check_penalty_nonneg, [sub_compliance, golden_compliance]),
            ("check_net_le_penalty", check_net_le_penalty, [sub_compliance, golden_compliance]),
            ("check_penalty_zero_compliant", check_penalty_zero_compliant, [sub_compliance, golden_compliance]),
            ("check_cqi_match", check_cqi_match, [sub_compliance, golden_compliance]),
            ("check_rcs_match", check_rcs_match, [sub_compliance, golden_compliance]),
            ("check_violation_class_match", check_violation_class_match, [sub_compliance, golden_compliance]),
            ("check_risk_trend_match", check_risk_trend_match, [sub_compliance, golden_compliance]),
            ("check_penalty_match", check_penalty_match, [sub_compliance, golden_compliance]),
            ("check_maint_credit_match", check_maint_credit_match, [sub_compliance, golden_compliance]),
            ("check_net_penalty_match", check_net_penalty_match, [sub_compliance, golden_compliance]),
        ])

    # regional_summary checks (23-30)
    checks.append(("check_regional_exists", check_regional_exists, [sub_regional, golden_regional]))
    if sub_regional is not None:
        checks.extend([
            ("check_regional_row_count", check_regional_row_count, [sub_regional, golden_regional]),
            ("check_regional_columns", check_regional_columns, [sub_regional, golden_regional]),
            ("check_regional_pairs", check_regional_pairs, [sub_regional, golden_regional]),
            ("check_compliance_rate_range", check_compliance_rate_range, [sub_regional, golden_regional]),
            ("check_regional_avg_cqi_match", check_regional_avg_cqi_match, [sub_regional, golden_regional]),
            ("check_regional_penalties_match", check_regional_penalties_match, [sub_regional, golden_regional]),
            ("check_regional_compliance_match", check_regional_compliance_match, [sub_regional, golden_regional]),
        ])

    # anomaly_report checks (31-37)
    checks.append(("check_anomaly_exists", check_anomaly_exists, [sub_anomalies, golden_anomalies]))
    if sub_anomalies is not None:
        checks.extend([
            ("check_anomaly_row_count", check_anomaly_row_count, [sub_anomalies, golden_anomalies]),
            ("check_anomaly_columns", check_anomaly_columns, [sub_anomalies, golden_anomalies]),
            ("check_anomaly_pdi_above_05", check_anomaly_pdi_above_05, [sub_anomalies, golden_anomalies]),
            ("check_anomaly_threshold_types", check_anomaly_threshold_types, [sub_anomalies, golden_anomalies]),
            ("check_anomaly_parameters", check_anomaly_parameters, [sub_anomalies, golden_anomalies]),
            ("check_anomaly_critical_match", check_anomaly_critical_match, [sub_anomalies, golden_anomalies]),
        ])

    # penalty_ledger checks (38-44)
    checks.append(("check_ledger_exists", check_ledger_exists, [sub_ledger, golden_ledger]))
    if sub_ledger is not None:
        checks.extend([
            ("check_ledger_row_count", check_ledger_row_count, [sub_ledger, golden_ledger]),
            ("check_ledger_columns", check_ledger_columns, [sub_ledger, golden_ledger]),
            ("check_ledger_tier_positive", check_ledger_tier_positive, [sub_ledger, golden_ledger]),
            ("check_ledger_workday_fraction", check_ledger_workday_fraction, [sub_ledger, golden_ledger]),
            ("check_ledger_maint_cost", check_ledger_maint_cost, [sub_ledger, golden_ledger]),
            ("check_ledger_net_penalty", check_ledger_net_penalty, [sub_ledger, golden_ledger]),
        ])

    # station_risk_profile checks (45-52)
    checks.append(("check_profile_exists", check_profile_exists, [sub_profile, golden_profile]))
    if sub_profile is not None:
        checks.extend([
            ("check_profile_row_count", check_profile_row_count, [sub_profile, golden_profile]),
            ("check_profile_columns", check_profile_columns, [sub_profile, golden_profile]),
            ("check_profile_risk_categories", check_profile_risk_categories, [sub_profile, golden_profile]),
            ("check_profile_category_match", check_profile_category_match, [sub_profile, golden_profile]),
            ("check_profile_violations_match", check_profile_violations_match, [sub_profile, golden_profile]),
            ("check_profile_worst_violation", check_profile_worst_violation, [sub_profile, golden_profile]),
            ("check_profile_avg_cqi", check_profile_avg_cqi, [sub_profile, golden_profile]),
        ])

    # Cross-file consistency checks (53-55)
    if sub_compliance is not None and sub_ledger is not None:
        checks.extend([
            ("check_cross_penalty_consistency", check_cross_penalty_consistency, [sub_compliance, sub_ledger]),
            ("check_cross_penalty_sums", check_cross_penalty_sums, [sub_compliance, sub_ledger]),
        ])
    if sub_compliance is not None and sub_profile is not None:
        checks.append(
            ("check_cross_profile_total_penalties", check_cross_profile_total_penalties, [sub_compliance, sub_profile])
        )

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

    print(f"\n{passed}/{max_score}")
    sys.exit(0 if passed == max_score else 1)


if __name__ == "__main__":
    run_verification()

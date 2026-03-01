import pandas as pd
import numpy as np
from pathlib import Path
import sys


GOLDEN_DIR = Path(__file__).parent
TASK_DIR = GOLDEN_DIR.parent / "task"

VALID_STATUSES = {"OK", "FAIL", "TIMEOUT", "REJECT"}
VALID_OUTCOME_CLASSES = {"SUCCESS", "DELAYED_SUCCESS", "EXHAUSTED",
                         "INCIDENT_FAILURE", "TIMEOUT_FAILURE", "ERROR_FAILURE"}
VALID_HEALTH_STATUSES = {"CRITICAL", "DEGRADED", "AT_RISK", "STABLE", "HEALTHY"}
VALID_SERVICE_LEVELS = {"PLATINUM", "GOLD", "SILVER", "BRONZE"}
VALID_BACKOFF_TYPES = {"DECORRELATED", "EXPONENTIAL", "LINEAR", "FIXED"}
VALID_REGIONS = {"ALPHA", "BETA", "GAMMA", "DELTA"}
VALID_SEVERITIES = {"S1", "S2", "S3", "S4"}
VALID_PENALTY_TIERS = {"NONE", "GRACE", "STANDARD", "SEVERE", "CRITICAL"}
VALID_TRENDS = {"IMPROVING", "DECLINING", "STABLE"}


def load_golden(name):
    return pd.read_csv(GOLDEN_DIR / f"golden_{name}.csv")


def load_sub(name):
    path = TASK_DIR / f"{name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


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


def check_column_values(merged, col, tol=0.01):
    col_g = f"{col}_g"
    col_s = f"{col}_s"
    if col_g not in merged.columns or col_s not in merged.columns:
        return 0
    total = len(merged)
    matches = sum(check_val(merged.loc[i, col_g], merged.loc[i, col_s], tol)
                  for i in merged.index)
    return matches / total if total > 0 else 0


# ============================================================
#  job_metrics.csv checks (1-12)
# ============================================================

def check_jm_exists(sub, golden):
    assert sub is not None, "job_metrics.csv not found"

def check_jm_row_count(sub, golden):
    assert len(sub) == len(golden), f"Expected {len(golden)} rows, got {len(sub)}"

def check_jm_columns(sub, golden):
    missing = set(golden.columns) - set(sub.columns)
    assert not missing, f"Missing columns: {missing}"

def check_jm_job_ids(sub, golden):
    expected = set(golden["job_id"])
    actual = set(sub["job_id"])
    missing = expected - actual
    assert len(missing) == 0, f"Missing job_ids: {missing}"

def check_jm_final_status(sub, golden):
    merged = merge_on_keys(sub, golden, ["job_id"])
    acc = check_column_values(merged, "final_status")
    assert acc >= 0.95, f"final_status accuracy {acc:.2%} < 95%"

def check_jm_outcome_class(sub, golden):
    merged = merge_on_keys(sub, golden, ["job_id"])
    acc = check_column_values(merged, "outcome_class")
    assert acc >= 0.95, f"outcome_class accuracy {acc:.2%} < 95%"

def check_jm_outcome_values(sub, golden):
    invalid = set(sub["outcome_class"].unique()) - VALID_OUTCOME_CLASSES
    assert not invalid, f"Invalid outcome classes: {invalid}"

def check_jm_sla_breached(sub, golden):
    merged = merge_on_keys(sub, golden, ["job_id"])
    acc = check_column_values(merged, "sla_breached")
    assert acc >= 0.95, f"sla_breached accuracy {acc:.2%} < 95%"

def check_jm_penalty_amount(sub, golden):
    merged = merge_on_keys(sub, golden, ["job_id"])
    acc = check_column_values(merged, "penalty_amount", tol=0.02)
    assert acc >= 0.95, f"penalty_amount accuracy {acc:.2%} < 95%"

def check_jm_expected_delay(sub, golden):
    merged = merge_on_keys(sub, golden, ["job_id"])
    acc = check_column_values(merged, "expected_delay_sec", tol=0.1)
    assert acc >= 0.95, f"expected_delay_sec accuracy {acc:.2%} < 95%"

def check_jm_retry_count(sub, golden):
    merged = merge_on_keys(sub, golden, ["job_id"])
    acc = check_column_values(merged, "retry_count")
    assert acc >= 0.95, f"retry_count accuracy {acc:.2%} < 95%"

def check_jm_incident_affected(sub, golden):
    merged = merge_on_keys(sub, golden, ["job_id"])
    acc = check_column_values(merged, "incident_affected")
    assert acc >= 0.95, f"incident_affected accuracy {acc:.2%} < 95%"


def check_jm_penalty_tier(sub, golden):
    merged = merge_on_keys(sub, golden, ["job_id"])
    acc = check_column_values(merged, "penalty_tier")
    assert acc >= 0.95, f"penalty_tier accuracy {acc:.2%} < 95%"

def check_jm_penalty_tier_values(sub, golden):
    invalid = set(sub["penalty_tier"].unique()) - VALID_PENALTY_TIERS
    assert not invalid, f"Invalid penalty tiers: {invalid}"

def check_jm_breach_seconds(sub, golden):
    merged = merge_on_keys(sub, golden, ["job_id"])
    acc = check_column_values(merged, "breach_seconds", tol=0.05)
    assert acc >= 0.95, f"breach_seconds accuracy {acc:.2%} < 95%"

def check_jm_actual_wait(sub, golden):
    merged = merge_on_keys(sub, golden, ["job_id"])
    acc = check_column_values(merged, "actual_wait_sec", tol=0.5)
    assert acc >= 0.90, f"actual_wait_sec accuracy {acc:.2%} < 90%"

def check_jm_total_duration(sub, golden):
    merged = merge_on_keys(sub, golden, ["job_id"])
    acc = check_column_values(merged, "total_duration_sec", tol=0.5)
    assert acc >= 0.95, f"total_duration_sec accuracy {acc:.2%} < 95%"

def check_jm_penalty_weight(sub, golden):
    merged = merge_on_keys(sub, golden, ["job_id"])
    acc = check_column_values(merged, "penalty_weight", tol=0.01)
    assert acc >= 0.95, f"penalty_weight accuracy {acc:.2%} < 95%"

def check_jm_penalty_weight_zeroes(sub, golden):
    non_breached = sub[sub["sla_breached"] == "NO"]
    bad = non_breached[non_breached["penalty_weight"] != 0.0]
    assert len(bad) == 0, f"{len(bad)} non-breached jobs have non-zero penalty_weight"


# ============================================================
#  tenant_monthly.csv checks (13-24)
# ============================================================

def check_tm_exists(sub, golden):
    assert sub is not None, "tenant_monthly.csv not found"

def check_tm_row_count(sub, golden):
    assert len(sub) == len(golden), f"Expected {len(golden)} rows, got {len(sub)}"

def check_tm_columns(sub, golden):
    missing = set(golden.columns) - set(sub.columns)
    assert not missing, f"Missing columns: {missing}"

def check_tm_keys(sub, golden):
    expected = set(zip(golden["tenant_id"], golden["month"]))
    actual = set(zip(sub["tenant_id"], sub["month"]))
    missing = expected - actual
    assert len(missing) == 0, f"Missing tenant-month keys: {missing}"

def check_tm_success_rate(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id", "month"])
    acc = check_column_values(merged, "success_rate", tol=0.005)
    assert acc >= 0.95, f"success_rate accuracy {acc:.2%} < 95%"

def check_tm_total_cost(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id", "month"])
    acc = check_column_values(merged, "total_cost", tol=0.05)
    assert acc >= 0.90, f"total_cost accuracy {acc:.2%} < 90%"

def check_tm_reliability_score(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id", "month"])
    acc = check_column_values(merged, "reliability_score", tol=0.5)
    assert acc >= 0.90, f"reliability_score accuracy {acc:.2%} < 90%"

def check_tm_health_status(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id", "month"])
    acc = check_column_values(merged, "health_status")
    assert acc >= 0.90, f"health_status accuracy {acc:.2%} < 90%"

def check_tm_health_values(sub, golden):
    invalid = set(sub["health_status"].unique()) - VALID_HEALTH_STATUSES
    assert not invalid, f"Invalid health statuses: {invalid}"

def check_tm_sla_breaches(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id", "month"])
    acc = check_column_values(merged, "sla_breaches")
    assert acc >= 0.90, f"sla_breaches accuracy {acc:.2%} < 90%"

def check_tm_retry_rate(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id", "month"])
    acc = check_column_values(merged, "retry_rate", tol=0.005)
    assert acc >= 0.90, f"retry_rate accuracy {acc:.2%} < 90%"

def check_tm_penalty_cost(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id", "month"])
    acc = check_column_values(merged, "penalty_cost", tol=0.05)
    assert acc >= 0.90, f"penalty_cost accuracy {acc:.2%} < 90%"


def check_tm_trend(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id", "month"])
    acc = check_column_values(merged, "trend")
    assert acc >= 0.90, f"trend accuracy {acc:.2%} < 90%"

def check_tm_trend_values(sub, golden):
    invalid = set(sub["trend"].unique()) - VALID_TRENDS
    assert not invalid, f"Invalid trend values: {invalid}"

def check_tm_base_cost(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id", "month"])
    acc = check_column_values(merged, "base_cost", tol=0.05)
    assert acc >= 0.85, f"base_cost accuracy {acc:.2%} < 85%"

def check_tm_cost_efficiency_ratio(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id", "month"])
    acc = check_column_values(merged, "cost_efficiency_ratio", tol=0.02)
    assert acc >= 0.90, f"cost_efficiency_ratio accuracy {acc:.2%} < 90%"

def check_tm_risk_score(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id", "month"])
    acc = check_column_values(merged, "risk_score", tol=0.01)
    assert acc >= 0.90, f"risk_score accuracy {acc:.2%} < 90%"

def check_tm_risk_score_range(sub, golden):
    bad = sub[(sub["risk_score"] < 0) | (sub["risk_score"] > 100)]
    assert len(bad) == 0, f"{len(bad)} rows have risk_score outside [0,100]"


# ============================================================
#  pool_monthly.csv checks (25-34)
# ============================================================

def check_pm_exists(sub, golden):
    assert sub is not None, "pool_monthly.csv not found"

def check_pm_row_count(sub, golden):
    assert len(sub) == len(golden), f"Expected {len(golden)} rows, got {len(sub)}"

def check_pm_columns(sub, golden):
    missing = set(golden.columns) - set(sub.columns)
    assert not missing, f"Missing columns: {missing}"

def check_pm_keys(sub, golden):
    expected = set(zip(golden["pool_id"], golden["month"]))
    actual = set(zip(sub["pool_id"], sub["month"]))
    missing = expected - actual
    assert len(missing) == 0, f"Missing pool-month keys: {missing}"

def check_pm_ok_count(sub, golden):
    merged = merge_on_keys(sub, golden, ["pool_id", "month"])
    acc = check_column_values(merged, "ok_count")
    assert acc >= 0.90, f"ok_count accuracy {acc:.2%} < 90%"

def check_pm_fail_count(sub, golden):
    merged = merge_on_keys(sub, golden, ["pool_id", "month"])
    acc = check_column_values(merged, "fail_count")
    assert acc >= 0.90, f"fail_count accuracy {acc:.2%} < 90%"

def check_pm_timeout_count(sub, golden):
    merged = merge_on_keys(sub, golden, ["pool_id", "month"])
    acc = check_column_values(merged, "timeout_count")
    assert acc >= 0.90, f"timeout_count accuracy {acc:.2%} < 90%"

def check_pm_total_cpu(sub, golden):
    merged = merge_on_keys(sub, golden, ["pool_id", "month"])
    acc = check_column_values(merged, "total_cpu_ms", tol=10)
    assert acc >= 0.90, f"total_cpu_ms accuracy {acc:.2%} < 90%"

def check_pm_top_error(sub, golden):
    merged = merge_on_keys(sub, golden, ["pool_id", "month"])
    acc = check_column_values(merged, "top_error")
    assert acc >= 0.90, f"top_error accuracy {acc:.2%} < 90%"

def check_pm_incident_minutes(sub, golden):
    merged = merge_on_keys(sub, golden, ["pool_id", "month"])
    acc = check_column_values(merged, "incident_minutes", tol=1.0)
    assert acc >= 0.90, f"incident_minutes accuracy {acc:.2%} < 90%"


def check_pm_cpu_utilization(sub, golden):
    merged = merge_on_keys(sub, golden, ["pool_id", "month"])
    acc = check_column_values(merged, "cpu_utilization_pct", tol=0.001)
    assert acc >= 0.85, f"cpu_utilization_pct accuracy {acc:.2%} < 85%"

def check_pm_reject_count(sub, golden):
    merged = merge_on_keys(sub, golden, ["pool_id", "month"])
    acc = check_column_values(merged, "reject_count")
    assert acc >= 0.90, f"reject_count accuracy {acc:.2%} < 90%"


# ============================================================
#  retry_effectiveness.csv checks (35-43)
# ============================================================

def check_re_exists(sub, golden):
    assert sub is not None, "retry_effectiveness.csv not found"

def check_re_row_count(sub, golden):
    assert len(sub) == len(golden), f"Expected {len(golden)} rows, got {len(sub)}"

def check_re_columns(sub, golden):
    missing = set(golden.columns) - set(sub.columns)
    assert not missing, f"Missing columns: {missing}"

def check_re_keys(sub, golden):
    expected = set(zip(golden["service_level"], golden["priority"]))
    actual = set(zip(sub["service_level"], sub["priority"]))
    missing = expected - actual
    assert len(missing) == 0, f"Missing SL-priority keys: {missing}"

def check_re_backoff_type(sub, golden):
    merged = merge_on_keys(sub, golden, ["service_level", "priority"])
    acc = check_column_values(merged, "backoff_type")
    assert acc >= 0.90, f"backoff_type accuracy {acc:.2%} < 90%"

def check_re_recovery_rate(sub, golden):
    merged = merge_on_keys(sub, golden, ["service_level", "priority"])
    acc = check_column_values(merged, "recovery_rate", tol=0.01)
    assert acc >= 0.90, f"recovery_rate accuracy {acc:.2%} < 90%"

def check_re_total_penalties(sub, golden):
    merged = merge_on_keys(sub, golden, ["service_level", "priority"])
    acc = check_column_values(merged, "total_penalties", tol=0.05)
    assert acc >= 0.90, f"total_penalties accuracy {acc:.2%} < 90%"

def check_re_exhaustion_rate(sub, golden):
    merged = merge_on_keys(sub, golden, ["service_level", "priority"])
    acc = check_column_values(merged, "exhaustion_rate", tol=0.005)
    assert acc >= 0.90, f"exhaustion_rate accuracy {acc:.2%} < 90%"

def check_re_delay_deviation(sub, golden):
    merged = merge_on_keys(sub, golden, ["service_level", "priority"])
    acc = check_column_values(merged, "avg_delay_deviation", tol=0.2)
    assert acc >= 0.85, f"avg_delay_deviation accuracy {acc:.2%} < 85%"


def check_re_first_attempt_success(sub, golden):
    merged = merge_on_keys(sub, golden, ["service_level", "priority"])
    acc = check_column_values(merged, "first_attempt_success_rate", tol=0.01)
    assert acc >= 0.90, f"first_attempt_success_rate accuracy {acc:.2%} < 90%"

def check_re_avg_expected_delay(sub, golden):
    merged = merge_on_keys(sub, golden, ["service_level", "priority"])
    acc = check_column_values(merged, "avg_expected_delay", tol=0.5)
    assert acc >= 0.85, f"avg_expected_delay accuracy {acc:.2%} < 85%"


# ============================================================
#  incident_impact.csv checks (44-51)
# ============================================================

def check_ii_exists(sub, golden):
    assert sub is not None, "incident_impact.csv not found"

def check_ii_row_count(sub, golden):
    assert len(sub) == len(golden), f"Expected {len(golden)} rows, got {len(sub)}"

def check_ii_columns(sub, golden):
    missing = set(golden.columns) - set(sub.columns)
    assert not missing, f"Missing columns: {missing}"

def check_ii_incident_ids(sub, golden):
    expected = set(golden["incident_id"])
    actual = set(sub["incident_id"])
    missing = expected - actual
    assert len(missing) == 0, f"Missing incident_ids: {missing}"

def check_ii_affected_jobs(sub, golden):
    merged = merge_on_keys(sub, golden, ["incident_id"])
    acc = check_column_values(merged, "affected_jobs")
    assert acc >= 0.85, f"affected_jobs accuracy {acc:.2%} < 85%"

def check_ii_failed_attempts(sub, golden):
    merged = merge_on_keys(sub, golden, ["incident_id"])
    acc = check_column_values(merged, "failed_attempts")
    assert acc >= 0.85, f"failed_attempts accuracy {acc:.2%} < 85%"

def check_ii_penalties_incurred(sub, golden):
    merged = merge_on_keys(sub, golden, ["incident_id"])
    acc = check_column_values(merged, "penalties_incurred", tol=0.05)
    assert acc >= 0.90, f"penalties_incurred accuracy {acc:.2%} < 90%"

def check_ii_duration_minutes(sub, golden):
    merged = merge_on_keys(sub, golden, ["incident_id"])
    acc = check_column_values(merged, "duration_minutes", tol=0.5)
    assert acc >= 0.95, f"duration_minutes accuracy {acc:.2%} < 95%"


def check_ii_affected_attempts(sub, golden):
    merged = merge_on_keys(sub, golden, ["incident_id"])
    acc = check_column_values(merged, "affected_attempts")
    assert acc >= 0.85, f"affected_attempts accuracy {acc:.2%} < 85%"

def check_ii_sla_breaches(sub, golden):
    merged = merge_on_keys(sub, golden, ["incident_id"])
    acc = check_column_values(merged, "sla_breaches")
    assert acc >= 0.85, f"sla_breaches accuracy {acc:.2%} < 85%"

def check_ii_affected_tenants(sub, golden):
    merged = merge_on_keys(sub, golden, ["incident_id"])
    acc = check_column_values(merged, "affected_tenants")
    assert acc >= 0.85, f"affected_tenants accuracy {acc:.2%} < 85%"


# ============================================================
#  tenant_profile.csv checks (52-60)
# ============================================================

def check_tp_exists(sub, golden):
    assert sub is not None, "tenant_profile.csv not found"

def check_tp_row_count(sub, golden):
    assert len(sub) == len(golden), f"Expected {len(golden)} rows, got {len(sub)}"

def check_tp_columns(sub, golden):
    missing = set(golden.columns) - set(sub.columns)
    assert not missing, f"Missing columns: {missing}"

def check_tp_tenant_ids(sub, golden):
    expected = set(golden["tenant_id"])
    actual = set(sub["tenant_id"])
    missing = expected - actual
    assert len(missing) == 0, f"Missing tenant_ids: {missing}"

def check_tp_total_cost(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id"])
    acc = check_column_values(merged, "total_cost", tol=0.10)
    assert acc >= 0.90, f"total_cost accuracy {acc:.2%} < 90%"

def check_tp_final_health(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id"])
    acc = check_column_values(merged, "final_health_status")
    assert acc >= 0.85, f"final_health_status accuracy {acc:.2%} < 85%"

def check_tp_final_health_values(sub, golden):
    invalid = set(sub["final_health_status"].unique()) - VALID_HEALTH_STATUSES
    assert not invalid, f"Invalid health statuses: {invalid}"

def check_tp_reliability_score(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id"])
    acc = check_column_values(merged, "final_reliability_score", tol=1.0)
    assert acc >= 0.85, f"final_reliability_score accuracy {acc:.2%} < 85%"

def check_tp_total_penalties(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id"])
    acc = check_column_values(merged, "total_penalties", tol=0.10)
    assert acc >= 0.90, f"total_penalties accuracy {acc:.2%} < 90%"


def check_tp_worst_health(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id"])
    acc = check_column_values(merged, "worst_health_status")
    assert acc >= 0.85, f"worst_health_status accuracy {acc:.2%} < 85%"

def check_tp_worst_health_values(sub, golden):
    invalid = set(sub["worst_health_status"].unique()) - VALID_HEALTH_STATUSES
    assert not invalid, f"Invalid worst_health_status values: {invalid}"

def check_tp_consecutive_healthy(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id"])
    acc = check_column_values(merged, "consecutive_healthy_months")
    assert acc >= 0.85, f"consecutive_healthy_months accuracy {acc:.2%} < 85%"

def check_tp_longest_healthy_streak(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id"])
    acc = check_column_values(merged, "longest_healthy_streak")
    assert acc >= 0.85, f"longest_healthy_streak accuracy {acc:.2%} < 85%"

def check_tp_overall_success_rate(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id"])
    acc = check_column_values(merged, "overall_success_rate", tol=0.005)
    assert acc >= 0.90, f"overall_success_rate accuracy {acc:.2%} < 90%"

def check_tp_avg_retry_rate(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id"])
    acc = check_column_values(merged, "avg_retry_rate", tol=0.005)
    assert acc >= 0.85, f"avg_retry_rate accuracy {acc:.2%} < 85%"

def check_tp_consecutive_breach_months(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id"])
    acc = check_column_values(merged, "consecutive_breach_months")
    assert acc >= 0.85, f"consecutive_breach_months accuracy {acc:.2%} < 85%"

def check_tp_avg_penalty_per_breach(sub, golden):
    merged = merge_on_keys(sub, golden, ["tenant_id"])
    acc = check_column_values(merged, "avg_penalty_per_breach", tol=0.05)
    assert acc >= 0.85, f"avg_penalty_per_breach accuracy {acc:.2%} < 85%"


# ============================================================
#  Cross-file consistency checks (61-65)
# ============================================================

def check_cross_penalty_sum(sub_jm, sub_tp, golden_jm, golden_tp):
    if sub_jm is None or sub_tp is None:
        return
    for tid in sub_tp["tenant_id"].unique():
        jm_penalty = sub_jm[sub_jm["tenant_id"] == tid]["penalty_amount"].sum()
        tp_penalty = sub_tp[sub_tp["tenant_id"] == tid]["total_penalties"].values[0]
        assert abs(jm_penalty - tp_penalty) < 1.0, \
            f"Penalty mismatch for {tid}: jm={jm_penalty:.2f} vs tp={tp_penalty:.2f}"

def check_cross_job_count(sub_jm, sub_tp, golden_jm, golden_tp):
    if sub_jm is None or sub_tp is None:
        return
    for tid in sub_tp["tenant_id"].unique():
        jm_count = len(sub_jm[sub_jm["tenant_id"] == tid])
        tp_count = sub_tp[sub_tp["tenant_id"] == tid]["total_jobs"].values[0]
        assert jm_count == tp_count, \
            f"Job count mismatch for {tid}: jm={jm_count} vs tp={tp_count}"

def check_cross_tm_sla_breaches(sub_jm, sub_tm, golden_jm, golden_tm):
    if sub_jm is None or sub_tm is None:
        return
    total_jm_breaches = len(sub_jm[sub_jm["sla_breached"] == "YES"])
    total_tm_breaches = sub_tm["sla_breaches"].sum()
    assert abs(total_jm_breaches - total_tm_breaches) <= 2, \
        f"SLA breach total mismatch: jm={total_jm_breaches} vs tm={total_tm_breaches}"

def check_cross_pool_attempts(sub_pm, sub_jm, golden_pm, golden_jm):
    if sub_pm is None:
        return
    for _, row in sub_pm.iterrows():
        total = row["ok_count"] + row["fail_count"] + row["timeout_count"] + row["reject_count"]
        assert total == row["total_attempts"], \
            f"Attempt count mismatch for {row['pool_id']} {row['month']}: " \
            f"sum={total} vs total={row['total_attempts']}"

def check_cross_tm_cost_components(sub_tm, golden_tm):
    if sub_tm is None:
        return
    for idx in sub_tm.index:
        base = sub_tm.loc[idx, "base_cost"]
        retry = sub_tm.loc[idx, "retry_cost"]
        penalty = sub_tm.loc[idx, "penalty_cost"]
        total = sub_tm.loc[idx, "total_cost"]
        expected = round(base + retry + penalty, 2)
        assert abs(expected - total) < 0.05, \
            f"Cost component mismatch row {idx}: {base}+{retry}+{penalty}={expected} vs {total}"


# ============================================================
#  Test runner
# ============================================================

CHECKS = [
    ("01_jm_exists", "job_metrics", check_jm_exists),
    ("02_jm_row_count", "job_metrics", check_jm_row_count),
    ("03_jm_columns", "job_metrics", check_jm_columns),
    ("04_jm_job_ids", "job_metrics", check_jm_job_ids),
    ("05_jm_final_status", "job_metrics", check_jm_final_status),
    ("06_jm_outcome_class", "job_metrics", check_jm_outcome_class),
    ("07_jm_outcome_values", "job_metrics", check_jm_outcome_values),
    ("08_jm_sla_breached", "job_metrics", check_jm_sla_breached),
    ("09_jm_penalty_amount", "job_metrics", check_jm_penalty_amount),
    ("10_jm_expected_delay", "job_metrics", check_jm_expected_delay),
    ("11_jm_retry_count", "job_metrics", check_jm_retry_count),
    ("12_jm_incident_affected", "job_metrics", check_jm_incident_affected),
    ("13_jm_penalty_tier", "job_metrics", check_jm_penalty_tier),
    ("14_jm_penalty_tier_values", "job_metrics", check_jm_penalty_tier_values),
    ("15_jm_breach_seconds", "job_metrics", check_jm_breach_seconds),
    ("16_jm_actual_wait", "job_metrics", check_jm_actual_wait),
    ("17_jm_total_duration", "job_metrics", check_jm_total_duration),
    ("18_tm_exists", "tenant_monthly", check_tm_exists),
    ("19_tm_row_count", "tenant_monthly", check_tm_row_count),
    ("20_tm_columns", "tenant_monthly", check_tm_columns),
    ("21_tm_keys", "tenant_monthly", check_tm_keys),
    ("22_tm_success_rate", "tenant_monthly", check_tm_success_rate),
    ("23_tm_total_cost", "tenant_monthly", check_tm_total_cost),
    ("24_tm_reliability_score", "tenant_monthly", check_tm_reliability_score),
    ("25_tm_health_status", "tenant_monthly", check_tm_health_status),
    ("26_tm_health_values", "tenant_monthly", check_tm_health_values),
    ("27_tm_sla_breaches", "tenant_monthly", check_tm_sla_breaches),
    ("28_tm_retry_rate", "tenant_monthly", check_tm_retry_rate),
    ("29_tm_penalty_cost", "tenant_monthly", check_tm_penalty_cost),
    ("30_tm_trend", "tenant_monthly", check_tm_trend),
    ("31_tm_trend_values", "tenant_monthly", check_tm_trend_values),
    ("32_tm_base_cost", "tenant_monthly", check_tm_base_cost),
    ("33_pm_exists", "pool_monthly", check_pm_exists),
    ("34_pm_row_count", "pool_monthly", check_pm_row_count),
    ("35_pm_columns", "pool_monthly", check_pm_columns),
    ("36_pm_keys", "pool_monthly", check_pm_keys),
    ("37_pm_ok_count", "pool_monthly", check_pm_ok_count),
    ("38_pm_fail_count", "pool_monthly", check_pm_fail_count),
    ("39_pm_timeout_count", "pool_monthly", check_pm_timeout_count),
    ("40_pm_total_cpu", "pool_monthly", check_pm_total_cpu),
    ("41_pm_top_error", "pool_monthly", check_pm_top_error),
    ("42_pm_incident_minutes", "pool_monthly", check_pm_incident_minutes),
    ("43_pm_cpu_utilization", "pool_monthly", check_pm_cpu_utilization),
    ("44_pm_reject_count", "pool_monthly", check_pm_reject_count),
    ("45_re_exists", "retry_effectiveness", check_re_exists),
    ("46_re_row_count", "retry_effectiveness", check_re_row_count),
    ("47_re_columns", "retry_effectiveness", check_re_columns),
    ("48_re_keys", "retry_effectiveness", check_re_keys),
    ("49_re_backoff_type", "retry_effectiveness", check_re_backoff_type),
    ("50_re_recovery_rate", "retry_effectiveness", check_re_recovery_rate),
    ("51_re_total_penalties", "retry_effectiveness", check_re_total_penalties),
    ("52_re_exhaustion_rate", "retry_effectiveness", check_re_exhaustion_rate),
    ("53_re_delay_deviation", "retry_effectiveness", check_re_delay_deviation),
    ("54_re_first_attempt_success", "retry_effectiveness", check_re_first_attempt_success),
    ("55_re_avg_expected_delay", "retry_effectiveness", check_re_avg_expected_delay),
    ("56_ii_exists", "incident_impact", check_ii_exists),
    ("57_ii_row_count", "incident_impact", check_ii_row_count),
    ("58_ii_columns", "incident_impact", check_ii_columns),
    ("59_ii_incident_ids", "incident_impact", check_ii_incident_ids),
    ("60_ii_affected_jobs", "incident_impact", check_ii_affected_jobs),
    ("61_ii_failed_attempts", "incident_impact", check_ii_failed_attempts),
    ("62_ii_penalties_incurred", "incident_impact", check_ii_penalties_incurred),
    ("63_ii_duration_minutes", "incident_impact", check_ii_duration_minutes),
    ("64_ii_affected_attempts", "incident_impact", check_ii_affected_attempts),
    ("65_ii_sla_breaches", "incident_impact", check_ii_sla_breaches),
    ("66_ii_affected_tenants", "incident_impact", check_ii_affected_tenants),
    ("67_tp_exists", "tenant_profile", check_tp_exists),
    ("68_tp_row_count", "tenant_profile", check_tp_row_count),
    ("69_tp_columns", "tenant_profile", check_tp_columns),
    ("70_tp_tenant_ids", "tenant_profile", check_tp_tenant_ids),
    ("71_tp_total_cost", "tenant_profile", check_tp_total_cost),
    ("72_tp_final_health", "tenant_profile", check_tp_final_health),
    ("73_tp_final_health_values", "tenant_profile", check_tp_final_health_values),
    ("74_tp_reliability_score", "tenant_profile", check_tp_reliability_score),
    ("75_tp_total_penalties", "tenant_profile", check_tp_total_penalties),
    ("76_tp_worst_health", "tenant_profile", check_tp_worst_health),
    ("77_tp_worst_health_values", "tenant_profile", check_tp_worst_health_values),
    ("78_tp_consecutive_healthy", "tenant_profile", check_tp_consecutive_healthy),
    ("79_tp_longest_healthy_streak", "tenant_profile", check_tp_longest_healthy_streak),
    ("80_tp_overall_success_rate", "tenant_profile", check_tp_overall_success_rate),
    ("81_tp_avg_retry_rate", "tenant_profile", check_tp_avg_retry_rate),
    ("82_jm_penalty_weight", "job_metrics", check_jm_penalty_weight),
    ("83_jm_penalty_weight_zeroes", "job_metrics", check_jm_penalty_weight_zeroes),
    ("84_tm_cost_efficiency_ratio", "tenant_monthly", check_tm_cost_efficiency_ratio),
    ("85_tm_risk_score", "tenant_monthly", check_tm_risk_score),
    ("86_tm_risk_score_range", "tenant_monthly", check_tm_risk_score_range),
    ("87_tp_consecutive_breach_months", "tenant_profile", check_tp_consecutive_breach_months),
    ("88_tp_avg_penalty_per_breach", "tenant_profile", check_tp_avg_penalty_per_breach),
]

CROSS_CHECKS = [
    ("89_cross_penalty_sum", "job_metrics", "tenant_profile", check_cross_penalty_sum),
    ("90_cross_job_count", "job_metrics", "tenant_profile", check_cross_job_count),
    ("91_cross_tm_sla", "job_metrics", "tenant_monthly", check_cross_tm_sla_breaches),
    ("92_cross_pool_attempts", "pool_monthly", "job_metrics", check_cross_pool_attempts),
    ("93_cross_tm_cost", "tenant_monthly", "tenant_monthly", check_cross_tm_cost_components),
]


def main():
    passed = 0
    failed = 0
    errors = []

    golden_cache = {}
    sub_cache = {}

    def get_golden(name):
        if name not in golden_cache:
            golden_cache[name] = load_golden(name)
        return golden_cache[name]

    def get_sub(name):
        if name not in sub_cache:
            sub_cache[name] = load_sub(name)
        return sub_cache[name]

    for check_id, file_name, check_fn in CHECKS:
        try:
            golden = get_golden(file_name)
            sub = get_sub(file_name)
            check_fn(sub, golden)
            print(f"PASS  {check_id}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {check_id}: {e}")
            failed += 1
            errors.append((check_id, str(e)))

    for check_id, file1, file2, check_fn in CROSS_CHECKS:
        try:
            golden1 = get_golden(file1)
            golden2 = get_golden(file2)
            sub1 = get_sub(file1)
            sub2 = get_sub(file2)
            if file1 == file2:
                check_fn(sub1, golden1)
            else:
                check_fn(sub1, sub2, golden1, golden2)
            print(f"PASS  {check_id}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {check_id}: {e}")
            failed += 1
            errors.append((check_id, str(e)))

    total = passed + failed
    print(f"{passed}/{total}")

    if errors:
        for cid, msg in errors:
            print(f"  {cid}: {msg}")

    return passed, total


if __name__ == "__main__":
    p, t = main()
    sys.exit(0 if p == t else 1)

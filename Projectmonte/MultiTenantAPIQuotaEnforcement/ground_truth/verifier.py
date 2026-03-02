import pandas as pd
import numpy as np
from pathlib import Path
import sys

PASS = 0
FAIL = 0


def check(condition, msg):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"[FAIL] {msg}")


def load_csv(path, name):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        check(False, f"{name} could not be loaded: {e}")
        return None


def verify_request_attribution(student_df, golden_df):
    check(student_df is not None, "request_attribution.csv exists and loaded")
    if student_df is None:
        return
    
    required_cols = ["request_id", "tenant_id", "endpoint_id", "timestamp", 
                     "effective_tokens", "cost_multiplier", "effective_cost",
                     "quota_bucket", "hourly_count", "daily_count", "daily_tokens",
                     "within_limit", "throttle_reason", "billable"]
    for col in required_cols:
        check(col in student_df.columns, f"request_attribution has column '{col}'")
    
    check(len(student_df) == len(golden_df), 
          f"request_attribution row count matches ({len(student_df)} vs {len(golden_df)})")
    
    merged = pd.merge(student_df, golden_df, on="request_id", suffixes=("_s", "_g"))
    
    token_match = 0
    for _, row in merged.iterrows():
        if row["effective_tokens_s"] == row["effective_tokens_g"]:
            token_match += 1
    token_pct = (token_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(token_pct >= 95, f"request_attribution effective_tokens >95% match ({token_pct:.1f}%)")
    check(token_pct == 100, f"request_attribution effective_tokens 100% match ({token_pct:.1f}%)")
    
    cost_match = 0
    for _, row in merged.iterrows():
        if abs(row["effective_cost_s"] - row["effective_cost_g"]) < 0.01:
            cost_match += 1
    cost_pct = (cost_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(cost_pct >= 95, f"request_attribution effective_cost >95% match ({cost_pct:.1f}%)")
    
    hourly_match = 0
    for _, row in merged.iterrows():
        if row["hourly_count_s"] == row["hourly_count_g"]:
            hourly_match += 1
    hourly_pct = (hourly_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(hourly_pct >= 90, f"request_attribution hourly_count >90% match ({hourly_pct:.1f}%)")
    check(hourly_pct >= 99, f"request_attribution hourly_count >99% match ({hourly_pct:.1f}%)")
    
    daily_match = 0
    for _, row in merged.iterrows():
        if row["daily_count_s"] == row["daily_count_g"]:
            daily_match += 1
    daily_pct = (daily_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(daily_pct >= 90, f"request_attribution daily_count >90% match ({daily_pct:.1f}%)")
    
    limit_match = 0
    for _, row in merged.iterrows():
        if row["within_limit_s"] == row["within_limit_g"]:
            limit_match += 1
    limit_pct = (limit_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(limit_pct >= 95, f"request_attribution within_limit >95% match ({limit_pct:.1f}%)")
    check(limit_pct == 100, f"request_attribution within_limit 100% match ({limit_pct:.1f}%)")
    
    billable_match = 0
    for _, row in merged.iterrows():
        if row["billable_s"] == row["billable_g"]:
            billable_match += 1
    bill_pct = (billable_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(bill_pct >= 95, f"request_attribution billable >95% match ({bill_pct:.1f}%)")


def verify_tenant_consumption(student_df, golden_df):
    check(student_df is not None, "tenant_consumption.csv exists and loaded")
    if student_df is None:
        return
    
    required_cols = ["tenant_id", "tier_id", "quota_bucket", "request_count",
                     "tokens_consumed", "total_cost", "request_limit", "token_limit",
                     "request_pct", "token_pct", "quota_used_pct", "status",
                     "violation_count", "overage_requests", "overage_tokens"]
    for col in required_cols:
        check(col in student_df.columns, f"tenant_consumption has column '{col}'")
    
    check(len(student_df) == len(golden_df),
          f"tenant_consumption row count matches ({len(student_df)} vs {len(golden_df)})")
    
    merged = pd.merge(student_df, golden_df, on=["tenant_id", "quota_bucket"], suffixes=("_s", "_g"))
    
    req_match = 0
    for _, row in merged.iterrows():
        if row["request_count_s"] == row["request_count_g"]:
            req_match += 1
    req_pct = (req_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(req_pct >= 95, f"tenant_consumption request_count >95% match ({req_pct:.1f}%)")
    check(req_pct == 100, f"tenant_consumption request_count 100% match ({req_pct:.1f}%)")
    
    tok_match = 0
    for _, row in merged.iterrows():
        if abs(row["tokens_consumed_s"] - row["tokens_consumed_g"]) <= 1:
            tok_match += 1
    tok_pct = (tok_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(tok_pct >= 95, f"tenant_consumption tokens_consumed >95% match ({tok_pct:.1f}%)")
    
    status_match = 0
    for _, row in merged.iterrows():
        if row["status_s"] == row["status_g"]:
            status_match += 1
    status_pct = (status_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(status_pct >= 90, f"tenant_consumption status >90% match ({status_pct:.1f}%)")
    check(status_pct >= 95, f"tenant_consumption status >95% match ({status_pct:.1f}%)")
    
    overage_req_match = 0
    for _, row in merged.iterrows():
        if row["overage_requests_s"] == row["overage_requests_g"]:
            overage_req_match += 1
    overage_pct = (overage_req_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(overage_pct >= 90, f"tenant_consumption overage_requests >90% match ({overage_pct:.1f}%)")


def verify_violation_ledger(student_df, golden_df):
    check(student_df is not None, "violation_ledger.csv exists and loaded")
    if student_df is None:
        return
    
    required_cols = ["violation_id", "tenant_id", "request_id", "violation_timestamp",
                     "violation_type", "violation_number", "severity", "flat_penalty",
                     "penalty_amount", "cumulative_penalty", "enforcement_action"]
    for col in required_cols:
        check(col in student_df.columns, f"violation_ledger has column '{col}'")
    
    len_diff = abs(len(student_df) - len(golden_df))
    check(len_diff <= len(golden_df) * 0.05,
          f"violation_ledger row count within 5% ({len(student_df)} vs {len(golden_df)})")
    
    golden_tenant_counts = golden_df.groupby("tenant_id").size()
    student_tenant_counts = student_df.groupby("tenant_id").size()
    
    common_tenants = set(golden_tenant_counts.index) & set(student_tenant_counts.index)
    count_match = 0
    for t in common_tenants:
        if abs(golden_tenant_counts[t] - student_tenant_counts.get(t, 0)) <= 2:
            count_match += 1
    tenant_pct = (count_match / len(common_tenants)) * 100 if len(common_tenants) > 0 else 0
    check(tenant_pct >= 80, f"violation_ledger tenant violation counts >80% match ({tenant_pct:.1f}%)")
    
    merged = pd.merge(student_df, golden_df, on="request_id", suffixes=("_s", "_g"))
    
    if len(merged) > 0:
        severity_match = 0
        for _, row in merged.iterrows():
            if row["severity_s"] == row["severity_g"]:
                severity_match += 1
        sev_pct = (severity_match / len(merged)) * 100
        check(sev_pct >= 85, f"violation_ledger severity >85% match ({sev_pct:.1f}%)")
        
        penalty_match = 0
        for _, row in merged.iterrows():
            if abs(row["penalty_amount_s"] - row["penalty_amount_g"]) < 0.01:
                penalty_match += 1
        pen_pct = (penalty_match / len(merged)) * 100
        check(pen_pct >= 85, f"violation_ledger penalty_amount >85% match ({pen_pct:.1f}%)")


def verify_throttle_decisions(student_df, golden_df):
    check(student_df is not None, "throttle_decisions.csv exists and loaded")
    if student_df is None:
        return
    
    required_cols = ["request_id", "tenant_id", "tier_id", "timestamp",
                     "throttle_reason", "hourly_count", "daily_count",
                     "decision", "decision_reason"]
    for col in required_cols:
        check(col in student_df.columns, f"throttle_decisions has column '{col}'")
    
    len_diff = abs(len(student_df) - len(golden_df))
    check(len_diff <= len(golden_df) * 0.05,
          f"throttle_decisions row count within 5% ({len(student_df)} vs {len(golden_df)})")
    
    merged = pd.merge(student_df, golden_df, on="request_id", suffixes=("_s", "_g"))
    
    if len(merged) > 0:
        decision_match = 0
        for _, row in merged.iterrows():
            if row["decision_s"] == row["decision_g"]:
                decision_match += 1
        dec_pct = (decision_match / len(merged)) * 100
        check(dec_pct >= 80, f"throttle_decisions decision >80% match ({dec_pct:.1f}%)")
        check(dec_pct >= 90, f"throttle_decisions decision >90% match ({dec_pct:.1f}%)")
        
        reason_match = 0
        for _, row in merged.iterrows():
            if row["decision_reason_s"] == row["decision_reason_g"]:
                reason_match += 1
        reason_pct = (reason_match / len(merged)) * 100
        check(reason_pct >= 75, f"throttle_decisions decision_reason >75% match ({reason_pct:.1f}%)")


def verify_billing_summary(student_df, golden_df):
    check(student_df is not None, "billing_summary.csv exists and loaded")
    if student_df is None:
        return
    
    required_cols = ["tenant_id", "tier_id", "billing_period", "total_requests",
                     "total_tokens", "overage_requests", "overage_tokens",
                     "base_cost", "overage_cost", "penalty_cost", "total_due",
                     "violation_count"]
    for col in required_cols:
        check(col in student_df.columns, f"billing_summary has column '{col}'")
    
    check(len(student_df) == len(golden_df),
          f"billing_summary row count matches ({len(student_df)} vs {len(golden_df)})")
    
    merged = pd.merge(student_df, golden_df, on="tenant_id", suffixes=("_s", "_g"))
    
    req_match = 0
    for _, row in merged.iterrows():
        if row["total_requests_s"] == row["total_requests_g"]:
            req_match += 1
    req_pct = (req_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(req_pct >= 95, f"billing_summary total_requests >95% match ({req_pct:.1f}%)")
    check(req_pct == 100, f"billing_summary total_requests 100% match ({req_pct:.1f}%)")
    
    base_match = 0
    for _, row in merged.iterrows():
        if abs(row["base_cost_s"] - row["base_cost_g"]) < 0.01:
            base_match += 1
    base_pct = (base_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(base_pct >= 95, f"billing_summary base_cost >95% match ({base_pct:.1f}%)")
    check(base_pct == 100, f"billing_summary base_cost 100% match ({base_pct:.1f}%)")
    
    overage_match = 0
    for _, row in merged.iterrows():
        if abs(row["overage_cost_s"] - row["overage_cost_g"]) < 0.10:
            overage_match += 1
    overage_pct = (overage_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(overage_pct >= 90, f"billing_summary overage_cost >90% match ({overage_pct:.1f}%)")
    
    penalty_match = 0
    for _, row in merged.iterrows():
        if abs(row["penalty_cost_s"] - row["penalty_cost_g"]) < 1.0:
            penalty_match += 1
    penalty_pct = (penalty_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(penalty_pct >= 85, f"billing_summary penalty_cost >85% match ({penalty_pct:.1f}%)")
    
    total_match = 0
    for _, row in merged.iterrows():
        if abs(row["total_due_s"] - row["total_due_g"]) < 1.0:
            total_match += 1
    total_pct = (total_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(total_pct >= 85, f"billing_summary total_due >85% match ({total_pct:.1f}%)")
    check(total_pct >= 95, f"billing_summary total_due >95% match ({total_pct:.1f}%)")


def verify_cross_file_consistency(request_df, consumption_df, violation_df, billing_df):
    if request_df is None or consumption_df is None:
        return
    
    req_tenant_counts = request_df.groupby(["tenant_id", "quota_bucket"]).size().reset_index(name="count")
    cons_counts = consumption_df[["tenant_id", "quota_bucket", "request_count"]]
    
    merged = pd.merge(req_tenant_counts, cons_counts, on=["tenant_id", "quota_bucket"])
    match = 0
    for _, row in merged.iterrows():
        if row["count"] == row["request_count"]:
            match += 1
    match_pct = (match / len(merged)) * 100 if len(merged) > 0 else 0
    check(match_pct >= 95, f"Cross-file: request counts consistent with consumption ({match_pct:.1f}%)")
    
    if violation_df is not None and len(violation_df) > 0:
        throttled_requests = set(request_df[request_df["within_limit"] == "NO"]["request_id"])
        violation_requests = set(violation_df["request_id"])
        overlap = len(throttled_requests & violation_requests)
        overlap_pct = (overlap / len(violation_requests)) * 100 if len(violation_requests) > 0 else 0
        check(overlap_pct >= 95, f"Cross-file: violations match throttled requests ({overlap_pct:.1f}%)")
    
    if billing_df is not None:
        total_billing = billing_df["total_due"].sum()
        check(total_billing > 0, f"Cross-file: total billing is positive ({total_billing:.2f})")


def verify_data_complexity(golden_request, golden_violation, golden_consumption):
    throttled = len(golden_request[golden_request["within_limit"] == "NO"])
    check(throttled > 100, f"Data complexity: sufficient throttled requests ({throttled} > 100)")
    
    unique_tenants_violated = golden_violation["tenant_id"].nunique() if len(golden_violation) > 0 else 0
    check(unique_tenants_violated > 5, f"Data complexity: multiple tenants with violations ({unique_tenants_violated} > 5)")
    
    statuses = golden_consumption["status"].value_counts()
    check(len(statuses) >= 3, f"Data complexity: multiple status types ({len(statuses)} >= 3)")


def main():
    base_dir = Path(__file__).parent
    task_dir = base_dir.parent / "task"
    
    golden_request = load_csv(base_dir / "golden_request_attribution.csv", "golden_request_attribution")
    golden_consumption = load_csv(base_dir / "golden_tenant_consumption.csv", "golden_tenant_consumption")
    golden_violation = load_csv(base_dir / "golden_violation_ledger.csv", "golden_violation_ledger")
    golden_throttle = load_csv(base_dir / "golden_throttle_decisions.csv", "golden_throttle_decisions")
    golden_billing = load_csv(base_dir / "golden_billing_summary.csv", "golden_billing_summary")
    
    student_request = load_csv(task_dir / "request_attribution.csv", "request_attribution")
    student_consumption = load_csv(task_dir / "tenant_consumption.csv", "tenant_consumption")
    student_violation = load_csv(task_dir / "violation_ledger.csv", "violation_ledger")
    student_throttle = load_csv(task_dir / "throttle_decisions.csv", "throttle_decisions")
    student_billing = load_csv(task_dir / "billing_summary.csv", "billing_summary")
    
    if golden_request is not None and golden_violation is not None and golden_consumption is not None:
        verify_data_complexity(golden_request, golden_violation, golden_consumption)
    
    verify_request_attribution(student_request, golden_request)
    verify_tenant_consumption(student_consumption, golden_consumption)
    verify_violation_ledger(student_violation, golden_violation)
    verify_throttle_decisions(student_throttle, golden_throttle)
    verify_billing_summary(student_billing, golden_billing)
    
    verify_cross_file_consistency(student_request, student_consumption, student_violation, student_billing)
    
    print(f"{PASS}/{PASS+FAIL}")
    
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

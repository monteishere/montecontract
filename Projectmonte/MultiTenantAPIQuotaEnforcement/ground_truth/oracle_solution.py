import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta


def load_data(task_dir):
    api_requests = pd.read_csv(task_dir / "api_requests.csv")
    tenants = pd.read_csv(task_dir / "tenants.csv")
    tier_policies = pd.read_csv(task_dir / "tier_policies.csv")
    endpoints = pd.read_csv(task_dir / "endpoints.csv")
    rate_windows = pd.read_csv(task_dir / "rate_windows.csv")
    penalty_rules = pd.read_csv(task_dir / "penalty_rules.csv")
    quota_overrides = pd.read_csv(task_dir / "quota_overrides.csv")
    historical_violations = pd.read_csv(task_dir / "historical_violations.csv")
    return (api_requests, tenants, tier_policies, endpoints, rate_windows,
            penalty_rules, quota_overrides, historical_violations)


def get_effective_limits(tenant_id, timestamp, tenants, tier_policies, quota_overrides):
    tenant_row = tenants[tenants["tenant_id"] == tenant_id].iloc[0]
    tier_id = tenant_row["tier_id"]
    tier_row = tier_policies[tier_policies["tier_id"] == tier_id].iloc[0]
    
    limits = {
        "daily_request_limit": tier_row["daily_request_limit"],
        "hourly_request_limit": tier_row["hourly_request_limit"],
        "daily_token_limit": tier_row["daily_token_limit"],
        "burst_multiplier": tier_row["burst_multiplier"],
        "overage_rate_per_1k_requests": tier_row["overage_rate_per_1k_requests"],
        "overage_rate_per_1k_tokens": tier_row["overage_rate_per_1k_tokens"],
        "grace_period_minutes": tier_row["grace_period_minutes"],
        "throttle_behavior": tier_row["throttle_behavior"],
        "priority_level": tier_row["priority_level"],
    }
    
    if pd.notna(tenant_row["custom_rate_limit"]):
        limits["hourly_request_limit"] = int(tenant_row["custom_rate_limit"])
    
    if tenant_row["exempt_from_throttling"] == "YES":
        limits["throttle_behavior"] = "EXEMPT"
    
    ts = pd.to_datetime(timestamp)
    for _, ovr in quota_overrides.iterrows():
        if ovr["tenant_id"] != tenant_id:
            continue
        start = pd.to_datetime(ovr["effective_start"])
        end = pd.to_datetime(ovr["effective_end"])
        if start <= ts <= end:
            field = ovr["override_field"]
            if field in limits:
                limits[field] = ovr["override_value"]
    
    return limits


def compute_request_attribution(api_requests, tenants, tier_policies, endpoints, quota_overrides):
    endpoint_info = {}
    for _, row in endpoints.iterrows():
        endpoint_info[row["endpoint_id"]] = {
            "cost_multiplier": row["cost_multiplier"],
            "avg_token_estimate": row["avg_token_estimate"],
            "category": row["category"],
            "billable": row["billable"],
        }
    
    tenant_tier_map = dict(zip(tenants["tenant_id"], tenants["tier_id"]))
    
    rows = []
    api_requests_sorted = api_requests.sort_values("timestamp").reset_index(drop=True)
    
    running_hourly = defaultdict(lambda: {"count": 0, "window_start": None})
    running_daily = defaultdict(lambda: {"count": 0, "tokens": 0, "date": None})
    
    for idx, req in api_requests_sorted.iterrows():
        tenant_id = req["tenant_id"]
        endpoint_id = req["endpoint_id"]
        ts = pd.to_datetime(req["timestamp"])
        tokens_used = req["tokens_used"]
        response_code = req["response_code"]
        
        ep_info = endpoint_info.get(endpoint_id, {
            "cost_multiplier": 1.0,
            "avg_token_estimate": 100,
            "category": "UNKNOWN",
            "billable": "YES",
        })
        
        if pd.isna(tokens_used):
            if response_code >= 500:
                effective_tokens = 0
            else:
                effective_tokens = int(ep_info["avg_token_estimate"])
        else:
            effective_tokens = int(tokens_used)
        
        cost_mult = ep_info["cost_multiplier"]
        effective_cost = round(effective_tokens * cost_mult, 4)
        
        limits = get_effective_limits(tenant_id, ts, tenants, tier_policies, quota_overrides)
        
        current_date = ts.date()
        current_hour_start = ts.replace(minute=0, second=0, microsecond=0)
        
        daily_key = (tenant_id, current_date)
        if running_daily[daily_key]["date"] != current_date:
            running_daily[daily_key] = {"count": 0, "tokens": 0, "date": current_date}
        
        hourly_key = (tenant_id, current_hour_start)
        if running_hourly[hourly_key]["window_start"] != current_hour_start:
            running_hourly[hourly_key] = {"count": 0, "window_start": current_hour_start}
        
        running_daily[daily_key]["count"] += 1
        running_daily[daily_key]["tokens"] += effective_tokens
        running_hourly[hourly_key]["count"] += 1
        
        daily_count = running_daily[daily_key]["count"]
        daily_tokens = running_daily[daily_key]["tokens"]
        hourly_count = running_hourly[hourly_key]["count"]
        
        daily_limit = limits["daily_request_limit"]
        hourly_limit = limits["hourly_request_limit"]
        token_limit = limits["daily_token_limit"]
        burst_mult = limits["burst_multiplier"]
        
        within_daily = daily_count <= daily_limit
        within_hourly = hourly_count <= hourly_limit
        within_tokens = daily_tokens <= token_limit
        within_burst = hourly_count <= int(hourly_limit * burst_mult)
        
        if within_daily and within_hourly and within_tokens:
            within_limit = "YES"
            throttle_reason = ""
        elif within_burst and within_daily and within_tokens:
            within_limit = "BURST"
            throttle_reason = "HOURLY_BURST"
        else:
            within_limit = "NO"
            reasons = []
            if not within_daily:
                reasons.append("DAILY_EXCEEDED")
            if not within_hourly and not within_burst:
                reasons.append("HOURLY_EXCEEDED")
            if not within_tokens:
                reasons.append("TOKEN_EXCEEDED")
            throttle_reason = "|".join(reasons)
        
        quota_bucket = f"{current_date.strftime('%Y-%m-%d')}"
        
        is_billable = ep_info["billable"] == "YES"
        if response_code >= 400 and response_code < 500:
            billable_flag = "NO"
        elif response_code >= 500:
            billable_flag = "NO"
        elif not is_billable:
            billable_flag = "NO"
        else:
            billable_flag = "YES"
        
        rows.append({
            "request_id": req["request_id"],
            "tenant_id": tenant_id,
            "endpoint_id": endpoint_id,
            "timestamp": req["timestamp"],
            "effective_tokens": effective_tokens,
            "cost_multiplier": cost_mult,
            "effective_cost": round(effective_cost, 2),
            "quota_bucket": quota_bucket,
            "hourly_count": hourly_count,
            "daily_count": daily_count,
            "daily_tokens": daily_tokens,
            "within_limit": within_limit,
            "throttle_reason": throttle_reason,
            "billable": billable_flag,
        })
    
    df = pd.DataFrame(rows)
    return df


def compute_tenant_consumption(request_attribution, tenants, tier_policies):
    tenant_tier_map = dict(zip(tenants["tenant_id"], tenants["tier_id"]))
    tier_limits = {}
    for _, row in tier_policies.iterrows():
        tier_limits[row["tier_id"]] = {
            "daily_request_limit": row["daily_request_limit"],
            "daily_token_limit": row["daily_token_limit"],
        }
    
    grouped = request_attribution.groupby(["tenant_id", "quota_bucket"]).agg({
        "request_id": "count",
        "effective_tokens": "sum",
        "effective_cost": "sum",
        "within_limit": lambda x: (x == "NO").sum(),
    }).reset_index()
    
    grouped.columns = ["tenant_id", "quota_bucket", "request_count", "tokens_consumed", "total_cost", "violation_count"]
    
    rows = []
    for _, row in grouped.iterrows():
        tenant_id = row["tenant_id"]
        tier_id = tenant_tier_map.get(tenant_id, "FREE")
        limits = tier_limits.get(tier_id, {"daily_request_limit": 1000, "daily_token_limit": 10000})
        
        req_limit = limits["daily_request_limit"]
        token_limit = limits["daily_token_limit"]
        
        request_pct = round((row["request_count"] / req_limit) * 100, 2) if req_limit > 0 else 0
        token_pct = round((row["tokens_consumed"] / token_limit) * 100, 2) if token_limit > 0 else 0
        
        quota_used_pct = round(max(request_pct, token_pct), 2)
        
        if row["violation_count"] > 10:
            status = "THROTTLED"
        elif quota_used_pct > 100:
            status = "OVERAGE"
        elif quota_used_pct > 80:
            status = "WARNING"
        else:
            status = "NORMAL"
        
        overage_requests = max(0, row["request_count"] - req_limit)
        overage_tokens = max(0, row["tokens_consumed"] - token_limit)
        
        rows.append({
            "tenant_id": tenant_id,
            "tier_id": tier_id,
            "quota_bucket": row["quota_bucket"],
            "request_count": int(row["request_count"]),
            "tokens_consumed": int(row["tokens_consumed"]),
            "total_cost": round(row["total_cost"], 2),
            "request_limit": req_limit,
            "token_limit": token_limit,
            "request_pct": request_pct,
            "token_pct": token_pct,
            "quota_used_pct": quota_used_pct,
            "status": status,
            "violation_count": int(row["violation_count"]),
            "overage_requests": int(overage_requests),
            "overage_tokens": int(overage_tokens),
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(["tenant_id", "quota_bucket"]).reset_index(drop=True)
    return df


def compute_violation_ledger(request_attribution, tenant_consumption, historical_violations, penalty_rules, tenants):
    tenant_tier_map = dict(zip(tenants["tenant_id"], tenants["tier_id"]))
    
    tier_rules = defaultdict(list)
    all_rules = []
    for _, rule in penalty_rules.iterrows():
        applies_to = rule["applies_to_tier"]
        if applies_to == "ALL":
            all_rules.append(rule)
        else:
            tier_rules[applies_to].append(rule)
    
    historical_counts = defaultdict(int)
    for _, viol in historical_violations.iterrows():
        if viol["resolved"] == "YES":
            historical_counts[viol["tenant_id"]] += 1
    
    violations_by_tenant = defaultdict(list)
    throttled_requests = request_attribution[request_attribution["within_limit"] == "NO"]
    
    for _, req in throttled_requests.iterrows():
        tenant_id = req["tenant_id"]
        violations_by_tenant[tenant_id].append({
            "request_id": req["request_id"],
            "timestamp": req["timestamp"],
            "reason": req["throttle_reason"],
        })
    
    rows = []
    violation_id = 1
    
    for tenant_id, viols in violations_by_tenant.items():
        tier_id = tenant_tier_map.get(tenant_id, "FREE")
        
        prior_count = historical_counts.get(tenant_id, 0)
        
        cumulative_penalty = 0.0
        
        for i, viol in enumerate(viols):
            current_violation_num = prior_count + i + 1
            
            applicable_rules = all_rules + tier_rules.get(tier_id, [])
            
            matched_rule = None
            for rule in applicable_rules:
                if rule["min_violations_in_period"] <= current_violation_num <= rule["max_violations_in_period"]:
                    if matched_rule is None or rule["flat_penalty_usd"] > matched_rule["flat_penalty_usd"]:
                        matched_rule = rule
            
            if matched_rule is None:
                severity = "MINOR"
                flat_penalty = 0.0
                pct_penalty = 0.0
                action = "WARNING"
            else:
                severity = matched_rule["severity_id"]
                flat_penalty = matched_rule["flat_penalty_usd"]
                pct_penalty = matched_rule["percentage_penalty"]
                action = matched_rule["enforcement_action"]
            
            penalty_amount = round(flat_penalty, 2)
            cumulative_penalty = round(cumulative_penalty + penalty_amount, 2)
            
            rows.append({
                "violation_id": f"VN{str(violation_id).zfill(6)}",
                "tenant_id": tenant_id,
                "request_id": viol["request_id"],
                "violation_timestamp": viol["timestamp"],
                "violation_type": viol["reason"].split("|")[0] if viol["reason"] else "UNKNOWN",
                "violation_number": current_violation_num,
                "severity": severity,
                "flat_penalty": flat_penalty,
                "pct_penalty": pct_penalty,
                "penalty_amount": penalty_amount,
                "cumulative_penalty": cumulative_penalty,
                "enforcement_action": action,
            })
            violation_id += 1
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["tenant_id", "violation_timestamp"]).reset_index(drop=True)
    return df


def compute_throttle_decisions(request_attribution, tenants, tier_policies):
    tenant_tier_map = dict(zip(tenants["tenant_id"], tenants["tier_id"]))
    tier_behavior = {}
    for _, row in tier_policies.iterrows():
        tier_behavior[row["tier_id"]] = {
            "throttle_behavior": row["throttle_behavior"],
            "grace_period_minutes": row["grace_period_minutes"],
        }
    
    exempt_tenants = set(tenants[tenants["exempt_from_throttling"] == "YES"]["tenant_id"])
    
    throttled = request_attribution[request_attribution["within_limit"] == "NO"].copy()
    
    rows = []
    grace_period_tracker = defaultdict(lambda: {"first_violation": None, "grace_used": False})
    
    for _, req in throttled.iterrows():
        tenant_id = req["tenant_id"]
        tier_id = tenant_tier_map.get(tenant_id, "FREE")
        behavior = tier_behavior.get(tier_id, {"throttle_behavior": "SOFT_THROTTLE", "grace_period_minutes": 0})
        
        ts = pd.to_datetime(req["timestamp"])
        
        if tenant_id in exempt_tenants:
            decision = "ALLOW"
            decision_reason = "EXEMPT_TENANT"
        else:
            grace_minutes = behavior["grace_period_minutes"]
            tracker = grace_period_tracker[tenant_id]
            
            if tracker["first_violation"] is None:
                tracker["first_violation"] = ts
                tracker["grace_used"] = False
            
            time_since_first = (ts - tracker["first_violation"]).total_seconds() / 60
            
            if time_since_first <= grace_minutes and not tracker["grace_used"]:
                decision = "GRACE"
                decision_reason = f"WITHIN_GRACE_{grace_minutes}MIN"
            else:
                tracker["grace_used"] = True
                
                if behavior["throttle_behavior"] == "HARD_BLOCK":
                    decision = "BLOCK"
                    decision_reason = "HARD_BLOCK_TIER"
                else:
                    hourly_excess = max(0, req["hourly_count"] - 100)
                    if hourly_excess > 50:
                        decision = "BLOCK"
                        decision_reason = "SEVERE_OVERAGE"
                    elif hourly_excess > 20:
                        decision = "THROTTLE_90"
                        decision_reason = "HIGH_OVERAGE"
                    else:
                        decision = "THROTTLE_50"
                        decision_reason = "MODERATE_OVERAGE"
        
        rows.append({
            "request_id": req["request_id"],
            "tenant_id": tenant_id,
            "tier_id": tier_id,
            "timestamp": req["timestamp"],
            "throttle_reason": req["throttle_reason"],
            "hourly_count": int(req["hourly_count"]),
            "daily_count": int(req["daily_count"]),
            "decision": decision,
            "decision_reason": decision_reason,
        })
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["tenant_id", "timestamp"]).reset_index(drop=True)
    return df


def compute_billing_summary(tenant_consumption, violation_ledger, tenants, tier_policies):
    tenant_tier_map = dict(zip(tenants["tenant_id"], tenants["tier_id"]))
    tier_pricing = {}
    for _, row in tier_policies.iterrows():
        tier_pricing[row["tier_id"]] = {
            "base_monthly_price": row["base_monthly_price"],
            "overage_rate_per_1k_requests": row["overage_rate_per_1k_requests"],
            "overage_rate_per_1k_tokens": row["overage_rate_per_1k_tokens"],
        }
    
    consumption_agg = tenant_consumption.groupby("tenant_id").agg({
        "request_count": "sum",
        "tokens_consumed": "sum",
        "total_cost": "sum",
        "overage_requests": "sum",
        "overage_tokens": "sum",
    }).reset_index()
    
    if len(violation_ledger) > 0:
        penalty_agg = violation_ledger.groupby("tenant_id").agg({
            "penalty_amount": "sum",
            "violation_id": "count",
        }).reset_index()
        penalty_agg.columns = ["tenant_id", "total_penalties", "violation_count"]
    else:
        penalty_agg = pd.DataFrame(columns=["tenant_id", "total_penalties", "violation_count"])
    
    rows = []
    for _, row in consumption_agg.iterrows():
        tenant_id = row["tenant_id"]
        tier_id = tenant_tier_map.get(tenant_id, "FREE")
        pricing = tier_pricing.get(tier_id, {
            "base_monthly_price": 0,
            "overage_rate_per_1k_requests": 0,
            "overage_rate_per_1k_tokens": 0,
        })
        
        period_fraction = 3 / 30
        base_cost = round(pricing["base_monthly_price"] * period_fraction, 2)
        
        overage_req = row["overage_requests"]
        overage_tok = row["overage_tokens"]
        
        overage_req_cost = round((overage_req / 1000) * pricing["overage_rate_per_1k_requests"], 2)
        overage_tok_cost = round((overage_tok / 1000) * pricing["overage_rate_per_1k_tokens"], 2)
        overage_cost = round(overage_req_cost + overage_tok_cost, 2)
        
        penalty_row = penalty_agg[penalty_agg["tenant_id"] == tenant_id]
        if len(penalty_row) > 0:
            penalty_cost = round(penalty_row.iloc[0]["total_penalties"], 2)
            violations = int(penalty_row.iloc[0]["violation_count"])
        else:
            penalty_cost = 0.0
            violations = 0
        
        total_due = round(base_cost + overage_cost + penalty_cost, 2)
        
        rows.append({
            "tenant_id": tenant_id,
            "tier_id": tier_id,
            "billing_period": "2025-01-01 to 2025-01-03",
            "total_requests": int(row["request_count"]),
            "total_tokens": int(row["tokens_consumed"]),
            "overage_requests": int(overage_req),
            "overage_tokens": int(overage_tok),
            "base_cost": base_cost,
            "overage_cost": overage_cost,
            "penalty_cost": penalty_cost,
            "total_due": total_due,
            "violation_count": violations,
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("tenant_id").reset_index(drop=True)
    return df


def main():
    base_dir = Path(__file__).parent
    task_dir = base_dir.parent / "task"
    ground_truth = base_dir
    
    data = load_data(task_dir)
    (api_requests, tenants, tier_policies, endpoints, rate_windows,
     penalty_rules, quota_overrides, historical_violations) = data
    
    request_attribution = compute_request_attribution(
        api_requests, tenants, tier_policies, endpoints, quota_overrides
    )
    request_attribution.to_csv(task_dir / "request_attribution.csv", index=False)
    print(f"request_attribution.csv: {len(request_attribution)} rows")
    
    tenant_consumption = compute_tenant_consumption(
        request_attribution, tenants, tier_policies
    )
    tenant_consumption.to_csv(task_dir / "tenant_consumption.csv", index=False)
    print(f"tenant_consumption.csv: {len(tenant_consumption)} rows")
    
    violation_ledger = compute_violation_ledger(
        request_attribution, tenant_consumption, historical_violations, penalty_rules, tenants
    )
    violation_ledger.to_csv(task_dir / "violation_ledger.csv", index=False)
    print(f"violation_ledger.csv: {len(violation_ledger)} rows")
    
    throttle_decisions = compute_throttle_decisions(
        request_attribution, tenants, tier_policies
    )
    throttle_decisions.to_csv(task_dir / "throttle_decisions.csv", index=False)
    print(f"throttle_decisions.csv: {len(throttle_decisions)} rows")
    
    billing_summary = compute_billing_summary(
        tenant_consumption, violation_ledger, tenants, tier_policies
    )
    billing_summary.to_csv(task_dir / "billing_summary.csv", index=False)
    print(f"billing_summary.csv: {len(billing_summary)} rows")
    
    request_attribution.to_csv(ground_truth / "golden_request_attribution.csv", index=False)
    tenant_consumption.to_csv(ground_truth / "golden_tenant_consumption.csv", index=False)
    violation_ledger.to_csv(ground_truth / "golden_violation_ledger.csv", index=False)
    throttle_decisions.to_csv(ground_truth / "golden_throttle_decisions.csv", index=False)
    billing_summary.to_csv(ground_truth / "golden_billing_summary.csv", index=False)


if __name__ == "__main__":
    main()

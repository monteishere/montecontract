import json
import math
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path

TASK = Path(__file__).parent.parent / "task"
GT   = Path(__file__).parent

DAY_MAP = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI", 5: "SAT", 6: "SUN"}


def load_inputs():
    api_requests     = pd.read_csv(TASK / "api_requests.csv")
    tenants          = pd.read_csv(TASK / "tenants.csv")
    tier_policies    = pd.read_csv(TASK / "tier_policies.csv")
    endpoints        = pd.read_csv(TASK / "endpoints.csv")
    surge_windows    = pd.read_csv(TASK / "surge_windows.csv")
    penalty_rules    = pd.read_csv(TASK / "penalty_rules.csv")
    quota_overrides  = pd.read_csv(TASK / "quota_overrides.csv")
    hist_violations  = pd.read_csv(TASK / "historical_violations.csv")

    with open(TASK / "endpoint_weights.json") as fh:
        endpoint_weights_raw = json.load(fh)

    sched = pd.read_excel(TASK / "sla_schedule.xlsx", sheet_name="active_policies")

    return (api_requests, tenants, tier_policies, endpoints, surge_windows,
            penalty_rules, quota_overrides, hist_violations,
            endpoint_weights_raw, sched)


def build_weight_lookup(endpoint_weights_raw, endpoints_df, tenants_df):
    """
    T1 trap: endpoint_weights.json uses UPPERCASE category keys ("READ", "WRITE", etc.)
             but endpoints.csv uses lowercase ("read", "write", etc.)
    T2 trap: sparse JSON — missing combos default to 1.0 (not NaN, not 0)
    """
    tier_ids = tenants_df["tier_id"].unique().tolist()
    ep_cat   = dict(zip(endpoints_df["endpoint_id"], endpoints_df["category"]))
    lookup   = {}
    for ep_id, cat in ep_cat.items():
        key = cat.upper()  # normalize to uppercase to match JSON keys
        for tier in tier_ids:
            w = endpoint_weights_raw.get(key, {}).get(tier, 1.0)
            lookup[(ep_id, tier)] = w
    return lookup


def build_sla_lookup(sched):
    """Build (tier_id, day_of_week) -> capacity_multiplier from active_policies sheet."""
    lookup = {}
    for _, row in sched.iterrows():
        lookup[(str(row["tier_id"]), str(row["day_of_week"]))] = float(row["capacity_multiplier"])
    return lookup


def get_sla_multiplier(sla_lookup, tier_id, ts):
    """
    T5 trap: lookup by exact day-of-week (3-char uppercase: MON/TUE/WED/THU/FRI/SAT/SUN),
             then fallback to ANY, then 1.0.
    """
    dow = DAY_MAP[ts.weekday()]
    exact = sla_lookup.get((tier_id, dow))
    if exact is not None:
        return exact
    fallback = sla_lookup.get((tier_id, "ANY"))
    return fallback if fallback is not None else 1.0


def get_surge_multiplier(surge_df, category, ts):
    """
    T6 trap: end is EXCLUSIVE. A request at exactly effective_end_utc does NOT get surge.
    """
    cat = category.lower()
    for _, row in surge_df.iterrows():
        if row["applies_to_category"].lower() != cat:
            continue
        start = pd.to_datetime(row["effective_start_utc"])
        end   = pd.to_datetime(row["effective_end_utc"])
        if start <= ts < end:   # end exclusive
            return float(row["surge_multiplier"])
    return 1.0


def get_effective_limits(tenant_id, ts_date, tenants_df, tier_policies_df, quota_overrides_df):
    """Apply quota overrides on top of tier baseline. Returns all limit/behavior fields."""
    t_row   = tenants_df[tenants_df["tenant_id"] == tenant_id].iloc[0]
    tier_id = t_row["tier_id"]
    tp_row  = tier_policies_df[tier_policies_df["tier_id"] == tier_id].iloc[0]

    hourly = int(tp_row["hourly_request_limit"])
    daily  = int(tp_row["daily_request_limit"])
    tokens = int(tp_row["daily_token_limit"])

    ts_date_only = pd.to_datetime(ts_date).date()
    for _, ovr in quota_overrides_df.iterrows():
        if ovr["tenant_id"] != tenant_id:
            continue
        start = pd.to_datetime(ovr["effective_start"]).date()
        end   = pd.to_datetime(ovr["effective_end"]).date()
        if start <= ts_date_only <= end:
            field = str(ovr["override_field"])
            val   = int(ovr["override_value"])
            if field == "hourly_request_limit":
                hourly = val
            elif field == "daily_request_limit":
                daily = val
            elif field == "daily_token_limit":
                tokens = val

    burst   = float(tp_row["burst_multiplier"])
    exempt  = str(t_row["exempt_from_throttling"]) == "YES"
    behavior= str(tp_row["throttle_behavior"])
    grace   = int(tp_row["grace_period_minutes"])
    return hourly, daily, tokens, burst, exempt, behavior, grace, tier_id


def compute_request_attribution(api_requests, tenants, tier_policies, endpoints,
                                 surge_windows, quota_overrides,
                                 weight_lookup, sla_lookup):
    ep_cat      = dict(zip(endpoints["endpoint_id"], endpoints["category"]))
    ep_billable = dict(zip(endpoints["endpoint_id"], endpoints["billable"]))
    ep_avg_tok  = dict(zip(endpoints["endpoint_id"], endpoints["avg_token_estimate"]))
    tenant_tier = dict(zip(tenants["tenant_id"], tenants["tier_id"]))

    sorted_reqs = api_requests.sort_values(["timestamp", "request_id"]).reset_index(drop=True)

    hourly_counts = defaultdict(int)
    daily_counts  = defaultdict(int)
    daily_tokens  = defaultdict(int)

    rows = []

    for _, req in sorted_reqs.iterrows():
        tid   = req["tenant_id"]
        ep_id = req["endpoint_id"]
        ts    = pd.to_datetime(req["timestamp"])
        rc    = int(req["response_code"])
        cat   = ep_cat.get(ep_id, "read")

        raw_tok = req["tokens_used"]
        if rc >= 500:
            base_tokens = 0
        elif pd.isna(raw_tok):
            base_tokens = int(ep_avg_tok.get(ep_id, 100))
        else:
            base_tokens = int(raw_tok)

        tier_id     = tenant_tier.get(tid, "FREE")
        weight_mult = weight_lookup.get((ep_id, tier_id), 1.0)
        surge_mult  = get_surge_multiplier(surge_windows, cat, ts)
        eff_tokens  = round(base_tokens * weight_mult * surge_mult)

        hour_key = (tid, ts.date(), ts.hour)
        day_key  = (tid, ts.date())

        hourly_counts[hour_key] += 1
        daily_counts[day_key]   += 1
        daily_tokens[day_key]   += eff_tokens

        h_count = hourly_counts[hour_key]
        d_count = daily_counts[day_key]
        d_tok   = daily_tokens[day_key]

        (base_hourly, base_daily, base_token_limit,
         burst_mult, exempt, behavior, grace, tier_id) = get_effective_limits(
            tid, ts, tenants, tier_policies, quota_overrides)

        sla_mult   = get_sla_multiplier(sla_lookup, tier_id, ts)
        eff_hourly = math.floor(base_hourly * sla_mult)
        burst_thr  = math.floor(eff_hourly * burst_mult)

        within_hourly = h_count <= eff_hourly
        within_daily  = d_count <= base_daily
        within_tokens = d_tok   <= base_token_limit
        within_burst  = h_count <= burst_thr

        if within_hourly and within_daily and within_tokens:
            within_limit    = "YES"
            throttle_reason = ""
        elif within_burst and within_daily and within_tokens:
            within_limit    = "BURST"
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

        billable_flag = "YES" if (rc < 400 and ep_billable.get(ep_id, "YES") == "YES") else "NO"

        rows.append({
            "request_id":             req["request_id"],
            "tenant_id":              tid,
            "endpoint_id":            ep_id,
            "timestamp":              req["timestamp"],
            "base_tokens":            base_tokens,
            "weight_multiplier":      round(weight_mult, 4),
            "surge_multiplier":       round(surge_mult, 4),
            "effective_tokens":       eff_tokens,
            "hourly_request_count":   h_count,
            "daily_request_count":    d_count,
            "daily_token_count":      d_tok,
            "effective_hourly_limit": eff_hourly,
            "within_limit":           within_limit,
            "throttle_reason":        throttle_reason,
            "billable":               billable_flag,
        })

    df = pd.DataFrame(rows)
    df.to_csv(TASK / "request_attribution.csv", index=False)
    return df


def compute_tenant_consumption(ra_df, tenants, tier_policies, quota_overrides, sla_lookup):
    tenant_tier = dict(zip(tenants["tenant_id"], tenants["tier_id"]))
    tp_idx      = tier_policies.set_index("tier_id")

    ra_df = ra_df.copy()
    ra_df["quota_bucket"] = pd.to_datetime(ra_df["timestamp"]).dt.strftime("%Y-%m-%d")

    grouped = ra_df.groupby(["tenant_id", "quota_bucket"]).agg(
        request_count   =("request_id",   "count"),
        tokens_consumed =("effective_tokens", "sum"),
        violation_count =("within_limit",  lambda x: (x == "NO").sum()),
    ).reset_index()

    rows = []
    for _, row in grouped.iterrows():
        tid    = row["tenant_id"]
        bucket = row["quota_bucket"]
        tier_id = tenant_tier.get(tid, "FREE")

        (base_hourly, base_daily, base_token_limit, burst_mult,
         exempt, behavior, grace, tier_id) = get_effective_limits(
            tid, bucket, tenants, tier_policies, quota_overrides)

        req_count = int(row["request_count"])
        tok_count = int(row["tokens_consumed"])

        req_pct   = round((req_count / base_daily)       * 100, 2) if base_daily       > 0 else 0.0
        tok_pct   = round((tok_count / base_token_limit) * 100, 2) if base_token_limit > 0 else 0.0
        quota_pct = round(max(req_pct, tok_pct), 2)

        viol = int(row["violation_count"])
        if viol > 10:
            status = "THROTTLED"
        elif quota_pct > 100:
            status = "OVERAGE"
        elif quota_pct > 80:
            status = "WARNING"
        else:
            status = "NORMAL"

        overage_req = max(0, req_count - base_daily)
        overage_tok = max(0, tok_count - base_token_limit)

        tp_row      = tp_idx.loc[tier_id]
        total_cost  = round(tok_count * float(tp_row["overage_rate_per_1k_tokens"]) / 1000, 2)

        rows.append({
            "tenant_id":        tid,
            "tier_id":          tier_id,
            "quota_bucket":     bucket,
            "request_count":    req_count,
            "tokens_consumed":  tok_count,
            "total_cost":       total_cost,
            "request_limit":    base_daily,
            "token_limit":      base_token_limit,
            "request_pct":      req_pct,
            "token_pct":        tok_pct,
            "quota_used_pct":   quota_pct,
            "status":           status,
            "violation_count":  viol,
            "overage_requests": int(overage_req),
            "overage_tokens":   int(overage_tok),
        })

    df = pd.DataFrame(rows).sort_values(["tenant_id", "quota_bucket"]).reset_index(drop=True)
    df.to_csv(TASK / "tenant_consumption.csv", index=False)
    return df


def compute_violation_ledger(ra_df, tenants, tier_policies, penalty_rules, hist_violations):
    """
    T4 trap: prior_violation_count = hist_count + current_position (0-indexed);
             violation_number = prior_violation_count + 1.
    T7 trap: penalty_amount = flat + min(pct_rate * eff_tokens/1000, cap_mult * flat)
    """
    tenant_tier = dict(zip(tenants["tenant_id"], tenants["tier_id"]))

    hist_counts = defaultdict(int)
    for _, hv in hist_violations.iterrows():
        hist_counts[hv["tenant_id"]] += 1

    throttled = ra_df[ra_df["within_limit"] == "NO"].copy()
    throttled  = throttled.sort_values(["tenant_id", "timestamp", "request_id"]).reset_index(drop=True)

    current_position   = defaultdict(int)
    cumulative_penalty = defaultdict(float)

    rows = []
    viol_id = 1

    for _, req in throttled.iterrows():
        tid     = req["tenant_id"]
        tier_id = tenant_tier.get(tid, "FREE")

        prior_count      = hist_counts[tid] + current_position[tid]
        violation_number = prior_count + 1
        current_position[tid] += 1

        matched = None
        for _, rule in penalty_rules.iterrows():
            applies = str(rule["applies_to_tier"])
            if applies not in ("ALL", tier_id):
                continue
            if rule["min_prior_violations"] <= prior_count <= rule["max_prior_violations"]:
                if matched is None or float(rule["flat_penalty_usd"]) > float(matched["flat_penalty_usd"]):
                    matched = rule

        if matched is None:
            severity = "WARNING"
            flat      = 0.0
            pct_rate  = 0.0
            cap_mult  = 1.0
            action    = "LOG"
        else:
            severity  = str(matched["severity_id"])
            flat      = float(matched["flat_penalty_usd"])
            pct_rate  = float(matched["pct_penalty_per_1k_tokens"])
            cap_mult  = float(matched["penalty_cap_multiplier"])
            action    = str(matched["enforcement_action"])

        eff_tok        = int(req["effective_tokens"])
        pct_component  = pct_rate * eff_tok / 1000.0
        cap_value      = cap_mult * flat
        penalty_amt    = round(flat + min(pct_component, cap_value), 4)

        cumulative_penalty[tid] = round(cumulative_penalty[tid] + penalty_amt, 4)

        rows.append({
            "violation_id":              f"VN{viol_id:06d}",
            "tenant_id":                 tid,
            "request_id":                req["request_id"],
            "violation_timestamp":       req["timestamp"],
            "violation_type":            req["throttle_reason"].split("|")[0] if req["throttle_reason"] else "UNKNOWN",
            "prior_violation_count":     prior_count,
            "violation_number":          violation_number,
            "severity":                  severity,
            "flat_penalty_usd":          flat,
            "pct_penalty_per_1k_tokens": pct_rate,
            "penalty_amount":            penalty_amt,
            "cumulative_penalty":        cumulative_penalty[tid],
            "enforcement_action":        action,
        })
        viol_id += 1

    df = pd.DataFrame(rows)
    df.to_csv(TASK / "violation_ledger.csv", index=False)
    return df


def compute_throttle_decisions(ra_df, tenants, tier_policies):
    tenant_tier = dict(zip(tenants["tenant_id"], tenants["tier_id"]))
    tp_idx      = tier_policies.set_index("tier_id")
    exempt_set  = set(tenants[tenants["exempt_from_throttling"] == "YES"]["tenant_id"])

    throttled = ra_df[ra_df["within_limit"] == "NO"].copy()
    throttled  = throttled.sort_values(["tenant_id", "timestamp", "request_id"]).reset_index(drop=True)

    first_viol_ts = {}
    grace_used    = set()
    rows = []

    for _, req in throttled.iterrows():
        tid     = req["tenant_id"]
        tier_id = tenant_tier.get(tid, "FREE")
        tp_row  = tp_idx.loc[tier_id]
        ts      = pd.to_datetime(req["timestamp"])

        if tid in exempt_set:
            decision = "ALLOW"
            reason   = "EXEMPT_TENANT"
        else:
            grace_min = int(tp_row["grace_period_minutes"])
            behavior  = str(tp_row["throttle_behavior"])

            if tid not in first_viol_ts:
                first_viol_ts[tid] = ts

            elapsed_min = (ts - first_viol_ts[tid]).total_seconds() / 60.0

            if elapsed_min <= grace_min and tid not in grace_used:
                decision = "GRACE"
                reason   = f"WITHIN_GRACE_{grace_min}MIN"
            else:
                grace_used.add(tid)
                if behavior == "HARD_BLOCK":
                    decision = "BLOCK"
                    reason   = "HARD_BLOCK_TIER"
                else:
                    hourly_excess = max(0, int(req["hourly_request_count"]) - int(req["effective_hourly_limit"]))
                    if hourly_excess > 50:
                        decision = "BLOCK"
                        reason   = "SEVERE_OVERAGE"
                    elif hourly_excess > 20:
                        decision = "THROTTLE_90"
                        reason   = "HIGH_OVERAGE"
                    else:
                        decision = "THROTTLE_50"
                        reason   = "MODERATE_OVERAGE"

        rows.append({
            "request_id":             req["request_id"],
            "tenant_id":              tid,
            "tier_id":                tier_id,
            "timestamp":              req["timestamp"],
            "throttle_reason":        req["throttle_reason"],
            "hourly_request_count":   int(req["hourly_request_count"]),
            "daily_request_count":    int(req["daily_request_count"]),
            "effective_hourly_limit": int(req["effective_hourly_limit"]),
            "decision":               decision,
            "decision_reason":        reason,
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["tenant_id", "timestamp", "request_id"]).reset_index(drop=True)
    df.to_csv(TASK / "throttle_decisions.csv", index=False)
    return df


def compute_billing_summary(tenant_consumption, violation_ledger, tenants, tier_policies):
    import calendar as cal_mod
    tenant_tier = dict(zip(tenants["tenant_id"], tenants["tier_id"]))
    tp_idx      = tier_policies.set_index("tier_id")

    cons_agg = tenant_consumption.groupby("tenant_id").agg(
        request_count   =("request_count",    "sum"),
        tokens_consumed =("tokens_consumed",   "sum"),
        overage_requests=("overage_requests",  "sum"),
        overage_tokens  =("overage_tokens",    "sum"),
        n_days          =("quota_bucket",      "nunique"),
        min_date        =("quota_bucket",      "min"),
    ).reset_index()

    if len(violation_ledger) > 0:
        pen_agg = violation_ledger.groupby("tenant_id").agg(
            penalty_cost    =("penalty_amount", "sum"),
            violation_count =("violation_id",   "count"),
        ).reset_index()
    else:
        pen_agg = pd.DataFrame(columns=["tenant_id", "penalty_cost", "violation_count"])

    rows = []
    for _, row in cons_agg.iterrows():
        tid     = row["tenant_id"]
        tier_id = tenant_tier.get(tid, "FREE")
        tp_row  = tp_idx.loc[tier_id]

        min_dt        = pd.to_datetime(row["min_date"])
        days_in_month = cal_mod.monthrange(min_dt.year, min_dt.month)[1]
        period_frac   = round(row["n_days"] / days_in_month, 4)
        base_cost     = round(float(tp_row["base_monthly_price"]) * period_frac, 2)

        ovc_req      = round((row["overage_requests"] / 1000) * float(tp_row["overage_rate_per_1k_requests"]), 2)
        ovc_tok      = round((row["overage_tokens"]   / 1000) * float(tp_row["overage_rate_per_1k_tokens"]),   2)
        overage_cost = round(ovc_req + ovc_tok, 2)

        pr = pen_agg[pen_agg["tenant_id"] == tid]
        penalty_cost    = round(float(pr.iloc[0]["penalty_cost"]),    2) if len(pr) > 0 else 0.0
        violation_count = int(pr.iloc[0]["violation_count"]) if len(pr) > 0 else 0

        total_due = round(base_cost + overage_cost + penalty_cost, 2)

        rows.append({
            "tenant_id":         tid,
            "tier_id":           tier_id,
            "billing_period":    "2025-01-01 to 2025-01-03",
            "total_requests":    int(row["request_count"]),
            "total_tokens":      int(row["tokens_consumed"]),
            "overage_requests":  int(row["overage_requests"]),
            "overage_tokens":    int(row["overage_tokens"]),
            "base_cost":         base_cost,
            "overage_cost":      overage_cost,
            "penalty_cost":      penalty_cost,
            "total_due":         total_due,
            "violation_count":   violation_count,
        })

    df = pd.DataFrame(rows).sort_values("tenant_id").reset_index(drop=True)
    df.to_csv(TASK / "billing_summary.csv", index=False)
    return df


if __name__ == "__main__":
    (api_requests, tenants, tier_policies, endpoints, surge_windows,
     penalty_rules, quota_overrides, hist_violations,
     endpoint_weights_raw, sched) = load_inputs()

    weight_lookup = build_weight_lookup(endpoint_weights_raw, endpoints, tenants)
    sla_lookup    = build_sla_lookup(sched)

    ra = compute_request_attribution(
        api_requests, tenants, tier_policies, endpoints,
        surge_windows, quota_overrides, weight_lookup, sla_lookup)

    tc = compute_tenant_consumption(ra, tenants, tier_policies, quota_overrides, sla_lookup)
    vl = compute_violation_ledger(ra, tenants, tier_policies, penalty_rules, hist_violations)
    td = compute_throttle_decisions(ra, tenants, tier_policies)
    bs = compute_billing_summary(tc, vl, tenants, tier_policies)

    print(f"request_attribution.csv:  {len(ra)} rows")
    print(f"tenant_consumption.csv:   {len(tc)} rows")
    print(f"violation_ledger.csv:     {len(vl)} rows")
    print(f"throttle_decisions.csv:   {len(td)} rows")
    print(f"billing_summary.csv:      {len(bs)} rows")

import pandas as pd
import numpy as np
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
import openpyxl

np.random.seed(99)
rng = np.random.default_rng(99)

TASK = Path(__file__).parent.parent / "task"

tenants_rows = []
tid = 1
for tier, count in [("FREE", 4), ("STARTER", 5), ("PROFESSIONAL", 5), ("BUSINESS", 4), ("ENTERPRISE", 2)]:
    for i in range(count):
        tenants_rows.append({
            "tenant_id": f"T{tid:03d}",
            "tenant_name": f"Company_{tid}",
            "tier_id": tier,
            "contract_start": "2024-01-01",
            "timezone": "UTC",
            "status": "ACTIVE",
            "exempt_from_throttling": "YES" if tid == 3 else "NO",
        })
        tid += 1

tenants = pd.DataFrame(tenants_rows)
tenants.to_csv(TASK / "tenants.csv", index=False)

tier_policies = pd.DataFrame([
    {"tier_id": "FREE",         "hourly_request_limit": 10,  "daily_request_limit": 80,   "daily_token_limit": 5000,    "burst_multiplier": 1.2,  "base_monthly_price": 0.0,    "overage_rate_per_1k_requests": 0.0,  "overage_rate_per_1k_tokens": 0.0,    "grace_period_minutes": 0,  "throttle_behavior": "HARD_BLOCK"},
    {"tier_id": "STARTER",      "hourly_request_limit": 25,  "daily_request_limit": 200,  "daily_token_limit": 25000,   "burst_multiplier": 1.3,  "base_monthly_price": 9.99,   "overage_rate_per_1k_requests": 0.50, "overage_rate_per_1k_tokens": 0.002,  "grace_period_minutes": 5,  "throttle_behavior": "SOFT_THROTTLE"},
    {"tier_id": "PROFESSIONAL", "hourly_request_limit": 75,  "daily_request_limit": 300,  "daily_token_limit": 150000,  "burst_multiplier": 1.2,  "base_monthly_price": 49.99,  "overage_rate_per_1k_requests": 0.30, "overage_rate_per_1k_tokens": 0.0015, "grace_period_minutes": 15, "throttle_behavior": "SOFT_THROTTLE"},
    {"tier_id": "BUSINESS",     "hourly_request_limit": 200, "daily_request_limit": 2000, "daily_token_limit": 1000000, "burst_multiplier": 2.0,  "base_monthly_price": 199.99, "overage_rate_per_1k_requests": 0.20, "overage_rate_per_1k_tokens": 0.001,  "grace_period_minutes": 30, "throttle_behavior": "SOFT_THROTTLE"},
    {"tier_id": "ENTERPRISE",   "hourly_request_limit": 500, "daily_request_limit": 5000, "daily_token_limit": 5000000, "burst_multiplier": 2.0,  "base_monthly_price": 999.99, "overage_rate_per_1k_requests": 0.10, "overage_rate_per_1k_tokens": 0.0005, "grace_period_minutes": 60, "throttle_behavior": "SOFT_THROTTLE"},
])
tier_policies.to_csv(TASK / "tier_policies.csv", index=False)

endpoints = pd.DataFrame([
    {"endpoint_id": "EP001", "method": "GET",    "path": "/api/v2/records",      "category": "read",  "avg_token_estimate": 200,  "billable": "YES"},
    {"endpoint_id": "EP002", "method": "POST",   "path": "/api/v2/records",      "category": "write", "avg_token_estimate": 500,  "billable": "YES"},
    {"endpoint_id": "EP003", "method": "GET",    "path": "/api/v2/search",       "category": "read",  "avg_token_estimate": 350,  "billable": "YES"},
    {"endpoint_id": "EP004", "method": "POST",   "path": "/api/v2/batch",        "category": "batch", "avg_token_estimate": 2500, "billable": "YES"},
    {"endpoint_id": "EP005", "method": "PUT",    "path": "/api/v2/records/{id}", "category": "write", "avg_token_estimate": 400,  "billable": "YES"},
    {"endpoint_id": "EP006", "method": "DELETE", "path": "/api/v2/records/{id}", "category": "write", "avg_token_estimate": 150,  "billable": "YES"},
    {"endpoint_id": "EP007", "method": "GET",    "path": "/api/v2/analytics",    "category": "read",  "avg_token_estimate": 800,  "billable": "YES"},
    {"endpoint_id": "EP008", "method": "POST",   "path": "/api/v2/admin/config", "category": "admin", "avg_token_estimate": 100,  "billable": "NO"},
    {"endpoint_id": "EP009", "method": "GET",    "path": "/api/v2/health",       "category": "health","avg_token_estimate": 10,   "billable": "NO"},
    {"endpoint_id": "EP010", "method": "POST",   "path": "/api/v2/export",       "category": "batch", "avg_token_estimate": 5000, "billable": "YES"},
])
endpoints.to_csv(TASK / "endpoints.csv", index=False)

# T1+T2: uppercase category keys; some combos intentionally missing -> 1.0
# MISSING: FREE+BATCH, ENTERPRISE+READ, FREE/STARTER/PROFESSIONAL+ADMIN, ENTERPRISE+HEALTH, BUSINESS+HEALTH
endpoint_weights = {
    "READ":   {"FREE": 0.5,  "STARTER": 0.7, "PROFESSIONAL": 1.0, "BUSINESS": 1.0},
    "WRITE":  {"FREE": 2.0,  "STARTER": 1.5, "PROFESSIONAL": 1.2, "BUSINESS": 1.1, "ENTERPRISE": 1.0},
    "BATCH":  {"STARTER": 3.0, "PROFESSIONAL": 2.5, "BUSINESS": 2.0, "ENTERPRISE": 1.5},
    "ADMIN":  {"BUSINESS": 0.8, "ENTERPRISE": 0.5},
    "HEALTH": {"FREE": 0.1, "STARTER": 0.1, "PROFESSIONAL": 0.1},
}
with open(TASK / "endpoint_weights.json", "w") as f:
    json.dump(endpoint_weights, f, indent=2)

# T5: sla_schedule.xlsx — two sheets; only active_policies is valid
# 2025-01-01=WED, 2025-01-02=THU, 2025-01-03=FRI
wb = openpyxl.Workbook()
ws1 = wb.active
ws1.title = "active_policies"
ws1.append(["tier_id", "day_of_week", "capacity_multiplier"])
sla_active = [
    ("FREE",         "ANY", 1.0),
    ("STARTER",      "MON", 0.9), ("STARTER",      "TUE", 0.9), ("STARTER",      "WED", 0.9),
    ("STARTER",      "THU", 0.9), ("STARTER",      "FRI", 0.9),
    ("STARTER",      "SAT", 1.3), ("STARTER",      "SUN", 1.3),
    ("STARTER",      "ANY", 1.0),
    ("PROFESSIONAL", "WED", 1.1), ("PROFESSIONAL", "THU", 1.1), ("PROFESSIONAL", "FRI", 1.1),
    ("PROFESSIONAL", "ANY", 1.0),
    ("BUSINESS",     "SAT", 1.4), ("BUSINESS",     "SUN", 1.4),
    ("BUSINESS",     "ANY", 1.2),
    ("ENTERPRISE",   "SAT", 1.6), ("ENTERPRISE",   "SUN", 1.6),
    ("ENTERPRISE",   "ANY", 1.3),
]
for row in sla_active:
    ws1.append(list(row))

ws2 = wb.create_sheet("deprecated_sla_2024")
ws2.append(["tier_id", "day_of_week", "capacity_multiplier"])
for tier_id, dow, mult in sla_active:
    ws2.append([tier_id, dow, round(mult * 0.65, 2)])

wb.save(TASK / "sla_schedule.xlsx")

# T6: surge_windows.csv — boundary traps
# SW001: WRITE, Jan1 00:00 to Jan2 00:00 (end EXCLUSIVE)
# SW002: BATCH, Jan2 06:00 to Jan3 18:00
# SW003: READ,  Jan15 onward — future, data only covers Jan1-3, won't apply
surge_windows = pd.DataFrame([
    {"surge_id": "SW001", "applies_to_category": "write", "effective_start_utc": "2025-01-01 00:00:00", "effective_end_utc": "2025-01-02 00:00:00", "surge_multiplier": 1.5},
    {"surge_id": "SW002", "applies_to_category": "batch", "effective_start_utc": "2025-01-02 06:00:00", "effective_end_utc": "2025-01-03 18:00:00", "surge_multiplier": 1.4},
    {"surge_id": "SW003", "applies_to_category": "read",  "effective_start_utc": "2025-01-15 00:00:00", "effective_end_utc": "2025-01-31 00:00:00", "surge_multiplier": 1.2},
])
surge_windows.to_csv(TASK / "surge_windows.csv", index=False)

# T7: penalty_rules — flat + pct with penalty_cap_multiplier
# penalty_amount = flat + min(pct_per_1k * effective_tokens/1000, cap_multiplier * flat)
penalty_rules = pd.DataFrame([
    {"severity_id": "WARNING",  "min_prior_violations": 0,  "max_prior_violations": 2,  "flat_penalty_usd": 0.0,   "pct_penalty_per_1k_tokens": 0.0,  "penalty_cap_multiplier": 1.0, "enforcement_action": "LOG",         "applies_to_tier": "ALL"},
    {"severity_id": "NOTICE",   "min_prior_violations": 3,  "max_prior_violations": 9,  "flat_penalty_usd": 5.0,   "pct_penalty_per_1k_tokens": 1.5,  "penalty_cap_multiplier": 0.6, "enforcement_action": "THROTTLE_50", "applies_to_tier": "ALL"},
    {"severity_id": "CAUTION",  "min_prior_violations": 10, "max_prior_violations": 24, "flat_penalty_usd": 20.0,  "pct_penalty_per_1k_tokens": 3.0,  "penalty_cap_multiplier": 0.4, "enforcement_action": "THROTTLE_90", "applies_to_tier": "ALL"},
    {"severity_id": "STRICT",   "min_prior_violations": 25, "max_prior_violations": 49, "flat_penalty_usd": 75.0,  "pct_penalty_per_1k_tokens": 5.0,  "penalty_cap_multiplier": 0.3, "enforcement_action": "SUSPEND_1HR", "applies_to_tier": "ALL"},
    {"severity_id": "CRITICAL", "min_prior_violations": 50, "max_prior_violations": 999,"flat_penalty_usd": 300.0, "pct_penalty_per_1k_tokens": 8.0,  "penalty_cap_multiplier": 0.2, "enforcement_action": "SUSPEND_24HR","applies_to_tier": "ALL"},
])
penalty_rules.to_csv(TASK / "penalty_rules.csv", index=False)

# T4: historical_violations — T007 has 3 prior, T011 has 11 prior
hist = []
for i in range(3):
    hist.append({"violation_id": f"H{i+1:04d}", "tenant_id": "T007",
                 "violation_timestamp": f"2024-12-{20+i:02d} 10:00:00", "violation_type": "HOURLY_EXCEEDED"})
for i in range(11):
    hist.append({"violation_id": f"H{i+10:04d}", "tenant_id": "T011",
                 "violation_timestamp": f"2024-12-{i+2:02d} 14:00:00", "violation_type": "DAILY_EXCEEDED"})
pd.DataFrame(hist).to_csv(TASK / "historical_violations.csv", index=False)

# quota_overrides — OVR001: T001 daily_limit=50 (Jan2-3 only), OVR002: T019 (future, doesn't apply)
quota_overrides = pd.DataFrame([
    {"override_id": "OVR001", "tenant_id": "T001", "effective_start": "2025-01-02", "effective_end": "2025-01-03",
     "override_field": "daily_request_limit", "override_value": 50, "reason": "Trial expansion"},
    {"override_id": "OVR002", "tenant_id": "T007", "effective_start": "2025-01-02", "effective_end": "2025-01-02",
     "override_field": "hourly_request_limit", "override_value": 40, "reason": "Campaign day boost"},
    {"override_id": "OVR003", "tenant_id": "T019", "effective_start": "2025-01-15", "effective_end": "2025-01-31",
     "override_field": "daily_request_limit", "override_value": 8000, "reason": "Future contract"},
])
quota_overrides.to_csv(TASK / "quota_overrides.csv", index=False)

# api_requests.csv
ALL_TIDS = [f"T{i:03d}" for i in range(1, 21)]
EPS_POOL = ["EP001", "EP002", "EP003", "EP004", "EP005", "EP006", "EP007", "EP008", "EP009", "EP010"]
requests = []
req_id = 1

def make_req(tenant_id, endpoint_id, timestamp, tokens_used=None, response_code=200):
    global req_id
    r = {"request_id": f"R{req_id:08d}", "tenant_id": tenant_id, "endpoint_id": endpoint_id,
         "timestamp": timestamp, "tokens_used": tokens_used, "response_code": response_code}
    req_id += 1
    return r

# Normal background traffic: ~5 per tenant per day, spread across day
for day in range(1, 4):
    ds = f"2025-01-{day:02d}"
    for tid in ALL_TIDS:
        for _ in range(5):
            h = rng.integers(1, 23)
            m = rng.integers(0, 60)
            s = rng.integers(0, 60)
            ts = f"{ds} {h:02d}:{m:02d}:{s:02d}"
            ep = EPS_POOL[rng.integers(0, len(EPS_POOL))]
            tok = None if rng.random() < 0.12 else int(rng.integers(50, 1500))
            rc = 200 if rng.random() < 0.87 else (400 if rng.random() < 0.65 else 500)
            requests.append(make_req(tid, ep, ts, tok, rc))

# T1-CASE explicit: T001 FREE + EP001 (read, weight=0.5)
# Correct: round(400*0.5*1.0)=200  Wrong (no case-norm): round(400*1.0*1.0)=400
requests.append(make_req("T001", "EP001", "2025-01-01 07:00:00", 400))

# T1 (FREE+BATCH missing->1.0): T001 uses EP004 BATCH
requests.append(make_req("T001", "EP004", "2025-01-01 08:00:00", 3000))
requests.append(make_req("T001", "EP004", "2025-01-01 08:05:00", 2500))

# T1 (ENTERPRISE+READ missing->1.0): T019 uses EP001 READ
requests.append(make_req("T019", "EP001", "2025-01-01 09:00:00", 800))
requests.append(make_req("T019", "EP003", "2025-01-01 09:10:00", 600))

# T1 (PROFESSIONAL+ADMIN missing->1.0): T010 uses EP008 ADMIN
requests.append(make_req("T010", "EP008", "2025-01-01 10:00:00", 500))

# T6 TRAP: WRITE endpoint at exact SW001 boundary
# SW001 covers [2025-01-01 00:00:00, 2025-01-02 00:00:00) — end is EXCLUSIVE
requests.append(make_req("T002", "EP002", "2025-01-01 23:59:59", 1000))  # surge=1.5 applies
requests.append(make_req("T002", "EP002", "2025-01-02 00:00:00", 1000))  # surge=1.0 (boundary, T6 trap)
requests.append(make_req("T002", "EP002", "2025-01-02 00:00:01", 1000))  # surge=1.0

# SW002 BATCH surge (Jan2 06:00 - Jan3 18:00, multiplier=1.4)
for i in range(6):
    h = rng.integers(7, 17)
    m = rng.integers(0, 60)
    ts = f"2025-01-02 {h:02d}:{m:02d}:00"
    requests.append(make_req("T015", "EP004", ts, int(rng.integers(2000, 6000))))

# T001 (FREE, hourly=10) violations: 14 requests in hour 10 on Jan1
# SLA(FREE, WED)=1.0 -> eff_hourly=10, burst_thresh=floor(10*1.2)=12
# Requests 11,12=BURST; 13,14=NO (2 violations)
for i in range(14):
    m = i * 3
    ts = f"2025-01-01 10:{m:02d}:00"
    ep = "EP001"
    requests.append(make_req("T001", ep, ts, 200))

# T007 (STARTER, hourly=25, 3 historical) violations on Jan1 (WED)
# SLA(STARTER,WED)=0.9 -> eff_hourly=floor(25*0.9)=22, burst_thresh=floor(22*1.3)=28
# 32 requests in hour 14: 1-22=YES, 23-28=BURST, 29-32=NO (4 violations)
# T7 trap: use BATCH EP004 with large tokens so pct component hits cap
# NOTICE rule: flat=5, pct=1.5/1k, cap_mult=0.6 -> cap=5*0.6=3.0
# tokens=2500: pct=1.5*2.5=3.75 > cap=3.0 -> penalty=5+3=8.0 (T7 triggered)
for i in range(32):
    m = i
    s = rng.integers(0, 58)
    ts = f"2025-01-01 14:{m:02d}:{s:02d}" if m < 60 else f"2025-01-01 14:59:{s:02d}"
    requests.append(make_req("T007", "EP004", ts, 2500))

# T011 (PROFESSIONAL, daily=300, 11 historical) violations on Jan2 (THU)
# SLA(PROFESSIONAL,THU)=1.1 -> eff_hourly=floor(75*1.1)=82, burst_thresh=floor(82*1.2)=98
# Daily limit=300. Send 320 total on Jan2 to create daily violations.
# Normal 5 already included above, add 315 more spread across the day
# CAUTION rule: flat=20, pct=3.0/1k, cap_mult=0.4 -> cap=20*0.4=8.0
# tokens=5000 (EP010 export): pct=3.0*5=15.0 > cap=8.0 -> penalty=20+8=28.0 (T7 triggered)
for i in range(315):
    h = rng.integers(0, 24)
    m = rng.integers(0, 60)
    s = rng.integers(0, 60)
    ts = f"2025-01-02 {h:02d}:{m:02d}:{s:02d}"
    ep = "EP010" if i % 5 == 0 else rng.choice(["EP001","EP002","EP003","EP004"])
    tok = 5000 if ep == "EP010" else int(rng.integers(200, 2000))
    requests.append(make_req("T011", ep, ts, tok))

# OVR002: T007 on Jan2 has hourly_request_limit overridden to 40
# SLA(STARTER,THU)=0.9 -> eff_hourly=floor(40*0.9)=36, burst_thresh=floor(36*1.3)=46
# Add 50 requests for T007 in hour 9 on Jan2 to create more violations
for i in range(50):
    m = rng.integers(0, 60)
    s = rng.integers(0, 60)
    ts = f"2025-01-02 09:{m:02d}:{s:02d}"
    requests.append(make_req("T007", "EP002", ts, int(rng.integers(300, 800))))

req_df = pd.DataFrame(requests).sort_values(["timestamp", "request_id"]).reset_index(drop=True)
req_df.to_csv(TASK / "api_requests.csv", index=False)
print(f"api_requests.csv: {len(req_df)} rows")
print("All task files generated.")

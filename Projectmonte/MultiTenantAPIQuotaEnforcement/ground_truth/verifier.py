import sys
import pandas as pd
from pathlib import Path

TASK = Path(__file__).parent.parent / "task"
GT   = Path(__file__).parent

errors = []
_total = [0]


def chk(ok, msg):
    _total[0] += 1
    if not ok:
        errors.append(f"[FAIL] {msg}")


def near(a, b, tol=0.01):
    if b == 0:
        return abs(a) <= tol
    return abs(a - b) / max(abs(b), 1e-9) <= tol


def load(path, name):
    try:
        return pd.read_csv(path)
    except Exception as e:
        errors.append(f"[FAIL] Cannot load {name}: {e}")
        return pd.DataFrame()


def solve(sol_dir=None):
    src = Path(sol_dir) if sol_dir else TASK

    ra = load(src / "request_attribution.csv",  "request_attribution.csv")
    tc = load(src / "tenant_consumption.csv",   "tenant_consumption.csv")
    vl = load(src / "violation_ledger.csv",     "violation_ledger.csv")
    td = load(src / "throttle_decisions.csv",   "throttle_decisions.csv")
    bs = load(src / "billing_summary.csv",      "billing_summary.csv")

    g_ra = load(GT / "request_attribution.csv", "golden request_attribution")
    g_tc = load(GT / "tenant_consumption.csv",  "golden tenant_consumption")
    g_vl = load(GT / "violation_ledger.csv",    "golden violation_ledger")
    g_td = load(GT / "throttle_decisions.csv",  "golden throttle_decisions")
    g_bs = load(GT / "billing_summary.csv",     "golden billing_summary")

    RA_COLS = ["request_id", "tenant_id", "endpoint_id", "timestamp", "base_tokens",
               "weight_multiplier", "surge_multiplier", "effective_tokens",
               "hourly_request_count", "daily_request_count", "daily_token_count",
               "effective_hourly_limit", "within_limit", "throttle_reason", "billable"]
    TC_COLS = ["tenant_id", "tier_id", "quota_bucket", "request_count", "tokens_consumed",
               "total_cost", "request_limit", "token_limit", "request_pct", "token_pct",
               "quota_used_pct", "status", "violation_count", "overage_requests", "overage_tokens"]
    VL_COLS = ["violation_id", "tenant_id", "request_id", "violation_timestamp",
               "violation_type", "prior_violation_count", "violation_number", "severity",
               "flat_penalty_usd", "pct_penalty_per_1k_tokens", "penalty_amount",
               "cumulative_penalty", "enforcement_action"]
    TD_COLS = ["request_id", "tenant_id", "tier_id", "timestamp", "throttle_reason",
               "hourly_request_count", "daily_request_count", "effective_hourly_limit",
               "decision", "decision_reason"]
    BS_COLS = ["tenant_id", "tier_id", "billing_period", "total_requests", "total_tokens",
               "overage_requests", "overage_tokens", "base_cost", "overage_cost",
               "penalty_cost", "total_due", "violation_count"]

    for fname, df, cols in [
        ("request_attribution.csv", ra, RA_COLS),
        ("tenant_consumption.csv",  tc, TC_COLS),
        ("violation_ledger.csv",    vl, VL_COLS),
        ("throttle_decisions.csv",  td, TD_COLS),
        ("billing_summary.csv",     bs, BS_COLS),
    ]:
        for col in cols:
            chk(not df.empty and col in df.columns, f"{fname} missing column '{col}'")

    chk(len(ra) == 726,            f"request_attribution row count {len(ra)} != 726")
    chk(len(tc) == 60,             f"tenant_consumption row count {len(tc)} != 60")
    chk(abs(len(vl) - 366) <= 20, f"violation_ledger row count {len(vl)} expected ~366")
    chk(abs(len(td) - 366) <= 20, f"throttle_decisions row count {len(td)} expected ~366")
    chk(len(bs) == 20,             f"billing_summary row count {len(bs)} != 20")

    if not ra.empty and "effective_tokens" in ra.columns and "weight_multiplier" in ra.columns:
        t1 = ra[(ra["tenant_id"] == "T001") & (ra["endpoint_id"] == "EP001") &
                (ra["timestamp"] == "2025-01-01 07:00:00")]
        chk(len(t1) == 1,
            "T1: T001/EP001/07:00:00 row exists in request_attribution")
        if len(t1) == 1:
            chk(float(t1.iloc[0]["weight_multiplier"]) == 0.5,
                f"T1: T001/EP001 weight_multiplier=0.5 (wrong=1.0 without case-norm), got {t1.iloc[0]['weight_multiplier']}")
            chk(int(t1.iloc[0]["effective_tokens"]) == 200,
                f"T1: T001/EP001/07:00 effective_tokens=200 (wrong=400 without case-norm), got {t1.iloc[0]['effective_tokens']}")

    if not ra.empty:
        t2a = ra[(ra["tenant_id"] == "T001") & (ra["endpoint_id"] == "EP004") &
                 (ra["timestamp"] == "2025-01-01 08:00:00")]
        chk(len(t2a) == 1, "T2: T001/EP004/08:00:00 row exists")
        if len(t2a) == 1:
            chk(float(t2a.iloc[0]["weight_multiplier"]) == 1.0,
                f"T2: FREE+BATCH missing combo default=1.0 (wrong=0.0), got {t2a.iloc[0]['weight_multiplier']}")
            chk(int(t2a.iloc[0]["effective_tokens"]) == 3000,
                f"T2: T001/EP004/08:00 effective_tokens=3000 (wrong=0 if default 0), got {t2a.iloc[0]['effective_tokens']}")

        t2b = ra[(ra["tenant_id"] == "T019") & (ra["endpoint_id"] == "EP001") &
                 (ra["timestamp"] == "2025-01-01 09:00:00")]
        chk(len(t2b) == 1, "T2: T019/EP001/09:00:00 row exists")
        if len(t2b) == 1:
            chk(float(t2b.iloc[0]["weight_multiplier"]) == 1.0,
                f"T2: ENTERPRISE+READ missing combo default=1.0 (wrong=0.0), got {t2b.iloc[0]['weight_multiplier']}")
            chk(int(t2b.iloc[0]["effective_tokens"]) == 800,
                f"T2: T019/EP001/09:00 effective_tokens=800 (wrong=0), got {t2b.iloc[0]['effective_tokens']}")

    if not ra.empty and "surge_multiplier" in ra.columns:
        t6_pre = ra[(ra["tenant_id"] == "T002") & (ra["endpoint_id"] == "EP002") &
                    (ra["timestamp"] == "2025-01-01 23:59:59")]
        t6_bnd = ra[(ra["tenant_id"] == "T002") & (ra["endpoint_id"] == "EP002") &
                    (ra["timestamp"] == "2025-01-02 00:00:00")]
        chk(len(t6_pre) == 1, "T6: T002/EP002/23:59:59 row exists (inside surge window)")
        chk(len(t6_bnd) == 1, "T6: T002/EP002/00:00:00 row exists (exclusive end boundary)")
        if len(t6_pre) == 1:
            chk(float(t6_pre.iloc[0]["surge_multiplier"]) == 1.5,
                f"T6: surge at 23:59:59=1.5 (inside SW001), got {t6_pre.iloc[0]['surge_multiplier']}")
            chk(int(t6_pre.iloc[0]["effective_tokens"]) == 3000,
                f"T6: 23:59:59 effective_tokens=3000, got {t6_pre.iloc[0]['effective_tokens']}")
        if len(t6_bnd) == 1:
            chk(float(t6_bnd.iloc[0]["surge_multiplier"]) == 1.0,
                f"T6: surge AT boundary 00:00:00=1.0 (end is EXCLUSIVE), got {t6_bnd.iloc[0]['surge_multiplier']}")
            chk(int(t6_bnd.iloc[0]["effective_tokens"]) == 2000,
                f"T6: 00:00:00 boundary effective_tokens=2000 (wrong=3000 if inclusive end), got {t6_bnd.iloc[0]['effective_tokens']}")

    if not vl.empty and "prior_violation_count" in vl.columns and "severity" in vl.columns:
        vl7 = vl[vl["tenant_id"] == "T007"].sort_values("violation_timestamp").reset_index(drop=True)
        chk(len(vl7) > 0, "T4: T007 appears in violation_ledger")
        if len(vl7) > 0:
            chk(int(vl7.iloc[0]["prior_violation_count"]) == 3,
                f"T4: T007 first prior_violation_count=3 (3 historical; wrong=0 if ignored), got {vl7.iloc[0]['prior_violation_count']}")
            chk(int(vl7.iloc[0]["violation_number"]) == 4,
                f"T4: T007 first violation_number=4 (wrong=1), got {vl7.iloc[0]['violation_number']}")
            chk(str(vl7.iloc[0]["severity"]) == "NOTICE",
                f"T4: T007 first severity=NOTICE (wrong=WARNING if no hist), got {vl7.iloc[0]['severity']}")
            chk(near(float(vl7.iloc[0]["penalty_amount"]), 8.0),
                f"T4/T7: T007 first penalty=8.0 (flat=5+min(3.75,3)=8; wrong if WARNING), got {vl7.iloc[0]['penalty_amount']}")

        vl11 = vl[vl["tenant_id"] == "T011"].sort_values("violation_timestamp").reset_index(drop=True)
        chk(len(vl11) > 0, "T4: T011 appears in violation_ledger")
        if len(vl11) > 0:
            chk(int(vl11.iloc[0]["prior_violation_count"]) == 11,
                f"T4: T011 first prior_violation_count=11 (11 historical), got {vl11.iloc[0]['prior_violation_count']}")
            chk(int(vl11.iloc[0]["violation_number"]) == 12,
                f"T4: T011 first violation_number=12 (wrong=1), got {vl11.iloc[0]['violation_number']}")
            chk(str(vl11.iloc[0]["severity"]) == "CAUTION",
                f"T4: T011 first severity=CAUTION (wrong=WARNING), got {vl11.iloc[0]['severity']}")
            chk(near(float(vl11.iloc[0]["penalty_amount"]), 28.0),
                f"T4/T7: T011 first penalty=28.0 (flat=20+min(15,8)=28; no-cap gives 35), got {vl11.iloc[0]['penalty_amount']}")

    if not ra.empty and "effective_hourly_limit" in ra.columns:
        t11_j2 = ra[(ra["tenant_id"] == "T011") & (ra["timestamp"].str.startswith("2025-01-02"))]
        chk(len(t11_j2) > 0, "T5: T011 has Jan2 requests")
        if len(t11_j2) > 0:
            vals = t11_j2["effective_hourly_limit"].unique()
            chk(all(int(v) == 82 for v in vals),
                f"T5: T011 Jan2 eff_hourly=82 (active_policies THU=1.1×75; deprecated gives 53), got {vals}")

        t11_j1 = ra[(ra["tenant_id"] == "T011") & (ra["timestamp"].str.startswith("2025-01-01"))]
        if len(t11_j1) > 0:
            chk(int(t11_j1.iloc[0]["effective_hourly_limit"]) == 82,
                f"T5: T011 Jan1 eff_hourly=82 (WED=1.1×75), got {t11_j1.iloc[0]['effective_hourly_limit']}")

    if not vl.empty and "penalty_amount" in vl.columns:
        notice_rows = vl[(vl["tenant_id"] == "T007") & (vl["severity"] == "NOTICE")]
        chk(len(notice_rows) > 0, "T7: T007 has NOTICE-severity violations")
        if len(notice_rows) > 0:
            chk(near(float(notice_rows.iloc[0]["penalty_amount"]), 8.0),
                f"T7: T007 NOTICE penalty=5+min(3.75,3.0)=8.0 (no-cap gives 12.5), got {notice_rows.iloc[0]['penalty_amount']}")

        caution_12 = vl[(vl["tenant_id"] == "T011") & (vl["severity"] == "CAUTION") &
                        (vl["violation_number"].astype(int) == 12)]
        chk(len(caution_12) > 0, "T7: T011 violation_number=12 with CAUTION exists")
        if len(caution_12) > 0:
            chk(near(float(caution_12.iloc[0]["penalty_amount"]), 28.0),
                f"T7: T011 CAUTION#12 penalty=20+min(15,8)=28.0 (no-cap gives 35), got {caution_12.iloc[0]['penalty_amount']}")

    if not td.empty and "decision" in td.columns and "decision_reason" in td.columns:
        td3 = td[td["tenant_id"] == "T003"]
        chk(len(td3) > 0, "Throttle: T003 (exempt) has decisions")
        if len(td3) > 0:
            chk((td3["decision"] == "ALLOW").all(),
                f"Throttle: T003 all decisions=ALLOW (exempt_from_throttling), got {td3['decision'].unique()}")
            chk((td3["decision_reason"] == "EXEMPT_TENANT").all(),
                f"Throttle: T003 all reasons=EXEMPT_TENANT, got {td3['decision_reason'].unique()}")

        td1 = td[td["tenant_id"] == "T001"].sort_values("timestamp").reset_index(drop=True)
        chk(len(td1) > 1, "Throttle: T001 has multiple decisions (HARD_BLOCK tier)")
        if len(td1) > 1:
            chk("BLOCK" in td1["decision"].values,
                f"Throttle: T001 must have BLOCK (FREE=HARD_BLOCK tier), got {td1['decision'].unique()}")
            chk(td1.iloc[-1]["decision"] == "BLOCK",
                f"Throttle: T001 last decision=BLOCK, got {td1.iloc[-1]['decision']}")

        td7 = td[td["tenant_id"] == "T007"].sort_values("timestamp").reset_index(drop=True)
        chk(len(td7) > 0, "Throttle: T007 has decisions")
        if len(td7) > 0:
            chk(td7.iloc[0]["decision"] == "GRACE",
                f"Throttle: T007 first decision=GRACE (5min grace period), got {td7.iloc[0]['decision']}")
            chk(str(td7.iloc[0]["decision_reason"]).upper() == "WITHIN_GRACE_5MIN",
                f"Throttle: T007 grace reason should be WITHIN_GRACE_5MIN (case-insensitive MIN/min), got {td7.iloc[0]['decision_reason']}")

        td11 = td[td["tenant_id"] == "T011"].sort_values("timestamp").reset_index(drop=True)
        chk(len(td11) > 0, "Throttle: T011 has decisions")
        if len(td11) > 0:
            post_grace = td11[td11["decision"] != "GRACE"]
            chk(len(post_grace) > 0, "Throttle: T011 has post-grace SOFT_THROTTLE decisions")
            if len(post_grace) > 0:
                valid = {"THROTTLE_50", "THROTTLE_90", "BLOCK"}
                chk(set(post_grace["decision"].unique()).issubset(valid),
                    f"Throttle: T011 post-grace decisions valid SOFT_THROTTLE outcomes, got {post_grace['decision'].unique()}")

    if not ra.empty and not g_ra.empty and "request_id" in ra.columns and "request_id" in g_ra.columns:
        m = pd.merge(ra, g_ra, on="request_id", suffixes=("_s", "_g"))
        if len(m) > 0:
            tok = ((pd.to_numeric(m["effective_tokens_s"], errors="coerce") -
                    pd.to_numeric(m["effective_tokens_g"], errors="coerce")).abs() < 0.001).mean()
            chk(tok >= 0.98, f"Golden RA: effective_tokens >=98% exact match ({tok*100:.1f}%)")
            chk(tok == 1.0,  f"Golden RA: effective_tokens 100% exact match ({tok*100:.1f}%)")
            lim = (m["within_limit_s"] == m["within_limit_g"]).mean()
            chk(lim >= 0.97, f"Golden RA: within_limit >=97% match ({lim*100:.1f}%)")
            chk(lim == 1.0,  f"Golden RA: within_limit 100% match ({lim*100:.1f}%)")
            hl  = (m["effective_hourly_limit_s"] == m["effective_hourly_limit_g"]).mean()
            chk(hl >= 0.97,  f"Golden RA: effective_hourly_limit >=97% match ({hl*100:.1f}%)")
            bl  = (m["billable_s"] == m["billable_g"]).mean()
            chk(bl >= 0.97,  f"Golden RA: billable >=97% match ({bl*100:.1f}%)")
            hc  = (m["hourly_request_count_s"] == m["hourly_request_count_g"]).mean()
            chk(hc >= 0.97,  f"Golden RA: hourly_request_count >=97% match ({hc*100:.1f}%)")

    if not tc.empty and not g_tc.empty and "quota_bucket" in tc.columns and "quota_bucket" in g_tc.columns:
        m = pd.merge(tc, g_tc, on=["tenant_id", "quota_bucket"], suffixes=("_s", "_g"))
        if len(m) > 0:
            rc = (m["request_count_s"] == m["request_count_g"]).mean()
            chk(rc >= 0.97, f"Golden TC: request_count >=97% ({rc*100:.1f}%)")
            chk(rc == 1.0,  f"Golden TC: request_count 100% ({rc*100:.1f}%)")
            st = (m["status_s"] == m["status_g"]).mean()
            chk(st >= 0.90, f"Golden TC: status >=90% ({st*100:.1f}%)")
            tk = ((m["tokens_consumed_s"] - m["tokens_consumed_g"]).abs() <= 1).mean()
            chk(tk >= 0.97, f"Golden TC: tokens_consumed >=97% ({tk*100:.1f}%)")

    if not vl.empty and not g_vl.empty and "request_id" in vl.columns and "request_id" in g_vl.columns:
        chk(abs(len(vl) - len(g_vl)) <= 20,
            f"Golden VL: row count within 20 of golden ({len(vl)} vs {len(g_vl)})")
        m = pd.merge(vl, g_vl, on="request_id", suffixes=("_s", "_g"))
        if len(m) > 0:
            sv = (m["severity_s"] == m["severity_g"]).mean()
            chk(sv >= 0.90, f"Golden VL: severity >=90% match ({sv*100:.1f}%)")
            pn = ((m["penalty_amount_s"] - m["penalty_amount_g"]).abs() < 0.01).mean()
            chk(pn >= 0.90, f"Golden VL: penalty_amount >=90% exact match ({pn*100:.1f}%)")

    if not bs.empty and not g_bs.empty:
        m = pd.merge(bs, g_bs, on="tenant_id", suffixes=("_s", "_g"))
        if len(m) > 0:
            bc = ((m["base_cost_s"] - m["base_cost_g"]).abs() < 0.01).mean()
            chk(bc >= 0.95, f"Golden BS: base_cost >=95% exact ({bc*100:.1f}%)")
            chk(bc == 1.0,  f"Golden BS: base_cost 100% exact ({bc*100:.1f}%)")
            td_m = ((m["total_due_s"] - m["total_due_g"]).abs() < 1.0).mean()
            chk(td_m >= 0.90, f"Golden BS: total_due >=90% within $1 ({td_m*100:.1f}%)")
            pc = ((m["penalty_cost_s"] - m["penalty_cost_g"]).abs() < 1.0).mean()
            chk(pc >= 0.85,  f"Golden BS: penalty_cost >=85% within $1 ({pc*100:.1f}%)")

    if not ra.empty and not tc.empty and "quota_bucket" in tc.columns:
        ra2 = ra.copy()
        ra2["quota_bucket"] = pd.to_datetime(ra2["timestamp"]).dt.strftime("%Y-%m-%d")
        ra_cnt = ra2.groupby(["tenant_id", "quota_bucket"])["request_id"].count().reset_index(name="ra_count")
        mc = pd.merge(ra_cnt, tc[["tenant_id", "quota_bucket", "request_count"]], on=["tenant_id", "quota_bucket"])
        if len(mc) > 0:
            match = (mc["ra_count"] == mc["request_count"]).mean()
            chk(match >= 0.97, f"Cross-file: request counts RA vs TC ({match*100:.1f}%)")

    if not vl.empty and not td.empty and "request_id" in vl.columns and "request_id" in td.columns:
        vl_ids = set(vl["request_id"])
        td_ids = set(td["request_id"])
        chk(len(vl_ids - td_ids) == 0,
            f"Cross-file: all VL request_ids in TD ({len(vl_ids - td_ids)} missing)")
        chk(len(td_ids - vl_ids) == 0,
            f"Cross-file: all TD request_ids in VL ({len(td_ids - vl_ids)} extra)")

    if not ra.empty:
        chk((ra["effective_tokens"] >= 0).all(), "Sanity: effective_tokens >= 0")
        chk(ra["within_limit"].isin(["YES", "BURST", "NO"]).all(), "Sanity: within_limit in {YES,BURST,NO}")
        chk(ra["billable"].isin(["YES", "NO"]).all(), "Sanity: billable in {YES,NO}")

    if not vl.empty and "violation_id" in vl.columns:
        chk(vl["violation_id"].str.match(r"VN\d{6}").all(), "Sanity: violation_id format VN######")
        chk(vl["severity"].isin(["WARNING", "NOTICE", "CAUTION", "STRICT", "CRITICAL"]).all(),
            "Sanity: severity values valid")

    if not tc.empty and "status" in tc.columns:
        chk(tc["status"].isin(["NORMAL", "WARNING", "OVERAGE", "THROTTLED"]).all(),
            "Sanity: tenant_consumption status values valid")

    if not bs.empty:
        chk((bs["total_due"] >= 0).all(), "Sanity: total_due >= 0")
        chk(bs["billing_period"].eq("2025-01-01 to 2025-01-03").all(),
            "Sanity: billing_period = '2025-01-01 to 2025-01-03'")


if __name__ == "__main__":
    import sys
    sol = sys.argv[1] if len(sys.argv) > 1 else None
    solve(sol)
    n_errors = len(errors)
    score = _total[0] - n_errors
    print(f"\n{'='*60}")
    print(f"MultiTenantAPIQuotaEnforcement — Verifier Results")
    print(f"{'='*60}")
    if errors:
        for e in errors:
            print(e)
    else:
        print("  All checks passed.")
    print(f"{'='*60}")
    print(f"Score: {score}/{_total[0]}")
    sys.exit(0 if n_errors == 0 else 1)

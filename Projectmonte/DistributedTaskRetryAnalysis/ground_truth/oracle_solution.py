import pandas as pd
import numpy as np
import math
from pathlib import Path
from datetime import datetime, timedelta

TASK_DIR = Path(__file__).parent.parent / "task"


def load_data():
    pools = pd.read_csv(TASK_DIR / "pools.csv")
    tenants = pd.read_csv(TASK_DIR / "tenants.csv")
    retry_configs = pd.read_csv(TASK_DIR / "retry_configs.csv")
    sla_terms = pd.read_csv(TASK_DIR / "sla_terms.csv")
    rate_cards = pd.read_csv(TASK_DIR / "rate_cards.csv")
    incidents = pd.read_csv(TASK_DIR / "incidents.csv")
    jobs = pd.read_csv(TASK_DIR / "jobs.csv")
    attempts = pd.read_csv(TASK_DIR / "attempts.csv")
    return pools, tenants, retry_configs, sla_terms, rate_cards, incidents, jobs, attempts


def clean_attempts(attempts_df, jobs_df):
    df = attempts_df.copy()

    df["ended_at"] = df["ended_at"].replace("", np.nan)
    df["cpu_ms"] = df["cpu_ms"].replace("", np.nan)
    df["mem_peak_mb"] = df["mem_peak_mb"].replace("", np.nan)
    df["error_code"] = df["error_code"].replace("", np.nan)

    df["queued_at"] = pd.to_datetime(df["queued_at"])
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce")

    job_sla = dict(zip(jobs_df["job_id"], jobs_df["sla_seconds"]))
    for idx in df.index:
        if pd.isna(df.loc[idx, "ended_at"]):
            sla = job_sla.get(df.loc[idx, "job_id"], 120)
            df.loc[idx, "ended_at"] = df.loc[idx, "started_at"] + pd.Timedelta(seconds=sla)

    df["cpu_ms"] = pd.to_numeric(df["cpu_ms"], errors="coerce").fillna(0).astype(int)
    df["mem_peak_mb"] = pd.to_numeric(df["mem_peak_mb"], errors="coerce").fillna(0.0)

    for idx in df.index:
        if df.loc[idx, "status"] == "FAIL" and pd.isna(df.loc[idx, "error_code"]):
            df.loc[idx, "error_code"] = "UNKNOWN"

    df["error_code"] = df["error_code"].fillna("")
    df["duration_sec"] = (df["ended_at"] - df["started_at"]).dt.total_seconds()
    df["duration_sec"] = df["duration_sec"].clip(lower=0)

    return df


def check_incident_overlap(attempt_row, incidents_df, pools_df):
    pool_id = attempt_row["pool_id"]
    started = attempt_row["started_at"]
    ended = attempt_row["ended_at"]

    overlaps = []
    pool_incidents = incidents_df[incidents_df["pool_id"] == pool_id]
    for _, inc in pool_incidents.iterrows():
        inc_start = pd.to_datetime(inc["start_time"])
        inc_end = pd.to_datetime(inc["end_time"])
        if started < inc_end and ended > inc_start:
            overlaps.append((inc["incident_id"], inc["severity"], inc["capacity_impact_pct"]))
    return overlaps


def get_retry_config(sl, priority, attempt_date, retry_configs_df):
    configs = retry_configs_df[
        (retry_configs_df["service_level"] == sl) &
        (retry_configs_df["priority"] == priority)
    ]
    for _, cfg in configs.iterrows():
        eff_from = pd.to_datetime(cfg["effective_from"])
        eff_to = pd.to_datetime(cfg["effective_to"])
        if eff_from <= attempt_date <= eff_to:
            return cfg
    return None


def compute_expected_delay(cfg, attempt_num, prev_delay):
    if cfg is None:
        return 0.0
    if attempt_num <= 1:
        return 0.0

    bt = cfg["backoff_type"]
    base = cfg["base_delay_sec"]
    max_d = cfg["max_delay_sec"]

    if bt == "FIXED":
        delay = base
    elif bt == "LINEAR":
        delay = base * attempt_num
    elif bt == "EXPONENTIAL":
        delay = base * (2 ** (attempt_num - 2))
    elif bt == "DECORRELATED":
        delay = min(max_d, base + (prev_delay if prev_delay > 0 else base) * 1.5)
    else:
        delay = base

    delay = min(delay, max_d)
    delay = round(delay, 2)
    return delay


def compute_job_metrics(jobs_df, attempts_df, tenants_df, pools_df,
                        incidents_df, retry_configs_df, sla_terms_df):
    tenant_sl = dict(zip(tenants_df["tenant_id"], tenants_df["service_level"]))
    tenant_region = dict(zip(tenants_df["tenant_id"], tenants_df["region"]))

    results = []

    for _, job in jobs_df.iterrows():
        job_id = job["job_id"]
        tenant_id = job["tenant_id"]
        priority = job["priority"]
        submitted = pd.to_datetime(job["submitted_at"])
        sla_sec = job["sla_seconds"]
        max_retries = job["max_retries"]
        sl = tenant_sl[tenant_id]

        job_attempts = attempts_df[attempts_df["job_id"] == job_id].sort_values("attempt_num")
        total_attempts = len(job_attempts)

        final_status = job_attempts.iloc[-1]["status"] if total_attempts > 0 else "UNKNOWN"
        final_ended = job_attempts.iloc[-1]["ended_at"] if total_attempts > 0 else submitted

        total_duration = (final_ended - submitted).total_seconds()
        total_cpu = job_attempts["cpu_ms"].sum()
        total_mem = round(job_attempts["mem_peak_mb"].max(), 1) if len(job_attempts) > 0 else 0.0
        total_exec = round(job_attempts["duration_sec"].sum(), 2)

        sla_breached = "YES" if total_duration > sla_sec else "NO"
        breach_sec = round(max(0, total_duration - sla_sec), 2) if sla_breached == "YES" else 0.0

        incident_affected = "NO"
        highest_severity = None
        severity_rank = {"S1": 4, "S2": 3, "S3": 2, "S4": 1}
        for _, att in job_attempts.iterrows():
            overlaps = check_incident_overlap(att, incidents_df, pools_df)
            for inc_id, inc_sev, inc_impact in overlaps:
                incident_affected = "YES"
                if highest_severity is None or severity_rank.get(inc_sev, 0) > severity_rank.get(highest_severity, 0):
                    highest_severity = inc_sev

        retry_count = max(0, total_attempts - 1)
        prev_delay = 0.0
        total_expected_delay = 0.0
        for i, (_, att) in enumerate(job_attempts.iterrows()):
            anum = att["attempt_num"]
            if anum <= 1:
                prev_delay = 0.0
                continue
            cfg = get_retry_config(sl, priority, att["queued_at"], retry_configs_df)
            ed = compute_expected_delay(cfg, anum, prev_delay)
            total_expected_delay += ed
            prev_delay = ed

        total_expected_delay = round(total_expected_delay, 2)

        actual_wait = 0.0
        prev_ended_at = None
        for _, att in job_attempts.iterrows():
            if prev_ended_at is not None:
                wait = (att["queued_at"] - prev_ended_at).total_seconds()
                actual_wait += max(0, wait)
            prev_ended_at = att["ended_at"]
        actual_wait = round(actual_wait, 2)

        delay_deviation = round(actual_wait - total_expected_delay, 2)

        sla_row = sla_terms_df[
            (sla_terms_df["service_level"] == sl) &
            (sla_terms_df["priority"] == priority)
        ]
        penalty_per_min = sla_row.iloc[0]["penalty_per_min"] if len(sla_row) > 0 else 0.0
        escalation_pct = sla_row.iloc[0]["escalation_pct"] if len(sla_row) > 0 else 0.0
        critical_mult = sla_row.iloc[0]["critical_mult"] if len(sla_row) > 0 else 1.0
        grace_sec = sla_row.iloc[0]["grace_sec"] if len(sla_row) > 0 else 0.0

        if sla_breached == "YES":
            effective_breach = max(0, breach_sec - grace_sec)

            if effective_breach == 0:
                penalty_tier = "GRACE"
            elif breach_sec <= sla_sec * 1.5:
                penalty_tier = "STANDARD"
            elif breach_sec <= sla_sec * 2:
                penalty_tier = "SEVERE"
            else:
                penalty_tier = "CRITICAL"

            breach_min = effective_breach / 60.0
            base_penalty = round(breach_min * penalty_per_min, 4)

            escalation_factor = 1.0 + escalation_pct * retry_count
            escalated_penalty = round(base_penalty * escalation_factor, 4)

            if retry_count > max_retries // 2:
                escalated_penalty = round(escalated_penalty * 1.15, 4)

            sl_mult = {"PLATINUM": 0.80, "GOLD": 0.95, "SILVER": 1.10, "BRONZE": 1.25}
            penalty_val = round(escalated_penalty * sl_mult.get(sl, 1.0), 4)

            incident_discount = {"S1": 0.60, "S2": 0.40, "S3": 0.20, "S4": 0.10}
            if incident_affected == "YES" and highest_severity:
                discount = incident_discount.get(highest_severity, 0.0)
                penalty_val = round(penalty_val * (1 - discount), 4)

            if breach_sec > sla_sec * 2:
                penalty_val = round(penalty_val * critical_mult, 4)

            final_penalty = round(penalty_val, 2)
        else:
            penalty_tier = "NONE"
            final_penalty = 0.0

        if sla_breached == "YES" and breach_sec > 0:
            penalty_weight = round(final_penalty * 60.0 / max(0.01, breach_sec), 4)
        else:
            penalty_weight = 0.0

        error_codes = job_attempts[job_attempts["error_code"] != ""]["error_code"].tolist()
        unique_errors = sorted(set(error_codes))
        error_summary = "|".join(unique_errors) if unique_errors else ""

        if final_status == "OK" and sla_breached == "NO":
            outcome_class = "SUCCESS"
        elif final_status == "OK" and sla_breached == "YES":
            outcome_class = "DELAYED_SUCCESS"
        elif retry_count >= max_retries:
            outcome_class = "EXHAUSTED"
        elif incident_affected == "YES":
            outcome_class = "INCIDENT_FAILURE"
        elif final_status == "TIMEOUT":
            outcome_class = "TIMEOUT_FAILURE"
        else:
            outcome_class = "ERROR_FAILURE"

        results.append({
            "job_id": job_id,
            "tenant_id": tenant_id,
            "priority": priority,
            "total_attempts": total_attempts,
            "retry_count": retry_count,
            "final_status": final_status,
            "total_duration_sec": round(total_duration, 2),
            "total_exec_sec": total_exec,
            "total_cpu_ms": total_cpu,
            "peak_mem_mb": total_mem,
            "sla_breached": sla_breached,
            "breach_seconds": breach_sec,
            "expected_delay_sec": total_expected_delay,
            "actual_wait_sec": actual_wait,
            "delay_deviation_sec": delay_deviation,
            "incident_affected": incident_affected,
            "penalty_amount": final_penalty,
            "error_summary": error_summary,
            "outcome_class": outcome_class,
            "penalty_tier": penalty_tier,
            "penalty_weight": penalty_weight,
        })

    return pd.DataFrame(results)


def compute_tenant_monthly(job_metrics_df, attempts_df, tenants_df, pools_df,
                           rate_cards_df, incidents_df):
    tenant_sl = dict(zip(tenants_df["tenant_id"], tenants_df["service_level"]))
    tenant_region = dict(zip(tenants_df["tenant_id"], tenants_df["region"]))

    attempts_df = attempts_df.copy()
    attempts_df["month"] = attempts_df["queued_at"].dt.to_period("M").astype(str)

    results = []
    for tenant_id in sorted(tenants_df["tenant_id"].unique()):
        sl = tenant_sl[tenant_id]
        region = tenant_region[tenant_id]

        tenant_jobs = job_metrics_df[job_metrics_df["tenant_id"] == tenant_id]
        tenant_attempts = attempts_df[attempts_df["job_id"].isin(tenant_jobs["job_id"])]

        for month in sorted(tenant_attempts["month"].unique()):
            month_attempts = tenant_attempts[tenant_attempts["month"] == month]
            month_job_ids = month_attempts["job_id"].unique()
            month_jobs = tenant_jobs[tenant_jobs["job_id"].isin(month_job_ids)]

            total_jobs = len(month_jobs)
            total_attempts_count = len(month_attempts)
            total_retries = month_attempts[month_attempts["attempt_num"] > 1].shape[0]

            ok_count = month_jobs[month_jobs["final_status"] == "OK"].shape[0]
            success_rate = round(ok_count / total_jobs, 4) if total_jobs > 0 else 0.0

            total_cpu = month_attempts["cpu_ms"].sum()
            total_mem_peak = round(month_attempts["mem_peak_mb"].max(), 1) if len(month_attempts) > 0 else 0.0

            rate_row = rate_cards_df[
                (rate_cards_df["region"] == region) &
                (rate_cards_df["month"] == month)
            ]
            if len(rate_row) > 0:
                cpu_mult = rate_row.iloc[0]["cpu_multiplier"]
                mem_mult = rate_row.iloc[0]["mem_multiplier"]
                retry_surcharge = rate_row.iloc[0]["retry_surcharge_pct"]
            else:
                cpu_mult = 1.0
                mem_mult = 1.0
                retry_surcharge = 0.0

            cpu_hours = total_cpu / (1000.0 * 3600.0)
            pool_ids_used = month_attempts["pool_id"].unique()
            avg_cpu_cost = pools_df[pools_df["pool_id"].isin(pool_ids_used)]["cost_per_cpu_hr"].mean()
            base_cpu_cost = round(cpu_hours * avg_cpu_cost * cpu_mult, 2)

            mem_gb_hours = (month_attempts["mem_peak_mb"].sum() / 1024.0) * (month_attempts["duration_sec"].sum() / 3600.0)
            base_mem_cost = round(mem_gb_hours * 0.01 * mem_mult, 2)

            base_cost = round(base_cpu_cost + base_mem_cost, 2)

            retry_cost = round(base_cost * retry_surcharge / 100.0, 2) if total_retries > 0 else 0.0

            total_penalties = round(month_jobs["penalty_amount"].sum(), 2)

            total_cost = round(base_cost + retry_cost + total_penalties, 2)

            sla_breached_count = month_jobs[month_jobs["sla_breached"] == "YES"].shape[0]
            incident_count = month_jobs[month_jobs["incident_affected"] == "YES"].shape[0]

            avg_duration = round(month_jobs["total_duration_sec"].mean(), 2) if total_jobs > 0 else 0.0
            retry_rate = round(total_retries / total_attempts_count, 4) if total_attempts_count > 0 else 0.0

            cost_efficiency_ratio = round(total_cost / max(1, total_jobs), 2)

            breach_ratio = sla_breached_count / max(1, total_jobs)
            risk_score = round((1 - success_rate) * 40 + retry_rate * 30 + breach_ratio * 30, 3)

            results.append({
                "tenant_id": tenant_id,
                "month": month,
                "service_level": sl,
                "total_jobs": total_jobs,
                "total_attempts": total_attempts_count,
                "total_retries": total_retries,
                "success_rate": success_rate,
                "avg_duration_sec": avg_duration,
                "retry_rate": retry_rate,
                "total_cpu_ms": total_cpu,
                "peak_mem_mb": total_mem_peak,
                "base_cost": base_cost,
                "retry_cost": retry_cost,
                "penalty_cost": total_penalties,
                "total_cost": total_cost,
                "sla_breaches": sla_breached_count,
                "incident_jobs": incident_count,
                "cost_efficiency_ratio": cost_efficiency_ratio,
                "risk_score": risk_score,
            })

    return pd.DataFrame(results)


def compute_reliability_score(tenant_monthly_df):
    df = tenant_monthly_df.copy()
    df = df.sort_values(["tenant_id", "month"]).reset_index(drop=True)

    df["reliability_score"] = 0.0
    df["trend"] = ""

    prev_scores = {}

    for idx in df.index:
        tenant_id = df.loc[idx, "tenant_id"]
        success_rate = df.loc[idx, "success_rate"]
        retry_rate = df.loc[idx, "retry_rate"]
        sla_breaches = df.loc[idx, "sla_breaches"]
        total_jobs = df.loc[idx, "total_jobs"]

        prev_rs = prev_scores.get(tenant_id, 50.0)

        sr_input = success_rate * 100.0
        rr_penalty = retry_rate * 30.0
        breach_penalty = min(20.0, (sla_breaches / max(1, total_jobs)) * 100.0)

        raw_score = sr_input - rr_penalty - breach_penalty

        if success_rate >= 0.90:
            rs = round(prev_rs * 0.70 + raw_score * 0.30, 3)
        elif success_rate >= 0.70:
            rs = round(prev_rs * 0.50 + raw_score * 0.50, 3)
        else:
            rs = round(raw_score, 3)

        rs = round(max(0, min(100, rs)), 3)

        if rs > prev_rs + 2:
            trend = "IMPROVING"
        elif rs < prev_rs - 2:
            trend = "DECLINING"
        else:
            trend = "STABLE"

        df.loc[idx, "reliability_score"] = rs
        df.loc[idx, "trend"] = trend
        prev_scores[tenant_id] = rs

    return df


def classify_tenant_health(row):
    rs = row["reliability_score"]
    sr = row["success_rate"]
    breaches = row["sla_breaches"]
    total = row["total_jobs"]

    breach_ratio = breaches / max(1, total)

    if rs < 30 or (sr < 0.50 and breach_ratio > 0.40):
        return "CRITICAL"
    if rs < 50 or (sr < 0.70 and breach_ratio > 0.25):
        return "DEGRADED"
    if rs < 70 or breach_ratio > 0.15:
        return "AT_RISK"
    if rs >= 85 and sr >= 0.90 and breach_ratio <= 0.05:
        return "HEALTHY"
    return "STABLE"


def compute_pool_monthly(attempts_df, pools_df, incidents_df):
    df = attempts_df.copy()
    df["month"] = df["queued_at"].dt.to_period("M").astype(str)

    results = []
    for pool_id in sorted(pools_df["pool_id"].unique()):
        pool_row = pools_df[pools_df["pool_id"] == pool_id].iloc[0]
        region = pool_row["region"]
        tier = pool_row["tier"]
        cpu_units = pool_row["cpu_units"]

        pool_attempts = df[df["pool_id"] == pool_id]

        for month in sorted(pool_attempts["month"].unique()):
            month_att = pool_attempts[pool_attempts["month"] == month]

            total_attempts = len(month_att)
            ok_count = month_att[month_att["status"] == "OK"].shape[0]
            fail_count = month_att[month_att["status"] == "FAIL"].shape[0]
            timeout_count = month_att[month_att["status"] == "TIMEOUT"].shape[0]
            reject_count = month_att[month_att["status"] == "REJECT"].shape[0]

            total_cpu = month_att["cpu_ms"].sum()
            total_duration = round(month_att["duration_sec"].sum(), 2)
            avg_duration = round(month_att["duration_sec"].mean(), 2) if total_attempts > 0 else 0.0
            max_mem = round(month_att["mem_peak_mb"].max(), 1) if total_attempts > 0 else 0.0

            cpu_utilization = round(total_cpu / (cpu_units * 1000 * 3600 * 730) * 100, 4) if cpu_units > 0 else 0.0

            error_counts = month_att[month_att["error_code"] != ""]["error_code"].value_counts().to_dict()
            if error_counts:
                max_count = max(error_counts.values())
                top_error = sorted(k for k, v in error_counts.items() if v == max_count)[0]
            else:
                top_error = ""

            pool_incidents = incidents_df[incidents_df["pool_id"] == pool_id]
            incident_minutes = 0.0
            for _, inc in pool_incidents.iterrows():
                inc_start = pd.to_datetime(inc["start_time"])
                inc_end = pd.to_datetime(inc["end_time"])
                inc_month = inc_start.to_period("M").strftime("%Y-%m")
                if inc_month == month:
                    incident_minutes += (inc_end - inc_start).total_seconds() / 60.0
            incident_minutes = round(incident_minutes, 2)

            results.append({
                "pool_id": pool_id,
                "month": month,
                "region": region,
                "tier": tier,
                "total_attempts": total_attempts,
                "ok_count": ok_count,
                "fail_count": fail_count,
                "timeout_count": timeout_count,
                "reject_count": reject_count,
                "total_cpu_ms": total_cpu,
                "total_duration_sec": total_duration,
                "avg_duration_sec": avg_duration,
                "peak_mem_mb": max_mem,
                "cpu_utilization_pct": cpu_utilization,
                "top_error": top_error,
                "incident_minutes": incident_minutes,
            })

    return pd.DataFrame(results)


def compute_retry_effectiveness(job_metrics_df, attempts_df, tenants_df, retry_configs_df):
    tenant_sl = dict(zip(tenants_df["tenant_id"], tenants_df["service_level"]))

    results = []
    for sl in sorted(tenants_df["service_level"].unique()):
        sl_tenants = tenants_df[tenants_df["service_level"] == sl]["tenant_id"].tolist()
        sl_jobs = job_metrics_df[job_metrics_df["tenant_id"].isin(sl_tenants)]

        for priority in sorted(sl_jobs["priority"].unique()):
            pri_jobs = sl_jobs[sl_jobs["priority"] == priority]

            total_jobs = len(pri_jobs)
            jobs_with_retries = pri_jobs[pri_jobs["retry_count"] > 0]
            retry_job_count = len(jobs_with_retries)

            recovered = jobs_with_retries[jobs_with_retries["final_status"] == "OK"]
            recovery_rate = round(len(recovered) / retry_job_count, 4) if retry_job_count > 0 else 0.0

            avg_retries = round(jobs_with_retries["retry_count"].mean(), 2) if retry_job_count > 0 else 0.0

            avg_expected = round(jobs_with_retries["expected_delay_sec"].mean(), 2) if retry_job_count > 0 else 0.0
            avg_actual = round(jobs_with_retries["actual_wait_sec"].mean(), 2) if retry_job_count > 0 else 0.0
            avg_deviation = round(jobs_with_retries["delay_deviation_sec"].mean(), 2) if retry_job_count > 0 else 0.0

            total_penalties = round(pri_jobs["penalty_amount"].sum(), 2)
            no_retry_penalties = round(pri_jobs[pri_jobs["retry_count"] == 0]["penalty_amount"].sum(), 2)
            retry_penalties = round(jobs_with_retries["penalty_amount"].sum(), 2)

            cfg = retry_configs_df[
                (retry_configs_df["service_level"] == sl) &
                (retry_configs_df["priority"] == priority)
            ]
            backoff_type = cfg.iloc[0]["backoff_type"] if len(cfg) > 0 else "UNKNOWN"

            first_attempt_ok = pri_jobs[pri_jobs["total_attempts"] == 1]
            first_ok_rate = round(len(first_attempt_ok[first_attempt_ok["final_status"] == "OK"]) / len(first_attempt_ok), 4) if len(first_attempt_ok) > 0 else 0.0

            exhausted = pri_jobs[pri_jobs["outcome_class"] == "EXHAUSTED"]
            exhaustion_rate = round(len(exhausted) / total_jobs, 4) if total_jobs > 0 else 0.0

            results.append({
                "service_level": sl,
                "priority": priority,
                "backoff_type": backoff_type,
                "total_jobs": total_jobs,
                "retry_job_count": retry_job_count,
                "recovery_rate": recovery_rate,
                "avg_retries": avg_retries,
                "avg_expected_delay": avg_expected,
                "avg_actual_delay": avg_actual,
                "avg_delay_deviation": avg_deviation,
                "total_penalties": total_penalties,
                "first_attempt_success_rate": first_ok_rate,
                "exhaustion_rate": exhaustion_rate,
            })

    return pd.DataFrame(results)


def compute_incident_impact(job_metrics_df, attempts_df, incidents_df, pools_df, tenants_df):
    results = []

    for _, inc in incidents_df.iterrows():
        inc_id = inc["incident_id"]
        pool_id = inc["pool_id"]
        inc_start = pd.to_datetime(inc["start_time"])
        inc_end = pd.to_datetime(inc["end_time"])
        severity = inc["severity"]
        impact_pct = inc["capacity_impact_pct"]
        duration_min = round((inc_end - inc_start).total_seconds() / 60.0, 2)

        pool_row = pools_df[pools_df["pool_id"] == pool_id].iloc[0]
        region = pool_row["region"]

        affected_attempts = attempts_df[
            (attempts_df["pool_id"] == pool_id) &
            (attempts_df["started_at"] < inc_end) &
            (attempts_df["ended_at"] > inc_start)
        ]

        affected_jobs = affected_attempts["job_id"].unique()
        affected_job_metrics = job_metrics_df[job_metrics_df["job_id"].isin(affected_jobs)]

        total_affected_jobs = len(affected_job_metrics)
        total_affected_attempts = len(affected_attempts)

        failed_during = affected_attempts[affected_attempts["status"].isin(["FAIL", "TIMEOUT", "REJECT"])].shape[0]

        penalties_incurred = round(affected_job_metrics["penalty_amount"].sum(), 2)
        sla_breaches = affected_job_metrics[affected_job_metrics["sla_breached"] == "YES"].shape[0]

        affected_tenants = affected_job_metrics["tenant_id"].nunique()

        results.append({
            "incident_id": inc_id,
            "pool_id": pool_id,
            "region": region,
            "severity": severity,
            "duration_minutes": duration_min,
            "capacity_impact_pct": impact_pct,
            "affected_jobs": total_affected_jobs,
            "affected_attempts": total_affected_attempts,
            "failed_attempts": failed_during,
            "sla_breaches": sla_breaches,
            "penalties_incurred": penalties_incurred,
            "affected_tenants": affected_tenants,
        })

    return pd.DataFrame(results)


def compute_tenant_profile(tenant_monthly_df, job_metrics_df, tenants_df):
    results = []

    for _, tenant in tenants_df.iterrows():
        tenant_id = tenant["tenant_id"]
        sl = tenant["service_level"]
        region = tenant["region"]

        t_monthly = tenant_monthly_df[tenant_monthly_df["tenant_id"] == tenant_id]
        t_jobs = job_metrics_df[job_metrics_df["tenant_id"] == tenant_id]

        if len(t_monthly) == 0:
            continue

        months_active = len(t_monthly)
        total_jobs = t_jobs.shape[0]
        total_cost = round(t_monthly["total_cost"].sum(), 2)
        avg_monthly_cost = round(total_cost / months_active, 2) if months_active > 0 else 0.0
        total_penalties = round(t_monthly["penalty_cost"].sum(), 2)
        total_sla_breaches = int(t_monthly["sla_breaches"].sum())

        overall_success = round(t_jobs[t_jobs["final_status"] == "OK"].shape[0] / total_jobs, 4) if total_jobs > 0 else 0.0

        last_month = t_monthly.sort_values("month").iloc[-1]
        final_rs = last_month["reliability_score"]
        final_health = classify_tenant_health(last_month)

        worst_health = "HEALTHY"
        health_order = {"CRITICAL": 0, "DEGRADED": 1, "AT_RISK": 2, "STABLE": 3, "HEALTHY": 4}
        for _, m_row in t_monthly.iterrows():
            h = classify_tenant_health(m_row)
            if health_order.get(h, 4) < health_order.get(worst_health, 4):
                worst_health = h

        consecutive_healthy = 0
        for _, m_row in t_monthly.sort_values("month", ascending=False).iterrows():
            h = classify_tenant_health(m_row)
            if h == "HEALTHY":
                consecutive_healthy += 1
            else:
                break

        longest_healthy_streak = 0
        current_streak = 0
        for _, m_row in t_monthly.sort_values("month").iterrows():
            h = classify_tenant_health(m_row)
            if h == "HEALTHY":
                current_streak += 1
                longest_healthy_streak = max(longest_healthy_streak, current_streak)
            else:
                current_streak = 0

        avg_retry_rate = round(t_monthly["retry_rate"].mean(), 4)
        total_incident_jobs = int(t_monthly["incident_jobs"].sum())

        consecutive_breach_months = 0
        for _, m_row in t_monthly.sort_values("month", ascending=False).iterrows():
            if m_row["sla_breaches"] > 0:
                consecutive_breach_months += 1
            else:
                break

        avg_penalty_per_breach = round(total_penalties / max(1, total_sla_breaches), 2)

        results.append({
            "tenant_id": tenant_id,
            "service_level": sl,
            "region": region,
            "months_active": months_active,
            "total_jobs": total_jobs,
            "overall_success_rate": overall_success,
            "total_cost": total_cost,
            "avg_monthly_cost": avg_monthly_cost,
            "total_penalties": total_penalties,
            "total_sla_breaches": total_sla_breaches,
            "avg_retry_rate": avg_retry_rate,
            "final_reliability_score": final_rs,
            "final_health_status": final_health,
            "worst_health_status": worst_health,
            "consecutive_healthy_months": consecutive_healthy,
            "longest_healthy_streak": longest_healthy_streak,
            "total_incident_jobs": total_incident_jobs,
            "consecutive_breach_months": consecutive_breach_months,
            "avg_penalty_per_breach": avg_penalty_per_breach,
        })

    return pd.DataFrame(results)


def main():
    pools, tenants, retry_configs, sla_terms, rate_cards, incidents, jobs, attempts = load_data()

    cleaned = clean_attempts(attempts, jobs)

    incidents["start_time"] = pd.to_datetime(incidents["start_time"])
    incidents["end_time"] = pd.to_datetime(incidents["end_time"])

    job_metrics = compute_job_metrics(jobs, cleaned, tenants, pools, incidents, retry_configs, sla_terms)
    job_metrics.to_csv(TASK_DIR / "job_metrics.csv", index=False)
    print(f"job_metrics.csv: {len(job_metrics)} rows")

    tenant_monthly = compute_tenant_monthly(job_metrics, cleaned, tenants, pools, rate_cards, incidents)
    tenant_monthly = compute_reliability_score(tenant_monthly)
    tenant_monthly["health_status"] = tenant_monthly.apply(classify_tenant_health, axis=1)
    tenant_monthly.to_csv(TASK_DIR / "tenant_monthly.csv", index=False)
    print(f"tenant_monthly.csv: {len(tenant_monthly)} rows")

    pool_monthly = compute_pool_monthly(cleaned, pools, incidents)
    pool_monthly.to_csv(TASK_DIR / "pool_monthly.csv", index=False)
    print(f"pool_monthly.csv: {len(pool_monthly)} rows")

    retry_eff = compute_retry_effectiveness(job_metrics, cleaned, tenants, retry_configs)
    retry_eff.to_csv(TASK_DIR / "retry_effectiveness.csv", index=False)
    print(f"retry_effectiveness.csv: {len(retry_eff)} rows")

    incident_impact = compute_incident_impact(job_metrics, cleaned, incidents, pools, tenants)
    incident_impact.to_csv(TASK_DIR / "incident_impact.csv", index=False)
    print(f"incident_impact.csv: {len(incident_impact)} rows")

    tenant_profile = compute_tenant_profile(tenant_monthly, job_metrics, tenants)
    tenant_profile.to_csv(TASK_DIR / "tenant_profile.csv", index=False)
    print(f"tenant_profile.csv: {len(tenant_profile)} rows")


if __name__ == "__main__":
    main()

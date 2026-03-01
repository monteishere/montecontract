import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

np.random.seed(42)

TASK_DIR = Path(__file__).parent.parent / "task"

REGIONS = ["ALPHA", "BETA", "GAMMA", "DELTA"]
SERVICE_LEVELS = ["BRONZE", "SILVER", "GOLD", "PLATINUM"]
PRIORITIES = [1, 2, 3, 4, 5]


def generate_pools():
    configs = [
        ("P01", "ALPHA", "T1", 32, 128, 0.12),
        ("P02", "ALPHA", "T2", 16, 64, 0.08),
        ("P03", "ALPHA", "T3", 64, 256, 0.18),
        ("P04", "BETA", "T1", 32, 128, 0.10),
        ("P05", "BETA", "T2", 16, 64, 0.07),
        ("P06", "BETA", "T3", 48, 192, 0.15),
        ("P07", "GAMMA", "T1", 24, 96, 0.11),
        ("P08", "GAMMA", "T2", 16, 64, 0.08),
        ("P09", "GAMMA", "T3", 64, 256, 0.20),
        ("P10", "DELTA", "T1", 32, 128, 0.09),
        ("P11", "DELTA", "T2", 16, 64, 0.06),
        ("P12", "DELTA", "T3", 48, 192, 0.14),
    ]
    rows = []
    for pid, region, tier, cpu, mem, cost in configs:
        rows.append({
            "pool_id": pid,
            "region": region,
            "tier": tier,
            "cpu_units": cpu,
            "mem_gb": mem,
            "cost_per_cpu_hr": cost,
        })
    return pd.DataFrame(rows)


def generate_tenants():
    configs = [
        ("T01", "PLATINUM", "ALPHA", "2023-01-15"),
        ("T02", "GOLD", "ALPHA", "2023-03-01"),
        ("T03", "SILVER", "BETA", "2023-06-15"),
        ("T04", "BRONZE", "BETA", "2023-09-01"),
        ("T05", "PLATINUM", "GAMMA", "2023-02-01"),
        ("T06", "GOLD", "GAMMA", "2023-04-15"),
        ("T07", "SILVER", "DELTA", "2023-07-01"),
        ("T08", "BRONZE", "DELTA", "2023-10-15"),
        ("T09", "GOLD", "ALPHA", "2023-05-01"),
        ("T10", "SILVER", "BETA", "2023-08-15"),
        ("T11", "PLATINUM", "GAMMA", "2023-11-01"),
        ("T12", "BRONZE", "ALPHA", "2023-12-01"),
        ("T13", "GOLD", "DELTA", "2023-01-01"),
        ("T14", "SILVER", "GAMMA", "2023-04-01"),
        ("T15", "BRONZE", "BETA", "2023-06-01"),
    ]
    rows = []
    for tid, sl, region, cs in configs:
        rows.append({
            "tenant_id": tid,
            "service_level": sl,
            "region": region,
            "contract_start": cs,
        })
    return pd.DataFrame(rows)


def generate_retry_configs():
    rows = []
    backoff_map = {
        ("PLATINUM", 1): ("DECORRELATED", 3.0, 60.0),
        ("PLATINUM", 2): ("EXPONENTIAL", 2.0, 45.0),
        ("PLATINUM", 3): ("EXPONENTIAL", 3.0, 60.0),
        ("PLATINUM", 4): ("LINEAR", 5.0, 90.0),
        ("PLATINUM", 5): ("FIXED", 10.0, 120.0),
        ("GOLD", 1): ("DECORRELATED", 2.0, 45.0),
        ("GOLD", 2): ("EXPONENTIAL", 3.0, 60.0),
        ("GOLD", 3): ("LINEAR", 4.0, 60.0),
        ("GOLD", 4): ("LINEAR", 6.0, 90.0),
        ("GOLD", 5): ("FIXED", 15.0, 120.0),
        ("SILVER", 1): ("EXPONENTIAL", 3.0, 45.0),
        ("SILVER", 2): ("LINEAR", 4.0, 60.0),
        ("SILVER", 3): ("LINEAR", 5.0, 75.0),
        ("SILVER", 4): ("FIXED", 8.0, 90.0),
        ("SILVER", 5): ("FIXED", 12.0, 120.0),
        ("BRONZE", 1): ("LINEAR", 5.0, 60.0),
        ("BRONZE", 2): ("FIXED", 8.0, 60.0),
        ("BRONZE", 3): ("FIXED", 10.0, 90.0),
        ("BRONZE", 4): ("FIXED", 12.0, 120.0),
        ("BRONZE", 5): ("FIXED", 15.0, 120.0),
    }
    for (sl, pri), (bt, base, maxd) in backoff_map.items():
        if sl in ("PLATINUM", "GOLD") and pri <= 2:
            rows.append({
                "service_level": sl,
                "priority": pri,
                "backoff_type": bt,
                "base_delay_sec": base,
                "max_delay_sec": maxd,
                "effective_from": "2024-01-01",
                "effective_to": "2024-03-31",
            })
            new_base = round(base * 0.8, 1)
            new_max = round(maxd * 0.9, 1)
            rows.append({
                "service_level": sl,
                "priority": pri,
                "backoff_type": bt,
                "base_delay_sec": new_base,
                "max_delay_sec": new_max,
                "effective_from": "2024-04-01",
                "effective_to": "2024-06-30",
            })
        else:
            rows.append({
                "service_level": sl,
                "priority": pri,
                "backoff_type": bt,
                "base_delay_sec": base,
                "max_delay_sec": maxd,
                "effective_from": "2024-01-01",
                "effective_to": "2024-06-30",
            })
    return pd.DataFrame(rows)


def generate_sla_terms():
    rows = []
    configs = {
        ("PLATINUM", 1): (0.50, 0.25, 2.5, 5.0),
        ("PLATINUM", 2): (0.35, 0.30, 2.0, 10.0),
        ("PLATINUM", 3): (0.20, 0.40, 1.8, 15.0),
        ("PLATINUM", 4): (0.10, 0.50, 1.5, 20.0),
        ("PLATINUM", 5): (0.05, 0.60, 1.3, 30.0),
        ("GOLD", 1): (0.40, 0.30, 2.0, 8.0),
        ("GOLD", 2): (0.25, 0.35, 1.8, 12.0),
        ("GOLD", 3): (0.15, 0.45, 1.5, 18.0),
        ("GOLD", 4): (0.08, 0.55, 1.3, 25.0),
        ("GOLD", 5): (0.04, 0.65, 1.2, 35.0),
        ("SILVER", 1): (0.30, 0.35, 1.8, 10.0),
        ("SILVER", 2): (0.18, 0.40, 1.5, 15.0),
        ("SILVER", 3): (0.10, 0.50, 1.3, 20.0),
        ("SILVER", 4): (0.05, 0.60, 1.2, 30.0),
        ("SILVER", 5): (0.03, 0.70, 1.1, 40.0),
        ("BRONZE", 1): (0.20, 0.40, 1.5, 15.0),
        ("BRONZE", 2): (0.12, 0.50, 1.3, 20.0),
        ("BRONZE", 3): (0.06, 0.60, 1.2, 25.0),
        ("BRONZE", 4): (0.03, 0.70, 1.1, 35.0),
        ("BRONZE", 5): (0.02, 0.80, 1.0, 45.0),
    }
    for (sl, pri), (penalty, esc, crit, grace) in configs.items():
        rows.append({
            "service_level": sl,
            "priority": pri,
            "penalty_per_min": penalty,
            "escalation_pct": esc,
            "critical_mult": crit,
            "grace_sec": grace,
        })
    return pd.DataFrame(rows)


def generate_rate_cards():
    rows = []
    base_cpu = {"ALPHA": 1.0, "BETA": 0.95, "GAMMA": 1.05, "DELTA": 0.90}
    base_mem = {"ALPHA": 1.0, "BETA": 0.92, "GAMMA": 1.08, "DELTA": 0.88}
    base_retry = {"ALPHA": 12.0, "BETA": 10.0, "GAMMA": 15.0, "DELTA": 8.0}
    for region in REGIONS:
        for month in range(1, 7):
            drift = (month - 1) * 0.02
            cpu_m = round(base_cpu[region] + drift + np.random.uniform(-0.03, 0.03), 4)
            mem_m = round(base_mem[region] + drift * 0.5 + np.random.uniform(-0.02, 0.02), 4)
            retry_s = round(base_retry[region] + month * 0.5 + np.random.uniform(-1, 1), 2)
            rows.append({
                "region": region,
                "month": f"2024-{month:02d}",
                "cpu_multiplier": cpu_m,
                "mem_multiplier": mem_m,
                "retry_surcharge_pct": retry_s,
            })
    return pd.DataFrame(rows)


def generate_incidents(pools_df):
    incident_configs = [
        ("INC01", "P01", "2024-01-15 08:00:00", "2024-01-15 14:30:00", "S2", 40),
        ("INC02", "P03", "2024-01-28 22:00:00", "2024-01-29 06:00:00", "S3", 25),
        ("INC03", "P05", "2024-02-10 03:00:00", "2024-02-10 11:00:00", "S1", 75),
        ("INC04", "P07", "2024-02-20 16:00:00", "2024-02-21 02:00:00", "S2", 50),
        ("INC05", "P02", "2024-03-05 10:00:00", "2024-03-05 18:00:00", "S3", 30),
        ("INC06", "P09", "2024-03-18 00:00:00", "2024-03-18 12:00:00", "S2", 45),
        ("INC07", "P04", "2024-04-02 06:00:00", "2024-04-02 20:00:00", "S1", 80),
        ("INC08", "P11", "2024-04-15 14:00:00", "2024-04-16 02:00:00", "S3", 20),
        ("INC09", "P06", "2024-05-01 08:00:00", "2024-05-01 20:00:00", "S2", 55),
        ("INC10", "P08", "2024-05-12 18:00:00", "2024-05-13 06:00:00", "S3", 35),
        ("INC11", "P10", "2024-05-25 02:00:00", "2024-05-25 14:00:00", "S1", 70),
        ("INC12", "P12", "2024-06-08 10:00:00", "2024-06-08 22:00:00", "S2", 45),
        ("INC13", "P01", "2024-06-20 00:00:00", "2024-06-20 16:00:00", "S3", 30),
        ("INC14", "P03", "2024-04-10 04:00:00", "2024-04-10 10:00:00", "S4", 15),
        ("INC15", "P07", "2024-06-15 12:00:00", "2024-06-15 18:00:00", "S3", 25),
    ]
    rows = []
    for inc_id, pool_id, start, end, sev, impact in incident_configs:
        rows.append({
            "incident_id": inc_id,
            "pool_id": pool_id,
            "start_time": start,
            "end_time": end,
            "severity": sev,
            "capacity_impact_pct": impact,
        })
    return pd.DataFrame(rows)


def random_datetime_in_month(year, month):
    if month == 12:
        days_in_month = 31
    else:
        next_month = datetime(year, month + 1, 1)
        days_in_month = (next_month - datetime(year, month, 1)).days
    day = np.random.randint(1, days_in_month + 1)
    hour = np.random.randint(0, 24)
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)
    return datetime(year, month, day, hour, minute, second)


def get_sla_for_priority(priority):
    sla_map = {1: 45, 2: 90, 3: 180, 4: 360, 5: 720}
    return sla_map[priority]


def get_max_retries_for_priority(priority):
    retry_map = {1: 5, 2: 4, 3: 3, 4: 3, 5: 2}
    return retry_map[priority]


def determine_outcome(attempt_num, priority, pool_tier, force_fail=False):
    if force_fail:
        r = np.random.random()
        if r < 0.50:
            return "FAIL"
        elif r < 0.80:
            return "TIMEOUT"
        else:
            return "REJECT"
    base_ok = {1: 0.45, 2: 0.50, 3: 0.60, 4: 0.65, 5: 0.70}
    tier_bonus = {"T1": 0.05, "T2": 0.0, "T3": 0.08}
    retry_bonus = min(0.12, (attempt_num - 1) * 0.04)
    ok_prob = min(0.85, base_ok[priority] + tier_bonus[pool_tier] + retry_bonus)
    fail_prob = (1 - ok_prob) * 0.50
    timeout_prob = (1 - ok_prob) * 0.30
    reject_prob = 1.0 - ok_prob - fail_prob - timeout_prob
    r = np.random.random()
    if r < ok_prob:
        return "OK"
    elif r < ok_prob + fail_prob:
        return "FAIL"
    elif r < ok_prob + fail_prob + timeout_prob:
        return "TIMEOUT"
    else:
        return "REJECT"


def generate_exec_duration(status, priority):
    if status == "OK":
        return round(np.random.uniform(2, 12), 2)
    elif status == "FAIL":
        return round(np.random.uniform(3, 20), 2)
    elif status == "TIMEOUT":
        sla = get_sla_for_priority(priority)
        return round(np.random.uniform(sla * 0.3, sla * 0.8), 2)
    else:
        return 0.0


def generate_cpu_ms(status, duration_sec):
    if status == "REJECT":
        return 0
    if status == "TIMEOUT":
        return round(duration_sec * np.random.uniform(500, 1500))
    return round(duration_sec * np.random.uniform(800, 2000))


def generate_mem_mb(status):
    if status == "REJECT":
        return 0.0
    return round(np.random.uniform(32, 512), 1)


def generate_error_code(status):
    if status == "OK":
        return ""
    if status == "FAIL":
        codes = ["E001", "E002", "E003", "E004", "E005", "E006"]
        return np.random.choice(codes)
    if status == "TIMEOUT":
        return "E100"
    if status == "REJECT":
        return "E200"
    return ""


def inject_nulls(attempts_df):
    df = attempts_df.copy()
    df["cpu_ms"] = df["cpu_ms"].astype(object)
    df["mem_peak_mb"] = df["mem_peak_mb"].astype(object)

    timeout_mask = df["status"] == "TIMEOUT"
    timeout_indices = df[timeout_mask].index.tolist()
    if len(timeout_indices) > 3:
        null_ended = np.random.choice(timeout_indices, size=max(3, int(len(timeout_indices) * 0.3)), replace=False)
        df.loc[null_ended, "ended_at"] = ""

    non_reject = df[df["status"] != "REJECT"].index.tolist()
    null_cpu = np.random.choice(non_reject, size=int(len(non_reject) * 0.04), replace=False)
    df.loc[null_cpu, "cpu_ms"] = ""

    null_mem = np.random.choice(non_reject, size=int(len(non_reject) * 0.05), replace=False)
    df.loc[null_mem, "mem_peak_mb"] = ""

    fail_mask = df["status"] == "FAIL"
    fail_indices = df[fail_mask].index.tolist()
    if len(fail_indices) > 2:
        null_err = np.random.choice(fail_indices, size=max(2, int(len(fail_indices) * 0.08)), replace=False)
        df.loc[null_err, "error_code"] = ""

    return df


def generate_jobs_and_attempts(pools_df, tenants_df, retry_configs_df, incidents_df):
    region_pools = {}
    for region in REGIONS:
        rp = pools_df[pools_df["region"] == region]["pool_id"].tolist()
        region_pools[region] = rp

    pool_tiers = dict(zip(pools_df["pool_id"], pools_df["tier"]))

    incident_windows = []
    for _, inc in incidents_df.iterrows():
        incident_windows.append({
            "pool_id": inc["pool_id"],
            "start": datetime.strptime(inc["start_time"], "%Y-%m-%d %H:%M:%S"),
            "end": datetime.strptime(inc["end_time"], "%Y-%m-%d %H:%M:%S"),
            "region": pools_df[pools_df["pool_id"] == inc["pool_id"]]["region"].values[0],
        })

    jobs_per_month = {"PLATINUM": 9, "GOLD": 7, "SILVER": 5, "BRONZE": 3}

    all_jobs = []
    all_attempts = []
    job_counter = 1
    attempt_counter = 1
    forced_fail_indices = set()
    incident_job_indices = set()

    total_expected = sum(
        (jobs_per_month[t["service_level"]] + 0) * 6
        for _, t in tenants_df.iterrows()
    )
    fail_targets = set(np.random.choice(range(1, total_expected + 50), size=int(total_expected * 0.10), replace=False))
    incident_targets = set(np.random.choice(range(1, total_expected + 50), size=int(total_expected * 0.08), replace=False))

    for month in range(1, 7):
        for _, tenant in tenants_df.iterrows():
            tenant_id = tenant["tenant_id"]
            sl = tenant["service_level"]
            region = tenant["region"]
            n_jobs = jobs_per_month[sl] + np.random.randint(-1, 2)

            for _ in range(n_jobs):
                priority = int(np.random.choice(PRIORITIES, p=[0.10, 0.20, 0.30, 0.25, 0.15]))
                submitted = random_datetime_in_month(2024, month)
                sla_sec = get_sla_for_priority(priority)
                max_retries = get_max_retries_for_priority(priority)

                force_fail = job_counter in fail_targets

                place_in_incident = job_counter in incident_targets
                if place_in_incident:
                    region_incidents = [iw for iw in incident_windows if iw["region"] == region]
                    if region_incidents:
                        chosen_inc = np.random.choice(len(region_incidents))
                        inc_w = region_incidents[chosen_inc]
                        inc_duration = (inc_w["end"] - inc_w["start"]).total_seconds()
                        offset = np.random.uniform(0, max(1, inc_duration - 30))
                        submitted = inc_w["start"] + timedelta(seconds=offset)

                job_id = f"J{job_counter:04d}"
                all_jobs.append({
                    "job_id": job_id,
                    "tenant_id": tenant_id,
                    "priority": priority,
                    "submitted_at": submitted.strftime("%Y-%m-%d %H:%M:%S"),
                    "sla_seconds": sla_sec,
                    "max_retries": max_retries,
                })

                pools_in_region = region_pools[region]
                current_time = submitted + timedelta(seconds=np.random.randint(1, 8))

                if place_in_incident and region_incidents:
                    target_pool = inc_w["pool_id"]
                    if target_pool in pools_in_region:
                        first_pool = target_pool
                    else:
                        first_pool = np.random.choice(pools_in_region)
                else:
                    first_pool = None

                prev_ended = None
                for attempt_num in range(1, max_retries + 2):
                    if first_pool and attempt_num == 1:
                        pool_id = first_pool
                    else:
                        pool_id = np.random.choice(pools_in_region)
                    pool_tier = pool_tiers[pool_id]
                    status = determine_outcome(attempt_num, priority, pool_tier, force_fail=force_fail)

                    queued_at = current_time
                    started_at = queued_at + timedelta(seconds=np.random.uniform(0.5, 3.0))
                    exec_dur = generate_exec_duration(status, priority)
                    if status == "REJECT":
                        ended_at = queued_at
                        started_at = queued_at
                    else:
                        ended_at = started_at + timedelta(seconds=exec_dur)

                    cpu_ms = generate_cpu_ms(status, exec_dur)
                    mem_mb = generate_mem_mb(status)
                    error_code = generate_error_code(status)

                    attempt_id = f"A{attempt_counter:04d}"
                    all_attempts.append({
                        "attempt_id": attempt_id,
                        "job_id": job_id,
                        "pool_id": pool_id,
                        "attempt_num": attempt_num,
                        "queued_at": queued_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "ended_at": ended_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "status": status,
                        "cpu_ms": cpu_ms,
                        "mem_peak_mb": mem_mb,
                        "error_code": error_code,
                    })
                    attempt_counter += 1
                    prev_ended = ended_at

                    if status == "OK":
                        break
                    if attempt_num >= max_retries + 1:
                        break

                    if status == "REJECT" and attempt_num >= 2 and np.random.random() < 0.40:
                        break
                    if status == "TIMEOUT" and attempt_num >= 2 and np.random.random() < 0.30:
                        break

                    backoff_base = np.random.uniform(3, 20)
                    current_time = ended_at + timedelta(seconds=backoff_base)

                job_counter += 1

    jobs_df = pd.DataFrame(all_jobs)
    attempts_df = pd.DataFrame(all_attempts)
    attempts_df = inject_nulls(attempts_df)
    return jobs_df, attempts_df


def main():
    TASK_DIR.mkdir(parents=True, exist_ok=True)

    pools = generate_pools()
    pools.to_csv(TASK_DIR / "pools.csv", index=False)

    tenants = generate_tenants()
    tenants.to_csv(TASK_DIR / "tenants.csv", index=False)

    retry_configs = generate_retry_configs()
    retry_configs.to_csv(TASK_DIR / "retry_configs.csv", index=False)

    sla_terms = generate_sla_terms()
    sla_terms.to_csv(TASK_DIR / "sla_terms.csv", index=False)

    rate_cards = generate_rate_cards()
    rate_cards.to_csv(TASK_DIR / "rate_cards.csv", index=False)

    incidents = generate_incidents(pools)
    incidents.to_csv(TASK_DIR / "incidents.csv", index=False)

    jobs, attempts = generate_jobs_and_attempts(pools, tenants, retry_configs, incidents)
    jobs.to_csv(TASK_DIR / "jobs.csv", index=False)
    attempts.to_csv(TASK_DIR / "attempts.csv", index=False)

    print(f"Generated {len(pools)} pools")
    print(f"Generated {len(tenants)} tenants")
    print(f"Generated {len(retry_configs)} retry_configs")
    print(f"Generated {len(sla_terms)} sla_terms")
    print(f"Generated {len(rate_cards)} rate_cards")
    print(f"Generated {len(incidents)} incidents")
    print(f"Generated {len(jobs)} jobs")
    print(f"Generated {len(attempts)} attempts")


if __name__ == "__main__":
    main()

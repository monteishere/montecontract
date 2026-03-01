import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re


GOLDEN_DIR = Path(__file__).parent
TASK_DIR = GOLDEN_DIR.parent / "task"

VALID_COST_GRADES = {"CRITICAL", "OVER_DEMAND", "LOW_EFFICIENCY", "PF_PENALTY", "EFFICIENT", "STANDARD"}
VALID_COST_CATEGORIES = {"HIGH_COST", "ELEVATED", "OPTIMIZED", "NORMAL"}
VALID_RATCHET = {"YES", "NO"}
VALID_TARGET_MET = {"YES", "NO"}


def normalize_columns(df):
    rename_map = {}
    for col in df.columns:
        clean = re.sub(r"\(.*\)$", "", col).strip()
        if clean != col:
            rename_map[col] = clean
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def normalize_quarter(df):
    if "quarter" in df.columns:
        qmap = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        df["quarter"] = df["quarter"].apply(lambda x: qmap.get(str(x).strip(), x))
        df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce").astype(int)
    return df


def load_golden(name):
    return pd.read_csv(GOLDEN_DIR / f"golden_{name}.csv")


def load_sub(name):
    path = TASK_DIR / f"{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = normalize_columns(df)
    df = normalize_quarter(df)
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
    return golden.merge(sub, on=keys, suffixes=("_g", "_s"), how="inner")


def check_bill_exists(sub, golden):
    assert sub is not None, "facility_monthly_bill.csv not found"

def check_bill_row_count(sub, golden):
    assert len(sub) == 150, f"Expected 150 rows, got {len(sub)}"

def check_bill_columns(sub, golden):
    missing = set(golden.columns) - set(sub.columns)
    assert not missing, f"Missing columns: {missing}"

def check_bill_facility_ids(sub, golden):
    expected = set(golden["facility_id"].unique())
    actual = set(sub["facility_id"].unique())
    assert expected == actual, f"Facility IDs mismatch: missing={expected-actual}, extra={actual-expected}"

def check_bill_months(sub, golden):
    expected = set(golden["month"].unique())
    actual = set(sub["month"].unique())
    assert expected == actual, f"Months mismatch"

def check_bill_kwh_nonneg(sub, golden):
    bad = sub[sub["total_kwh"] < -0.01]
    assert len(bad) == 0, f"{len(bad)} rows with negative total_kwh"

def check_bill_energy_cost_nonneg(sub, golden):
    bad = sub[sub["energy_cost"] < -0.01]
    assert len(bad) == 0, f"{len(bad)} rows with negative energy_cost"

def check_bill_demand_nonneg(sub, golden):
    bad = sub[sub["billed_demand"] < -0.01]
    assert len(bad) == 0, f"{len(bad)} rows with negative billed_demand"

def check_bill_demand_charge_nonneg(sub, golden):
    bad = sub[sub["demand_charge"] < -0.01]
    assert len(bad) == 0, f"{len(bad)} rows with negative demand_charge"

def check_bill_pf_range(sub, golden):
    pf = sub["power_factor"]
    bad = pf[(pf < 0) | (pf > 1.01)]
    assert len(bad) == 0, f"{len(bad)} power_factor values out of [0,1] range"

def check_bill_pf_rounding(sub, golden):
    pf = sub["power_factor"].dropna()
    rounded = pf.round(4)
    mismatches = (pf - rounded).abs() > 1e-9
    assert mismatches.sum() == 0, f"{mismatches.sum()} power_factor values not rounded to 4 decimals"

def check_bill_solar_nonneg(sub, golden):
    bad = sub[sub["solar_credit"] < -0.01]
    assert len(bad) == 0, f"{len(bad)} rows with negative solar_credit"

def check_bill_cost_grade_values(sub, golden):
    invalid = set(sub["cost_grade"].unique()) - VALID_COST_GRADES
    assert not invalid, f"Invalid cost_grade values: {invalid}"

def check_bill_billed_ge_contracted_floor(sub, golden):
    facilities = pd.read_csv(TASK_DIR / "facilities.csv")
    merged = sub.merge(facilities[["facility_id", "contracted_demand_kw"]], on="facility_id")
    floor = merged["contracted_demand_kw"] * 0.70
    bad = merged[merged["billed_demand"] < floor - 0.5]
    assert len(bad) == 0, f"{len(bad)} rows with billed_demand below contracted floor"

def check_bill_total_kwh_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["total_kwh_g"], row["total_kwh_s"], tol=5.0):
            mismatches += 1
    assert mismatches <= 3, f"{mismatches} total_kwh mismatches (tolerance 5.0)"

def check_bill_energy_cost_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["energy_cost_g"], row["energy_cost_s"], tol=2.0):
            mismatches += 1
    assert mismatches <= 3, f"{mismatches} energy_cost mismatches (tolerance 2.0)"

def check_bill_demand_charge_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["demand_charge_g"], row["demand_charge_s"], tol=2.0):
            mismatches += 1
    assert mismatches <= 3, f"{mismatches} demand_charge mismatches (tolerance 2.0)"

def check_bill_pf_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["power_factor_g"], row["power_factor_s"], tol=0.005):
            mismatches += 1
    assert mismatches <= 3, f"{mismatches} power_factor mismatches (tolerance 0.005)"

def check_bill_pf_adjustment_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["pf_adjustment_g"], row["pf_adjustment_s"], tol=1.0):
            mismatches += 1
    assert mismatches <= 5, f"{mismatches} pf_adjustment mismatches (tolerance 1.0)"

def check_bill_solar_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["solar_credit_g"], row["solar_credit_s"], tol=1.0):
            mismatches += 1
    assert mismatches <= 3, f"{mismatches} solar_credit mismatches (tolerance 1.0)"

def check_bill_equip_surcharge_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["equipment_surcharge_g"], row["equipment_surcharge_s"], tol=1.5):
            mismatches += 1
    assert mismatches <= 8, f"{mismatches} equipment_surcharge mismatches (tolerance 1.5)"

def check_bill_overhead_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["overhead_cost_g"], row["overhead_cost_s"], tol=2.0):
            mismatches += 1
    assert mismatches <= 3, f"{mismatches} overhead_cost mismatches (tolerance 2.0)"

def check_bill_total_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["total_bill_g"], row["total_bill_s"], tol=5.0):
            mismatches += 1
    assert mismatches <= 5, f"{mismatches} total_bill mismatches (tolerance 5.0)"

def check_bill_cost_grade_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "month"])
    mismatches = (m["cost_grade_g"] != m["cost_grade_s"]).sum()
    assert mismatches <= 5, f"{mismatches} cost_grade mismatches"

def check_bill_ratchet_demand_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["ratchet_demand_g"], row["ratchet_demand_s"], tol=1.0):
            mismatches += 1
    assert mismatches <= 3, f"{mismatches} ratchet_demand mismatches (tolerance 1.0)"

def check_bill_billed_demand_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["billed_demand_g"], row["billed_demand_s"], tol=1.0):
            mismatches += 1
    assert mismatches <= 3, f"{mismatches} billed_demand mismatches (tolerance 1.0)"

def check_bill_consec_penalty_nonneg(sub, golden):
    bad = sub[sub["consecutive_penalty_months"] < 0]
    assert len(bad) == 0, f"{len(bad)} rows have negative consecutive_penalty_months"

def check_bill_consec_penalty_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["consecutive_penalty_months_g"], row["consecutive_penalty_months_s"], tol=0):
            mismatches += 1
    assert mismatches <= 3, f"{mismatches} consecutive_penalty_months mismatches"

def check_bill_fes_range(sub, golden):
    fes = sub["facility_efficiency_score"]
    bad = fes[(fes < -0.01) | (fes > 100.01)]
    assert len(bad) == 0, f"{len(bad)} facility_efficiency_score values out of [0,100]"

def check_bill_fes_rounding(sub, golden):
    fes = sub["facility_efficiency_score"].dropna()
    rounded = fes.round(3)
    mismatches = (fes - rounded).abs() > 1e-9
    assert mismatches.sum() == 0, f"{mismatches.sum()} FES values not rounded to 3 decimals"

def check_bill_fes_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["facility_efficiency_score_g"], row["facility_efficiency_score_s"], tol=1.0):
            mismatches += 1
    assert mismatches <= 5, f"{mismatches} facility_efficiency_score mismatches (tolerance 1.0)"


def check_curt_exists(sub, golden):
    assert sub is not None, "curtailment_credit_ledger.csv not found"

def check_curt_row_count(sub, golden):
    diff = abs(len(sub) - len(golden))
    assert diff <= 5, f"Row count: expected ~{len(golden)}, got {len(sub)} (diff {diff})"

def check_curt_columns(sub, golden):
    missing = set(golden.columns) - set(sub.columns)
    assert not missing, f"Missing columns: {missing}"

def check_curt_target_met_values(sub, golden):
    invalid = set(sub["target_met"].unique()) - VALID_TARGET_MET
    assert not invalid, f"Invalid target_met values: {invalid}"

def check_curt_credit_nonneg(sub, golden):
    bad = sub[sub["credit_amount"] < -0.01]
    assert len(bad) == 0, f"{len(bad)} rows with negative credit_amount"

def check_curt_credit_zero_when_not_met(sub, golden):
    not_met = sub[sub["target_met"] == "NO"]
    bad = not_met[not_met["credit_amount"] > 0.01]
    assert len(bad) == 0, f"{len(bad)} rows with credit > 0 when target not met"

def check_curt_baseline_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "event_date"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["baseline_kwh_g"], row["baseline_kwh_s"], tol=5.0):
            mismatches += 1
    assert mismatches <= 5, f"{mismatches} curtailment baseline_kwh mismatches (tolerance 5.0)"

def check_curt_credit_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "event_date"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["credit_amount_g"], row["credit_amount_s"], tol=2.0):
            mismatches += 1
    assert mismatches <= 5, f"{mismatches} curtailment credit_amount mismatches (tolerance 2.0)"

def check_curt_target_met_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "event_date"])
    mismatches = (m["target_met_g"] != m["target_met_s"]).sum()
    assert mismatches <= 5, f"{mismatches} target_met mismatches"


def check_campus_exists(sub, golden):
    assert sub is not None, "campus_quarterly_summary.csv not found"

def check_campus_row_count(sub, golden):
    assert len(sub) == len(golden), f"Expected {len(golden)} rows, got {len(sub)}"

def check_campus_columns(sub, golden):
    missing = set(golden.columns) - set(sub.columns)
    assert not missing, f"Missing columns: {missing}"

def check_campus_pairs(sub, golden):
    expected = set(zip(golden["campus"], golden["quarter"]))
    actual = set(zip(sub["campus"], sub["quarter"]))
    assert expected == actual, f"Campus-quarter pairs mismatch"

def check_campus_kwh_match(sub, golden):
    m = merge_on_keys(sub, golden, ["campus", "quarter"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["total_kwh_g"], row["total_kwh_s"], tol=50.0):
            mismatches += 1
    assert mismatches == 0, f"{mismatches} campus total_kwh mismatches"

def check_campus_bills_match(sub, golden):
    m = merge_on_keys(sub, golden, ["campus", "quarter"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["total_bills_g"], row["total_bills_s"], tol=50.0):
            mismatches += 1
    assert mismatches <= 1, f"{mismatches} campus total_bills mismatches"

def check_campus_pf_match(sub, golden):
    m = merge_on_keys(sub, golden, ["campus", "quarter"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["avg_power_factor_g"], row["avg_power_factor_s"], tol=0.01):
            mismatches += 1
    assert mismatches == 0, f"{mismatches} campus avg_power_factor mismatches"

def check_campus_fes_match(sub, golden):
    m = merge_on_keys(sub, golden, ["campus", "quarter"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["avg_fes_g"], row["avg_fes_s"], tol=1.0):
            mismatches += 1
    assert mismatches == 0, f"{mismatches} campus avg_fes mismatches"

def check_campus_efficiency_rate_match(sub, golden):
    m = merge_on_keys(sub, golden, ["campus", "quarter"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["cost_efficiency_rate_g"], row["cost_efficiency_rate_s"], tol=0.02):
            mismatches += 1
    assert mismatches == 0, f"{mismatches} cost_efficiency_rate mismatches"


def check_ratchet_exists(sub, golden):
    assert sub is not None, "demand_ratchet_tracker.csv not found"

def check_ratchet_row_count(sub, golden):
    assert len(sub) == 150, f"Expected 150 rows, got {len(sub)}"

def check_ratchet_columns(sub, golden):
    missing = set(golden.columns) - set(sub.columns)
    assert not missing, f"Missing columns: {missing}"

def check_ratchet_applied_values(sub, golden):
    invalid = set(sub["ratchet_applied"].unique()) - VALID_RATCHET
    assert not invalid, f"Invalid ratchet_applied values: {invalid}"

def check_ratchet_billed_ge_floor(sub, golden):
    bad = sub[sub["billed_demand"] < sub["contracted_floor"] - 0.5]
    assert len(bad) == 0, f"{len(bad)} rows with billed_demand below contracted_floor"

def check_ratchet_billed_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["billed_demand_g"], row["billed_demand_s"], tol=1.0):
            mismatches += 1
    assert mismatches <= 3, f"{mismatches} ratchet billed_demand mismatches"

def check_ratchet_peak_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "month"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["month_peak_demand_g"], row["month_peak_demand_s"], tol=1.0):
            mismatches += 1
    assert mismatches <= 3, f"{mismatches} ratchet month_peak_demand mismatches"

def check_ratchet_applied_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id", "month"])
    mismatches = (m["ratchet_applied_g"] != m["ratchet_applied_s"]).sum()
    assert mismatches <= 5, f"{mismatches} ratchet_applied mismatches"


def check_profile_exists(sub, golden):
    assert sub is not None, "facility_cost_profile.csv not found"

def check_profile_row_count(sub, golden):
    assert len(sub) == 25, f"Expected 25 rows, got {len(sub)}"

def check_profile_columns(sub, golden):
    missing = set(golden.columns) - set(sub.columns)
    assert not missing, f"Missing columns: {missing}"

def check_profile_cost_categories(sub, golden):
    invalid = set(sub["cost_category"].unique()) - VALID_COST_CATEGORIES
    assert not invalid, f"Invalid cost_category values: {invalid}"

def check_profile_cost_grade_values(sub, golden):
    invalid = set(sub["worst_cost_grade"].unique()) - VALID_COST_GRADES
    assert not invalid, f"Invalid worst_cost_grade values: {invalid}"

def check_profile_category_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id"])
    mismatches = (m["cost_category_g"] != m["cost_category_s"]).sum()
    assert mismatches <= 2, f"{mismatches} cost_category mismatches"

def check_profile_worst_grade_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id"])
    mismatches = (m["worst_cost_grade_g"] != m["worst_cost_grade_s"]).sum()
    assert mismatches <= 2, f"{mismatches} worst_cost_grade mismatches"

def check_profile_avg_bill_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["avg_monthly_bill_g"], row["avg_monthly_bill_s"], tol=5.0):
            mismatches += 1
    assert mismatches <= 2, f"{mismatches} avg_monthly_bill mismatches"

def check_profile_total_kwh_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["total_kwh_g"], row["total_kwh_s"], tol=10.0):
            mismatches += 1
    assert mismatches <= 2, f"{mismatches} profile total_kwh mismatches"

def check_profile_demand_charges_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["total_demand_charges_g"], row["total_demand_charges_s"], tol=10.0):
            mismatches += 1
    assert mismatches <= 2, f"{mismatches} total_demand_charges mismatches"

def check_profile_fes_match(sub, golden):
    m = merge_on_keys(sub, golden, ["facility_id"])
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["final_fes_g"], row["final_fes_s"], tol=2.0):
            mismatches += 1
    assert mismatches <= 2, f"{mismatches} final_fes mismatches"


def check_cross_bill_ratchet_demand(sub_bill, sub_ratchet):
    if sub_bill is None or sub_ratchet is None:
        assert False, "Missing files for cross-check"
    m = sub_bill.merge(sub_ratchet, on=["facility_id", "month"], suffixes=("_bill", "_ratchet"))
    mismatches = 0
    for _, row in m.iterrows():
        if not check_val(row["billed_demand_bill"], row["billed_demand_ratchet"], tol=0.5):
            mismatches += 1
    assert mismatches <= 2, f"{mismatches} billed_demand cross-file mismatches"

def check_cross_bill_profile_totals(sub_bill, sub_profile):
    if sub_bill is None or sub_profile is None:
        assert False, "Missing files for cross-check"
    for _, prow in sub_profile.iterrows():
        sid = prow["facility_id"]
        bill_sum = sub_bill[sub_bill["facility_id"] == sid]["total_bill"].sum()
        profile_avg = prow["avg_monthly_bill"]
        profile_months = prow["months_active"]
        if profile_months > 0:
            implied_total = profile_avg * profile_months
            if not check_val(bill_sum, implied_total, tol=10.0):
                assert False, f"Station {sid}: bill sum={bill_sum:.2f} vs profile implied total={implied_total:.2f}"

def check_cross_campus_curtailment(sub_campus, sub_curt):
    if sub_campus is None or sub_curt is None:
        assert False, "Missing files for cross-check"
    facilities = pd.read_csv(TASK_DIR / "facilities.csv")
    for _, crow in sub_campus.iterrows():
        campus = crow["campus"]
        quarter = crow["quarter"]
        campus_facs = facilities[facilities["campus"] == campus]["facility_id"].tolist()
        sub_curt_copy = sub_curt.copy()
        sub_curt_copy["event_date_dt"] = pd.to_datetime(sub_curt_copy["event_date"])
        sub_curt_copy["quarter"] = sub_curt_copy["event_date_dt"].dt.quarter
        curt_q = sub_curt_copy[
            (sub_curt_copy["facility_id"].isin(campus_facs)) &
            (sub_curt_copy["quarter"] == quarter)
        ]
        curt_sum = curt_q["credit_amount"].sum()
        campus_val = crow["total_curtailment_credits"]
        if not check_val(curt_sum, campus_val, tol=5.0):
            assert False, f"Campus {campus} Q{quarter}: curtailment sum={curt_sum:.2f} vs campus={campus_val:.2f}"


def run_verification():
    golden_bill = load_golden("facility_monthly_bill")
    golden_curt = load_golden("curtailment_credit_ledger")
    golden_campus = load_golden("campus_quarterly_summary")
    golden_ratchet = load_golden("demand_ratchet_tracker")
    golden_profile = load_golden("facility_cost_profile")

    sub_bill = load_sub("facility_monthly_bill")
    sub_curt = load_sub("curtailment_credit_ledger")
    sub_campus = load_sub("campus_quarterly_summary")
    sub_ratchet = load_sub("demand_ratchet_tracker")
    sub_profile = load_sub("facility_cost_profile")

    checks = []

    checks.append(("check_bill_exists", check_bill_exists, [sub_bill, golden_bill]))
    if sub_bill is not None:
        checks.extend([
            ("check_bill_row_count", check_bill_row_count, [sub_bill, golden_bill]),
            ("check_bill_columns", check_bill_columns, [sub_bill, golden_bill]),
            ("check_bill_facility_ids", check_bill_facility_ids, [sub_bill, golden_bill]),
            ("check_bill_months", check_bill_months, [sub_bill, golden_bill]),
            ("check_bill_kwh_nonneg", check_bill_kwh_nonneg, [sub_bill, golden_bill]),
            ("check_bill_energy_cost_nonneg", check_bill_energy_cost_nonneg, [sub_bill, golden_bill]),
            ("check_bill_demand_nonneg", check_bill_demand_nonneg, [sub_bill, golden_bill]),
            ("check_bill_demand_charge_nonneg", check_bill_demand_charge_nonneg, [sub_bill, golden_bill]),
            ("check_bill_pf_range", check_bill_pf_range, [sub_bill, golden_bill]),
            ("check_bill_pf_rounding", check_bill_pf_rounding, [sub_bill, golden_bill]),
            ("check_bill_solar_nonneg", check_bill_solar_nonneg, [sub_bill, golden_bill]),
            ("check_bill_cost_grade_values", check_bill_cost_grade_values, [sub_bill, golden_bill]),
            ("check_bill_billed_ge_contracted_floor", check_bill_billed_ge_contracted_floor, [sub_bill, golden_bill]),
            ("check_bill_total_kwh_match", check_bill_total_kwh_match, [sub_bill, golden_bill]),
            ("check_bill_energy_cost_match", check_bill_energy_cost_match, [sub_bill, golden_bill]),
            ("check_bill_demand_charge_match", check_bill_demand_charge_match, [sub_bill, golden_bill]),
            ("check_bill_pf_match", check_bill_pf_match, [sub_bill, golden_bill]),
            ("check_bill_pf_adjustment_match", check_bill_pf_adjustment_match, [sub_bill, golden_bill]),
            ("check_bill_solar_match", check_bill_solar_match, [sub_bill, golden_bill]),
            ("check_bill_equip_surcharge_match", check_bill_equip_surcharge_match, [sub_bill, golden_bill]),
            ("check_bill_overhead_match", check_bill_overhead_match, [sub_bill, golden_bill]),
            ("check_bill_total_match", check_bill_total_match, [sub_bill, golden_bill]),
            ("check_bill_cost_grade_match", check_bill_cost_grade_match, [sub_bill, golden_bill]),
            ("check_bill_ratchet_demand_match", check_bill_ratchet_demand_match, [sub_bill, golden_bill]),
            ("check_bill_billed_demand_match", check_bill_billed_demand_match, [sub_bill, golden_bill]),
            ("check_bill_consec_penalty_nonneg", check_bill_consec_penalty_nonneg, [sub_bill, golden_bill]),
            ("check_bill_consec_penalty_match", check_bill_consec_penalty_match, [sub_bill, golden_bill]),
            ("check_bill_fes_range", check_bill_fes_range, [sub_bill, golden_bill]),
            ("check_bill_fes_rounding", check_bill_fes_rounding, [sub_bill, golden_bill]),
            ("check_bill_fes_match", check_bill_fes_match, [sub_bill, golden_bill]),
        ])

    checks.append(("check_curt_exists", check_curt_exists, [sub_curt, golden_curt]))
    if sub_curt is not None:
        checks.extend([
            ("check_curt_row_count", check_curt_row_count, [sub_curt, golden_curt]),
            ("check_curt_columns", check_curt_columns, [sub_curt, golden_curt]),
            ("check_curt_target_met_values", check_curt_target_met_values, [sub_curt, golden_curt]),
            ("check_curt_credit_nonneg", check_curt_credit_nonneg, [sub_curt, golden_curt]),
            ("check_curt_credit_zero_when_not_met", check_curt_credit_zero_when_not_met, [sub_curt, golden_curt]),
            ("check_curt_baseline_match", check_curt_baseline_match, [sub_curt, golden_curt]),
            ("check_curt_credit_match", check_curt_credit_match, [sub_curt, golden_curt]),
            ("check_curt_target_met_match", check_curt_target_met_match, [sub_curt, golden_curt]),
        ])

    checks.append(("check_campus_exists", check_campus_exists, [sub_campus, golden_campus]))
    if sub_campus is not None:
        checks.extend([
            ("check_campus_row_count", check_campus_row_count, [sub_campus, golden_campus]),
            ("check_campus_columns", check_campus_columns, [sub_campus, golden_campus]),
            ("check_campus_pairs", check_campus_pairs, [sub_campus, golden_campus]),
            ("check_campus_kwh_match", check_campus_kwh_match, [sub_campus, golden_campus]),
            ("check_campus_bills_match", check_campus_bills_match, [sub_campus, golden_campus]),
            ("check_campus_pf_match", check_campus_pf_match, [sub_campus, golden_campus]),
            ("check_campus_fes_match", check_campus_fes_match, [sub_campus, golden_campus]),
            ("check_campus_efficiency_rate_match", check_campus_efficiency_rate_match, [sub_campus, golden_campus]),
        ])

    checks.append(("check_ratchet_exists", check_ratchet_exists, [sub_ratchet, golden_ratchet]))
    if sub_ratchet is not None:
        checks.extend([
            ("check_ratchet_row_count", check_ratchet_row_count, [sub_ratchet, golden_ratchet]),
            ("check_ratchet_columns", check_ratchet_columns, [sub_ratchet, golden_ratchet]),
            ("check_ratchet_applied_values", check_ratchet_applied_values, [sub_ratchet, golden_ratchet]),
            ("check_ratchet_billed_ge_floor", check_ratchet_billed_ge_floor, [sub_ratchet, golden_ratchet]),
            ("check_ratchet_billed_match", check_ratchet_billed_match, [sub_ratchet, golden_ratchet]),
            ("check_ratchet_peak_match", check_ratchet_peak_match, [sub_ratchet, golden_ratchet]),
            ("check_ratchet_applied_match", check_ratchet_applied_match, [sub_ratchet, golden_ratchet]),
        ])

    checks.append(("check_profile_exists", check_profile_exists, [sub_profile, golden_profile]))
    if sub_profile is not None:
        checks.extend([
            ("check_profile_row_count", check_profile_row_count, [sub_profile, golden_profile]),
            ("check_profile_columns", check_profile_columns, [sub_profile, golden_profile]),
            ("check_profile_cost_categories", check_profile_cost_categories, [sub_profile, golden_profile]),
            ("check_profile_cost_grade_values", check_profile_cost_grade_values, [sub_profile, golden_profile]),
            ("check_profile_category_match", check_profile_category_match, [sub_profile, golden_profile]),
            ("check_profile_worst_grade_match", check_profile_worst_grade_match, [sub_profile, golden_profile]),
            ("check_profile_avg_bill_match", check_profile_avg_bill_match, [sub_profile, golden_profile]),
            ("check_profile_total_kwh_match", check_profile_total_kwh_match, [sub_profile, golden_profile]),
            ("check_profile_demand_charges_match", check_profile_demand_charges_match, [sub_profile, golden_profile]),
            ("check_profile_fes_match", check_profile_fes_match, [sub_profile, golden_profile]),
        ])

    if sub_bill is not None and sub_ratchet is not None:
        checks.append(("check_cross_bill_ratchet_demand", check_cross_bill_ratchet_demand, [sub_bill, sub_ratchet]))
    if sub_bill is not None and sub_profile is not None:
        checks.append(("check_cross_bill_profile_totals", check_cross_bill_profile_totals, [sub_bill, sub_profile]))
    if sub_campus is not None and sub_curt is not None:
        checks.append(("check_cross_campus_curtailment", check_cross_campus_curtailment, [sub_campus, sub_curt]))

    passed = 0
    total = len(checks)

    for name, check_func, args in checks:
        try:
            check_func(*args)
            print(f"PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {name}: {e}")
        except Exception as e:
            print(f"ERROR: {name}: {e}")

    print(f"\n{passed}/{total}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    run_verification()

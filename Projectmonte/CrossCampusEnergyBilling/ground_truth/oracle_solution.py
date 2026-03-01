import pandas as pd
import numpy as np
from pathlib import Path
import math
import sys


def load_data(task_dir):
    facilities = pd.read_csv(task_dir / "facilities.csv")
    readings = pd.read_csv(task_dir / "meter_readings.csv")
    rate_schedules = pd.read_csv(task_dir / "rate_schedules.csv")
    day_class = pd.read_csv(task_dir / "day_classifications.csv")
    curtailment = pd.read_csv(task_dir / "curtailment_events.csv")
    zone_holidays = pd.read_csv(task_dir / "zone_holidays.csv")
    pf_tariffs = pd.read_csv(task_dir / "pf_tariffs.csv")
    equipment = pd.read_csv(task_dir / "equipment_efficiency.csv")
    return facilities, readings, rate_schedules, day_class, curtailment, zone_holidays, pf_tariffs, equipment


def get_facility_zone(facility_id, facilities_df):
    row = facilities_df[facilities_df["facility_id"] == facility_id].iloc[0]
    return row["utility_zone"]


def get_facility_campus(facility_id, facilities_df):
    row = facilities_df[facilities_df["facility_id"] == facility_id].iloc[0]
    return row["campus"]


def fill_nulls(readings_df, facilities_df):
    readings_df = readings_df.copy()
    readings_df["reading_date"] = pd.to_datetime(readings_df["reading_date"])
    readings_df = readings_df.sort_values(["facility_id", "reading_date"]).reset_index(drop=True)

    readings_df["original_kwh_null"] = readings_df["kwh_consumed"].isna()
    readings_df["original_peak_null"] = readings_df["peak_demand_kw"].isna()

    readings_df["solar_kwh"] = readings_df["solar_kwh"].fillna(0.0)

    for fid in facilities_df["facility_id"].unique():
        mask = readings_df["facility_id"] == fid
        fac_idx = readings_df[mask].index.tolist()

        last_valid_kwh = 0.0
        for idx in fac_idx:
            kwh_null = pd.isna(readings_df.loc[idx, "kwh_consumed"])
            peak_null = pd.isna(readings_df.loc[idx, "peak_demand_kw"])

            if kwh_null and not peak_null:
                readings_df.loc[idx, "kwh_consumed"] = readings_df.loc[idx, "peak_demand_kw"] * 8
            elif kwh_null and peak_null:
                readings_df.loc[idx, "kwh_consumed"] = last_valid_kwh
                readings_df.loc[idx, "peak_demand_kw"] = 0.0
            else:
                last_valid_kwh = readings_df.loc[idx, "kwh_consumed"]

            if not kwh_null and peak_null:
                readings_df.loc[idx, "peak_demand_kw"] = round(readings_df.loc[idx, "kwh_consumed"] * 0.15, 2)

    readings_df["month"] = readings_df["reading_date"].dt.to_period("M").astype(str)

    for fid in facilities_df["facility_id"].unique():
        mask = readings_df["facility_id"] == fid
        fac_data = readings_df[mask]

        for month in fac_data["month"].unique():
            month_mask = mask & (readings_df["month"] == month)
            month_kvar = readings_df.loc[month_mask, "kvar_reactive"]
            valid_kvar = month_kvar.dropna()

            if len(valid_kvar) > 0:
                median_val = valid_kvar.median()
            else:
                median_val = 0.0

            null_kvar_mask = month_mask & readings_df["kvar_reactive"].isna()
            readings_df.loc[null_kvar_mask, "kvar_reactive"] = median_val

    return readings_df


def compute_daily_tou(readings_df, facilities_df, day_class_df, rate_sched_df):
    readings_df = readings_df.copy()
    readings_df["peak_kwh"] = 0.0
    readings_df["shoulder_kwh"] = 0.0
    readings_df["offpeak_kwh"] = 0.0
    readings_df["daily_energy_cost"] = 0.0

    day_class_df = day_class_df.copy()
    day_class_df["date"] = pd.to_datetime(day_class_df["date"])

    for idx, row in readings_df.iterrows():
        fid = row["facility_id"]
        zone = get_facility_zone(fid, facilities_df)
        rd = row["reading_date"]
        kwh = row["kwh_consumed"]

        dc_match = day_class_df[
            (day_class_df["utility_zone"] == zone) &
            (day_class_df["date"] == rd)
        ]

        if len(dc_match) == 0:
            continue

        dc = dc_match.iloc[0]
        season = dc["season"]
        pf = dc["peak_frac"]
        sf = dc["shoulder_frac"]
        of = dc["offpeak_frac"]

        p_kwh = round(kwh * pf, 3)
        s_kwh = round(kwh * sf, 3)
        o_kwh = round(kwh * of, 3)

        readings_df.loc[idx, "peak_kwh"] = p_kwh
        readings_df.loc[idx, "shoulder_kwh"] = s_kwh
        readings_df.loc[idx, "offpeak_kwh"] = o_kwh

        peak_rate_row = rate_sched_df[
            (rate_sched_df["utility_zone"] == zone) &
            (rate_sched_df["season"] == season) &
            (rate_sched_df["period_type"] == "PEAK")
        ]
        shoulder_rate_row = rate_sched_df[
            (rate_sched_df["utility_zone"] == zone) &
            (rate_sched_df["season"] == season) &
            (rate_sched_df["period_type"] == "SHOULDER")
        ]
        offpeak_rate_row = rate_sched_df[
            (rate_sched_df["utility_zone"] == zone) &
            (rate_sched_df["season"] == season) &
            (rate_sched_df["period_type"] == "OFF_PEAK")
        ]

        pr = peak_rate_row.iloc[0]["rate_per_kwh"] if len(peak_rate_row) > 0 else 0
        sr = shoulder_rate_row.iloc[0]["rate_per_kwh"] if len(shoulder_rate_row) > 0 else 0
        orr = offpeak_rate_row.iloc[0]["rate_per_kwh"] if len(offpeak_rate_row) > 0 else 0

        daily_cost = round(p_kwh * pr + s_kwh * sr + o_kwh * orr, 2)
        readings_df.loc[idx, "daily_energy_cost"] = daily_cost

    return readings_df


def compute_monthly_aggregation(readings_df, facilities_df):
    all_months = sorted(readings_df["month"].unique())
    all_facilities = facilities_df["facility_id"].unique()

    results = []
    for fid in all_facilities:
        for month in all_months:
            fac_month = readings_df[
                (readings_df["facility_id"] == fid) &
                (readings_df["month"] == month)
            ]

            if len(fac_month) == 0:
                results.append({
                    "facility_id": fid,
                    "month": month,
                    "total_kwh": 0.0,
                    "valid_reading_days": 0,
                    "peak_kwh": 0.0,
                    "shoulder_kwh": 0.0,
                    "offpeak_kwh": 0.0,
                    "energy_cost": 0.0,
                    "month_peak_demand": 0.0,
                    "total_kvar": 0.0,
                    "total_solar": 0.0,
                    "has_readings": False,
                })
                continue

            total_kwh = round(fac_month["kwh_consumed"].sum(), 2)
            valid_days = int((~fac_month["original_kwh_null"]).sum())
            peak_kwh = round(fac_month["peak_kwh"].sum(), 3)
            shoulder_kwh = round(fac_month["shoulder_kwh"].sum(), 3)
            offpeak_kwh = round(fac_month["offpeak_kwh"].sum(), 3)
            energy_cost = round(fac_month["daily_energy_cost"].sum(), 2)
            month_peak = round(fac_month["peak_demand_kw"].max(), 1) if fac_month["peak_demand_kw"].notna().any() else 0.0
            total_kvar = round(fac_month["kvar_reactive"].sum(), 2)
            total_solar = round(fac_month["solar_kwh"].sum(), 2)

            results.append({
                "facility_id": fid,
                "month": month,
                "total_kwh": total_kwh,
                "valid_reading_days": valid_days,
                "peak_kwh": peak_kwh,
                "shoulder_kwh": shoulder_kwh,
                "offpeak_kwh": offpeak_kwh,
                "energy_cost": energy_cost,
                "month_peak_demand": month_peak,
                "total_kvar": total_kvar,
                "total_solar": total_solar,
                "has_readings": True,
            })

    return pd.DataFrame(results)


def compute_solar_credits(monthly_df):
    monthly_df = monthly_df.copy()
    monthly_df["solar_credit"] = 0.0

    for idx, row in monthly_df.iterrows():
        total_solar = row["total_solar"]
        total_kwh = row["total_kwh"]
        energy_cost = row["energy_cost"]

        if total_solar <= 0 or total_kwh <= 0:
            continue

        credit_kwh = min(total_solar, total_kwh)
        weighted_avg_rate = energy_cost / total_kwh if total_kwh > 0 else 0.0
        solar_credit = round(credit_kwh * weighted_avg_rate * 0.80, 2)
        monthly_df.loc[idx, "solar_credit"] = solar_credit

    return monthly_df


def compute_demand_ratchet(monthly_df, facilities_df):
    monthly_df = monthly_df.sort_values(["facility_id", "month"]).copy()
    monthly_df["ratchet_demand"] = 0.0
    monthly_df["contracted_floor"] = 0.0
    monthly_df["billed_demand"] = 0.0
    monthly_df["ratchet_applied"] = "NO"

    for fid in facilities_df["facility_id"].unique():
        fac_row = facilities_df[facilities_df["facility_id"] == fid].iloc[0]
        contracted = fac_row["contracted_demand_kw"]
        contracted_floor = round(contracted * 0.70, 1)

        mask = monthly_df["facility_id"] == fid
        fac_data = monthly_df[mask].sort_values("month")
        indices = fac_data.index.tolist()
        month_peaks = []

        for i, idx in enumerate(indices):
            mp = fac_data.loc[idx, "month_peak_demand"]
            month_peaks.append(mp)

            if i == 0:
                ratchet_demand = mp
            else:
                lookback = month_peaks[max(0, i - 3):i]
                prior_max = max(lookback) if lookback else 0.0
                ratchet_demand = max(mp, round(0.80 * prior_max, 1))

            billed = round(max(ratchet_demand, contracted_floor), 1)
            ratchet_applied = "YES" if billed > mp else "NO"

            monthly_df.loc[idx, "ratchet_demand"] = round(ratchet_demand, 1)
            monthly_df.loc[idx, "contracted_floor"] = contracted_floor
            monthly_df.loc[idx, "billed_demand"] = billed
            monthly_df.loc[idx, "ratchet_applied"] = ratchet_applied

    return monthly_df


def compute_demand_charge(monthly_df, facilities_df, rate_sched_df, day_class_df):
    monthly_df = monthly_df.copy()
    monthly_df["demand_charge"] = 0.0

    day_class_df = day_class_df.copy()
    day_class_df["date"] = pd.to_datetime(day_class_df["date"])

    for idx, row in monthly_df.iterrows():
        fid = row["facility_id"]
        month_str = row["month"]
        zone = get_facility_zone(fid, facilities_df)

        period = pd.Period(month_str, freq="M")
        last_day = period.end_time

        dc_match = day_class_df[
            (day_class_df["utility_zone"] == zone) &
            (day_class_df["date"].dt.date == last_day.date())
        ]
        if len(dc_match) > 0:
            season = dc_match.iloc[0]["season"]
        else:
            month_num = period.month
            season = "WINTER" if month_num <= 4 else "SUMMER"

        rate_match = rate_sched_df[
            (rate_sched_df["utility_zone"] == zone) &
            (rate_sched_df["season"] == season) &
            (rate_sched_df["period_type"] == "PEAK")
        ]
        if len(rate_match) > 0:
            demand_rate = rate_match.iloc[0]["demand_charge_per_kw"]
        else:
            demand_rate = 0.0

        billed = row["billed_demand"]
        demand_charge = round(billed * demand_rate, 2)
        monthly_df.loc[idx, "demand_charge"] = demand_charge

    return monthly_df


def compute_power_factor(monthly_df, facilities_df, pf_tariffs_df):
    monthly_df = monthly_df.copy()
    monthly_df["power_factor"] = 1.0
    monthly_df["pf_adjustment"] = 0.0

    for idx, row in monthly_df.iterrows():
        total_kwh = row["total_kwh"]
        total_kvar = row["total_kvar"]
        fid = row["facility_id"]
        zone = get_facility_zone(fid, facilities_df)

        if total_kwh <= 0 and total_kvar <= 0:
            monthly_df.loc[idx, "power_factor"] = 1.0
            continue

        apparent = round(math.sqrt(total_kwh ** 2 + total_kvar ** 2), 2)
        if apparent > 0:
            pf = round(total_kwh / apparent, 4)
        else:
            pf = 1.0

        monthly_df.loc[idx, "power_factor"] = pf

        tariff = pf_tariffs_df[pf_tariffs_df["utility_zone"] == zone]
        if len(tariff) == 0:
            continue
        tariff = tariff.iloc[0]
        min_pf = tariff["min_pf"]
        penalty_rate = tariff["penalty_pct_per_point"]
        bonus_rate = tariff["bonus_pct_per_point_above_95"]
        demand_charge = row["demand_charge"]

        if pf < min_pf:
            points_below = round((min_pf - pf) * 100, 1)
            pf_adj = round(points_below * penalty_rate * demand_charge / 100, 2)
            monthly_df.loc[idx, "pf_adjustment"] = pf_adj
        elif pf >= 0.95:
            points_above = round((pf - 0.95) * 100, 1)
            pf_adj = -round(points_above * bonus_rate * demand_charge / 100, 2)
            monthly_df.loc[idx, "pf_adjustment"] = pf_adj

    return monthly_df


def compute_equipment_surcharge(monthly_df, equipment_df):
    monthly_df = monthly_df.copy()
    monthly_df["equipment_surcharge"] = 0.0

    equipment_df = equipment_df.copy()
    equipment_df["install_date"] = pd.to_datetime(equipment_df["install_date"])

    for idx, row in monthly_df.iterrows():
        fid = row["facility_id"]
        month_str = row["month"]
        energy_cost = row["energy_cost"]
        period = pd.Period(month_str, freq="M")
        month_start = period.start_time

        fac_equip = equipment_df[
            (equipment_df["facility_id"] == fid) &
            (equipment_df["install_date"] < month_start + pd.DateOffset(months=1))
        ]

        if len(fac_equip) == 0:
            continue

        efficiencies = []
        for _, eq in fac_equip.iterrows():
            install_dt = eq["install_date"]
            months_since = (month_start.year - install_dt.year) * 12 + (month_start.month - install_dt.month)
            if months_since < 0:
                months_since = 0
            current_eff = eq["rated_efficiency"] - eq["degradation_rate_monthly"] * months_since
            current_eff = max(0.50, current_eff)
            current_eff = round(current_eff, 3)
            efficiencies.append(current_eff)

        avg_eff = round(np.mean(efficiencies), 3)

        if avg_eff < 0.85:
            surcharge = round((0.85 - avg_eff) * energy_cost * 0.25, 2)
            monthly_df.loc[idx, "equipment_surcharge"] = surcharge

    return monthly_df


def compute_overhead(monthly_df, facilities_df):
    monthly_df = monthly_df.copy()
    monthly_df["overhead_cost"] = 0.0

    overhead_rates = {1: 0.05, 2: 0.08, 3: 0.12, 4: 0.15}

    for idx, row in monthly_df.iterrows():
        fid = row["facility_id"]
        fac_row = facilities_df[facilities_df["facility_id"] == fid].iloc[0]
        dc = fac_row["demand_class"]
        energy_cost = row["energy_cost"]
        demand_charge = row["demand_charge"]
        rate = overhead_rates.get(dc, 0.05)

        base_overhead = round((energy_cost + demand_charge) * rate, 2)
        admin_surcharge = round(energy_cost * 0.03, 2) if dc >= 3 else 0.0
        total_overhead = round(base_overhead + admin_surcharge, 2)

        monthly_df.loc[idx, "overhead_cost"] = total_overhead

    return monthly_df


def compute_total_bill(monthly_df):
    monthly_df = monthly_df.copy()
    monthly_df["total_bill"] = 0.0

    for idx, row in monthly_df.iterrows():
        total = (
            row["energy_cost"]
            + row["demand_charge"]
            + row["pf_adjustment"]
            + row["equipment_surcharge"]
            + row["overhead_cost"]
            - row["solar_credit"]
        )
        monthly_df.loc[idx, "total_bill"] = round(total, 2)

    return monthly_df


def classify_cost_grade(monthly_df, facilities_df, pf_tariffs_df):
    monthly_df = monthly_df.copy()
    monthly_df["cost_grade"] = "STANDARD"

    for idx, row in monthly_df.iterrows():
        fid = row["facility_id"]
        fac_row = facilities_df[facilities_df["facility_id"] == fid].iloc[0]
        contracted = fac_row["contracted_demand_kw"]
        zone = fac_row["utility_zone"]

        tariff = pf_tariffs_df[pf_tariffs_df["utility_zone"] == zone]
        min_pf = tariff.iloc[0]["min_pf"] if len(tariff) > 0 else 0.90

        billed = row["billed_demand"]
        pf = row["power_factor"]
        pf_adj = row["pf_adjustment"]
        solar_credit = row["solar_credit"]
        equip_surcharge = row["equipment_surcharge"]

        if pf < 0.75 and billed > contracted * 1.20:
            grade = "CRITICAL"
        elif billed > contracted * 1.30:
            grade = "OVER_DEMAND"
        elif equip_surcharge > 0 and pf < min_pf:
            grade = "LOW_EFFICIENCY"
        elif pf_adj > 0:
            grade = "PF_PENALTY"
        elif solar_credit > 0 and pf >= 0.95 and billed <= contracted:
            grade = "EFFICIENT"
        else:
            grade = "STANDARD"

        monthly_df.loc[idx, "cost_grade"] = grade

    return monthly_df


def compute_consecutive_penalty_months(monthly_df):
    monthly_df = monthly_df.sort_values(["facility_id", "month"]).copy()
    monthly_df["consecutive_penalty_months"] = 0

    for fid in monthly_df["facility_id"].unique():
        mask = monthly_df["facility_id"] == fid
        fac_data = monthly_df[mask].sort_values("month")
        indices = fac_data.index.tolist()

        consecutive = 0
        for idx in indices:
            grade = monthly_df.loc[idx, "cost_grade"]
            if grade in {"CRITICAL", "OVER_DEMAND", "LOW_EFFICIENCY", "PF_PENALTY"}:
                monthly_df.loc[idx, "consecutive_penalty_months"] = consecutive
                consecutive += 1
            else:
                consecutive = 0
                monthly_df.loc[idx, "consecutive_penalty_months"] = 0

    return monthly_df


def compute_facility_efficiency_score(monthly_df):
    monthly_df = monthly_df.sort_values(["facility_id", "month"]).copy()
    monthly_df["facility_efficiency_score"] = 0.0

    for fid in monthly_df["facility_id"].unique():
        mask = monthly_df["facility_id"] == fid
        fac_data = monthly_df[mask].sort_values("month")
        indices = fac_data.index.tolist()

        prev_fes = 50.0
        for idx in indices:
            grade = monthly_df.loc[idx, "cost_grade"]
            pf = monthly_df.loc[idx, "power_factor"]
            has_data = monthly_df.loc[idx, "has_readings"]

            if not has_data:
                fes = round(prev_fes * 0.90, 3)
            elif grade in {"EFFICIENT", "STANDARD"}:
                current_input = round(pf * 100, 2)
                fes = round(prev_fes * 0.85 + current_input * 0.15, 3)
            elif grade in {"PF_PENALTY", "LOW_EFFICIENCY"}:
                current_input = round(pf * 100, 2)
                fes = round(prev_fes * 0.60 + current_input * 0.40, 3)
            else:
                fes = round(min(prev_fes * 0.50, 40.0), 3)

            fes = round(max(0.0, min(100.0, fes)), 3)
            monthly_df.loc[idx, "facility_efficiency_score"] = fes
            prev_fes = fes

    return monthly_df


def compute_curtailment_credits(readings_df, curtailment_df, facilities_df, zone_holidays_df):
    readings_df = readings_df.copy()
    curtailment_df = curtailment_df.copy()
    curtailment_df["event_date"] = pd.to_datetime(curtailment_df["event_date"])
    zone_holidays_df = zone_holidays_df.copy()
    zone_holidays_df["date"] = pd.to_datetime(zone_holidays_df["date"])

    curtailment_dates_by_campus = {}
    for _, ev in curtailment_df.iterrows():
        campus = ev["campus"]
        if campus not in curtailment_dates_by_campus:
            curtailment_dates_by_campus[campus] = set()
        curtailment_dates_by_campus[campus].add(ev["event_date"])

    ledger = []

    for _, event in curtailment_df.iterrows():
        campus = event["campus"]
        event_date = event["event_date"]
        target_pct = event["target_reduction_pct"]
        credit_rate = event["credit_rate_per_kwh"]

        campus_facilities = facilities_df[facilities_df["campus"] == campus]["facility_id"].tolist()

        for fid in campus_facilities:
            zone = get_facility_zone(fid, facilities_df)

            zone_hols = set(
                zone_holidays_df[
                    (zone_holidays_df["utility_zone"] == zone) |
                    (zone_holidays_df["utility_zone"] == "ALL")
                ]["date"].tolist()
            )

            campus_curt_dates = curtailment_dates_by_campus.get(campus, set())

            fac_readings = readings_df[readings_df["facility_id"] == fid].sort_values("reading_date")

            qualifying_days = []
            check_date = event_date - pd.Timedelta(days=1)
            days_checked = 0
            max_lookback = 60

            while len(qualifying_days) < 10 and days_checked < max_lookback:
                if check_date.weekday() < 5:
                    if check_date not in zone_hols and check_date not in campus_curt_dates:
                        day_reading = fac_readings[fac_readings["reading_date"] == check_date]
                        if len(day_reading) > 0:
                            qualifying_days.append(day_reading.iloc[0]["kwh_consumed"])
                check_date -= pd.Timedelta(days=1)
                days_checked += 1

            if len(qualifying_days) > 0:
                baseline_kwh = round(np.mean(qualifying_days), 2)
            else:
                baseline_kwh = 0.0

            event_reading = fac_readings[fac_readings["reading_date"] == event_date]
            if len(event_reading) > 0:
                actual_kwh = event_reading.iloc[0]["kwh_consumed"]
            else:
                actual_kwh = 0.0

            reduction_kwh = round(max(0, baseline_kwh - actual_kwh), 2)
            reduction_pct = round(reduction_kwh / baseline_kwh, 4) if baseline_kwh > 0 else 0.0
            target_met = "YES" if reduction_pct >= target_pct else "NO"
            credit_amount = round(reduction_kwh * credit_rate, 2) if target_met == "YES" else 0.0

            ledger.append({
                "facility_id": fid,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "baseline_kwh": baseline_kwh,
                "actual_kwh": round(actual_kwh, 2),
                "reduction_kwh": reduction_kwh,
                "reduction_pct": reduction_pct,
                "target_met": target_met,
                "credit_amount": credit_amount,
            })

    return pd.DataFrame(ledger)


def compute_campus_quarterly_summary(monthly_df, curtailment_ledger_df, facilities_df):
    monthly_df = monthly_df.copy()
    monthly_df["month_dt"] = pd.to_datetime(monthly_df["month"])
    monthly_df["quarter"] = monthly_df["month_dt"].dt.quarter

    curtailment_ledger_df = curtailment_ledger_df.copy()
    curtailment_ledger_df["event_date_dt"] = pd.to_datetime(curtailment_ledger_df["event_date"])
    curtailment_ledger_df["quarter"] = curtailment_ledger_df["event_date_dt"].dt.quarter

    summaries = []
    for campus in sorted(facilities_df["campus"].unique()):
        campus_facs = facilities_df[facilities_df["campus"] == campus]["facility_id"].tolist()
        facility_count = len(campus_facs)

        for quarter in sorted(monthly_df["quarter"].unique()):
            qdata = monthly_df[
                (monthly_df["facility_id"].isin(campus_facs)) &
                (monthly_df["quarter"] == quarter)
            ]

            active_facs = qdata[qdata["has_readings"]]["facility_id"].nunique()
            total_kwh = round(qdata["total_kwh"].sum(), 2)
            total_energy_cost = round(qdata["energy_cost"].sum(), 2)
            total_demand_charges = round(qdata["demand_charge"].sum(), 2)
            total_solar_credits = round(qdata["solar_credit"].sum(), 2)

            curt_q = curtailment_ledger_df[
                (curtailment_ledger_df["facility_id"].isin(campus_facs)) &
                (curtailment_ledger_df["quarter"] == quarter)
            ]
            total_curtailment_credits = round(curt_q["credit_amount"].sum(), 2)

            total_bills = round(qdata["total_bill"].sum(), 2)

            pf_values = qdata[qdata["has_readings"] & (qdata["total_kwh"] > 0)]["power_factor"]
            avg_pf = round(pf_values.mean(), 4) if len(pf_values) > 0 else 0.0

            avg_cost_per_kwh = round(total_bills / total_kwh, 4) if total_kwh > 0 else 0.0

            last_month_map = {1: "2024-03", 2: "2024-06"}
            last_month = last_month_map.get(quarter, None)

            fes_data = monthly_df[
                (monthly_df["facility_id"].isin(campus_facs)) &
                (monthly_df["month"] == last_month) &
                (monthly_df["has_readings"])
            ] if last_month else pd.DataFrame()
            fes_values = fes_data["facility_efficiency_score"] if len(fes_data) > 0 else pd.Series(dtype=float)
            avg_fes = round(fes_values.mean(), 3) if len(fes_values) > 0 else 0.0
            if last_month:
                end_data = monthly_df[
                    (monthly_df["facility_id"].isin(campus_facs)) &
                    (monthly_df["month"] == last_month) &
                    (monthly_df["has_readings"])
                ]
                efficient = end_data[
                    end_data["cost_grade"].isin({"EFFICIENT", "STANDARD"})
                ]["facility_id"].nunique()
                efficiency_rate = round(efficient / active_facs, 4) if active_facs > 0 else 0.0
            else:
                efficiency_rate = 0.0

            summaries.append({
                "campus": campus,
                "quarter": quarter,
                "facility_count": facility_count,
                "active_facilities": active_facs,
                "total_kwh": total_kwh,
                "total_energy_cost": total_energy_cost,
                "total_demand_charges": total_demand_charges,
                "total_solar_credits": total_solar_credits,
                "total_curtailment_credits": total_curtailment_credits,
                "total_bills": total_bills,
                "avg_power_factor": avg_pf,
                "avg_fes": avg_fes,
                "cost_efficiency_rate": efficiency_rate,
            })

    return pd.DataFrame(summaries)


def build_demand_ratchet_tracker(monthly_df, facilities_df):
    monthly_df = monthly_df.sort_values(["facility_id", "month"]).copy()
    rows = []

    for fid in facilities_df["facility_id"].unique():
        fac_row = facilities_df[facilities_df["facility_id"] == fid].iloc[0]
        contracted_floor = round(fac_row["contracted_demand_kw"] * 0.70, 1)

        mask = monthly_df["facility_id"] == fid
        fac_data = monthly_df[mask].sort_values("month")

        month_peaks = []
        for _, r in fac_data.iterrows():
            mp = r["month_peak_demand"]
            month_peaks.append(mp)
            i = len(month_peaks) - 1

            if i == 0:
                prior_max = 0.0
            else:
                lookback = month_peaks[max(0, i - 3):i]
                prior_max = max(lookback) if lookback else 0.0

            rows.append({
                "facility_id": fid,
                "month": r["month"],
                "month_peak_demand": mp,
                "prior_3mo_max": round(prior_max, 1),
                "ratchet_demand": r["ratchet_demand"],
                "contracted_floor": contracted_floor,
                "billed_demand": r["billed_demand"],
                "ratchet_applied": r["ratchet_applied"],
            })

    return pd.DataFrame(rows)


def compute_facility_cost_profile(monthly_df, facilities_df):
    profiles = []

    grade_priority = {
        "CRITICAL": 6,
        "OVER_DEMAND": 5,
        "LOW_EFFICIENCY": 4,
        "PF_PENALTY": 3,
        "EFFICIENT": 1,
        "STANDARD": 0,
    }

    for _, fac in facilities_df.iterrows():
        fid = fac["facility_id"]
        campus = fac["campus"]
        zone = fac["utility_zone"]
        dc = fac["demand_class"]
        contracted = fac["contracted_demand_kw"]

        fac_data = monthly_df[monthly_df["facility_id"] == fid].sort_values("month")
        active_months = fac_data[fac_data["has_readings"]]
        months_active = len(active_months)

        total_kwh = round(fac_data["total_kwh"].sum(), 2)

        bills = active_months["total_bill"]
        avg_monthly_bill = round(bills.mean(), 2) if len(bills) > 0 else 0.0
        min_monthly_bill = round(bills.min(), 2) if len(bills) > 0 else 0.0
        max_monthly_bill = round(bills.max(), 2) if len(bills) > 0 else 0.0

        last_row = fac_data.iloc[-1] if len(fac_data) > 0 else None
        final_billed_demand = round(last_row["billed_demand"], 1) if last_row is not None else 0.0
        final_fes = round(last_row["facility_efficiency_score"], 3) if last_row is not None else 0.0

        total_demand = round(fac_data["demand_charge"].sum(), 2)
        total_solar = round(fac_data["solar_credit"].sum(), 2)
        total_pf = round(fac_data["pf_adjustment"].sum(), 2)

        worst_priority = -1
        worst_grade = "STANDARD"
        for _, mr in fac_data.iterrows():
            g = mr["cost_grade"]
            p = grade_priority.get(g, 0)
            if p > worst_priority:
                worst_priority = p
                worst_grade = g

        over_demand_months = len(fac_data[fac_data["billed_demand"] > contracted * 1.20])
        pf_penalty_months = len(fac_data[fac_data["pf_adjustment"] > 0])

        if final_billed_demand > contracted * 1.20 or over_demand_months >= 3:
            cost_category = "HIGH_COST"
        elif final_billed_demand > contracted or pf_penalty_months >= 2:
            cost_category = "ELEVATED"
        elif avg_monthly_bill > 0 and total_pf < 0:
            cost_category = "OPTIMIZED"
        else:
            cost_category = "NORMAL"

        profiles.append({
            "facility_id": fid,
            "campus": campus,
            "utility_zone": zone,
            "demand_class": dc,
            "months_active": months_active,
            "total_kwh": total_kwh,
            "avg_monthly_bill": avg_monthly_bill,
            "min_monthly_bill": min_monthly_bill,
            "max_monthly_bill": max_monthly_bill,
            "final_billed_demand": final_billed_demand,
            "total_demand_charges": total_demand,
            "total_solar_credits": total_solar,
            "total_pf_adjustments": total_pf,
            "worst_cost_grade": worst_grade,
            "final_fes": final_fes,
            "cost_category": cost_category,
        })

    return pd.DataFrame(profiles)


def main():
    task_dir = Path(__file__).parent.parent / "task"
    ground_truth = Path(__file__).parent

    facilities, readings, rate_sched, day_class, curtailment, zone_holidays, pf_tariffs, equipment = load_data(task_dir)

    processed = fill_nulls(readings, facilities)

    processed = compute_daily_tou(processed, facilities, day_class, rate_sched)

    monthly = compute_monthly_aggregation(processed, facilities)

    monthly = compute_solar_credits(monthly)

    monthly = compute_demand_ratchet(monthly, facilities)

    monthly = compute_demand_charge(monthly, facilities, rate_sched, day_class)

    monthly = compute_power_factor(monthly, facilities, pf_tariffs)

    monthly = compute_equipment_surcharge(monthly, equipment)

    monthly = compute_overhead(monthly, facilities)

    monthly = compute_total_bill(monthly)

    monthly = classify_cost_grade(monthly, facilities, pf_tariffs)

    monthly = compute_consecutive_penalty_months(monthly)

    monthly = compute_facility_efficiency_score(monthly)

    bill_output = monthly[[
        "facility_id", "month", "total_kwh", "valid_reading_days",
        "peak_kwh", "shoulder_kwh", "offpeak_kwh", "energy_cost",
        "month_peak_demand", "ratchet_demand", "billed_demand", "demand_charge",
        "power_factor", "pf_adjustment", "solar_credit", "equipment_surcharge",
        "overhead_cost", "total_bill", "cost_grade",
        "consecutive_penalty_months", "facility_efficiency_score",
    ]].copy()
    bill_output.to_csv(task_dir / "facility_monthly_bill.csv", index=False)

    curt_ledger = compute_curtailment_credits(processed, curtailment, facilities, zone_holidays)
    curt_ledger.to_csv(task_dir / "curtailment_credit_ledger.csv", index=False)

    campus_summary = compute_campus_quarterly_summary(monthly, curt_ledger, facilities)
    campus_summary.to_csv(task_dir / "campus_quarterly_summary.csv", index=False)

    ratchet_tracker = build_demand_ratchet_tracker(monthly, facilities)
    ratchet_tracker.to_csv(task_dir / "demand_ratchet_tracker.csv", index=False)

    cost_profile = compute_facility_cost_profile(monthly, facilities)
    cost_profile.to_csv(task_dir / "facility_cost_profile.csv", index=False)

    bill_output.to_csv(ground_truth / "golden_facility_monthly_bill.csv", index=False)
    curt_ledger.to_csv(ground_truth / "golden_curtailment_credit_ledger.csv", index=False)
    campus_summary.to_csv(ground_truth / "golden_campus_quarterly_summary.csv", index=False)
    ratchet_tracker.to_csv(ground_truth / "golden_demand_ratchet_tracker.csv", index=False)
    cost_profile.to_csv(ground_truth / "golden_facility_cost_profile.csv", index=False)

    print(f"facility_monthly_bill.csv: {len(bill_output)} rows")
    print(f"curtailment_credit_ledger.csv: {len(curt_ledger)} rows")
    print(f"campus_quarterly_summary.csv: {len(campus_summary)} rows")
    print(f"demand_ratchet_tracker.csv: {len(ratchet_tracker)} rows")
    print(f"facility_cost_profile.csv: {len(cost_profile)} rows")


if __name__ == "__main__":
    main()

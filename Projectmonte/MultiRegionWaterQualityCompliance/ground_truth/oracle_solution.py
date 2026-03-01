import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
import calendar


def load_input_data(task_dir):
    stations = pd.read_csv(task_dir / "stations.csv")
    readings = pd.read_csv(task_dir / "readings.csv")
    thresholds = pd.read_csv(task_dir / "thresholds.csv")
    inspections = pd.read_csv(task_dir / "inspections.csv")
    maintenance = pd.read_csv(task_dir / "maintenance_log.csv")
    penalty_rates = pd.read_csv(task_dir / "penalty_rates.csv")
    regional_adj = pd.read_csv(task_dir / "regional_adjustments.csv")
    holidays = pd.read_csv(task_dir / "holidays.csv")
    return stations, readings, thresholds, inspections, maintenance, penalty_rates, regional_adj, holidays


def get_threshold_for_station(station_row, thresholds_df, parameter):
    tier = station_row["zone_tier"]
    match = thresholds_df[(thresholds_df["zone_tier"] == tier) & (thresholds_df["parameter"] == parameter)]
    if len(match) == 0:
        return None, None, None
    row = match.iloc[0]
    return row["caution_limit"], row["violation_limit"], row["critical_limit"]


def compute_pdi(measured, caution, violation, critical):
    if pd.isna(measured):
        return np.nan
    if measured <= caution:
        return 0.0
    elif measured <= violation:
        return (measured - caution) / (violation - caution) * 0.5
    elif measured <= critical:
        return 0.5 + (measured - violation) / (critical - violation) * 0.5
    else:
        return 1.0


def fill_nulls_carry_forward(readings_sorted, station_id, parameter, stations_df, thresholds_df):
    station_readings = readings_sorted[readings_sorted["station_id"] == station_id].copy()
    station_row = stations_df[stations_df["station_id"] == station_id].iloc[0]
    caution, _, _ = get_threshold_for_station(station_row, thresholds_df, parameter)
    default_value = caution if caution is not None else 0.0

    last_valid = default_value
    filled_values = {}
    for idx, row in station_readings.iterrows():
        val = row[parameter]
        if pd.isna(val):
            filled_values[idx] = last_valid
        else:
            last_valid = val
            filled_values[idx] = val
    return filled_values


def process_readings(stations_df, readings_df, thresholds_df):
    readings_df = readings_df.copy()
    readings_df["sample_date"] = pd.to_datetime(readings_df["sample_date"])
    readings_df = readings_df.sort_values(["station_id", "sample_date"]).reset_index(drop=True)

    carry_forward_params = ["ph", "turbidity_ntu", "chlorine_mg_l"]
    for param in carry_forward_params:
        for station_id in stations_df["station_id"].unique():
            filled = fill_nulls_carry_forward(readings_df, station_id, param, stations_df, thresholds_df)
            for idx, val in filled.items():
                readings_df.loc[idx, param] = val

    readings_df["lead_ppb"] = readings_df["lead_ppb"].fillna(0.0)

    readings_df["exclude_coliform"] = readings_df["coliform_count"].isna()
    readings_df["exclude_offline"] = readings_df["flow_rate_lpm"].isna()

    readings_df["month"] = readings_df["sample_date"].dt.to_period("M").astype(str)

    for idx, row in readings_df.iterrows():
        station_row = stations_df[stations_df["station_id"] == row["station_id"]].iloc[0]
        for param in ["ph", "turbidity_ntu", "chlorine_mg_l", "lead_ppb", "coliform_count"]:
            caution, violation, critical = get_threshold_for_station(station_row, thresholds_df, param)
            if caution is None:
                readings_df.loc[idx, f"pdi_{param}"] = np.nan
                continue
            measured = row[param]
            pdi = compute_pdi(measured, caution, violation, critical)
            readings_df.loc[idx, f"pdi_{param}"] = pdi

    return readings_df


def compute_monthly_station_metrics(readings_df, stations_df):
    all_months = sorted(readings_df["month"].unique())
    all_stations = stations_df["station_id"].unique()

    results = []
    for station_id in all_stations:
        for month in all_months:
            station_month = readings_df[
                (readings_df["station_id"] == station_id) &
                (readings_df["month"] == month)
            ]

            total_readings = len(station_month)
            if total_readings == 0:
                results.append({
                    "station_id": station_id,
                    "month": month,
                    "readings_count": 0,
                    "valid_readings_count": 0,
                    "avg_pdi_ph": np.nan,
                    "avg_pdi_turbidity": np.nan,
                    "avg_pdi_chlorine": np.nan,
                    "avg_pdi_lead": np.nan,
                    "avg_pdi_coliform": np.nan,
                    "max_daily_pdi": np.nan,
                    "params_above_05": 0,
                    "cqi": np.nan,
                    "has_readings": False,
                })
                continue

            online_readings = station_month[~station_month["exclude_offline"]]
            valid_readings_count = len(online_readings)

            quality_base = online_readings.copy()

            if len(quality_base) == 0:
                results.append({
                    "station_id": station_id,
                    "month": month,
                    "readings_count": total_readings,
                    "valid_readings_count": 0,
                    "avg_pdi_ph": np.nan,
                    "avg_pdi_turbidity": np.nan,
                    "avg_pdi_chlorine": np.nan,
                    "avg_pdi_lead": np.nan,
                    "avg_pdi_coliform": np.nan,
                    "max_daily_pdi": np.nan,
                    "params_above_05": 0,
                    "cqi": np.nan,
                    "has_readings": False,
                })
                continue

            avg_pdi_ph = quality_base["pdi_ph"].mean() if quality_base["pdi_ph"].notna().any() else 0.0
            avg_pdi_turb = quality_base["pdi_turbidity_ntu"].mean() if quality_base["pdi_turbidity_ntu"].notna().any() else 0.0
            avg_pdi_chlor = quality_base["pdi_chlorine_mg_l"].mean() if quality_base["pdi_chlorine_mg_l"].notna().any() else 0.0
            avg_pdi_lead = quality_base["pdi_lead_ppb"].mean() if quality_base["pdi_lead_ppb"].notna().any() else 0.0

            coliform_valid = quality_base[~quality_base["exclude_coliform"]]
            avg_pdi_coli = coliform_valid["pdi_coliform_count"].mean() if len(coliform_valid) > 0 and coliform_valid["pdi_coliform_count"].notna().any() else 0.0

            weighted_sum = (
                0.15 * avg_pdi_ph +
                0.20 * avg_pdi_turb +
                0.15 * avg_pdi_chlor +
                0.30 * avg_pdi_lead +
                0.20 * avg_pdi_coli
            )
            cqi = round(100.0 * (1.0 - weighted_sum), 2)
            cqi = max(0.0, min(100.0, cqi))

            pdi_columns_for_max = ["pdi_ph", "pdi_turbidity_ntu", "pdi_chlorine_mg_l", "pdi_lead_ppb", "pdi_coliform_count"]
            max_daily_pdi = 0.0
            for _, r in quality_base.iterrows():
                for col in pdi_columns_for_max:
                    val = r.get(col, 0)
                    if pd.notna(val) and val > max_daily_pdi:
                        max_daily_pdi = val

            params_above_05 = 0
            for pdi_val in [avg_pdi_ph, avg_pdi_turb, avg_pdi_chlor, avg_pdi_lead, avg_pdi_coli]:
                if pdi_val > 0.5:
                    params_above_05 += 1

            results.append({
                "station_id": station_id,
                "month": month,
                "readings_count": total_readings,
                "valid_readings_count": valid_readings_count,
                "avg_pdi_ph": round(avg_pdi_ph, 6),
                "avg_pdi_turbidity": round(avg_pdi_turb, 6),
                "avg_pdi_chlorine": round(avg_pdi_chlor, 6),
                "avg_pdi_lead": round(avg_pdi_lead, 6),
                "avg_pdi_coliform": round(avg_pdi_coli, 6),
                "max_daily_pdi": round(max_daily_pdi, 6),
                "params_above_05": params_above_05,
                "cqi": cqi,
                "has_readings": True,
            })

    return pd.DataFrame(results)


def compute_rcs(monthly_metrics):
    monthly_metrics = monthly_metrics.sort_values(["station_id", "month"]).copy()
    monthly_metrics["rcs"] = np.nan

    for station_id in monthly_metrics["station_id"].unique():
        mask = monthly_metrics["station_id"] == station_id
        station_data = monthly_metrics[mask].copy()
        indices = station_data.index.tolist()

        prev_rcs = None
        for i, idx in enumerate(indices):
            cqi = station_data.loc[idx, "cqi"]
            has_readings = station_data.loc[idx, "has_readings"]

            if not has_readings or pd.isna(cqi):
                if prev_rcs is not None:
                    rcs = round(prev_rcs * 0.95, 3)
                else:
                    rcs = np.nan
            elif prev_rcs is None:
                rcs = cqi
            else:
                if cqi >= 80:
                    rcs = round(prev_rcs * 0.85 + cqi * 0.15, 3)
                elif cqi >= 60:
                    rcs = round(prev_rcs * 0.6 + cqi * 0.4, 3)
                else:
                    rcs = cqi

            monthly_metrics.loc[idx, "rcs"] = rcs
            prev_rcs = rcs

    return monthly_metrics


def compute_risk_trend(monthly_metrics):
    monthly_metrics = monthly_metrics.sort_values(["station_id", "month"]).copy()
    monthly_metrics["risk_trend"] = ""

    for station_id in monthly_metrics["station_id"].unique():
        mask = monthly_metrics["station_id"] == station_id
        station_data = monthly_metrics[mask].copy()
        indices = station_data.index.tolist()

        prev_cqi = None
        first_data_seen = False
        for idx in indices:
            cqi = station_data.loc[idx, "cqi"]
            has_readings = station_data.loc[idx, "has_readings"]

            if not has_readings or pd.isna(cqi):
                monthly_metrics.loc[idx, "risk_trend"] = "INSUFFICIENT"
                continue

            if not first_data_seen:
                monthly_metrics.loc[idx, "risk_trend"] = "NEW"
                first_data_seen = True
                prev_cqi = cqi
                continue

            if cqi > prev_cqi + 2:
                monthly_metrics.loc[idx, "risk_trend"] = "IMPROVING"
            elif cqi < prev_cqi - 2:
                monthly_metrics.loc[idx, "risk_trend"] = "DECLINING"
            else:
                monthly_metrics.loc[idx, "risk_trend"] = "STABLE"

            prev_cqi = cqi

    return monthly_metrics


def classify_violations(monthly_metrics, inspections_df):
    inspections_df = inspections_df.copy()
    inspections_df["inspection_date"] = pd.to_datetime(inspections_df["inspection_date"])
    monthly_metrics["violation_class"] = "COMPLIANT"

    for idx, row in monthly_metrics.iterrows():
        if not row["has_readings"]:
            monthly_metrics.loc[idx, "violation_class"] = "NO_DATA"
            continue

        station_id = row["station_id"]
        month_str = row["month"]
        month_end = pd.Period(month_str, freq="M").end_time
        month_start = pd.Period(month_str, freq="M").start_time

        day_90_ago = month_end - timedelta(days=90)
        recent_failed = inspections_df[
            (inspections_df["station_id"] == station_id) &
            (inspections_df["result"] == "FAIL") &
            (inspections_df["inspection_date"] >= day_90_ago) &
            (inspections_df["inspection_date"] <= month_end)
        ]
        has_recent_fail = len(recent_failed) > 0

        max_pdi = row["max_daily_pdi"]
        params_above = row["params_above_05"]
        avg_lead = row["avg_pdi_lead"]
        avg_coli = row["avg_pdi_coliform"]
        avg_ph = row["avg_pdi_ph"]
        avg_turb = row["avg_pdi_turbidity"]
        avg_chlor = row["avg_pdi_chlorine"]

        any_param_above_05 = any(v > 0.5 for v in [avg_ph, avg_turb, avg_chlor, avg_lead, avg_coli] if pd.notna(v))
        any_param_above_0 = any(v > 0.0 for v in [avg_ph, avg_turb, avg_chlor, avg_lead, avg_coli] if pd.notna(v))

        if max_pdi > 0.9 and has_recent_fail:
            violation_class = "CRITICAL_REPEAT"
        elif max_pdi > 0.9:
            violation_class = "CRITICAL"
        elif params_above >= 3:
            violation_class = "MULTI_VIOLATION"
        elif avg_lead > 0.5 and avg_coli > 0.5:
            violation_class = "HEALTH_HAZARD"
        elif any_param_above_05:
            violation_class = "SINGLE_VIOLATION"
        elif any_param_above_0:
            violation_class = "WATCH"
        else:
            violation_class = "COMPLIANT"

        monthly_metrics.loc[idx, "violation_class"] = violation_class

    return monthly_metrics


def compute_consecutive_violations(monthly_metrics):
    monthly_metrics = monthly_metrics.sort_values(["station_id", "month"]).copy()
    monthly_metrics["consecutive_violations"] = 0
    non_violation_classes = {"COMPLIANT", "WATCH", "NO_DATA"}

    for station_id in monthly_metrics["station_id"].unique():
        mask = monthly_metrics["station_id"] == station_id
        station_data = monthly_metrics[mask]
        indices = station_data.index.tolist()

        consecutive = 0
        for idx in indices:
            vclass = station_data.loc[idx, "violation_class"]
            if vclass not in non_violation_classes:
                monthly_metrics.loc[idx, "consecutive_violations"] = consecutive
                consecutive += 1
            else:
                consecutive = 0
                monthly_metrics.loc[idx, "consecutive_violations"] = 0

    return monthly_metrics


def get_business_days_in_month(year, month, region, holidays_df):
    holidays_df = holidays_df.copy()
    holidays_df["date"] = pd.to_datetime(holidays_df["date"])

    num_days = calendar.monthrange(year, month)[1]

    business_days = 0
    for day in range(1, num_days + 1):
        dt = datetime(year, month, day)
        if dt.weekday() >= 5:
            continue
        is_holiday = holidays_df[
            (holidays_df["date"] == dt) &
            ((holidays_df["region"] == region) | (holidays_df["region"] == "ALL"))
        ]
        if len(is_holiday) > 0:
            continue
        business_days += 1

    return business_days, num_days


def compute_workday_fraction(month_str, region, holidays_df):
    period = pd.Period(month_str, freq="M")
    year = period.year
    month = period.month
    business_days, calendar_days = get_business_days_in_month(year, month, region, holidays_df)
    return round(business_days / calendar_days, 4)


def compute_penalties(monthly_metrics, penalty_rates_df, regional_adj_df, stations_df, holidays_df, maintenance_df):
    penalty_rates_df = penalty_rates_df.copy()
    penalty_rates_df["effective_from"] = pd.to_datetime(penalty_rates_df["effective_from"])
    penalty_rates_df["effective_to"] = pd.to_datetime(penalty_rates_df["effective_to"])

    maintenance_df = maintenance_df.copy()
    maintenance_df["service_date"] = pd.to_datetime(maintenance_df["service_date"])
    maintenance_df["maint_month"] = maintenance_df["service_date"].dt.to_period("M").astype(str)

    violation_tier_map = {
        "CRITICAL_REPEAT": 5,
        "CRITICAL": 4,
        "MULTI_VIOLATION": 3,
        "HEALTH_HAZARD": 3,
        "SINGLE_VIOLATION": 2,
        "WATCH": 1,
        "COMPLIANT": 0,
        "NO_DATA": 0,
    }

    monthly_metrics["violation_tier"] = monthly_metrics["violation_class"].map(violation_tier_map).fillna(0).astype(int)
    monthly_metrics["penalty_amount"] = 0.0
    monthly_metrics["maintenance_credit"] = 0.0
    monthly_metrics["net_penalty"] = 0.0
    monthly_metrics["base_amount_used"] = 0.0
    monthly_metrics["adjustment_factor_used"] = 1.0
    monthly_metrics["workday_fraction"] = 0.0
    monthly_metrics["maintenance_cost"] = 0.0

    for idx, row in monthly_metrics.iterrows():
        station_id = row["station_id"]
        month_str = row["month"]
        station_row = stations_df[stations_df["station_id"] == station_id].iloc[0]
        region = station_row["region"]

        wf = compute_workday_fraction(month_str, region, holidays_df)
        monthly_metrics.loc[idx, "workday_fraction"] = wf

        vclass = row["violation_class"]
        tier = violation_tier_map.get(vclass, 0)

        month_start = pd.Period(month_str, freq="M").start_time
        quarter = (month_start.month - 1) // 3 + 1
        adj_row = regional_adj_df[
            (regional_adj_df["region"] == region) &
            (regional_adj_df["quarter"] == quarter)
        ]
        if len(adj_row) > 0 and pd.notna(adj_row.iloc[0]["adjustment_factor"]):
            regional_factor = adj_row.iloc[0]["adjustment_factor"]
        else:
            regional_factor = 1.0
        monthly_metrics.loc[idx, "adjustment_factor_used"] = regional_factor

        if tier == 0:
            continue

        rate_match = penalty_rates_df[
            (penalty_rates_df["violation_tier"] == tier) &
            (penalty_rates_df["effective_from"] <= month_start) &
            (penalty_rates_df["effective_to"] >= month_start)
        ]
        if len(rate_match) == 0:
            continue

        base_amount = rate_match.iloc[0]["base_amount"]
        consecutive = row["consecutive_violations"]

        gross_penalty = round(base_amount * regional_factor * wf * (1 + 0.15 * consecutive), 2)

        station_maint = maintenance_df[
            (maintenance_df["station_id"] == station_id) &
            (maintenance_df["maint_month"] == month_str)
        ]
        maint_cost = 0.0
        if len(station_maint) > 0:
            maint_cost = round(station_maint["parts_cost"].sum() + station_maint["labor_cost"].sum(), 2)

        maint_credit = round(min(maint_cost * 0.05, gross_penalty * 0.20), 2)
        net_pen = round(max(0.0, gross_penalty - maint_credit), 2)

        monthly_metrics.loc[idx, "penalty_amount"] = gross_penalty
        monthly_metrics.loc[idx, "maintenance_cost"] = maint_cost
        monthly_metrics.loc[idx, "maintenance_credit"] = maint_credit
        monthly_metrics.loc[idx, "net_penalty"] = net_pen
        monthly_metrics.loc[idx, "base_amount_used"] = base_amount

    return monthly_metrics


def compute_regional_summary(monthly_metrics, stations_df):
    monthly_metrics = monthly_metrics.copy()
    monthly_metrics["month_dt"] = pd.to_datetime(monthly_metrics["month"])
    monthly_metrics["quarter"] = monthly_metrics["month_dt"].dt.quarter

    summaries = []
    for region in sorted(stations_df["region"].unique()):
        region_stations = stations_df[stations_df["region"] == region]["station_id"].tolist()
        station_count = len(region_stations)

        for quarter in sorted(monthly_metrics["quarter"].unique()):
            qdata = monthly_metrics[
                (monthly_metrics["station_id"].isin(region_stations)) &
                (monthly_metrics["quarter"] == quarter)
            ]

            active_stations = qdata[qdata["has_readings"]]["station_id"].nunique()
            total_readings = int(qdata["readings_count"].sum())
            valid_readings = int(qdata["valid_readings_count"].sum())

            cqi_values = qdata[qdata["has_readings"] & qdata["cqi"].notna()]["cqi"]
            avg_cqi = round(cqi_values.mean(), 2) if len(cqi_values) > 0 else np.nan

            rcs_values = qdata[qdata["has_readings"] & qdata["rcs"].notna()]["rcs"]
            avg_rcs = round(rcs_values.mean(), 2) if len(rcs_values) > 0 else np.nan

            non_violation = {"COMPLIANT", "WATCH", "NO_DATA"}
            violations_count = int(qdata[~qdata["violation_class"].isin(non_violation)].shape[0])
            critical_count = int(qdata[qdata["violation_class"].isin({"CRITICAL", "CRITICAL_REPEAT"})].shape[0])
            total_net_penalties = round(qdata["net_penalty"].sum(), 2)

            last_month_map = {1: "2024-03", 2: "2024-06"}
            last_month = last_month_map.get(quarter, None)

            if last_month:
                end_data = monthly_metrics[
                    (monthly_metrics["station_id"].isin(region_stations)) &
                    (monthly_metrics["month"] == last_month) &
                    (monthly_metrics["has_readings"])
                ]
                compliant_at_end = end_data[end_data["rcs"] >= 70]["station_id"].nunique()
                compliance_rate = round(compliant_at_end / active_stations, 4) if active_stations > 0 else 0.0
            else:
                compliance_rate = 0.0

            summaries.append({
                "region": region,
                "quarter": quarter,
                "station_count": station_count,
                "active_stations": active_stations,
                "total_readings": total_readings,
                "valid_readings": valid_readings,
                "avg_cqi": avg_cqi,
                "avg_rcs": avg_rcs,
                "violations_count": violations_count,
                "critical_count": critical_count,
                "total_net_penalties": total_net_penalties,
                "compliance_rate": compliance_rate,
            })

    return pd.DataFrame(summaries)


def compute_anomaly_report(readings_df, stations_df, thresholds_df):
    anomalies = []
    for idx, row in readings_df.iterrows():
        if row.get("exclude_offline", False):
            continue

        station_row = stations_df[stations_df["station_id"] == row["station_id"]].iloc[0]

        for param in ["ph", "turbidity_ntu", "chlorine_mg_l", "lead_ppb", "coliform_count"]:
            if param == "coliform_count" and row.get("exclude_coliform", False):
                continue

            measured = row[param]
            if pd.isna(measured):
                continue

            caution, violation, critical = get_threshold_for_station(station_row, thresholds_df, param)
            if caution is None:
                continue

            pdi = compute_pdi(measured, caution, violation, critical)
            if pdi > 0.5:
                if measured > critical:
                    threshold_type = "CRITICAL"
                    threshold_value = critical
                elif measured > violation:
                    threshold_type = "VIOLATION"
                    threshold_value = violation
                else:
                    threshold_type = "CAUTION"
                    threshold_value = caution

                anomalies.append({
                    "station_id": row["station_id"],
                    "reading_date": row["sample_date"].strftime("%Y-%m-%d") if isinstance(row["sample_date"], datetime) else str(row["sample_date"])[:10],
                    "parameter": param,
                    "measured_value": round(float(measured), 4),
                    "threshold_type": threshold_type,
                    "threshold_value": threshold_value,
                    "pdi_value": round(pdi, 6),
                })

    return pd.DataFrame(anomalies)


def compute_penalty_ledger(monthly_metrics):
    penalized = monthly_metrics[monthly_metrics["violation_tier"] > 0].copy()
    if len(penalized) == 0:
        return pd.DataFrame(columns=[
            "station_id", "month", "violation_class", "violation_tier",
            "base_amount", "adjustment_factor", "workday_fraction",
            "consecutive_months", "gross_penalty", "maintenance_cost",
            "maintenance_credit", "net_penalty"
        ])

    ledger = pd.DataFrame({
        "station_id": penalized["station_id"].values,
        "month": penalized["month"].values,
        "violation_class": penalized["violation_class"].values,
        "violation_tier": penalized["violation_tier"].values,
        "base_amount": penalized["base_amount_used"].values,
        "adjustment_factor": penalized["adjustment_factor_used"].values,
        "workday_fraction": penalized["workday_fraction"].values,
        "consecutive_months": penalized["consecutive_violations"].values,
        "gross_penalty": penalized["penalty_amount"].values,
        "maintenance_cost": penalized["maintenance_cost"].values,
        "maintenance_credit": penalized["maintenance_credit"].values,
        "net_penalty": penalized["net_penalty"].values,
    })

    return ledger


def compute_station_risk_profile(monthly_metrics, stations_df):
    violation_priority = {
        "CRITICAL_REPEAT": 7,
        "CRITICAL": 6,
        "MULTI_VIOLATION": 5,
        "HEALTH_HAZARD": 4,
        "SINGLE_VIOLATION": 3,
        "WATCH": 2,
        "COMPLIANT": 1,
        "NO_DATA": 0,
    }

    non_violation_classes = {"COMPLIANT", "WATCH", "NO_DATA"}

    profiles = []
    for _, station_row in stations_df.iterrows():
        station_id = station_row["station_id"]
        region = station_row["region"]
        zone_tier = station_row["zone_tier"]

        station_data = monthly_metrics[monthly_metrics["station_id"] == station_id].sort_values("month")

        months_with_readings = station_data[station_data["has_readings"]]
        months_monitored = len(months_with_readings)

        total_readings = int(station_data["readings_count"].sum())

        cqi_values = months_with_readings["cqi"].dropna()
        avg_cqi = round(cqi_values.mean(), 2) if len(cqi_values) > 0 else np.nan
        min_cqi = round(cqi_values.min(), 2) if len(cqi_values) > 0 else np.nan
        max_cqi = round(cqi_values.max(), 2) if len(cqi_values) > 0 else np.nan

        last_row = station_data.iloc[-1]
        final_rcs = last_row["rcs"] if pd.notna(last_row["rcs"]) else np.nan

        violation_months = station_data[~station_data["violation_class"].isin(non_violation_classes)]
        total_violations = len(violation_months)

        total_net_penalties = round(station_data["net_penalty"].sum(), 2)

        worst_priority = 0
        worst_class = "COMPLIANT"
        for _, mrow in station_data.iterrows():
            vclass = mrow["violation_class"]
            p = violation_priority.get(vclass, 0)
            if p > worst_priority:
                worst_priority = p
                worst_class = vclass

        if pd.isna(final_rcs):
            risk_category = "HIGH_RISK"
        elif final_rcs < 60 or total_violations >= 4:
            risk_category = "HIGH_RISK"
        elif final_rcs < 75 or total_violations >= 2:
            risk_category = "ELEVATED"
        elif final_rcs < 90:
            risk_category = "MODERATE"
        else:
            risk_category = "LOW_RISK"

        profiles.append({
            "station_id": station_id,
            "region": region,
            "zone_tier": zone_tier,
            "months_monitored": months_monitored,
            "total_readings": total_readings,
            "avg_cqi": avg_cqi,
            "min_cqi": min_cqi,
            "max_cqi": max_cqi,
            "final_rcs": round(final_rcs, 3) if pd.notna(final_rcs) else np.nan,
            "total_violations": total_violations,
            "total_net_penalties": total_net_penalties,
            "worst_violation": worst_class,
            "risk_category": risk_category,
        })

    return pd.DataFrame(profiles)


def main():
    task_dir = Path(__file__).parent.parent / "task"
    ground_truth = Path(__file__).parent

    stations, readings, thresholds, inspections, maintenance, penalty_rates, regional_adj, holidays = load_input_data(task_dir)

    processed_readings = process_readings(stations, readings, thresholds)

    monthly = compute_monthly_station_metrics(processed_readings, stations)

    monthly = compute_rcs(monthly)

    monthly = compute_risk_trend(monthly)

    monthly = classify_violations(monthly, inspections)

    monthly = compute_consecutive_violations(monthly)

    monthly = compute_penalties(monthly, penalty_rates, regional_adj, stations, holidays, maintenance)

    compliance_output = monthly[[
        "station_id", "month", "readings_count", "valid_readings_count",
        "avg_pdi_ph", "avg_pdi_turbidity", "avg_pdi_chlorine", "avg_pdi_lead", "avg_pdi_coliform",
        "cqi", "rcs", "violation_class", "risk_trend", "consecutive_violations",
        "penalty_amount", "maintenance_credit", "net_penalty"
    ]].copy()
    compliance_output.to_csv(task_dir / "station_compliance.csv", index=False)

    regional = compute_regional_summary(monthly, stations)
    regional.to_csv(task_dir / "regional_summary.csv", index=False)

    anomalies = compute_anomaly_report(processed_readings, stations, thresholds)
    anomalies.to_csv(task_dir / "anomaly_report.csv", index=False)

    ledger = compute_penalty_ledger(monthly)
    ledger.to_csv(task_dir / "penalty_ledger.csv", index=False)

    risk_profile = compute_station_risk_profile(monthly, stations)
    risk_profile.to_csv(task_dir / "station_risk_profile.csv", index=False)

    print(f"station_compliance.csv: {len(compliance_output)} rows")
    print(f"regional_summary.csv: {len(regional)} rows")
    print(f"anomaly_report.csv: {len(anomalies)} rows")
    print(f"penalty_ledger.csv: {len(ledger)} rows")
    print(f"station_risk_profile.csv: {len(risk_profile)} rows")


if __name__ == "__main__":
    main()

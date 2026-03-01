import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "task")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. facilities.csv - 25 facilities, 5 campuses, 4 utility zones
# ============================================================
campuses = ["NORTH", "SOUTH", "EAST", "WEST", "CENTRAL"]
zone_map = {
    "NORTH":   ["ZA", "ZA", "ZB", "ZC", "ZA"],
    "SOUTH":   ["ZB", "ZC", "ZB", "ZD", "ZB"],
    "EAST":    ["ZC", "ZC", "ZD", "ZA", "ZC"],
    "WEST":    ["ZD", "ZA", "ZD", "ZD", "ZB"],
    "CENTRAL": ["ZA", "ZB", "ZC", "ZD", "ZA"],
}

facilities = []
fid = 1
for campus in campuses:
    for i in range(5):
        contracted = int(np.random.choice([100, 120, 150, 180, 200, 250, 300, 350, 400]))
        demand_class = int(np.random.choice([1, 2, 3, 4], p=[0.28, 0.32, 0.24, 0.16]))
        facilities.append({
            "facility_id": f"F{fid:03d}",
            "campus": campus,
            "utility_zone": zone_map[campus][i],
            "demand_class": demand_class,
            "contracted_demand_kw": contracted,
        })
        fid += 1

facilities_df = pd.DataFrame(facilities)
facilities_df.to_csv(os.path.join(OUTPUT_DIR, "facilities.csv"), index=False)
print(f"facilities.csv: {len(facilities_df)} rows")

# ============================================================
# 2. rate_schedules.csv - rates by zone, season, period_type
# ============================================================
rate_rows = []
base_rates = {
    "ZA": {"PEAK": 0.185, "SHOULDER": 0.125, "OFF_PEAK": 0.078},
    "ZB": {"PEAK": 0.198, "SHOULDER": 0.132, "OFF_PEAK": 0.082},
    "ZC": {"PEAK": 0.172, "SHOULDER": 0.118, "OFF_PEAK": 0.071},
    "ZD": {"PEAK": 0.210, "SHOULDER": 0.140, "OFF_PEAK": 0.088},
}
demand_rates = {
    "ZA": {"PEAK": 12.50, "SHOULDER": 8.00, "OFF_PEAK": 5.00},
    "ZB": {"PEAK": 13.80, "SHOULDER": 8.50, "OFF_PEAK": 5.20},
    "ZC": {"PEAK": 11.90, "SHOULDER": 7.60, "OFF_PEAK": 4.80},
    "ZD": {"PEAK": 14.50, "SHOULDER": 9.00, "OFF_PEAK": 5.50},
}

for zone in ["ZA", "ZB", "ZC", "ZD"]:
    for season in ["WINTER", "SUMMER"]:
        season_mult = 1.0 if season == "WINTER" else 1.15
        for period in ["PEAK", "SHOULDER", "OFF_PEAK"]:
            rate_rows.append({
                "utility_zone": zone,
                "season": season,
                "period_type": period,
                "rate_per_kwh": round(base_rates[zone][period] * season_mult, 4),
                "demand_charge_per_kw": round(demand_rates[zone][period] * season_mult, 2),
            })

rate_df = pd.DataFrame(rate_rows)
rate_df.to_csv(os.path.join(OUTPUT_DIR, "rate_schedules.csv"), index=False)
print(f"rate_schedules.csv: {len(rate_df)} rows")

# ============================================================
# 3. zone_holidays.csv - holidays per zone (some ALL)
# ============================================================
holidays = [
    ("ALL", "2024-01-01"),
    ("ALL", "2024-01-15"),
    ("ALL", "2024-05-27"),
    ("ZA", "2024-02-12"),
    ("ZA", "2024-04-15"),
    ("ZB", "2024-03-08"),
    ("ZB", "2024-06-14"),
    ("ZC", "2024-02-19"),
    ("ZC", "2024-04-22"),
    ("ZD", "2024-03-01"),
    ("ZD", "2024-05-10"),
    ("ALL", "2024-02-19"),
    ("ZA", "2024-06-19"),
    ("ZB", "2024-01-29"),
    ("ZD", "2024-06-28"),
]

holidays_df = pd.DataFrame(holidays, columns=["utility_zone", "date"])
holidays_df.to_csv(os.path.join(OUTPUT_DIR, "zone_holidays.csv"), index=False)
print(f"zone_holidays.csv: {len(holidays_df)} rows")

# ============================================================
# 4. day_classifications.csv - per zone per date
# ============================================================
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 6, 30)
all_dates = pd.date_range(start_date, end_date)

holiday_set = {}
for zone in ["ZA", "ZB", "ZC", "ZD"]:
    zone_hols = set()
    for _, row in holidays_df.iterrows():
        if row["utility_zone"] == zone or row["utility_zone"] == "ALL":
            zone_hols.add(row["date"])
    holiday_set[zone] = zone_hols

peak_fracs = {
    "ZA": {"WINTER": {"WEEKDAY": 0.35, "WEEKEND": 0.15, "HOLIDAY": 0.10},
            "SUMMER": {"WEEKDAY": 0.40, "WEEKEND": 0.18, "HOLIDAY": 0.12}},
    "ZB": {"WINTER": {"WEEKDAY": 0.33, "WEEKEND": 0.14, "HOLIDAY": 0.09},
            "SUMMER": {"WEEKDAY": 0.38, "WEEKEND": 0.16, "HOLIDAY": 0.11}},
    "ZC": {"WINTER": {"WEEKDAY": 0.36, "WEEKEND": 0.16, "HOLIDAY": 0.11},
            "SUMMER": {"WEEKDAY": 0.42, "WEEKEND": 0.20, "HOLIDAY": 0.13}},
    "ZD": {"WINTER": {"WEEKDAY": 0.34, "WEEKEND": 0.13, "HOLIDAY": 0.08},
            "SUMMER": {"WEEKDAY": 0.39, "WEEKEND": 0.17, "HOLIDAY": 0.10}},
}

shoulder_fracs = {
    "ZA": {"WINTER": {"WEEKDAY": 0.35, "WEEKEND": 0.30, "HOLIDAY": 0.25},
            "SUMMER": {"WEEKDAY": 0.30, "WEEKEND": 0.27, "HOLIDAY": 0.23}},
    "ZB": {"WINTER": {"WEEKDAY": 0.37, "WEEKEND": 0.31, "HOLIDAY": 0.26},
            "SUMMER": {"WEEKDAY": 0.32, "WEEKEND": 0.29, "HOLIDAY": 0.24}},
    "ZC": {"WINTER": {"WEEKDAY": 0.34, "WEEKEND": 0.29, "HOLIDAY": 0.24},
            "SUMMER": {"WEEKDAY": 0.28, "WEEKEND": 0.25, "HOLIDAY": 0.22}},
    "ZD": {"WINTER": {"WEEKDAY": 0.36, "WEEKEND": 0.32, "HOLIDAY": 0.27},
            "SUMMER": {"WEEKDAY": 0.31, "WEEKEND": 0.28, "HOLIDAY": 0.25}},
}

day_class_rows = []
for zone in ["ZA", "ZB", "ZC", "ZD"]:
    for dt in all_dates:
        date_str = dt.strftime("%Y-%m-%d")
        month = dt.month
        season = "WINTER" if month <= 4 else "SUMMER"

        if date_str in holiday_set[zone]:
            day_type = "HOLIDAY"
        elif dt.weekday() >= 5:
            day_type = "WEEKEND"
        else:
            day_type = "WEEKDAY"

        pf = peak_fracs[zone][season][day_type]
        sf = shoulder_fracs[zone][season][day_type]
        of = round(1.0 - pf - sf, 2)

        day_class_rows.append({
            "utility_zone": zone,
            "date": date_str,
            "day_type": day_type,
            "season": season,
            "peak_frac": pf,
            "shoulder_frac": sf,
            "offpeak_frac": of,
        })

day_class_df = pd.DataFrame(day_class_rows)
day_class_df.to_csv(os.path.join(OUTPUT_DIR, "day_classifications.csv"), index=False)
print(f"day_classifications.csv: {len(day_class_df)} rows")

# ============================================================
# 5. pf_tariffs.csv - power factor tariffs per zone
# ============================================================
pf_tariffs = [
    {"utility_zone": "ZA", "min_pf": 0.90, "penalty_pct_per_point": 1.50, "bonus_pct_per_point_above_95": 0.80},
    {"utility_zone": "ZB", "min_pf": 0.88, "penalty_pct_per_point": 1.75, "bonus_pct_per_point_above_95": 0.90},
    {"utility_zone": "ZC", "min_pf": 0.90, "penalty_pct_per_point": 1.40, "bonus_pct_per_point_above_95": 0.75},
    {"utility_zone": "ZD", "min_pf": 0.85, "penalty_pct_per_point": 2.00, "bonus_pct_per_point_above_95": 1.00},
]
pf_df = pd.DataFrame(pf_tariffs)
pf_df.to_csv(os.path.join(OUTPUT_DIR, "pf_tariffs.csv"), index=False)
print(f"pf_tariffs.csv: {len(pf_df)} rows")

# ============================================================
# 6. curtailment_events.csv - ~15 events across campuses
# ============================================================
curtailment_events = [
    {"campus": "NORTH", "event_date": "2024-01-18", "target_reduction_pct": 0.15, "credit_rate_per_kwh": 0.25},
    {"campus": "SOUTH", "event_date": "2024-01-25", "target_reduction_pct": 0.20, "credit_rate_per_kwh": 0.30},
    {"campus": "EAST",  "event_date": "2024-02-08", "target_reduction_pct": 0.10, "credit_rate_per_kwh": 0.22},
    {"campus": "WEST",  "event_date": "2024-02-15", "target_reduction_pct": 0.18, "credit_rate_per_kwh": 0.28},
    {"campus": "CENTRAL","event_date": "2024-02-22", "target_reduction_pct": 0.12, "credit_rate_per_kwh": 0.24},
    {"campus": "NORTH", "event_date": "2024-03-07", "target_reduction_pct": 0.20, "credit_rate_per_kwh": 0.32},
    {"campus": "SOUTH", "event_date": "2024-03-14", "target_reduction_pct": 0.15, "credit_rate_per_kwh": 0.26},
    {"campus": "EAST",  "event_date": "2024-04-04", "target_reduction_pct": 0.22, "credit_rate_per_kwh": 0.35},
    {"campus": "WEST",  "event_date": "2024-04-18", "target_reduction_pct": 0.10, "credit_rate_per_kwh": 0.20},
    {"campus": "CENTRAL","event_date": "2024-04-25", "target_reduction_pct": 0.16, "credit_rate_per_kwh": 0.27},
    {"campus": "NORTH", "event_date": "2024-05-09", "target_reduction_pct": 0.25, "credit_rate_per_kwh": 0.38},
    {"campus": "SOUTH", "event_date": "2024-05-16", "target_reduction_pct": 0.18, "credit_rate_per_kwh": 0.29},
    {"campus": "EAST",  "event_date": "2024-05-30", "target_reduction_pct": 0.14, "credit_rate_per_kwh": 0.23},
    {"campus": "WEST",  "event_date": "2024-06-06", "target_reduction_pct": 0.20, "credit_rate_per_kwh": 0.31},
    {"campus": "CENTRAL","event_date": "2024-06-20", "target_reduction_pct": 0.15, "credit_rate_per_kwh": 0.25},
]
curtailment_df = pd.DataFrame(curtailment_events)
curtailment_df.to_csv(os.path.join(OUTPUT_DIR, "curtailment_events.csv"), index=False)
print(f"curtailment_events.csv: {len(curtailment_df)} rows")

# ============================================================
# 7. equipment_efficiency.csv - ~80 equipment records
# ============================================================
equipment_types = ["HVAC", "COMPRESSOR", "LIGHTING", "MOTOR", "TRANSFORMER"]
equipment_rows = []
eq_id = 1
for _, fac in facilities_df.iterrows():
    n_equip = np.random.randint(2, 5)
    for _ in range(n_equip):
        install_months_ago = np.random.randint(3, 48)
        install_date = datetime(2024, 1, 1) - timedelta(days=install_months_ago * 30)
        equipment_rows.append({
            "facility_id": fac["facility_id"],
            "equipment_id": f"EQ{eq_id:04d}",
            "install_date": install_date.strftime("%Y-%m-%d"),
            "rated_efficiency": round(np.random.uniform(0.82, 0.98), 2),
            "degradation_rate_monthly": round(np.random.uniform(0.002, 0.012), 4),
        })
        eq_id += 1

equipment_df = pd.DataFrame(equipment_rows)
equipment_df.to_csv(os.path.join(OUTPUT_DIR, "equipment_efficiency.csv"), index=False)
print(f"equipment_efficiency.csv: {len(equipment_df)} rows")

# ============================================================
# 8. meter_readings.csv - ~4300 daily readings with nulls/gaps
# ============================================================
reading_rows = []
solar_facilities = set(np.random.choice(facilities_df["facility_id"].tolist(), size=15, replace=False))

for _, fac in facilities_df.iterrows():
    fid = fac["facility_id"]
    contracted = fac["contracted_demand_kw"]
    has_solar = fid in solar_facilities

    base_kwh = contracted * np.random.uniform(3, 8)
    base_peak = contracted * np.random.uniform(0.7, 1.3)
    base_kvar = base_kwh * np.random.uniform(0.3, 0.6)

    for dt in all_dates:
        if np.random.random() < 0.06:
            continue

        date_str = dt.strftime("%Y-%m-%d")
        dow_factor = 0.65 if dt.weekday() >= 5 else 1.0
        seasonal_factor = 1.0 + 0.15 * (dt.month - 1) / 5

        kwh = round(base_kwh * dow_factor * seasonal_factor * np.random.uniform(0.7, 1.3), 1)
        peak = round(base_peak * dow_factor * seasonal_factor * np.random.uniform(0.75, 1.25), 1)
        kvar = round(base_kvar * dow_factor * np.random.uniform(0.5, 1.5), 1)
        solar = round(np.random.uniform(20, 120) * seasonal_factor, 1) if has_solar else 0.0

        kwh_val = kwh
        peak_val = peak
        kvar_val = kvar
        solar_val = solar

        r = np.random.random()
        if r < 0.03:
            kwh_val = None
            peak_val = None
        elif r < 0.06:
            kwh_val = None
        elif r < 0.08:
            peak_val = None

        if np.random.random() < 0.07:
            kvar_val = None

        if has_solar and np.random.random() < 0.10:
            solar_val = None

        reading_rows.append({
            "facility_id": fid,
            "reading_date": date_str,
            "kwh_consumed": kwh_val,
            "kvar_reactive": kvar_val,
            "peak_demand_kw": peak_val,
            "solar_kwh": solar_val,
        })

readings_df = pd.DataFrame(reading_rows)
readings_df = readings_df.sort_values(["facility_id", "reading_date"]).reset_index(drop=True)
readings_df.to_csv(os.path.join(OUTPUT_DIR, "meter_readings.csv"), index=False)
print(f"meter_readings.csv: {len(readings_df)} rows")

# Summary stats
print(f"\nSummary:")
print(f"  kwh_consumed nulls: {readings_df['kwh_consumed'].isna().sum()}")
print(f"  peak_demand_kw nulls: {readings_df['peak_demand_kw'].isna().sum()}")
print(f"  kvar_reactive nulls: {readings_df['kvar_reactive'].isna().sum()}")
print(f"  solar_kwh nulls: {readings_df['solar_kwh'].isna().sum()}")
print(f"  solar facilities: {len(solar_facilities)}")
print(f"  total reading days: {len(readings_df)}")

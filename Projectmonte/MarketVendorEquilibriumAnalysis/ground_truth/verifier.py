import pandas as pd
import numpy as np
import sys
from pathlib import Path


def load_golden(name):
    p = Path(__file__).parent / f"golden_{name}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_submission(task_dir, name):
    p = task_dir / f"{name}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def check_value(expected, actual, tol=0.01):
    if pd.isna(expected) and pd.isna(actual):
        return True
    if pd.isna(expected) or pd.isna(actual):
        return False
    try:
        e, a = float(expected), float(actual)
        return abs(a - e) <= tol
    except (ValueError, TypeError):
        return str(expected).strip().lower() == str(actual).strip().lower()


def run_checks(task_dir):
    checks = []
    passed = 0

    g_vm = load_golden("vendor_monthly")
    s_vm = load_submission(task_dir, "vendor_monthly")

    checks.append("VM_EXISTS")
    if s_vm is not None:
        passed += 1
        print("  PASS  VM_EXISTS")
    else:
        print("  FAIL  VM_EXISTS: vendor_monthly.csv not found")
        checks_for_vm = ["VM_ROW_COUNT", "VM_COLUMNS", "VM_KEYS", "VM_GROSS_REVENUE",
                         "VM_NET_REVENUE", "VM_COMMISSION", "VM_SEASONAL_REV",
                         "VM_RETURN_RATE", "VM_AVG_RATING", "VM_COMPLAINT_COUNT",
                         "VM_CRITICAL_COMPLAINTS", "VM_RESOLUTION_DAYS",
                         "VM_WEIGHTED_SHIPPING", "VM_VSI"]
        for c in checks_for_vm:
            checks.append(c)
            print(f"  FAIL  {c}: file missing")
        g_se = load_golden("segment_equilibrium")
        s_se = load_submission(task_dir, "segment_equilibrium")
        checks.append("SE_EXISTS")
        if s_se is not None:
            passed += 1
            print("  PASS  SE_EXISTS")
        else:
            print("  FAIL  SE_EXISTS: segment_equilibrium.csv not found")

        remaining_checks = [
            "SE_ROW_COUNT", "SE_COLUMNS", "SE_KEYS", "SE_VENDOR_COUNT",
            "SE_TOTAL_REVENUE", "SE_HHI", "SE_ENTROPY", "SE_VCI", "SE_MSI",
            "SE_CONCENTRATED", "SE_HEALTH_CLASS", "SE_STREAK",
            "VRP_EXISTS", "VRP_ROW_COUNT", "VRP_COLUMNS", "VRP_AVG_VSI",
            "VRP_COMPLAINTS", "VRP_REVENUE_VOL", "VRP_DECLINE_MONTHS",
            "VRP_RISK_CATEGORY", "VRP_WORST_MONTH",
            "PE_EXISTS", "PE_ROW_COUNT", "PE_COLUMNS", "PE_REVENUE",
            "PE_ROI", "PE_LIFT", "PE_DISCOUNT",
            "MHL_EXISTS", "MHL_ROW_COUNT", "MHL_COLUMNS", "MHL_HEALTH_CLASS",
            "MHL_ENTROPY_GAP", "MHL_PENALTY_SCORE", "MHL_DECAY",
            "MHL_CUMULATIVE", "MHL_NET_PENALTY",
            "CROSS_SEG_VENDOR_COUNT", "CROSS_RISK_DISTRIBUTION", "CROSS_PENALTY_CONSISTENCY"
        ]
        for c in remaining_checks:
            checks.append(c)
            print(f"  FAIL  {c}: dependency missing")
        total = len(checks)
        print(f"\n{passed}/{total}")
        sys.exit(1)

    checks.append("VM_ROW_COUNT")
    if len(s_vm) == len(g_vm):
        passed += 1
        print(f"  PASS  VM_ROW_COUNT ({len(s_vm)} rows)")
    else:
        print(f"  FAIL  VM_ROW_COUNT: expected {len(g_vm)}, got {len(s_vm)}")

    required_vm_cols = list(g_vm.columns)
    checks.append("VM_COLUMNS")
    missing_cols = [c for c in required_vm_cols if c not in s_vm.columns]
    if len(missing_cols) == 0:
        passed += 1
        print("  PASS  VM_COLUMNS")
    else:
        print(f"  FAIL  VM_COLUMNS: missing {missing_cols}")

    checks.append("VM_KEYS")
    try:
        merged_vm = g_vm.merge(s_vm, on=["vendor_id", "month"], suffixes=("_exp", "_act"))
        if len(merged_vm) == len(g_vm):
            passed += 1
            print("  PASS  VM_KEYS")
        else:
            print(f"  FAIL  VM_KEYS: matched {len(merged_vm)} of {len(g_vm)} rows")
    except Exception as e:
        print(f"  FAIL  VM_KEYS: {e}")
        merged_vm = pd.DataFrame()

    if len(merged_vm) > 0:
        for col, tol, max_mismatches, check_name in [
            ("gross_revenue", 1.0, 3, "VM_GROSS_REVENUE"),
            ("net_revenue", 1.0, 3, "VM_NET_REVENUE"),
            ("commission_amount", 0.5, 3, "VM_COMMISSION"),
            ("seasonal_adjusted_revenue", 1.0, 3, "VM_SEASONAL_REV"),
            ("adjusted_return_rate", 0.005, 3, "VM_RETURN_RATE"),
            ("avg_rating", 0.1, 3, "VM_AVG_RATING"),
            ("complaint_count", 0, 2, "VM_COMPLAINT_COUNT"),
            ("critical_complaints", 0, 2, "VM_CRITICAL_COMPLAINTS"),
            ("avg_resolution_days", 1.0, 3, "VM_RESOLUTION_DAYS"),
            ("weighted_shipping", 1.0, 5, "VM_WEIGHTED_SHIPPING"),
            ("vsi", 0.5, 3, "VM_VSI"),
        ]:
            checks.append(check_name)
            col_exp = f"{col}_exp"
            col_act = f"{col}_act"
            if col_exp not in merged_vm.columns or col_act not in merged_vm.columns:
                print(f"  FAIL  {check_name}: column not found")
                continue
            mismatches = 0
            for _, row in merged_vm.iterrows():
                if not check_value(row[col_exp], row[col_act], tol):
                    mismatches += 1
            if mismatches <= max_mismatches:
                passed += 1
                print(f"  PASS  {check_name} ({mismatches} mismatches, tol={tol})")
            else:
                samples = []
                for _, row in merged_vm.iterrows():
                    if not check_value(row[col_exp], row[col_act], tol):
                        samples.append(f"{row['vendor_id']}/{row['month']}: exp={row[col_exp]}, got={row[col_act]}")
                        if len(samples) >= 3:
                            break
                print(f"  FAIL  {check_name}: {mismatches} mismatches (max {max_mismatches}). Samples: {'; '.join(samples)}")
    else:
        for check_name in ["VM_GROSS_REVENUE", "VM_NET_REVENUE", "VM_COMMISSION",
                           "VM_SEASONAL_REV", "VM_RETURN_RATE", "VM_AVG_RATING",
                           "VM_COMPLAINT_COUNT", "VM_CRITICAL_COMPLAINTS",
                           "VM_RESOLUTION_DAYS", "VM_WEIGHTED_SHIPPING", "VM_VSI"]:
            checks.append(check_name)
            print(f"  FAIL  {check_name}: no matched rows")

    g_se = load_golden("segment_equilibrium")
    s_se = load_submission(task_dir, "segment_equilibrium")

    checks.append("SE_EXISTS")
    if s_se is not None:
        passed += 1
        print("  PASS  SE_EXISTS")
    else:
        print("  FAIL  SE_EXISTS: segment_equilibrium.csv not found")
        for c in ["SE_ROW_COUNT", "SE_COLUMNS", "SE_KEYS", "SE_VENDOR_COUNT",
                   "SE_TOTAL_REVENUE", "SE_HHI", "SE_ENTROPY", "SE_VCI", "SE_MSI",
                   "SE_CONCENTRATED", "SE_HEALTH_CLASS", "SE_STREAK"]:
            checks.append(c)
            print(f"  FAIL  {c}: file missing")
        s_se = None

    if s_se is not None:
        checks.append("SE_ROW_COUNT")
        if len(s_se) == len(g_se):
            passed += 1
            print(f"  PASS  SE_ROW_COUNT ({len(s_se)} rows)")
        else:
            print(f"  FAIL  SE_ROW_COUNT: expected {len(g_se)}, got {len(s_se)}")

        checks.append("SE_COLUMNS")
        missing_se = [c for c in g_se.columns if c not in s_se.columns]
        if len(missing_se) == 0:
            passed += 1
            print("  PASS  SE_COLUMNS")
        else:
            print(f"  FAIL  SE_COLUMNS: missing {missing_se}")

        checks.append("SE_KEYS")
        try:
            merged_se = g_se.merge(s_se, on=["segment", "month"], suffixes=("_exp", "_act"))
            if len(merged_se) == len(g_se):
                passed += 1
                print("  PASS  SE_KEYS")
            else:
                print(f"  FAIL  SE_KEYS: matched {len(merged_se)} of {len(g_se)} rows")
        except Exception as e:
            print(f"  FAIL  SE_KEYS: {e}")
            merged_se = pd.DataFrame()

        if len(merged_se) > 0:
            for col, tol, max_mm, check_name in [
                ("vendor_count", 0, 0, "SE_VENDOR_COUNT"),
                ("total_revenue", 5.0, 2, "SE_TOTAL_REVENUE"),
                ("hhi", 0.005, 2, "SE_HHI"),
                ("shannon_entropy", 0.01, 2, "SE_ENTROPY"),
                ("vci", 0.05, 2, "SE_VCI"),
                ("msi", 0.05, 2, "SE_MSI"),
                ("concentrated_vendors", 0, 1, "SE_CONCENTRATED"),
                ("health_class", 0, 2, "SE_HEALTH_CLASS"),
                ("unhealthy_streak", 0, 2, "SE_STREAK"),
            ]:
                checks.append(check_name)
                col_exp = f"{col}_exp" if f"{col}_exp" in merged_se.columns else col
                col_act = f"{col}_act" if f"{col}_act" in merged_se.columns else col
                if col_exp not in merged_se.columns or col_act not in merged_se.columns:
                    if col in ["health_class"]:
                        col_exp = col + "_exp"
                        col_act = col + "_act"
                    if col_exp not in merged_se.columns:
                        print(f"  FAIL  {check_name}: column not found")
                        continue
                mismatches = 0
                for _, row in merged_se.iterrows():
                    if not check_value(row[col_exp], row[col_act], tol):
                        mismatches += 1
                if mismatches <= max_mm:
                    passed += 1
                    print(f"  PASS  {check_name} ({mismatches} mismatches)")
                else:
                    print(f"  FAIL  {check_name}: {mismatches} mismatches (max {max_mm})")
        else:
            for c in ["SE_VENDOR_COUNT", "SE_TOTAL_REVENUE", "SE_HHI", "SE_ENTROPY",
                       "SE_VCI", "SE_MSI", "SE_CONCENTRATED", "SE_HEALTH_CLASS", "SE_STREAK"]:
                checks.append(c)
                print(f"  FAIL  {c}: no matched rows")

    g_vrp = load_golden("vendor_risk_profile")
    s_vrp = load_submission(task_dir, "vendor_risk_profile")

    checks.append("VRP_EXISTS")
    if s_vrp is not None:
        passed += 1
        print("  PASS  VRP_EXISTS")
    else:
        print("  FAIL  VRP_EXISTS: vendor_risk_profile.csv not found")
        for c in ["VRP_ROW_COUNT", "VRP_COLUMNS", "VRP_AVG_VSI", "VRP_COMPLAINTS",
                   "VRP_REVENUE_VOL", "VRP_DECLINE_MONTHS", "VRP_RISK_CATEGORY", "VRP_WORST_MONTH"]:
            checks.append(c)
            print(f"  FAIL  {c}: file missing")
        s_vrp = None

    if s_vrp is not None:
        checks.append("VRP_ROW_COUNT")
        if len(s_vrp) == len(g_vrp):
            passed += 1
            print(f"  PASS  VRP_ROW_COUNT ({len(s_vrp)} rows)")
        else:
            print(f"  FAIL  VRP_ROW_COUNT: expected {len(g_vrp)}, got {len(s_vrp)}")

        checks.append("VRP_COLUMNS")
        missing_vrp = [c for c in g_vrp.columns if c not in s_vrp.columns]
        if len(missing_vrp) == 0:
            passed += 1
            print("  PASS  VRP_COLUMNS")
        else:
            print(f"  FAIL  VRP_COLUMNS: missing {missing_vrp}")

        try:
            merged_vrp = g_vrp.merge(s_vrp, on="vendor_id", suffixes=("_exp", "_act"))
        except Exception:
            merged_vrp = pd.DataFrame()

        if len(merged_vrp) > 0:
            for col, tol, max_mm, check_name in [
                ("avg_vsi", 0.5, 2, "VRP_AVG_VSI"),
                ("total_complaints", 0, 2, "VRP_COMPLAINTS"),
                ("revenue_volatility", 0.05, 3, "VRP_REVENUE_VOL"),
                ("consecutive_decline_months", 0, 2, "VRP_DECLINE_MONTHS"),
                ("risk_category", 0, 2, "VRP_RISK_CATEGORY"),
                ("worst_month", 0, 3, "VRP_WORST_MONTH"),
            ]:
                checks.append(check_name)
                col_exp = f"{col}_exp" if f"{col}_exp" in merged_vrp.columns else col
                col_act = f"{col}_act" if f"{col}_act" in merged_vrp.columns else col
                if col_exp not in merged_vrp.columns:
                    print(f"  FAIL  {check_name}: column not found")
                    continue
                mismatches = 0
                for _, row in merged_vrp.iterrows():
                    if not check_value(row[col_exp], row[col_act], tol):
                        mismatches += 1
                if mismatches <= max_mm:
                    passed += 1
                    print(f"  PASS  {check_name} ({mismatches} mismatches)")
                else:
                    print(f"  FAIL  {check_name}: {mismatches} mismatches (max {max_mm})")
        else:
            for c in ["VRP_AVG_VSI", "VRP_COMPLAINTS", "VRP_REVENUE_VOL",
                       "VRP_DECLINE_MONTHS", "VRP_RISK_CATEGORY", "VRP_WORST_MONTH"]:
                checks.append(c)
                print(f"  FAIL  {c}: no matched rows")

    g_pe = load_golden("promotion_effectiveness")
    s_pe = load_submission(task_dir, "promotion_effectiveness")

    checks.append("PE_EXISTS")
    if s_pe is not None:
        passed += 1
        print("  PASS  PE_EXISTS")
    else:
        print("  FAIL  PE_EXISTS: promotion_effectiveness.csv not found")
        for c in ["PE_ROW_COUNT", "PE_COLUMNS", "PE_REVENUE", "PE_ROI", "PE_LIFT", "PE_DISCOUNT"]:
            checks.append(c)
            print(f"  FAIL  {c}: file missing")
        s_pe = None

    if s_pe is not None:
        checks.append("PE_ROW_COUNT")
        row_diff = abs(len(s_pe) - len(g_pe))
        if row_diff <= 5:
            passed += 1
            print(f"  PASS  PE_ROW_COUNT ({len(s_pe)} rows, expected {len(g_pe)})")
        else:
            print(f"  FAIL  PE_ROW_COUNT: expected {len(g_pe)}, got {len(s_pe)}")

        checks.append("PE_COLUMNS")
        missing_pe = [c for c in g_pe.columns if c not in s_pe.columns]
        if len(missing_pe) == 0:
            passed += 1
            print("  PASS  PE_COLUMNS")
        else:
            print(f"  FAIL  PE_COLUMNS: missing {missing_pe}")

        try:
            merged_pe = g_pe.merge(s_pe, on=["promo_id", "vendor_id"], suffixes=("_exp", "_act"))
        except Exception:
            merged_pe = pd.DataFrame()

        if len(merged_pe) > 0:
            for col, tol, max_mm, check_name in [
                ("total_promo_revenue", 2.0, 5, "PE_REVENUE"),
                ("roi", 0.05, 5, "PE_ROI"),
                ("lift_ratio", 0.1, 5, "PE_LIFT"),
                ("avg_discount_applied", 0.01, 3, "PE_DISCOUNT"),
            ]:
                checks.append(check_name)
                col_exp = f"{col}_exp" if f"{col}_exp" in merged_pe.columns else col
                col_act = f"{col}_act" if f"{col}_act" in merged_pe.columns else col
                if col_exp not in merged_pe.columns:
                    print(f"  FAIL  {check_name}: column not found")
                    continue
                mismatches = 0
                for _, row in merged_pe.iterrows():
                    if not check_value(row[col_exp], row[col_act], tol):
                        mismatches += 1
                if mismatches <= max_mm:
                    passed += 1
                    print(f"  PASS  {check_name} ({mismatches} mismatches)")
                else:
                    print(f"  FAIL  {check_name}: {mismatches} mismatches (max {max_mm})")
        else:
            for c in ["PE_REVENUE", "PE_ROI", "PE_LIFT", "PE_DISCOUNT"]:
                checks.append(c)
                print(f"  FAIL  {c}: no matched rows")

    g_mhl = load_golden("market_health_ledger")
    s_mhl = load_submission(task_dir, "market_health_ledger")

    checks.append("MHL_EXISTS")
    if s_mhl is not None:
        passed += 1
        print("  PASS  MHL_EXISTS")
    else:
        print("  FAIL  MHL_EXISTS: market_health_ledger.csv not found")
        for c in ["MHL_ROW_COUNT", "MHL_COLUMNS", "MHL_HEALTH_CLASS",
                   "MHL_ENTROPY_GAP", "MHL_PENALTY_SCORE", "MHL_DECAY",
                   "MHL_CUMULATIVE", "MHL_NET_PENALTY"]:
            checks.append(c)
            print(f"  FAIL  {c}: file missing")
        s_mhl = None

    if s_mhl is not None:
        checks.append("MHL_ROW_COUNT")
        row_diff = abs(len(s_mhl) - len(g_mhl))
        if row_diff <= 3:
            passed += 1
            print(f"  PASS  MHL_ROW_COUNT ({len(s_mhl)} rows, expected {len(g_mhl)})")
        else:
            print(f"  FAIL  MHL_ROW_COUNT: expected {len(g_mhl)}, got {len(s_mhl)}")

        checks.append("MHL_COLUMNS")
        missing_mhl = [c for c in g_mhl.columns if c not in s_mhl.columns]
        if len(missing_mhl) == 0:
            passed += 1
            print("  PASS  MHL_COLUMNS")
        else:
            print(f"  FAIL  MHL_COLUMNS: missing {missing_mhl}")

        try:
            merged_mhl = g_mhl.merge(s_mhl, on=["segment", "month"], suffixes=("_exp", "_act"))
        except Exception:
            merged_mhl = pd.DataFrame()

        if len(merged_mhl) > 0:
            for col, tol, max_mm, check_name in [
                ("health_class", 0, 1, "MHL_HEALTH_CLASS"),
                ("entropy_gap", 0.01, 1, "MHL_ENTROPY_GAP"),
                ("penalty_score", 0.5, 1, "MHL_PENALTY_SCORE"),
                ("decay_factor", 0.01, 1, "MHL_DECAY"),
                ("cumulative_penalty", 1.0, 2, "MHL_CUMULATIVE"),
                ("net_penalty", 1.0, 2, "MHL_NET_PENALTY"),
            ]:
                checks.append(check_name)
                col_exp = f"{col}_exp" if f"{col}_exp" in merged_mhl.columns else col
                col_act = f"{col}_act" if f"{col}_act" in merged_mhl.columns else col
                if col_exp not in merged_mhl.columns:
                    print(f"  FAIL  {check_name}: column not found")
                    continue
                mismatches = 0
                for _, row in merged_mhl.iterrows():
                    if not check_value(row[col_exp], row[col_act], tol):
                        mismatches += 1
                if mismatches <= max_mm:
                    passed += 1
                    print(f"  PASS  {check_name} ({mismatches} mismatches)")
                else:
                    print(f"  FAIL  {check_name}: {mismatches} mismatches (max {max_mm})")
        else:
            for c in ["MHL_HEALTH_CLASS", "MHL_ENTROPY_GAP", "MHL_PENALTY_SCORE",
                       "MHL_DECAY", "MHL_CUMULATIVE", "MHL_NET_PENALTY"]:
                checks.append(c)
                print(f"  FAIL  {c}: no matched rows")

    checks.append("CROSS_SEG_VENDOR_COUNT")
    try:
        if s_vm is not None and s_se is not None:
            for seg in g_se["segment"].unique():
                vm_vendors = s_vm["vendor_id"].nunique()
                se_row = s_se[s_se["segment"] == seg]
            seg_counts = s_se.groupby("segment")["vendor_count"].first()
            total_from_se = seg_counts.sum()
            total_from_vm = s_vm["vendor_id"].nunique()
            if total_from_se == total_from_vm:
                passed += 1
                print(f"  PASS  CROSS_SEG_VENDOR_COUNT")
            else:
                print(f"  FAIL  CROSS_SEG_VENDOR_COUNT: SE sums to {total_from_se}, VM has {total_from_vm}")
        else:
            print("  FAIL  CROSS_SEG_VENDOR_COUNT: files missing")
    except Exception as e:
        print(f"  FAIL  CROSS_SEG_VENDOR_COUNT: {e}")

    checks.append("CROSS_RISK_DISTRIBUTION")
    try:
        if s_vrp is not None:
            risk_counts = s_vrp["risk_category"].value_counts()
            g_risk_counts = g_vrp["risk_category"].value_counts()
            match = True
            for cat in g_risk_counts.index:
                expected = g_risk_counts.get(cat, 0)
                actual = risk_counts.get(cat, 0)
                if abs(expected - actual) > 2:
                    match = False
                    break
            if match:
                passed += 1
                print("  PASS  CROSS_RISK_DISTRIBUTION")
            else:
                print(f"  FAIL  CROSS_RISK_DISTRIBUTION: expected {dict(g_risk_counts)}, got {dict(risk_counts)}")
        else:
            print("  FAIL  CROSS_RISK_DISTRIBUTION: file missing")
    except Exception as e:
        print(f"  FAIL  CROSS_RISK_DISTRIBUTION: {e}")

    checks.append("CROSS_PENALTY_CONSISTENCY")
    try:
        if s_mhl is not None and s_se is not None:
            unhealthy_in_se = s_se[~s_se["health_class"].isin(["HEALTHY", "STABLE"])]
            ledger_entries = s_mhl[["segment", "month"]].drop_duplicates()
            match_count = unhealthy_in_se.merge(ledger_entries, on=["segment", "month"])
            if abs(len(match_count) - len(ledger_entries)) <= 2:
                passed += 1
                print("  PASS  CROSS_PENALTY_CONSISTENCY")
            else:
                print(f"  FAIL  CROSS_PENALTY_CONSISTENCY: ledger has {len(ledger_entries)} entries, SE unhealthy has {len(unhealthy_in_se)}")
        else:
            print("  FAIL  CROSS_PENALTY_CONSISTENCY: files missing")
    except Exception as e:
        print(f"  FAIL  CROSS_PENALTY_CONSISTENCY: {e}")

    total = len(checks)
    print(f"\n{passed}/{total}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    task_dir = Path(__file__).parent.parent / "task"
    run_checks(task_dir)

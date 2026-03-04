import pandas as pd
import numpy as np
from pathlib import Path
import sys

PASS = 0
FAIL = 0


def check(condition, msg):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"[PASS] {msg}")
    else:
        FAIL += 1
        print(f"[FAIL] {msg}")


def load_csv(path, name):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        check(False, f"{name} could not be loaded: {e}")
        return None


def verify_exploded_requirements(student_df, golden_df):
    check(student_df is not None, "exploded_requirements.csv exists and loaded")
    if student_df is None:
        return
    
    required_cols = ["component_id", "total_required_qty", "num_parent_assemblies", 
                     "deepest_level", "category", "unit_cost", "extended_cost"]
    for col in required_cols:
        check(col in student_df.columns, f"exploded_requirements has column '{col}'")
    
    check(len(student_df) == len(golden_df), 
          f"exploded_requirements row count matches ({len(student_df)} == {len(golden_df)})")
    
    golden_comps = set(golden_df["component_id"])
    student_comps = set(student_df["component_id"])
    check(golden_comps == student_comps, "exploded_requirements component_id set matches")
    
    merged = pd.merge(student_df, golden_df, on="component_id", suffixes=("_s", "_g"))
    
    qty_match = 0
    for _, row in merged.iterrows():
        if abs(row["total_required_qty_s"] - row["total_required_qty_g"]) <= 0.01:
            qty_match += 1
    
    pct = (qty_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(pct >= 95, f"exploded_requirements total_required_qty >95% match ({pct:.1f}%)")
    check(pct >= 98, f"exploded_requirements total_required_qty >98% match ({pct:.1f}%)")
    
    cost_match = 0
    for _, row in merged.iterrows():
        if abs(row["extended_cost_s"] - row["extended_cost_g"]) < 1.0:
            cost_match += 1
    
    cost_pct = (cost_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(cost_pct >= 95, f"exploded_requirements extended_cost >95% match ({cost_pct:.1f}%)")
    
    comp_with_multiple = len(merged[merged["num_parent_assemblies_g"] > 1])
    check(comp_with_multiple > 5, f"exploded_requirements has components with multi-parent ({comp_with_multiple} > 5)")
    
    deep_levels = len(merged[merged["deepest_level_g"] > 2])
    check(deep_levels > 3, f"exploded_requirements has deep BOM levels ({deep_levels} > 3)")


def verify_material_shortages(student_df, golden_df, exploded_df):
    check(student_df is not None, "material_shortages.csv exists and loaded")
    if student_df is None:
        return
    
    required_cols = ["component_id", "required_qty", "available_qty", "on_order_qty",
                     "gross_shortage", "net_shortage", "is_critical", "substitute_id",
                     "substitute_factor", "substitute_available", "substitute_can_cover"]
    for col in required_cols:
        check(col in student_df.columns, f"material_shortages has column '{col}'")
    
    check(len(student_df) == len(golden_df),
          f"material_shortages row count matches ({len(student_df)} == {len(golden_df)})")
    
    check(len(student_df) < len(exploded_df), 
          "material_shortages is subset of exploded (only includes shortages)")
    
    merged = pd.merge(student_df, golden_df, on="component_id", suffixes=("_s", "_g"))
    
    gross_match = 0
    for _, row in merged.iterrows():
        if abs(row["gross_shortage_s"] - row["gross_shortage_g"]) < 0.1:
            gross_match += 1
    
    gross_pct = (gross_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(gross_pct >= 90, f"material_shortages gross_shortage >90% match ({gross_pct:.1f}%)")
    
    net_match = 0
    for _, row in merged.iterrows():
        if abs(row["net_shortage_s"] - row["net_shortage_g"]) < 0.1:
            net_match += 1
    
    net_pct = (net_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(net_pct >= 90, f"material_shortages net_shortage >90% match ({net_pct:.1f}%)")
    
    with_subs = len(merged[(merged["substitute_id_g"].notna()) & (merged["substitute_id_g"] != "")])
    check(with_subs > 0, f"material_shortages has items with substitutes ({with_subs} > 0)")
    
    sub_avail_match = 0
    for _, row in merged.iterrows():
        if abs(float(row["substitute_available_s"]) - float(row["substitute_available_g"])) < 0.5:
            sub_avail_match += 1
    sub_avail_pct = (sub_avail_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(sub_avail_pct >= 90, f"material_shortages substitute_available >90% match ({sub_avail_pct:.1f}%)")
    
    sub_match = 0
    for _, row in merged.iterrows():
        s = str(row.get("substitute_can_cover_s", ""))
        g = str(row.get("substitute_can_cover_g", ""))
        if s == g:
            sub_match += 1
    
    sub_pct = (sub_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(sub_pct >= 85, f"material_shortages substitute_can_cover >85% match ({sub_pct:.1f}%)")
    
    critical_rows = merged[merged["is_critical_g"] == "YES"]
    check(len(critical_rows) > 0, "material_shortages has critical items")


def verify_purchase_orders(student_df, golden_df):
    check(student_df is not None, "purchase_orders.csv exists and loaded")
    if student_df is None:
        return
    
    required_cols = ["po_number", "supplier_id", "num_line_items", "total_qty",
                     "total_value", "max_lead_time_days", "meets_minimum", "min_order_value"]
    for col in required_cols:
        check(col in student_df.columns, f"purchase_orders has column '{col}'")
    
    check(len(student_df) == len(golden_df),
          f"purchase_orders row count matches ({len(student_df)} == {len(golden_df)})")
    
    golden_suppliers = set(golden_df["supplier_id"])
    student_suppliers = set(student_df["supplier_id"])
    check(golden_suppliers == student_suppliers, "purchase_orders supplier_id set matches")
    
    merged = pd.merge(student_df, golden_df, on="supplier_id", suffixes=("_s", "_g"))
    
    value_match = 0
    for _, row in merged.iterrows():
        if abs(row["total_value_s"] - row["total_value_g"]) < 1.0:
            value_match += 1
    
    val_pct = (value_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(val_pct >= 90, f"purchase_orders total_value >90% match ({val_pct:.1f}%)")
    
    lines_match = 0
    for _, row in merged.iterrows():
        if row["num_line_items_s"] == row["num_line_items_g"]:
            lines_match += 1
    
    lines_pct = (lines_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(lines_pct >= 90, f"purchase_orders num_line_items >90% match ({lines_pct:.1f}%)")
    
    meets_match = 0
    for _, row in merged.iterrows():
        if row["meets_minimum_s"] == row["meets_minimum_g"]:
            meets_match += 1
    
    meets_pct = (meets_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(meets_pct >= 90, f"purchase_orders meets_minimum >90% match ({meets_pct:.1f}%)")


def verify_critical_path(student_df, golden_df):
    check(student_df is not None, "critical_path_analysis.csv exists and loaded")
    if student_df is None:
        return
    
    required_cols = ["assembly_id", "priority", "due_date", "total_lead_time_days",
                     "num_direct_children", "critical_path_child", "critical_child_lead_time",
                     "buffer_days", "required_start_days", "capacity_adjusted_days", "required_start_date"]
    for col in required_cols:
        check(col in student_df.columns, f"critical_path_analysis has column '{col}'")
    
    check(len(student_df) == len(golden_df),
          f"critical_path_analysis row count matches ({len(student_df)} == {len(golden_df)})")
    
    merged = pd.merge(student_df, golden_df, on="assembly_id", suffixes=("_s", "_g"))
    
    lt_match = 0
    for _, row in merged.iterrows():
        if row["total_lead_time_days_s"] == row["total_lead_time_days_g"]:
            lt_match += 1
    
    lt_pct = (lt_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(lt_pct >= 80, f"critical_path total_lead_time_days >80% match ({lt_pct:.1f}%)")
    check(lt_pct >= 95, f"critical_path total_lead_time_days >95% match ({lt_pct:.1f}%)")
    
    crit_match = 0
    for _, row in merged.iterrows():
        if row["critical_path_child_s"] == row["critical_path_child_g"]:
            crit_match += 1
    
    crit_pct = (crit_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(crit_pct >= 80, f"critical_path critical_path_child >80% match ({crit_pct:.1f}%)")
    
    buf_match = 0
    for _, row in merged.iterrows():
        if row["buffer_days_s"] == row["buffer_days_g"]:
            buf_match += 1
    
    buf_pct = (buf_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(buf_pct == 100, f"critical_path buffer_days 100% match ({buf_pct:.1f}%)")
    
    start_match = 0
    for _, row in merged.iterrows():
        if row["required_start_days_s"] == row["required_start_days_g"]:
            start_match += 1
    
    start_pct = (start_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(start_pct >= 90, f"critical_path required_start_days >90% match ({start_pct:.1f}%)")
    
    adj_match = 0
    for _, row in merged.iterrows():
        if row["capacity_adjusted_days_s"] == row["capacity_adjusted_days_g"]:
            adj_match += 1
    
    adj_pct = (adj_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(adj_pct >= 80, f"critical_path capacity_adjusted_days >80% match ({adj_pct:.1f}%)")
    check(adj_pct == 100, f"critical_path capacity_adjusted_days 100% match ({adj_pct:.1f}%)")
    
    rsd_match = 0
    for _, row in merged.iterrows():
        if str(row["required_start_date_s"]) == str(row["required_start_date_g"]):
            rsd_match += 1
    
    rsd_pct = (rsd_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(rsd_pct >= 80, f"critical_path required_start_date >80% match ({rsd_pct:.1f}%)")
    check(rsd_pct == 100, f"critical_path required_start_date 100% match ({rsd_pct:.1f}%)")


def verify_cost_rollup(student_df, golden_df):
    check(student_df is not None, "cost_rollup.csv exists and loaded")
    if student_df is None:
        return
    
    required_cols = ["assembly_id", "demand_quantity", "unit_material_cost", "unit_labor_cost",
                     "unit_overhead_cost", "unit_total_cost", "extended_material", "extended_labor",
                     "extended_overhead", "extended_total", "margin_rate", "margin_amount", "selling_price"]
    for col in required_cols:
        check(col in student_df.columns, f"cost_rollup has column '{col}'")
    
    check(len(student_df) == len(golden_df),
          f"cost_rollup row count matches ({len(student_df)} == {len(golden_df)})")
    
    merged = pd.merge(student_df, golden_df, on="assembly_id", suffixes=("_s", "_g"))
    
    mat_match = 0
    for _, row in merged.iterrows():
        if abs(row["unit_material_cost_s"] - row["unit_material_cost_g"]) < 0.50:
            mat_match += 1
    
    mat_pct = (mat_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(mat_pct >= 90, f"cost_rollup unit_material_cost >90% match ({mat_pct:.1f}%)")
    
    total_match = 0
    for _, row in merged.iterrows():
        if abs(row["unit_total_cost_s"] - row["unit_total_cost_g"]) < 0.50:
            total_match += 1
    
    total_pct = (total_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(total_pct >= 90, f"cost_rollup unit_total_cost >90% match ({total_pct:.1f}%)")
    
    ext_match = 0
    for _, row in merged.iterrows():
        if abs(row["extended_total_s"] - row["extended_total_g"]) < 2.0:
            ext_match += 1
    
    ext_pct = (ext_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(ext_pct >= 95, f"cost_rollup extended_total >95% match ({ext_pct:.1f}%)")
    
    sell_match = 0
    for _, row in merged.iterrows():
        if abs(row["selling_price_s"] - row["selling_price_g"]) < 5.0:
            sell_match += 1
    
    sell_pct = (sell_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(sell_pct >= 95, f"cost_rollup selling_price >95% match ({sell_pct:.1f}%)")
    
    margin_match = 0
    for _, row in merged.iterrows():
        if abs(row["margin_rate_s"] - row["margin_rate_g"]) < 0.001:
            margin_match += 1
    
    margin_pct = (margin_match / len(merged)) * 100 if len(merged) > 0 else 0
    check(margin_pct == 100, f"cost_rollup margin_rate 100% match ({margin_pct:.1f}%)")


def verify_cross_file_consistency(exploded_df, shortages_df, po_df, cost_df):
    if exploded_df is None or shortages_df is None:
        return
    
    shortage_comps = set(shortages_df["component_id"])
    exploded_comps = set(exploded_df["component_id"])
    check(shortage_comps.issubset(exploded_comps), 
          "material_shortages components are subset of exploded_requirements")
    
    total_short_qty = shortages_df["net_shortage"].sum()
    check(total_short_qty > 0, f"material_shortages has positive net_shortage sum ({total_short_qty:.0f})")
    
    if po_df is not None and cost_df is not None:
        total_po_value = po_df["total_value"].sum()
        total_cost = cost_df["extended_total"].sum()
        check(total_po_value > 0, f"purchase_orders has positive total_value ({total_po_value:.2f})")
        check(total_cost > 0, f"cost_rollup has positive extended_total ({total_cost:.2f})")


def verify_bom_explosion_logic(golden_exploded):
    check(len(golden_exploded) > 30, f"Sufficient component explosion ({len(golden_exploded)} > 30)")
    
    multi_parent = golden_exploded[golden_exploded["num_parent_assemblies"] > 1]
    check(len(multi_parent) > 5, f"BOM has shared components ({len(multi_parent)} > 5)")
    
    multi_level = golden_exploded[golden_exploded["deepest_level"] > 2]
    check(len(multi_level) > 3, f"BOM has deep levels ({len(multi_level)} > 3)")
    
    categories = golden_exploded["category"].nunique()
    check(categories >= 3, f"Multiple component categories ({categories} >= 3)")


def main():
    base_dir = Path(__file__).parent
    task_dir = base_dir.parent / "task"
    
    golden_exploded = load_csv(base_dir / "golden_exploded_requirements.csv", "golden_exploded_requirements")
    golden_shortages = load_csv(base_dir / "golden_material_shortages.csv", "golden_material_shortages")
    golden_po = load_csv(base_dir / "golden_purchase_orders.csv", "golden_purchase_orders")
    golden_critical = load_csv(base_dir / "golden_critical_path_analysis.csv", "golden_critical_path_analysis")
    golden_cost = load_csv(base_dir / "golden_cost_rollup.csv", "golden_cost_rollup")
    
    student_exploded = load_csv(task_dir / "exploded_requirements.csv", "exploded_requirements")
    student_shortages = load_csv(task_dir / "material_shortages.csv", "material_shortages")
    student_po = load_csv(task_dir / "purchase_orders.csv", "purchase_orders")
    student_critical = load_csv(task_dir / "critical_path_analysis.csv", "critical_path_analysis")
    student_cost = load_csv(task_dir / "cost_rollup.csv", "cost_rollup")
    
    if golden_exploded is not None:
        verify_bom_explosion_logic(golden_exploded)
    
    verify_exploded_requirements(student_exploded, golden_exploded)
    
    if golden_exploded is not None:
        verify_material_shortages(student_shortages, golden_shortages, golden_exploded)
    
    verify_purchase_orders(student_po, golden_po)
    verify_critical_path(student_critical, golden_critical)
    verify_cost_rollup(student_cost, golden_cost)
    
    verify_cross_file_consistency(student_exploded, student_shortages, student_po, student_cost)
    
    print(f"{PASS}/{PASS+FAIL}")
    
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

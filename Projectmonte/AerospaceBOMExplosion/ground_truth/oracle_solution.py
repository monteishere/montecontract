
import json
import math
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime
from pathlib import Path

TASK          = Path(__file__).parent.parent / "task"
GT            = Path(__file__).parent
ANALYSIS_DATE = datetime(2024, 6, 1)

def load_inputs():
    asm   = pd.read_csv(TASK / "assemblies.csv")
    comp  = pd.read_csv(TASK / "components.csv")
    bom   = pd.read_csv(TASK / "bom_structure.csv")
    inv   = pd.read_csv(TASK / "inventory.csv")
    subs  = pd.read_csv(TASK / "substitutions.csv")
    pric  = pd.read_csv(TASK / "supplier_pricing.csv")
    lab   = pd.read_csv(TASK / "labor_rates.csv")
    exp   = pd.read_csv(TASK / "expedite_fees.csv")
    with open(TASK / "supplier_contracts.json") as fh:
        ctrs = json.load(fh)
    dlv   = pq.read_table(TASK / "delivery_performance.parquet").to_pandas()

    sched = pd.read_excel(TASK / "production_schedule.xlsx",
                          sheet_name="operational_schedule", dtype={"month": str})
    return asm, comp, bom, inv, subs, pric, lab, exp, ctrs, dlv, sched

def get_contract_discount(ctrs, supplier_id, category):

    sup = ctrs.get(supplier_id, {})

    cat_list = sup.get(category, None)
    if cat_list is None:
        return 0.0

    for entry in cat_list:
        ef = datetime.strptime(entry["effective_from"], "%Y-%m-%d")
        et = datetime.strptime(entry["effective_to"],   "%Y-%m-%d")
        if ef <= ANALYSIS_DATE <= et:
            return entry["discount_rate"]
    return 0.0  

def get_unit_price(pric_df, component_id, qty):

    mask = ((pric_df["component_id"] == component_id) &
            (pric_df["min_qty"] <= qty) &
            (pric_df["max_qty"] >= qty))
    hits = pric_df[mask]
    return float(hits.iloc[0]["unit_price"]) if not hits.empty else None

def get_capacity_multiplier(sched, year_month, priority):

    month_rows = sched[sched["month"] == year_month]
    exact = month_rows[month_rows["priority_scope"] == priority]
    if not exact.empty:
        return float(exact.iloc[0]["capacity_multiplier"])
    sentinel = month_rows[month_rows["priority_scope"] == "ALL"]
    if not sentinel.empty:
        return float(sentinel.iloc[0]["capacity_multiplier"])
    return 1.0

def explode_bom(parent_id, demand, bom_df, depth=0):

    if depth > 10:
        return {}
    result = {}
    for _, row in bom_df[bom_df["parent_id"] == parent_id].iterrows():
        child = row["child_id"]
        qty   = demand * float(row["quantity_per"]) * (1.0 + float(row["scrap_rate"]))
        if child[:3] in ("CMP", "PRT", "MAT", "RAW"):
            result[child] = result.get(child, 0.0) + qty
        else:
            for cid, cqty in explode_bom(child, qty, bom_df, depth + 1).items():
                result[cid] = result.get(cid, 0.0) + cqty
    return result

def compute_supplier_scorecard(dlv):

    rows = []
    for sup_id, grp in dlv.sort_values("scheduled_date").groupby("supplier_id"):
        statuses = grp["status"].tolist()
        total    = len(statuses)
        on_time  = sum(1 for s in statuses if s == "ON_TIME")

        consec = 0
        for s in reversed(statuses):
            if s == "LATE":
                consec += 1
            else:
                break
        grp2 = grp.assign(
            sched_dt  = pd.to_datetime(grp["scheduled_date"]),
            actual_dt = pd.to_datetime(grp["actual_delivery_date"]),
        )
        grp2["delay"] = (grp2["actual_dt"] - grp2["sched_dt"]).dt.days
        late_mask = grp2["status"] == "LATE"
        avg_delay = round(float(grp2.loc[late_mask, "delay"].mean()), 2) if late_mask.any() else 0.0
        rows.append({
            "supplier_id":            sup_id,
            "total_deliveries":       total,
            "on_time_deliveries":     on_time,
            "reliability_score":      round(on_time / total, 4),
            "consecutive_late_count": consec,
            "average_delay_days":     avg_delay,
        })
    df = pd.DataFrame(rows).sort_values("supplier_id").reset_index(drop=True)
    df.to_csv(GT / "supplier_scorecard.csv", index=False)
    return df

def compute_exploded_requirements(asm, comp, bom, inv, pric, ctrs):
    comp_idx = comp.set_index("component_id")
    inv_idx  = inv.set_index("component_id")
    rows = []
    for _, a in asm.iterrows():
        asm_id = a["assembly_id"]
        demand = int(a["demand_quantity"])
        for cid, gross in explode_bom(asm_id, demand, bom).items():
            gross    = round(gross, 6)
            c        = comp_idx.loc[cid]
            sup_id   = str(c["default_supplier"])
            category = str(c["category"])
            unit_cost = float(c["unit_cost"])
            on_hand   = float(inv_idx.loc[cid, "on_hand_qty"])  if cid in inv_idx.index else 0.0
            alloc     = float(inv_idx.loc[cid, "allocated_qty"]) if cid in inv_idx.index else 0.0
            available = max(0.0, on_hand - alloc)
            net_qty   = max(0.0, gross - available)

            price    = get_unit_price(pric, cid, max(1, math.ceil(gross)))
            if price is None:
                price = unit_cost

            discount   = get_contract_discount(ctrs, sup_id, category)
            disc_price = round(price * (1.0 - discount), 6)
            rows.append({
                "assembly_id":          asm_id,
                "component_id":         cid,
                "demand_qty":           demand,
                "gross_qty_required":   round(gross, 4),
                "net_qty_required":     round(net_qty, 4),
                "unit_price":           round(price, 4),
                "contract_discount":    discount,
                "discounted_unit_price": round(disc_price, 4),
            })
    df = (pd.DataFrame(rows)
            .sort_values(["assembly_id", "component_id"])
            .reset_index(drop=True))
    df.to_csv(GT / "exploded_requirements.csv", index=False)
    return df

def compute_material_shortages(expl_df, subs, inv, comp):
    inv_idx  = inv.set_index("component_id")
    comp_idx = comp.set_index("component_id")

    sub_cov = {}
    for _, s in subs.iterrows():
        if s["approval_status"] != "APPROVED":
            continue
        prim    = s["primary_component"]
        sub_c   = s["substitute_component"]
        sub_oh  = float(inv_idx.loc[sub_c, "on_hand_qty"]) if sub_c in inv_idx.index else 0.0
        cf      = float(s["conversion_factor"])
        eff     = sub_oh / cf if cf > 0 else 0.0
        sub_cov[prim] = sub_cov.get(prim, 0.0) + eff
    rows = []
    for _, req in expl_df.iterrows():
        net = float(req["net_qty_required"])
        if net < 1e-6:
            continue
        cid       = req["component_id"]
        avail_sub = round(sub_cov.get(cid, 0.0), 2)
        covered   = "YES" if avail_sub >= net else "NO"
        is_crit   = str(comp_idx.loc[cid, "is_critical"]) if cid in comp_idx.index else "NO"
        severity  = "CRITICAL" if is_crit == "YES" or net > 50.0 else "STANDARD"
        rows.append({
            "assembly_id":             req["assembly_id"],
            "component_id":            cid,
            "net_qty_required":        round(net, 4),
            "available_substitute_qty": avail_sub,
            "shortage_covered":        covered,
            "shortage_severity":       severity,
        })
    df = (pd.DataFrame(rows)
            .sort_values(["assembly_id", "component_id"])
            .reset_index(drop=True))
    df.to_csv(GT / "material_shortages.csv", index=False)
    return df

def compute_purchase_orders(shortages_df, comp, pric, exp_df, scorecard_df):

    comp_idx  = comp.set_index("component_id")
    exp_idx   = exp_df.set_index("supplier_id")
    sc_idx    = scorecard_df.set_index("supplier_id")
    rows = []
    for _, s in shortages_df.iterrows():

        if s["shortage_covered"] == "YES":
            continue
        cid       = s["component_id"]
        net_qty   = float(s["net_qty_required"])
        c         = comp_idx.loc[cid]
        sup_id    = str(c["default_supplier"])
        moq       = int(c["min_order_qty"])
        base_lead = int(c["lead_time_days"])
        qty_ord   = math.ceil(max(net_qty, moq))

        price = get_unit_price(pric, cid, qty_ord)
        if price is None:
            price = float(c["unit_cost"])
        po_value = round(qty_ord * price, 4)

        if sup_id in exp_idx.index:
            pct     = float(exp_idx.loc[sup_id, "expedite_pct"])
            max_fee = float(exp_idx.loc[sup_id, "max_expedite_fee"])
            exp_fee = round(min(po_value * pct, max_fee), 4)
        else:
            exp_fee = 0.0

        consec   = int(sc_idx.loc[sup_id, "consecutive_late_count"]) if sup_id in sc_idx.index else 0
        adj_lead = math.ceil(base_lead * (1.0 + 0.1 * consec))
        rows.append({
            "assembly_id":            s["assembly_id"],
            "component_id":           cid,
            "supplier_id":            sup_id,
            "qty_ordered":            qty_ord,
            "unit_price":             round(price, 4),
            "po_value":               po_value,
            "expedite_fee":           exp_fee,
            "base_lead_time":         base_lead,
            "adjusted_lead_time":     adj_lead,
            "consecutive_late_count": consec,
            "reliability_adjusted":   "YES" if consec > 0 else "NO",
        })
    df = (pd.DataFrame(rows)
            .sort_values(["assembly_id", "component_id"])
            .reset_index(drop=True))
    df.to_csv(GT / "purchase_orders.csv", index=False)
    return df

def compute_cost_rollup(asm, expl_df, comp, lab, sched):

    comp_idx = comp.set_index("component_id")
    lab_idx  = lab.set_index("category")
    rows = []
    for _, a in asm.iterrows():
        asm_id   = a["assembly_id"]
        demand   = int(a["demand_quantity"])
        month    = str(a["due_date"])[:7]          
        priority = str(a["priority"])
        cap_mult = get_capacity_multiplier(sched, month, priority)
        asm_reqs = expl_df[expl_df["assembly_id"] == asm_id]
        mat_cost = disc_mat = lab_cost = 0.0
        for _, req in asm_reqs.iterrows():
            cid   = req["component_id"]
            gross = float(req["gross_qty_required"])
            mat_cost += gross * float(req["unit_price"])
            disc_mat += gross * float(req["discounted_unit_price"])
            cat = str(comp_idx.loc[cid, "category"]) if cid in comp_idx.index else None
            if cat and cat in lab_idx.index:
                lr       = lab_idx.loc[cat]
                lab_cost += gross * float(lr["run_hours_per_unit"]) * \
                            float(lr["labor_rate_per_hour"]) * float(lr["overhead_rate"])
        eff_mat = round(disc_mat * cap_mult, 4)
        rows.append({
            "assembly_id":               asm_id,
            "demand_qty":                demand,
            "capacity_multiplier":       cap_mult,
            "material_cost":             round(mat_cost,  4),
            "discounted_material_cost":  round(disc_mat,  4),
            "effective_material_cost":   eff_mat,
            "labor_cost":                round(lab_cost,  4),
            "total_cost":                round(eff_mat + lab_cost, 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(GT / "cost_rollup.csv", index=False)
    return df

if __name__ == "__main__":
    (asm, comp, bom, inv, subs, pric, lab, exp, ctrs, dlv, sched) = load_inputs()
    scorecard  = compute_supplier_scorecard(dlv)
    expl       = compute_exploded_requirements(asm, comp, bom, inv, pric, ctrs)
    shortages  = compute_material_shortages(expl, subs, inv, comp)
    pos        = compute_purchase_orders(shortages, comp, pric, exp, scorecard)
    rollup     = compute_cost_rollup(asm, expl, comp, lab, sched)
    print(f"supplier_scorecard.csv:      {len(scorecard)} rows")
    print(f"exploded_requirements.csv:   {len(expl)} rows")
    print(f"material_shortages.csv:      {len(shortages)} rows")
    print(f"purchase_orders.csv:         {len(pos)} rows")
    print(f"cost_rollup.csv:             {len(rollup)} rows")

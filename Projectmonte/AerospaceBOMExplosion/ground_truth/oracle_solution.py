import math
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

ANALYSIS_DATE = "2024-06-01"


def load_data(task_dir):
    assemblies = pd.read_csv(task_dir / "assemblies.csv")
    components = pd.read_csv(task_dir / "components.csv")
    bom_structure = pd.read_csv(task_dir / "bom_structure.csv")
    suppliers = pd.read_csv(task_dir / "suppliers.csv")
    inventory = pd.read_csv(task_dir / "inventory.csv")
    substitutions = pd.read_csv(task_dir / "substitutions.csv")
    supplier_pricing = pd.read_csv(task_dir / "supplier_pricing.csv")
    production_calendar = pd.read_csv(task_dir / "production_calendar.csv")
    labor_rates = pd.read_csv(task_dir / "labor_rates.csv")
    return (assemblies, components, bom_structure, suppliers, inventory,
            substitutions, supplier_pricing, production_calendar, labor_rates)


def explode_bom(assemblies, bom_structure):
    bom_dict = defaultdict(list)
    for _, row in bom_structure.iterrows():
        bom_dict[row["parent_id"]].append({
            "child_id": row["child_id"],
            "quantity_per": row["quantity_per"],
            "scrap_rate": row["scrap_rate"],
            "level": row["level"],
        })
    
    requirements = defaultdict(lambda: {"total_qty": 0.0, "sources": []})
    
    def explode_recursive(parent_id, parent_qty, path_level, source_assembly):
        children = bom_dict.get(parent_id, [])
        for child in children:
            child_id = child["child_id"]
            scrap_rate = child["scrap_rate"] if pd.notna(child["scrap_rate"]) else 0.0
            adj_qty = child["quantity_per"] * (1 + scrap_rate)
            exploded_qty = round(parent_qty * adj_qty, 2)
            
            requirements[child_id]["total_qty"] += exploded_qty
            requirements[child_id]["sources"].append({
                "assembly": source_assembly,
                "parent": parent_id,
                "qty": exploded_qty,
                "level": path_level,
            })
            
            if child_id.startswith("SUB") or child_id.startswith("MOD"):
                explode_recursive(child_id, exploded_qty, path_level + 1, source_assembly)
    
    for _, asm in assemblies.iterrows():
        asm_id = asm["assembly_id"]
        demand = asm["demand_quantity"]
        explode_recursive(asm_id, demand, 1, asm_id)
    
    return requirements


def compute_exploded_requirements(assemblies, bom_structure, components):
    requirements = explode_bom(assemblies, bom_structure)
    
    comp_info = {}
    for _, row in components.iterrows():
        comp_info[row["component_id"]] = {
            "category": row["category"],
            "unit_cost": row["unit_cost"],
            "is_critical": row["is_critical"],
        }
    
    rows = []
    for comp_id, req in requirements.items():
        if comp_id.startswith("SUB") or comp_id.startswith("MOD"):
            continue
        
        total_qty = round(req["total_qty"], 2)
        num_assemblies = len(set(s["assembly"] for s in req["sources"]))
        max_level = max(s["level"] for s in req["sources"])
        
        info = comp_info.get(comp_id, {"category": "UNKNOWN", "unit_cost": 0.0})
        category = info["category"]
        unit_cost = info["unit_cost"]
        
        extended_cost = round(total_qty * unit_cost, 2)
        
        rows.append({
            "component_id": comp_id,
            "total_required_qty": total_qty,
            "num_parent_assemblies": num_assemblies,
            "deepest_level": max_level,
            "category": category,
            "unit_cost": round(unit_cost, 2),
            "extended_cost": extended_cost,
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("component_id").reset_index(drop=True)
    return df


def compute_material_shortages(exploded_requirements, inventory, substitutions, components):
    inv_map = {}
    for _, row in inventory.iterrows():
        comp_id = row["component_id"]
        available = row["on_hand_qty"] - row["allocated_qty"]
        on_order = row["on_order_qty"] if pd.notna(row["on_order_qty"]) else 0
        inv_map[comp_id] = {
            "available": available,
            "on_order": on_order,
            "expected_date": row["expected_receipt_date"],
        }
    
    sub_map = defaultdict(list)
    for _, row in substitutions.iterrows():
        if row["approval_status"] == "APPROVED":
            # Only use substitutes valid on the analysis date
            if str(row["effective_from"]) <= ANALYSIS_DATE <= str(row["effective_to"]):
                sub_map[row["primary_component"]].append({
                    "substitute": row["substitute_component"],
                    "factor": row["conversion_factor"],
                    "priority": row["priority_rank"],
                })
    
    comp_info = {}
    for _, row in components.iterrows():
        comp_info[row["component_id"]] = {
            "is_critical": row["is_critical"],
            "default_supplier": row["default_supplier"],
            "min_order_qty": row["min_order_qty"],
            "unit_cost": row["unit_cost"],
        }
    
    sub_pool = {comp_id: info["available"] for comp_id, info in inv_map.items()}
    
    rows = []
    for _, req in exploded_requirements.iterrows():
        comp_id = req["component_id"]
        needed = req["total_required_qty"]
        
        inv = inv_map.get(comp_id, {"available": 0, "on_order": 0, "expected_date": ""})
        available = inv["available"]
        on_order = inv["on_order"]
        
        gross_shortage = round(max(0, needed - available), 2)
        net_shortage = round(max(0, needed - available - on_order), 2)
        
        if gross_shortage <= 0:
            continue
        
        substitute_id = ""
        substitute_factor = 0.0
        substitute_available = 0
        substitute_can_cover = "NO"
        
        subs = sub_map.get(comp_id, [])
        subs_sorted = sorted(subs, key=lambda x: x["priority"])
        
        for sub in subs_sorted:
            sub_comp = sub["substitute"]
            factor = sub["factor"]
            sub_avail = sub_pool.get(sub_comp, 0)
            
            qty_needed_from_sub = round(gross_shortage * factor, 2)
            
            if sub_avail >= qty_needed_from_sub:
                substitute_id = sub_comp
                substitute_factor = factor
                substitute_available = sub_avail
                substitute_can_cover = "YES"
                break
            elif sub_avail > 0:
                substitute_id = sub_comp
                substitute_factor = factor
                substitute_available = sub_avail
                substitute_can_cover = "PARTIAL"
        
        if substitute_id and substitute_can_cover in ("YES", "PARTIAL"):
            if substitute_can_cover == "YES":
                sub_pool[substitute_id] = max(0, sub_pool.get(substitute_id, 0) - round(gross_shortage * substitute_factor, 2))
            else:
                sub_pool[substitute_id] = 0
        
        info = comp_info.get(comp_id, {"is_critical": "NO"})
        is_critical = info["is_critical"]

        # Compute shortage_severity
        if is_critical == "YES" and substitute_can_cover == "NO":
            shortage_severity = "CRITICAL"
        elif is_critical == "YES" and substitute_can_cover in ("YES", "PARTIAL"):
            shortage_severity = "HIGH"
        elif is_critical == "NO" and net_shortage >= 500:
            shortage_severity = "HIGH"
        elif is_critical == "NO" and net_shortage >= 100:
            shortage_severity = "MEDIUM"
        else:
            shortage_severity = "LOW"

        rows.append({
            "component_id": comp_id,
            "required_qty": round(needed, 2),
            "available_qty": available,
            "on_order_qty": on_order,
            "gross_shortage": gross_shortage,
            "net_shortage": net_shortage,
            "is_critical": is_critical,
            "substitute_id": substitute_id,
            "substitute_factor": substitute_factor,
            "substitute_available": substitute_available,
            "substitute_can_cover": substitute_can_cover,
            "shortage_severity": shortage_severity,
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("component_id").reset_index(drop=True)
    return df


def compute_purchase_orders(material_shortages, components, suppliers, supplier_pricing):
    comp_info = {}
    for _, row in components.iterrows():
        comp_info[row["component_id"]] = {
            "default_supplier": row["default_supplier"],
            "min_order_qty": row["min_order_qty"],
            "unit_cost": row["unit_cost"],
        }
    
    sup_info = {}
    for _, row in suppliers.iterrows():
        sup_info[row["supplier_id"]] = {
            "base_lead_time": row["base_lead_time"],
            "min_order_value": row["min_order_value"],
        }
    
    pricing_map = defaultdict(list)
    for _, row in supplier_pricing.iterrows():
        key = (row["component_id"], row["supplier_id"])
        pricing_map[key].append({
            "min_qty": row["min_qty"],
            "max_qty": row["max_qty"],
            "unit_price": row["unit_price"],
        })
    
    po_lines = defaultdict(list)
    
    for _, shortage in material_shortages.iterrows():
        if shortage["net_shortage"] <= 0:
            continue
        
        comp_id = shortage["component_id"]
        needed_qty = shortage["net_shortage"]
        
        info = comp_info.get(comp_id, {"default_supplier": "SUP001", "min_order_qty": 1, "unit_cost": 0.0})
        supplier_id = info["default_supplier"]
        min_order = info["min_order_qty"]
        
        order_qty = int(math.ceil(max(needed_qty, 0.001) / min_order) * min_order)
        
        pricing_tiers = pricing_map.get((comp_id, supplier_id), [])
        unit_price = info["unit_cost"]
        
        for tier in pricing_tiers:
            if tier["min_qty"] <= order_qty <= tier["max_qty"]:
                unit_price = tier["unit_price"]
                break
        
        line_total = round(order_qty * unit_price, 2)
        
        sup = sup_info.get(supplier_id, {"base_lead_time": 30})
        lead_time = sup["base_lead_time"]
        
        po_lines[supplier_id].append({
            "component_id": comp_id,
            "order_qty": int(order_qty),
            "unit_price": round(unit_price, 2),
            "line_total": line_total,
            "lead_time": lead_time,
        })
    
    rows = []
    po_num = 1
    for supplier_id, lines in po_lines.items():
        sup = sup_info.get(supplier_id, {"min_order_value": 0})
        min_order_value = sup["min_order_value"]
        
        total_value = sum(line["line_total"] for line in lines)
        total_qty = sum(line["order_qty"] for line in lines)
        max_lead = max(line["lead_time"] for line in lines)
        num_lines = len(lines)
        
        meets_minimum = "YES" if total_value >= min_order_value else "NO"
        
        rows.append({
            "po_number": f"PO{str(po_num).zfill(5)}",
            "supplier_id": supplier_id,
            "num_line_items": num_lines,
            "total_qty": total_qty,
            "total_value": round(total_value, 2),
            "max_lead_time_days": max_lead,
            "meets_minimum": meets_minimum,
            "min_order_value": round(min_order_value, 2),
        })
        po_num += 1
    
    df = pd.DataFrame(rows)
    df = df.sort_values("supplier_id").reset_index(drop=True)
    return df


def compute_critical_path(assemblies, bom_structure, components, suppliers, production_calendar):
    bom_dict = defaultdict(list)
    for _, row in bom_structure.iterrows():
        bom_dict[row["parent_id"]].append(row["child_id"])
    
    comp_info = {}
    for _, row in components.iterrows():
        comp_info[row["component_id"]] = {
            "lead_time_days": row["lead_time_days"],
            "default_supplier": row["default_supplier"],
        }
    
    cal_map = {}
    for _, row in production_calendar.iterrows():
        cal_map[row["month"]] = row["capacity_multiplier"]
    
    lead_time_cache = {}
    
    def get_lead_time(item_id):
        if item_id in lead_time_cache:
            return lead_time_cache[item_id]
        
        children = bom_dict.get(item_id, [])
        
        if len(children) == 0:
            info = comp_info.get(item_id, {"lead_time_days": 0})
            lt = info["lead_time_days"]
            lead_time_cache[item_id] = lt
            return lt
        
        max_child_lt = 0
        for child in children:
            child_lt = get_lead_time(child)
            max_child_lt = max(max_child_lt, child_lt)
        
        own_processing = 7
        if item_id.startswith("SUB"):
            own_processing = 5
        elif item_id.startswith("ASM"):
            own_processing = 10
        
        total_lt = max_child_lt + own_processing
        lead_time_cache[item_id] = total_lt
        return total_lt
    
    rows = []
    for _, asm in assemblies.iterrows():
        asm_id = asm["assembly_id"]
        priority = asm["priority"]
        due_date = asm["due_date"]
        
        total_lead_time = get_lead_time(asm_id)
        
        children = bom_dict.get(asm_id, [])
        num_direct_children = len(children)
        
        critical_child = ""
        critical_child_lt = 0
        for child in children:
            child_lt = get_lead_time(child)
            if child_lt > critical_child_lt:
                critical_child_lt = child_lt
                critical_child = child
        
        buffer_days = 14 if priority == "CRITICAL" else 7
        required_start_days = total_lead_time + buffer_days
        
        due_month = str(due_date)[:7]
        capacity_mult = cal_map.get(due_month, 1.0)
        capacity_adjusted_days = math.ceil(required_start_days / capacity_mult)
        required_start_date = (pd.to_datetime(due_date) - pd.Timedelta(days=capacity_adjusted_days)).strftime("%Y-%m-%d")
        
        rows.append({
            "assembly_id": asm_id,
            "priority": priority,
            "due_date": due_date,
            "total_lead_time_days": total_lead_time,
            "num_direct_children": num_direct_children,
            "critical_path_child": critical_child,
            "critical_child_lead_time": critical_child_lt,
            "buffer_days": buffer_days,
            "required_start_days": required_start_days,
            "capacity_adjusted_days": capacity_adjusted_days,
            "required_start_date": required_start_date,
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("assembly_id").reset_index(drop=True)
    return df


def compute_cost_rollup(assemblies, bom_structure, components, labor_rates):
    bom_dict = defaultdict(list)
    for _, row in bom_structure.iterrows():
        bom_dict[row["parent_id"]].append({
            "child_id": row["child_id"],
            "quantity_per": row["quantity_per"],
            "scrap_rate": row["scrap_rate"] if pd.notna(row["scrap_rate"]) else 0.0,
        })
    
    comp_info = {}
    for _, row in components.iterrows():
        comp_info[row["component_id"]] = {
            "unit_cost": row["unit_cost"],
            "category": row["category"],
        }
    
    labor_info = {}
    for _, row in labor_rates.iterrows():
        labor_info[row["category"]] = {
            "labor_rate_per_hour": row["labor_rate_per_hour"],
            "run_hours_per_unit": row["run_hours_per_unit"],
            "overhead_rate": row["overhead_rate"],
        }
    
    cost_cache = {}
    
    def get_cost(item_id):
        if item_id in cost_cache:
            return cost_cache[item_id]
        
        children = bom_dict.get(item_id, [])
        
        if len(children) == 0:
            info = comp_info.get(item_id, {"unit_cost": 0.0, "category": "STRUCTURAL"})
            mat_cost = info["unit_cost"]
            category = info["category"]
            
            labor = labor_info.get(category, {"labor_rate_per_hour": 50.0, "run_hours_per_unit": 0.5, "overhead_rate": 1.5})
            labor_rate = labor["labor_rate_per_hour"]
            run_hours = labor["run_hours_per_unit"]
            overhead_rate = labor["overhead_rate"]
            
            labor_cost = round(labor_rate * run_hours, 2)
            overhead_cost = round(labor_cost * (overhead_rate - 1), 2)
            
            total = round(mat_cost + labor_cost + overhead_cost, 2)
            cost_cache[item_id] = {
                "material": mat_cost,
                "labor": labor_cost,
                "overhead": overhead_cost,
                "total": total,
            }
            return cost_cache[item_id]
        
        total_material = 0.0
        total_labor = 0.0
        total_overhead = 0.0
        
        for child in children:
            child_id = child["child_id"]
            qty = child["quantity_per"]
            scrap = child["scrap_rate"]
            adj_qty = qty * (1 + scrap)
            
            child_cost = get_cost(child_id)
            total_material += round(child_cost["total"] * adj_qty, 2)
        
        if item_id.startswith("SUB"):
            labor_add = 25.0
            overhead_add = 12.5
        elif item_id.startswith("ASM"):
            labor_add = 75.0
            overhead_add = 37.5
        else:
            labor_add = 10.0
            overhead_add = 5.0
        
        total_material = round(total_material, 2)
        total_labor = round(labor_add, 2)
        total_overhead = round(overhead_add, 2)
        total = round(total_material + total_labor + total_overhead, 2)
        
        cost_cache[item_id] = {
            "material": total_material,
            "labor": total_labor,
            "overhead": total_overhead,
            "total": total,
        }
        return cost_cache[item_id]
    
    rows = []
    for _, asm in assemblies.iterrows():
        asm_id = asm["assembly_id"]
        demand = asm["demand_quantity"]
        
        unit_cost = get_cost(asm_id)
        
        extended_material = round(unit_cost["material"] * demand, 2)
        extended_labor = round(unit_cost["labor"] * demand, 2)
        extended_overhead = round(unit_cost["overhead"] * demand, 2)
        extended_total = round(unit_cost["total"] * demand, 2)
        
        margin_rate = 0.15
        margin_amount = round(extended_total * margin_rate, 2)
        selling_price = round(extended_total + margin_amount, 2)
        
        rows.append({
            "assembly_id": asm_id,
            "demand_quantity": demand,
            "unit_material_cost": round(unit_cost["material"], 2),
            "unit_labor_cost": round(unit_cost["labor"], 2),
            "unit_overhead_cost": round(unit_cost["overhead"], 2),
            "unit_total_cost": round(unit_cost["total"], 2),
            "extended_material": extended_material,
            "extended_labor": extended_labor,
            "extended_overhead": extended_overhead,
            "extended_total": extended_total,
            "margin_rate": margin_rate,
            "margin_amount": margin_amount,
            "selling_price": selling_price,
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("assembly_id").reset_index(drop=True)
    return df


def main():
    base_dir = Path(__file__).parent
    task_dir = base_dir.parent / "task"
    
    data = load_data(task_dir)
    assemblies, components, bom_structure, suppliers, inventory, substitutions, supplier_pricing, production_calendar, labor_rates = data
    
    exploded = compute_exploded_requirements(assemblies, bom_structure, components)
    exploded.to_csv(task_dir / "exploded_requirements.csv", index=False)
    
    shortages = compute_material_shortages(exploded, inventory, substitutions, components)
    shortages.to_csv(task_dir / "material_shortages.csv", index=False)
    
    purchase_orders = compute_purchase_orders(shortages, components, suppliers, supplier_pricing)
    purchase_orders.to_csv(task_dir / "purchase_orders.csv", index=False)
    
    critical_path = compute_critical_path(assemblies, bom_structure, components, suppliers, production_calendar)
    critical_path.to_csv(task_dir / "critical_path_analysis.csv", index=False)
    
    cost_rollup = compute_cost_rollup(assemblies, bom_structure, components, labor_rates)
    cost_rollup.to_csv(task_dir / "cost_rollup.csv", index=False)
    
    print(f"exploded_requirements.csv: {len(exploded)} rows")
    print(f"material_shortages.csv: {len(shortages)} rows")
    print(f"purchase_orders.csv: {len(purchase_orders)} rows")
    print(f"critical_path_analysis.csv: {len(critical_path)} rows")
    print(f"cost_rollup.csv: {len(cost_rollup)} rows")


if __name__ == "__main__":
    main()

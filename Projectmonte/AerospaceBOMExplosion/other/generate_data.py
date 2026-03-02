import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "task")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ASSEMBLY_PREFIXES = ["ASM", "SUB", "MOD"]
COMPONENT_PREFIXES = ["CMP", "PRT", "MAT", "RAW"]
SUPPLIERS = [f"SUP{str(i).zfill(3)}" for i in range(1, 16)]
CATEGORIES = ["STRUCTURAL", "ELECTRICAL", "HYDRAULIC", "AVIONICS", "INTERIOR", "FASTENER"]
UNITS = ["EA", "KG", "M", "L", "SET"]


def generate_assemblies():
    rows = []
    for i in range(1, 13):
        asm_id = f"ASM{str(i).zfill(4)}"
        demand = int(np.random.choice([5, 10, 15, 20, 25, 30]))
        priority = np.random.choice(["CRITICAL", "HIGH", "STANDARD", "LOW"])
        due_date = datetime(2024, 6, 1) + timedelta(days=np.random.randint(30, 180))
        rows.append({
            "assembly_id": asm_id,
            "assembly_name": f"Assembly_{i}",
            "demand_quantity": demand,
            "priority": priority,
            "due_date": due_date.strftime("%Y-%m-%d"),
            "customer_id": f"CUST{np.random.randint(1, 6):03d}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "assemblies.csv"), index=False)
    return df


def generate_components():
    rows = []
    comp_id = 1
    for cat in CATEGORIES:
        n_comps = np.random.randint(8, 15)
        for _ in range(n_comps):
            prefix = np.random.choice(COMPONENT_PREFIXES)
            cid = f"{prefix}{str(comp_id).zfill(5)}"
            unit_cost = round(np.random.lognormal(3.5, 1.2), 2)
            weight = round(np.random.uniform(0.01, 50.0), 3)
            lead_time = int(np.random.choice([7, 14, 21, 30, 45, 60, 90]))
            supplier = np.random.choice(SUPPLIERS)
            rows.append({
                "component_id": cid,
                "component_name": f"{cat}_{comp_id}",
                "category": cat,
                "unit": np.random.choice(UNITS),
                "unit_cost": unit_cost,
                "weight_kg": weight,
                "default_supplier": supplier,
                "lead_time_days": lead_time,
                "min_order_qty": int(np.random.choice([1, 5, 10, 25, 50, 100])),
                "is_critical": "YES" if np.random.random() < 0.15 else "NO",
            })
            comp_id += 1
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "components.csv"), index=False)
    return df


def generate_bom_structure(assemblies_df, components_df):
    rows = []
    comp_ids = components_df["component_id"].tolist()
    
    for _, asm in assemblies_df.iterrows():
        asm_id = asm["assembly_id"]
        n_direct = np.random.randint(3, 8)
        selected = np.random.choice(comp_ids, size=min(n_direct, len(comp_ids)), replace=False)
        for comp in selected:
            qty = round(np.random.choice([1, 2, 3, 4, 5, 0.5, 0.25, 1.5, 2.5]) * np.random.uniform(0.8, 3.0), 2)
            rows.append({
                "parent_id": asm_id,
                "child_id": comp,
                "quantity_per": qty,
                "level": 1,
                "scrap_rate": round(np.random.choice([0.0, 0.01, 0.02, 0.03, 0.05]), 2),
                "operation_sequence": int(np.random.randint(10, 100)),
            })
    
    sub_assemblies = []
    for i in range(1, 9):
        sub_id = f"SUB{str(i).zfill(4)}"
        sub_assemblies.append(sub_id)
        n_children = np.random.randint(2, 6)
        selected_comps = np.random.choice(comp_ids, size=min(n_children, len(comp_ids)), replace=False)
        for comp in selected_comps:
            qty = round(np.random.uniform(1.0, 5.0), 2)
            rows.append({
                "parent_id": sub_id,
                "child_id": comp,
                "quantity_per": qty,
                "level": 2,
                "scrap_rate": round(np.random.choice([0.0, 0.01, 0.02, 0.05]), 2),
                "operation_sequence": int(np.random.randint(10, 100)),
            })
    
    mod_assemblies = []
    for i in range(1, 5):
        mod_id = f"MOD{str(i).zfill(4)}"
        mod_assemblies.append(mod_id)
        n_children = np.random.randint(2, 4)
        selected_comps = np.random.choice(comp_ids, size=min(n_children, len(comp_ids)), replace=False)
        for comp in selected_comps:
            qty = round(np.random.uniform(1.0, 3.0), 2)
            rows.append({
                "parent_id": mod_id,
                "child_id": comp,
                "quantity_per": qty,
                "level": 3,
                "scrap_rate": round(np.random.choice([0.0, 0.01, 0.02]), 2),
                "operation_sequence": int(np.random.randint(10, 50)),
            })
    
    for i, sub_id in enumerate(sub_assemblies[:4]):
        if i < len(mod_assemblies):
            mod_id = mod_assemblies[i]
            qty = round(np.random.uniform(1.0, 2.0), 2)
            rows.append({
                "parent_id": sub_id,
                "child_id": mod_id,
                "quantity_per": qty,
                "level": 2,
                "scrap_rate": 0.01,
                "operation_sequence": 50,
            })
    
    for _, asm in assemblies_df.iterrows():
        asm_id = asm["assembly_id"]
        n_subs = np.random.randint(1, 4)
        selected_subs = np.random.choice(sub_assemblies, size=min(n_subs, len(sub_assemblies)), replace=False)
        for sub in selected_subs:
            qty = round(np.random.uniform(1.0, 3.0), 2)
            rows.append({
                "parent_id": asm_id,
                "child_id": sub,
                "quantity_per": qty,
                "level": 1,
                "scrap_rate": round(np.random.choice([0.0, 0.01, 0.02]), 2),
                "operation_sequence": int(np.random.randint(100, 200)),
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "bom_structure.csv"), index=False)
    return df


def generate_suppliers():
    rows = []
    for sup_id in SUPPLIERS:
        base_lead = int(np.random.choice([7, 14, 21, 30]))
        reliability = round(np.random.uniform(0.80, 0.99), 2)
        rows.append({
            "supplier_id": sup_id,
            "supplier_name": f"Supplier_{sup_id}",
            "country": np.random.choice(["USA", "GERMANY", "JAPAN", "FRANCE", "UK", "CHINA"]),
            "base_lead_time": base_lead,
            "reliability_score": reliability,
            "payment_terms": int(np.random.choice([15, 30, 45, 60])),
            "min_order_value": round(np.random.choice([100, 250, 500, 1000, 2500]), 2),
            "shipping_cost_per_kg": round(np.random.uniform(0.50, 5.00), 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "suppliers.csv"), index=False)
    return df


def generate_inventory(components_df):
    rows = []
    for _, comp in components_df.iterrows():
        on_hand = int(np.random.choice([0, 0, 0, 5, 10, 25, 50, 100, 200, 500]))
        allocated = int(min(on_hand, np.random.randint(0, max(1, on_hand // 2))))
        on_order = int(np.random.choice([0, 0, 0, 25, 50, 100]))
        expected_date = ""
        if on_order > 0:
            expected_date = (datetime(2024, 6, 1) + timedelta(days=np.random.randint(7, 60))).strftime("%Y-%m-%d")
        rows.append({
            "component_id": comp["component_id"],
            "warehouse": np.random.choice(["WH_MAIN", "WH_OVERFLOW", "WH_BONDED"]),
            "on_hand_qty": on_hand,
            "allocated_qty": allocated,
            "on_order_qty": on_order,
            "expected_receipt_date": expected_date,
            "last_count_date": (datetime(2024, 5, 1) + timedelta(days=np.random.randint(0, 30))).strftime("%Y-%m-%d"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "inventory.csv"), index=False)
    return df


def generate_substitutions(components_df):
    rows = []
    comp_ids = components_df["component_id"].tolist()
    comp_cats = dict(zip(components_df["component_id"], components_df["category"]))
    
    for comp_id in comp_ids:
        if np.random.random() < 0.35:
            cat = comp_cats[comp_id]
            same_cat = [c for c in comp_ids if comp_cats[c] == cat and c != comp_id]
            if len(same_cat) > 0:
                substitute = np.random.choice(same_cat)
                factor = round(np.random.choice([0.8, 0.9, 1.0, 1.1, 1.2, 1.25, 1.5]), 2)
                priority = int(np.random.randint(1, 4))
                rows.append({
                    "primary_component": comp_id,
                    "substitute_component": substitute,
                    "conversion_factor": factor,
                    "priority_rank": priority,
                    "effective_from": "2024-01-01",
                    "effective_to": "2024-12-31",
                    "approval_status": np.random.choice(["APPROVED", "APPROVED", "PENDING"]),
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "substitutions.csv"), index=False)
    return df


def generate_supplier_pricing(components_df, suppliers_df):
    rows = []
    for _, comp in components_df.iterrows():
        default_sup = comp["default_supplier"]
        base_cost = comp["unit_cost"]
        
        rows.append({
            "component_id": comp["component_id"],
            "supplier_id": default_sup,
            "min_qty": 1,
            "max_qty": 49,
            "unit_price": round(base_cost * 1.0, 2),
            "effective_date": "2024-01-01",
        })
        rows.append({
            "component_id": comp["component_id"],
            "supplier_id": default_sup,
            "min_qty": 50,
            "max_qty": 199,
            "unit_price": round(base_cost * 0.92, 2),
            "effective_date": "2024-01-01",
        })
        rows.append({
            "component_id": comp["component_id"],
            "supplier_id": default_sup,
            "min_qty": 200,
            "max_qty": 999999,
            "unit_price": round(base_cost * 0.85, 2),
            "effective_date": "2024-01-01",
        })
        
        if np.random.random() < 0.4:
            alt_sup = np.random.choice([s for s in SUPPLIERS if s != default_sup])
            alt_factor = np.random.uniform(0.90, 1.15)
            rows.append({
                "component_id": comp["component_id"],
                "supplier_id": alt_sup,
                "min_qty": 1,
                "max_qty": 999999,
                "unit_price": round(base_cost * alt_factor, 2),
                "effective_date": "2024-01-01",
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "supplier_pricing.csv"), index=False)
    return df


def generate_production_calendar():
    rows = []
    months = ["2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11"]
    for month in months:
        working_days = int(np.random.randint(20, 23))
        overtime_factor = round(np.random.uniform(1.0, 1.25), 2)
        rows.append({
            "month": month,
            "working_days": working_days,
            "overtime_factor": overtime_factor,
            "holiday_count": int(np.random.randint(0, 3)),
            "capacity_multiplier": round(np.random.uniform(0.9, 1.1), 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "production_calendar.csv"), index=False)
    return df


def generate_labor_rates():
    rows = []
    for cat in CATEGORIES:
        rows.append({
            "category": cat,
            "labor_rate_per_hour": round(np.random.uniform(25, 75), 2),
            "setup_hours": round(np.random.uniform(0.5, 4.0), 1),
            "run_hours_per_unit": round(np.random.uniform(0.1, 2.0), 2),
            "overhead_rate": round(np.random.uniform(1.2, 1.8), 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "labor_rates.csv"), index=False)
    return df


if __name__ == "__main__":
    assemblies = generate_assemblies()
    components = generate_components()
    bom = generate_bom_structure(assemblies, components)
    suppliers = generate_suppliers()
    inventory = generate_inventory(components)
    substitutions = generate_substitutions(components)
    pricing = generate_supplier_pricing(components, suppliers)
    calendar = generate_production_calendar()
    labor = generate_labor_rates()
    
    print(f"assemblies.csv: {len(assemblies)} rows")
    print(f"components.csv: {len(components)} rows")
    print(f"bom_structure.csv: {len(bom)} rows")
    print(f"suppliers.csv: {len(suppliers)} rows")
    print(f"inventory.csv: {len(inventory)} rows")
    print(f"substitutions.csv: {len(substitutions)} rows")
    print(f"supplier_pricing.csv: {len(pricing)} rows")
    print(f"production_calendar.csv: {len(calendar)} rows")
    print(f"labor_rates.csv: {len(labor)} rows")

import json
import math
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from pathlib import Path
from openpyxl import Workbook

np.random.seed(99)

TASK = Path(__file__).parent.parent / "task"
TASK.mkdir(exist_ok=True)

ANALYSIS_DATE = datetime(2024, 6, 1)
SUPPLIERS = [f"SUP{i:03d}" for i in range(1, 13)]
CATEGORIES = ["STRUCTURAL", "ELECTRICAL", "HYDRAULIC", "AVIONICS", "INTERIOR", "FASTENER"]
COMP_PREFIXES = ["CMP", "PRT", "MAT", "RAW"]
FIRST_BATCH_SUP = {1: "SUP006", 2: "SUP008", 3: "SUP006", 4: "SUP008", 5: "SUP003", 6: "SUP003"}
FASTENER_COVERED_IDS = set()


def generate_assemblies():
    priorities = ["CRITICAL", "HIGH", "STANDARD", "LOW",
                  "CRITICAL", "HIGH", "STANDARD", "LOW",
                  "CRITICAL", "HIGH", "STANDARD", "LOW"]
    due_dates = [
        "2024-07-15", "2024-08-01", "2024-09-15", "2024-10-01",
        "2024-10-15", "2024-11-01", "2024-11-15", "2024-07-01",
        "2024-09-01", "2024-09-30", "2024-10-30", "2024-12-01",
    ]
    rows = []
    for i in range(1, 13):
        rows.append({
            "assembly_id": f"ASM{i:04d}",
            "assembly_name": f"Assembly_ASM{i:04d}",
            "demand_quantity": int(np.random.choice([5, 10, 15, 20, 25])),
            "priority": priorities[i - 1],
            "due_date": due_dates[i - 1],
            "customer_id": f"CUST{((i - 1) % 5) + 1:03d}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(TASK / "assemblies.csv", index=False)
    return df


def generate_components():
    global FASTENER_COVERED_IDS
    rows = []
    comp_id = 1
    for cat in CATEGORIES:
        for j in range(11):
            prefix = COMP_PREFIXES[j % len(COMP_PREFIXES)]
            cid = f"{prefix}{comp_id:05d}"
            if cat == "FASTENER" and j >= 3:
                FASTENER_COVERED_IDS.add(cid)
            unit_cost = round(np.random.lognormal(3.5, 1.1), 2)
            lead_time = int(np.random.choice([7, 14, 21, 30, 45, 60]))
            if comp_id in FIRST_BATCH_SUP:
                supplier = FIRST_BATCH_SUP[comp_id]
            else:
                supplier = SUPPLIERS[comp_id % len(SUPPLIERS)]
            rows.append({
                "component_id": cid,
                "component_name": f"{cat}_{comp_id}",
                "category": cat,
                "unit": np.random.choice(["EA", "KG", "M", "L", "SET"]),
                "unit_cost": unit_cost,
                "default_supplier": supplier,
                "lead_time_days": lead_time,
                "min_order_qty": int(np.random.choice([1, 5, 10, 25, 50])),
                "is_critical": "YES" if comp_id % 7 == 0 else "NO",
            })
            comp_id += 1
    df = pd.DataFrame(rows)
    df.to_csv(TASK / "components.csv", index=False)
    return df


def generate_bom_structure(assemblies_df, components_df):
    comp_ids = components_df["component_id"].tolist()
    non_covered = [c for c in comp_ids if c not in FASTENER_COVERED_IDS]
    sub_assemblies = [f"SUB{i:04d}" for i in range(1, 9)]
    mod_assemblies = [f"MOD{i:04d}" for i in range(1, 5)]
    rows = []

    for _, asm in assemblies_df.iterrows():
        asm_id = asm["assembly_id"]
        n_direct = np.random.randint(3, 7)
        selected = np.random.choice(non_covered, size=n_direct, replace=False).tolist()
        for comp in selected:
            rows.append({"parent_id": asm_id, "child_id": comp,
                         "quantity_per": round(np.random.uniform(1.0, 4.0), 2),
                         "level": 1,
                         "scrap_rate": round(np.random.choice([0.0, 0.01, 0.02, 0.03, 0.05]), 2)})

    for sub_id in sub_assemblies:
        n = np.random.randint(3, 6)
        selected = np.random.choice(non_covered, size=n, replace=False).tolist()
        for comp in selected:
            rows.append({"parent_id": sub_id, "child_id": comp,
                         "quantity_per": round(np.random.uniform(1.0, 4.0), 2),
                         "level": 2,
                         "scrap_rate": round(np.random.choice([0.0, 0.01, 0.02]), 2)})

    for mod_id in mod_assemblies:
        n = np.random.randint(2, 4)
        selected = np.random.choice(non_covered, size=n, replace=False).tolist()
        for comp in selected:
            rows.append({"parent_id": mod_id, "child_id": comp,
                         "quantity_per": round(np.random.uniform(1.0, 3.0), 2),
                         "level": 3,
                         "scrap_rate": round(np.random.choice([0.0, 0.01]), 2)})

    for i, sub_id in enumerate(sub_assemblies[:4]):
        mod_id = mod_assemblies[i]
        rows.append({"parent_id": sub_id, "child_id": mod_id,
                     "quantity_per": round(np.random.uniform(1.0, 2.0), 2),
                     "level": 2, "scrap_rate": 0.01})

    for _, asm in assemblies_df.iterrows():
        n_subs = np.random.randint(1, 4)
        selected_subs = np.random.choice(sub_assemblies, size=n_subs, replace=False).tolist()
        for sub in selected_subs:
            rows.append({"parent_id": asm["assembly_id"], "child_id": sub,
                         "quantity_per": round(np.random.uniform(1.0, 2.5), 2),
                         "level": 1, "scrap_rate": round(np.random.choice([0.0, 0.01, 0.02]), 2)})

    covered_list = sorted(FASTENER_COVERED_IDS)
    for fc in covered_list[:4]:
        rows.append({"parent_id": "ASM0001", "child_id": fc,
                     "quantity_per": 8.0, "level": 1, "scrap_rate": 0.0})
    for fc in covered_list[4:]:
        rows.append({"parent_id": "ASM0002", "child_id": fc,
                     "quantity_per": 6.0, "level": 1, "scrap_rate": 0.0})

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["parent_id", "child_id"])
    df.to_csv(TASK / "bom_structure.csv", index=False)
    return df


def generate_suppliers():
    lead_times = [14, 21, 7, 30, 21, 14, 30, 7, 21, 14, 30, 21]
    rows = []
    for i, sup_id in enumerate(SUPPLIERS):
        rows.append({
            "supplier_id": sup_id,
            "supplier_name": f"Supplier_{sup_id}",
            "country": np.random.choice(["USA", "GERMANY", "JAPAN", "FRANCE", "UK", "CHINA"]),
            "base_lead_time": lead_times[i],
            "reliability_score": round(np.random.uniform(0.78, 0.99), 2),
            "payment_terms": int(np.random.choice([15, 30, 45, 60])),
            "min_order_value": round(np.random.choice([100, 250, 500, 1000, 2500]), 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TASK / "suppliers.csv", index=False)
    return df


def generate_inventory(components_df, substitutions_df):
    sub_avail = {}
    for _, row in substitutions_df.iterrows():
        if row["primary_component"] in FASTENER_COVERED_IDS:
            sc = row["substitute_component"]
            sub_avail[sc] = sub_avail.get(sc, 0) + 9999
    rows = []
    for _, comp in components_df.iterrows():
        cid = comp["component_id"]
        if cid in FASTENER_COVERED_IDS:
            on_hand, allocated, on_order = 0, 0, 2
            expected = "2024-07-15"
        elif cid in sub_avail:
            on_hand = sub_avail[cid]
            allocated, on_order = 0, 0
            expected = ""
        else:
            on_hand = int(np.random.choice([0, 0, 0, 5, 10, 25, 50, 100, 200, 500]))
            allocated = int(min(on_hand, np.random.randint(0, max(1, on_hand // 2))))
            on_order = int(np.random.choice([0, 0, 0, 25, 50, 100]))
            expected = ""
            if on_order > 0:
                expected = (ANALYSIS_DATE + timedelta(days=int(np.random.randint(7, 60)))).strftime("%Y-%m-%d")
        rows.append({
            "component_id": cid,
            "on_hand_qty": on_hand,
            "allocated_qty": allocated,
            "on_order_qty": on_order,
            "expected_receipt_date": expected,
        })
    df = pd.DataFrame(rows)
    df.to_csv(TASK / "inventory.csv", index=False)
    return df


def generate_substitutions(components_df):
    comp_ids = components_df["component_id"].tolist()
    comp_cats = dict(zip(components_df["component_id"], components_df["category"]))
    rows = []
    fastener_noncovered = [c for c in comp_ids if comp_cats[c] == "FASTENER" and c not in FASTENER_COVERED_IDS]
    for cid in sorted(FASTENER_COVERED_IDS):
        candidates = [c for c in fastener_noncovered if c != cid]
        if candidates:
            rows.append({
                "primary_component": cid,
                "substitute_component": candidates[0],
                "conversion_factor": 1.0,
                "priority_rank": 1,
                "effective_from": "2024-01-01",
                "effective_to": "2024-12-31",
                "approval_status": "APPROVED",
            })
    non_covered = [c for c in comp_ids if c not in FASTENER_COVERED_IDS]
    for cid in non_covered:
        if np.random.random() < 0.28:
            cat = comp_cats[cid]
            same_cat = [c for c in non_covered if comp_cats[c] == cat and c != cid]
            if same_cat:
                sub = np.random.choice(same_cat)
                rows.append({
                    "primary_component": cid,
                    "substitute_component": sub,
                    "conversion_factor": round(np.random.choice([0.9, 1.0, 1.1, 1.25]), 2),
                    "priority_rank": int(np.random.randint(1, 4)),
                    "effective_from": "2024-01-01",
                    "effective_to": "2024-12-31",
                    "approval_status": np.random.choice(["APPROVED", "APPROVED", "PENDING"]),
                })
    df = pd.DataFrame(rows)
    df.to_csv(TASK / "substitutions.csv", index=False)
    return df


def generate_supplier_pricing(components_df):
    rows = []
    for _, comp in components_df.iterrows():
        if comp["component_id"] in FASTENER_COVERED_IDS:
            continue
        sup = comp["default_supplier"]
        base = comp["unit_cost"]
        rows.append({"component_id": comp["component_id"], "supplier_id": sup,
                     "min_qty": 1, "max_qty": 49, "unit_price": round(base * 1.00, 4)})
        rows.append({"component_id": comp["component_id"], "supplier_id": sup,
                     "min_qty": 50, "max_qty": 199, "unit_price": round(base * 0.92, 4)})
        rows.append({"component_id": comp["component_id"], "supplier_id": sup,
                     "min_qty": 200, "max_qty": 999999, "unit_price": round(base * 0.85, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(TASK / "supplier_pricing.csv", index=False)
    return df


def generate_labor_rates():
    rows = []
    for cat in CATEGORIES:
        rows.append({
            "category": cat,
            "labor_rate_per_hour": round(np.random.uniform(25, 75), 2),
            "run_hours_per_unit": round(np.random.uniform(0.1, 2.0), 2),
            "overhead_rate": round(np.random.uniform(1.2, 1.8), 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TASK / "labor_rates.csv", index=False)
    return df


def generate_expedite_fees():
    max_fees = [250.0, 180.0, 320.0, 150.0, 400.0, 220.0, 175.0, 350.0, 200.0, 280.0, 140.0, 300.0]
    rows = []
    for i, sup_id in enumerate(SUPPLIERS):
        rows.append({
            "supplier_id": sup_id,
            "expedite_pct": 0.03,
            "max_expedite_fee": max_fees[i],
        })
    df = pd.DataFrame(rows)
    df.to_csv(TASK / "expedite_fees.csv", index=False)
    return df


def generate_supplier_contracts():
    contracts = {
        "SUP001": {
            "STRUCTURAL": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.08}],
            "ELECTRICAL": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.05}],
            # T6 trap: not yet active as of ANALYSIS_DATE 2024-06-01 → PRT00024 must get 0.0, not 0.06
            "HYDRAULIC":  [{"effective_from": "2024-07-01", "effective_to": "2024-12-31", "discount_rate": 0.06}],
        },
        "SUP002": {
            "HYDRAULIC": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.10}],
            "AVIONICS": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.07}],
        },
        "SUP003": {
            "ELECTRICAL": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.07}],
            "STRUCTURAL": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.06}],
        },
        "SUP004": {
            "INTERIOR": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.09}],
            "FASTENER": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.04}],
        },
        "SUP005": {
            "STRUCTURAL": [{"effective_from": "2024-01-01", "effective_to": "2024-05-31", "discount_rate": 0.11}],
            "AVIONICS":   [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.09}],
            # T6 trap: expired before ANALYSIS_DATE 2024-06-01 → CMP00016 must get 0.0, not 0.07
            "ELECTRICAL": [{"effective_from": "2024-01-01", "effective_to": "2024-05-31", "discount_rate": 0.07}],
        },
        "SUP006": {
            "HYDRAULIC": [{"effective_from": "2024-07-01", "effective_to": "2024-12-31", "discount_rate": 0.08}],
            "ELECTRICAL": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.06}],
        },
        "SUP007": {
            "FASTENER": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.05}],
            "INTERIOR": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.08}],
        },
        "SUP008": {
            "AVIONICS": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.12}],
            "STRUCTURAL": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.07}],
        },
        "SUP009": {
            "INTERIOR": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.06}],
            "HYDRAULIC": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.09}],
        },
        "SUP010": {
            "ELECTRICAL": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.08}],
            "FASTENER": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.05}],
        },
        "SUP011": {
            "STRUCTURAL": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.06}],
        },
        "SUP012": {
            "HYDRAULIC": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.07}],
            "AVIONICS": [{"effective_from": "2024-01-01", "effective_to": "2024-12-31", "discount_rate": 0.10}],
        },
    }
    with open(TASK / "supplier_contracts.json", "w") as fh:
        json.dump(contracts, fh, indent=2)


def generate_production_schedule():
    wb = Workbook()
    ws = wb.active
    ws.title = "operational_schedule"
    ws.append(["month", "priority_scope", "capacity_multiplier"])
    sched = [
        ("2024-07", "CRITICAL", 0.85),
        ("2024-08", "ALL", 0.95),
        ("2024-09", "CRITICAL", 0.80),
        ("2024-09", "STANDARD", 1.15),
        ("2024-10", "ALL", 1.05),
        ("2024-11", "ALL", 0.90),
        ("2024-12", "ALL", 1.00),
    ]
    for r in sched:
        ws.append(list(r))
    ws2 = wb.create_sheet("historical_2023")
    ws2.append(["month", "priority_scope", "capacity_multiplier"])
    hist = [
        ("2023-07", "ALL", 1.10), ("2023-08", "ALL", 0.88),
        ("2023-09", "ALL", 0.92), ("2023-10", "ALL", 1.02),
    ]
    for r in hist:
        ws2.append(list(r))
    wb.save(TASK / "production_schedule.xlsx")


DELIVERY_SEQUENCES = {
    "SUP001": ["ON_TIME", "ON_TIME", "LATE", "ON_TIME", "LATE"],
    "SUP002": ["ON_TIME", "LATE", "LATE", "ON_TIME"],
    "SUP003": ["LATE", "ON_TIME", "LATE", "ON_TIME"],
    "SUP004": ["ON_TIME", "LATE", "ON_TIME", "ON_TIME"],
    "SUP005": ["ON_TIME", "LATE", "LATE", "LATE", "ON_TIME"],
    "SUP006": ["LATE", "LATE", "ON_TIME", "LATE"],
    "SUP007": ["ON_TIME", "LATE", "LATE", "LATE"],
    "SUP008": ["LATE", "ON_TIME", "LATE", "LATE"],
    "SUP009": ["LATE", "ON_TIME", "ON_TIME", "ON_TIME"],
    "SUP010": ["ON_TIME", "LATE", "LATE", "LATE", "LATE"],
    "SUP011": ["LATE", "LATE", "ON_TIME", "LATE", "ON_TIME"],
    "SUP012": ["ON_TIME", "ON_TIME", "LATE", "ON_TIME", "ON_TIME"],
}


def generate_delivery_performance():
    rows = []
    base_date = datetime(2023, 1, 1)
    for sup_id, seq in DELIVERY_SEQUENCES.items():
        scheduled = base_date
        for status in seq:
            offset = 0 if status == "ON_TIME" else int(np.random.randint(3, 14))
            rows.append({
                "supplier_id": sup_id,
                "scheduled_date": scheduled.strftime("%Y-%m-%d"),
                "actual_delivery_date": (scheduled + timedelta(days=offset)).strftime("%Y-%m-%d"),
                "status": status,
            })
            scheduled += timedelta(days=int(np.random.randint(45, 90)))
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, TASK / "delivery_performance.parquet")
    return df


if __name__ == "__main__":
    assemblies = generate_assemblies()
    components = generate_components()
    substitutions_pre = generate_substitutions(components)
    bom = generate_bom_structure(assemblies, components)
    suppliers = generate_suppliers()
    inventory = generate_inventory(components, substitutions_pre)
    pricing = generate_supplier_pricing(components)
    labor = generate_labor_rates()
    expedite = generate_expedite_fees()
    generate_supplier_contracts()
    generate_production_schedule()
    delivery = generate_delivery_performance()

    print(f"assemblies.csv: {len(assemblies)} rows")
    print(f"components.csv: {len(components)} rows")
    print(f"bom_structure.csv: {len(bom)} rows")
    print(f"suppliers.csv: {len(suppliers)} rows")
    print(f"inventory.csv: {len(inventory)} rows")
    print(f"substitutions.csv: {len(substitutions_pre)} rows")
    print(f"supplier_pricing.csv: {len(pricing)} rows")
    print(f"labor_rates.csv: {len(labor)} rows")
    print(f"expedite_fees.csv: {len(expedite)} rows")
    print(f"supplier_contracts.json: written")
    print(f"production_schedule.xlsx: written (2 sheets)")
    print(f"delivery_performance.parquet: {len(delivery)} rows")
    print(f"FASTENER covered IDs (no stock, subs exist): {sorted(FASTENER_COVERED_IDS)}")

                      
\
\
\
\
\
\
\
\
\
\
\
\
\
\
import sys, math
import pandas as pd
import numpy as np
from pathlib import Path

SOL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
GT  = Path(__file__).parent

FASTENER_COVERED = {
    "RAW00059","CMP00060","PRT00061","MAT00062",
    "RAW00063","CMP00064","PRT00065","MAT00066",
}

errors = []

def err(code, msg):
    errors.append(f"[{code}] {msg}")

def near(a, b, tol=0.05):
    \
    if b == 0:
        return abs(a) <= tol
    return abs(a - b) / max(abs(b), 1e-9) <= tol

def load(path, name):
    try:
        return pd.read_csv(path)
    except Exception as e:
        err("FILE", f"{name} could not be loaded: {e}")
        return pd.DataFrame()

                                                                              

sc  = load(SOL / "supplier_scorecard.csv",      "supplier_scorecard.csv")               
er  = load(SOL / "exploded_requirements.csv",   "exploded_requirements.csv")             
sh  = load(SOL / "material_shortages.csv",      "material_shortages.csv")                
po  = load(SOL / "purchase_orders.csv",         "purchase_orders.csv")                   
cr  = load(SOL / "cost_rollup.csv",             "cost_rollup.csv")                       

             
g_sc = pd.read_csv(GT / "supplier_scorecard.csv")
g_er = pd.read_csv(GT / "exploded_requirements.csv")
g_sh = pd.read_csv(GT / "material_shortages.csv")
g_po = pd.read_csv(GT / "purchase_orders.csv")
g_cr = pd.read_csv(GT / "cost_rollup.csv")
task = Path(__file__).parent.parent / "task"
exp  = pd.read_csv(task / "expedite_fees.csv").set_index("supplier_id")


                                                                              

def get_df_val(df, filter_col, filter_val, target_col, default=None):
    rows = df[df[filter_col] == filter_val]
    if rows.empty or target_col not in rows.columns:
        return default
    return rows.iloc[0][target_col]

def get_df_val2(df, c1, v1, c2, v2, target_col, default=None):
    rows = df[(df[c1] == v1) & (df[c2] == v2)]
    if rows.empty or target_col not in rows.columns:
        return default
    return rows.iloc[0][target_col]


                               

req_cols = {
    "supplier_scorecard.csv":    ["supplier_id","total_deliveries","on_time_deliveries",
                                   "reliability_score","consecutive_late_count","average_delay_days"],
    "exploded_requirements.csv": ["assembly_id","component_id","demand_qty","gross_qty_required",
                                   "net_qty_required","unit_price","contract_discount","discounted_unit_price"],
    "material_shortages.csv":    ["assembly_id","component_id","net_qty_required",
                                   "available_substitute_qty","shortage_covered","shortage_severity"],
    "purchase_orders.csv":       ["assembly_id","component_id","supplier_id","qty_ordered",
                                   "unit_price","po_value","expedite_fee","base_lead_time",
                                   "adjusted_lead_time","consecutive_late_count","reliability_adjusted"],
    "cost_rollup.csv":           ["assembly_id","demand_qty","capacity_multiplier",
                                   "material_cost","discounted_material_cost",
                                   "effective_material_cost","labor_cost","total_cost"],
}
for fname, cols in req_cols.items():
    df = {"supplier_scorecard.csv":sc,"exploded_requirements.csv":er,
          "material_shortages.csv":sh,"purchase_orders.csv":po,"cost_rollup.csv":cr}[fname]
    for col in cols:
        if df.empty:
            break
        if col not in df.columns:
            err("SCHEMA", f"{fname} missing column '{col}'")


                                   

def chk_rows(df, name, expected, tolerance=20):
    n = len(df)
    if abs(n - expected) > tolerance:
        err("ROWS", f"{name} has {n} rows; expected ~{expected}")

if not sc.empty:  chk_rows(sc, "supplier_scorecard", 12, 0)
if not er.empty:  chk_rows(er, "exploded_requirements", 165, 30)
if not sh.empty:  chk_rows(sh, "material_shortages", 126, 30)
if not po.empty:  chk_rows(po, "purchase_orders", 109, 30)
if not cr.empty:  chk_rows(cr, "cost_rollup", 12, 0)


                                                

if not sc.empty and "supplier_id" in sc.columns and "consecutive_late_count" in sc.columns:
    sc_idx = sc.set_index("supplier_id")

                                                                         
    T4_EXPECTED = {
        "SUP001": 1, "SUP002": 0, "SUP003": 0, "SUP004": 0,
        "SUP005": 0, "SUP006": 1, "SUP007": 3, "SUP008": 2,
        "SUP009": 0, "SUP010": 4, "SUP011": 0, "SUP012": 0,
    }
    for sup, expected_consec in T4_EXPECTED.items():
        if sup in sc_idx.index:
            got = int(sc_idx.loc[sup, "consecutive_late_count"])
            if got != expected_consec:
                err("T4", f"{sup} consecutive_late_count={got}, expected {expected_consec}")
        else:
            err("T4", f"{sup} missing from supplier_scorecard")

                             
    if "reliability_score" in sc.columns:
        bad = sc[(sc["reliability_score"] < 0) | (sc["reliability_score"] > 1)]
        if len(bad):
            err("SC", f"{len(bad)} rows have reliability_score outside [0,1]")

                                      
    REL = {"SUP010": 0.20, "SUP012": 0.80, "SUP001": 0.60}
    for sup, expected_rel in REL.items():
        if sup in sc_idx.index:
            got = float(sc_idx.loc[sup, "reliability_score"])
            if not near(got, expected_rel, 0.01):
                err("SC", f"{sup} reliability_score={got:.4f}, expected {expected_rel}")

                                  
    TOTALS = {"SUP001": 5, "SUP010": 5, "SUP012": 5, "SUP002": 4}
    for sup, expected_total in TOTALS.items():
        if sup in sc_idx.index:
            got = int(sc_idx.loc[sup, "total_deliveries"])
            if got != expected_total:
                err("SC", f"{sup} total_deliveries={got}, expected {expected_total}")

                                                               
    for sup in ["SUP001","SUP007","SUP008","SUP010"]:
        if sup in sc_idx.index and "average_delay_days" in sc.columns:
            v = float(sc_idx.loc[sup, "average_delay_days"])
            if v <= 0:
                err("SC", f"{sup} average_delay_days={v}, expected > 0")


                                                       

if not er.empty and "component_id" in er.columns and "contract_discount" in er.columns:
    er_disc = er.drop_duplicates("component_id").set_index("component_id")

                                                         
    T2_CHECKS = ["CMP00001", "MAT00011", "RAW00030", "CMP00045", "MAT00047"]
    for cid in T2_CHECKS:
        if cid in er_disc.index:
            got = float(er_disc.loc[cid, "contract_discount"])
            if abs(got) > 1e-9:
                err("T2", f"{cid} contract_discount={got:.4f}, expected 0.0 (missing supplier+category key)")

                                                                             
    if "CMP00016" in er_disc.index:
        got = float(er_disc.loc["CMP00016", "contract_discount"])
        if abs(got) > 1e-9:
            err("T6", f"CMP00016 contract_discount={got:.4f}, expected 0.0 (contract expired 2024-05-31)")
    else:
        err("T6", "CMP00016 missing from exploded_requirements (needed for T6 expired-contract check)")

                                                                                  
    if "PRT00024" in er_disc.index:
        got = float(er_disc.loc["PRT00024", "contract_discount"])
        if abs(got) > 1e-9:
            err("T6", f"PRT00024 contract_discount={got:.4f}, expected 0.0 (contract not yet active 2024-07-01)")
    else:
        err("T6", "PRT00024 missing from exploded_requirements (needed for T6 future-contract check)")

                                          
    ACTIVE_DISC = {
        "MAT00007": 0.07,                                        
        "PRT00006": 0.06,                              
        "PRT00032": 0.09,                             
        "PRT00043": 0.12,                            
        "MAT00040": 0.09,                                                                            
    }
    for cid, expected_disc in ACTIVE_DISC.items():
        if cid in er_disc.index:
            got = float(er_disc.loc[cid, "contract_discount"])
            if not near(got, expected_disc, 0.001):
                err("T6", f"{cid} contract_discount={got:.4f}, expected {expected_disc} (active contract)")

                                                                           
    if "unit_price" in er.columns and "discounted_unit_price" in er.columns:
        sample = er.sample(min(20, len(er)), random_state=7)
        for _, row in sample.iterrows():
            up   = float(row["unit_price"])
            disc = float(row["contract_discount"])
            dup  = float(row["discounted_unit_price"])
            expected = up * (1 - disc)
            if not near(dup, expected, 0.02):
                err("T3", f"{row['component_id']}: discounted_unit_price={dup:.4f}, "
                          f"expected unit_price*(1-discount)={expected:.4f}")

                               
    if "net_qty_required" in er.columns:
        neg = er[er["net_qty_required"] < -1e-6]
        if len(neg):
            err("ER", f"{len(neg)} rows with negative net_qty_required")

                                      
    if "contract_discount" in er.columns:
        bad = er[(er["contract_discount"] < 0) | (er["contract_discount"] > 0.5)]
        if len(bad):
            err("ER", f"{len(bad)} rows with contract_discount outside [0, 0.5]")


                                                

if not sh.empty and "component_id" in sh.columns and "shortage_covered" in sh.columns:

                                                                     
    sh_idx = sh.set_index(["assembly_id","component_id"]) if "assembly_id" in sh.columns else None
    sh_comp_covered = sh.groupby("component_id")["shortage_covered"].first()

    for fc in FASTENER_COVERED:
        if fc in sh_comp_covered.index:
            if sh_comp_covered[fc] != "YES":
                err("T1", f"{fc} (covered FASTENER) has shortage_covered={sh_comp_covered[fc]}, expected YES")
        else:
                                                                         
            err("T1", f"{fc} (covered FASTENER, on_hand=0) absent from material_shortages")

                                               
    valid_covered = sh["shortage_covered"].isin(["YES","NO"])
    if not valid_covered.all():
        err("SH", f"{(~valid_covered).sum()} rows have invalid shortage_covered value")

                                                           
    if "shortage_severity" in sh.columns:
        valid_sev = sh["shortage_severity"].isin(["CRITICAL","STANDARD"])
        if not valid_sev.all():
            err("SH", f"{(~valid_sev).sum()} rows have invalid shortage_severity value")

                                                                                    
    if "net_qty_required" in sh.columns:
        non_pos = sh[sh["net_qty_required"] <= 0]
        if len(non_pos):
            err("SH", f"{len(non_pos)} rows with net_qty_required <= 0 in material_shortages")

                                    
    n_covered = sh["shortage_covered"].eq("YES").sum()
    if n_covered < 5:
        err("T1", f"Only {n_covered} shortages marked as covered; expected >= 8 (FASTENER substitutes)")

                                           
    if "available_substitute_qty" in sh.columns:
        neg = sh[sh["available_substitute_qty"] < 0]
        if len(neg):
            err("SH", f"{len(neg)} rows with negative available_substitute_qty")


                                                         

if not po.empty and "component_id" in po.columns:

                                                    
    if "component_id" in po.columns:
        fc_in_po = set(po["component_id"]) & FASTENER_COVERED
        if fc_in_po:
            err("T1", f"Covered FASTENER components found in purchase_orders (should be substituted): {fc_in_po}")

                          
    po_sup = po.set_index(["assembly_id","component_id"]) if "assembly_id" in po.columns else po

                                                               
    if all(c in po.columns for c in ["supplier_id","po_value","expedite_fee"]):
        for _, row in po.iterrows():
            sup = row["supplier_id"]
            pv  = float(row["po_value"])
            fee = float(row["expedite_fee"])
            if sup in exp.index:
                pct     = float(exp.loc[sup, "expedite_pct"])
                max_fee = float(exp.loc[sup, "max_expedite_fee"])
                expected = min(pv * pct, max_fee)
                if not near(fee, expected, 0.01):
                    err("T7", f"{row['component_id']} expedite_fee={fee:.4f}, expected min({pv}*{pct},{max_fee})={expected:.4f}")

                                                                          
        n_capped = sum(
            1 for _, row in po.iterrows()
            if row["supplier_id"] in exp.index and
               float(row["po_value"]) * float(exp.loc[row["supplier_id"], "expedite_pct"]) > float(exp.loc[row["supplier_id"], "max_expedite_fee"])
        )
        if n_capped == 0:
            err("T7", "No capped expedite fees found; expected at least 1 (e.g. RAW00048/SUP001)")

                                                                                        
    if all(c in po.columns for c in ["base_lead_time","consecutive_late_count","adjusted_lead_time"]):
        for _, row in po.iterrows():
            base    = float(row["base_lead_time"])
            consec  = int(row["consecutive_late_count"])
            adj     = int(row["adjusted_lead_time"])
            expected = math.ceil(base * (1.0 + 0.1 * consec))
            if adj != expected:
                err("T4", f"{row['component_id']} adjusted_lead_time={adj}, expected ceil({base}*(1+0.1*{consec}))={expected}")

                                                                    
    if all(c in po.columns for c in ["consecutive_late_count","reliability_adjusted"]):
        mismatch = po[
            (po["consecutive_late_count"] > 0) & (po["reliability_adjusted"] != "YES") |
            (po["consecutive_late_count"] == 0) & (po["reliability_adjusted"] != "NO")
        ]
        if len(mismatch):
            err("T4", f"{len(mismatch)} rows with inconsistent reliability_adjusted vs consecutive_late_count")

                                             
    if all(c in po.columns for c in ["qty_ordered","unit_price","po_value"]):
        for _, row in po.iterrows():
            expected_pv = float(row["qty_ordered"]) * float(row["unit_price"])
            if not near(float(row["po_value"]), expected_pv, 0.01):
                err("T3", f"{row['component_id']} po_value={row['po_value']:.4f}, expected qty*price={expected_pv:.4f}")

                                                                    
    if "unit_price" in po.columns:
        bad = po[po["unit_price"] <= 0]
        if len(bad):
            err("T3", f"{len(bad)} PO rows with unit_price <= 0")

                                                   
    if not sh.empty and "shortage_covered" in sh.columns and "assembly_id" in sh.columns:
        covered_pairs = set(
            zip(sh[sh["shortage_covered"]=="YES"]["assembly_id"],
                sh[sh["shortage_covered"]=="YES"]["component_id"])
        )
        if "assembly_id" in po.columns:
            po_pairs = set(zip(po["assembly_id"], po["component_id"]))
            bad_pairs = po_pairs & covered_pairs
            if bad_pairs:
                err("T1", f"POs exist for shortage-covered pairs: {list(bad_pairs)[:5]}")


                                         

if not cr.empty and "assembly_id" in cr.columns and "capacity_multiplier" in cr.columns:
    cr_idx = cr.set_index("assembly_id")

                                                                         
    T5_CAP = {
        "ASM0001": 0.85,                                            
        "ASM0002": 0.95,                                                
        "ASM0003": 1.15,                                            
        "ASM0009": 0.80,                                            
        "ASM0010": 1.00,                                                           
        "ASM0004": 1.05,                                 
        "ASM0008": 1.00,                                                                    
        "ASM0006": 0.90,                                 
    }
    for asm_id, expected_cap in T5_CAP.items():
        if asm_id in cr_idx.index:
            got = float(cr_idx.loc[asm_id, "capacity_multiplier"])
            if not near(got, expected_cap, 0.001):
                err("T5", f"{asm_id} capacity_multiplier={got:.4f}, expected {expected_cap}")
        else:
            err("T5", f"{asm_id} missing from cost_rollup (needed for T5 check)")

                                                                              
    if all(c in cr.columns for c in ["discounted_material_cost","capacity_multiplier","effective_material_cost"]):
        for _, row in cr.iterrows():
            expected_eff = float(row["discounted_material_cost"]) * float(row["capacity_multiplier"])
            if not near(float(row["effective_material_cost"]), expected_eff, 0.02):
                err("T5", f"{row['assembly_id']} effective_material_cost={row['effective_material_cost']:.4f}, "
                          f"expected disc_mat*cap={expected_eff:.4f}")

                                                       
    if all(c in cr.columns for c in ["effective_material_cost","labor_cost","total_cost"]):
        for _, row in cr.iterrows():
            expected = float(row["effective_material_cost"]) + float(row["labor_cost"])
            if not near(float(row["total_cost"]), expected, 0.02):
                err("T5", f"{row['assembly_id']} total_cost={row['total_cost']:.4f}, "
                          f"expected eff_mat+labor={expected:.4f}")

                                                                        
    if all(c in cr.columns for c in ["material_cost","discounted_material_cost"]):
        tot_mat  = cr["material_cost"].sum()
        tot_disc = cr["discounted_material_cost"].sum()
        if tot_disc >= tot_mat:
            err("ER", f"Sum discounted_material_cost ({tot_disc:.2f}) >= material_cost ({tot_mat:.2f}); "
                      f"some discounts must be applied")

                        
    for col in ["material_cost","labor_cost","total_cost"]:
        if col in cr.columns:
            nonpos = cr[cr[col] <= 0]
            if len(nonpos):
                err("CR", f"{len(nonpos)} rows with {col} <= 0")

                                                       
    GOLDEN_TOTALS = {
        "ASM0001": 36375.86,
        "ASM0002": 139724.96,
        "ASM0003": 90677.90,
        "ASM0009": 86287.42,
    }
    for asm_id, expected_total in GOLDEN_TOTALS.items():
        if asm_id in cr_idx.index:
            got = float(cr_idx.loc[asm_id, "total_cost"])
            if not near(got, expected_total, 0.03):                
                err("CR", f"{asm_id} total_cost={got:.2f}, expected ~{expected_total:.2f}")


                                                    

if not er.empty and not sh.empty:
                                                                    
    if "assembly_id" in sh.columns and "assembly_id" in er.columns:
        sh_asms = set(sh["assembly_id"])
        er_asms = set(er["assembly_id"])
        orphan = sh_asms - er_asms
        if orphan:
            err("T3", f"Assemblies in shortages but not in exploded_requirements: {orphan}")

if not er.empty and not po.empty:
                                                                   
    if "component_id" in er.columns and "component_id" in po.columns:
        po_comps = set(po["component_id"])
        er_comps = set(er["component_id"])
        orphan = po_comps - er_comps
        if orphan:
            err("T3", f"Components in purchase_orders absent from exploded_requirements: {orphan}")

                                                                                                      
                                                                                             
                                                                          
    if all(c in er.columns for c in ["component_id","contract_discount"]) and\
       all(c in po.columns for c in ["component_id","consecutive_late_count"]):
        er_disc_map = er.drop_duplicates("component_id").set_index("component_id")["contract_discount"]
                                                                                                
        er_per_component = er.groupby("component_id")["contract_discount"].nunique()
        multi_disc = er_per_component[er_per_component > 1]
        if len(multi_disc):
            err("T3", f"Components with inconsistent contract_discount across assemblies: {list(multi_disc.index)[:5]}")

                     

total = 120                                                                
n_errors = len(errors)

print(f"\n{'='*60}")
print(f"AerospaceBOMExplosion V2 — Verifier Results")
print(f"{'='*60}")
if errors:
    for e in errors:
        print(f"  FAIL  {e}")
else:
    print("  All checks passed.")
print(f"{'='*60}")
print(f"Reasoning errors: {n_errors}")
print(f"Result: {'PASS' if n_errors == 0 else 'FAIL'}")
sys.exit(0 if n_errors == 0 else 1)

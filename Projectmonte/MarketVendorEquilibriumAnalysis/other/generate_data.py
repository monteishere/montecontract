import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "task")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VENDORS = [f"V{str(i).zfill(2)}" for i in range(1, 21)]
SEGMENTS = ["ELECTRONICS", "APPAREL", "HOME_GOODS", "GROCERY"]
REGIONS = ["NORTH", "SOUTH", "WEST"]
MONTHS = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
TIERS = ["PLATINUM", "GOLD", "SILVER", "BRONZE"]
COMPLAINT_TYPES = ["QUALITY", "SHIPPING", "DESCRIPTION", "SERVICE", "BILLING"]
PROMO_TYPES = ["FLASH_SALE", "BUNDLE", "LOYALTY_DISCOUNT", "CLEARANCE", "SEASONAL"]
STATUSES = ["COMPLETED", "RETURNED", "CANCELLED", "DISPUTED"]


def generate_vendors():
    rows = []
    for i, vid in enumerate(VENDORS):
        segment = SEGMENTS[i % len(SEGMENTS)]
        region = REGIONS[i % len(REGIONS)]
        tier = TIERS[i % len(TIERS)]
        join_date = datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 300))
        rows.append({
            "vendor_id": vid,
            "vendor_name": f"Vendor_{vid}",
            "segment": segment,
            "region": region,
            "tier": tier,
            "join_date": join_date.strftime("%Y-%m-%d"),
            "min_order_value": round(np.random.choice([5.0, 10.0, 15.0, 25.0]), 2),
            "max_return_pct": round(np.random.uniform(0.10, 0.35), 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "vendors.csv"), index=False)
    return df


def generate_products(vendors_df):
    rows = []
    pid = 1
    for _, v in vendors_df.iterrows():
        n_products = np.random.randint(3, 8)
        for _ in range(n_products):
            price = round(np.random.lognormal(3.0, 0.8), 2)
            cost = round(price * np.random.uniform(0.3, 0.7), 2)
            weight = round(np.random.uniform(0.1, 15.0), 2)
            rows.append({
                "product_id": f"P{str(pid).zfill(4)}",
                "vendor_id": v["vendor_id"],
                "category": np.random.choice(["CAT_A", "CAT_B", "CAT_C", "CAT_D", "CAT_E"]),
                "base_price": price,
                "cost_price": cost,
                "weight_kg": weight,
                "launch_date": (datetime(2023, 6, 1) + timedelta(days=np.random.randint(0, 400))).strftime("%Y-%m-%d"),
                "is_fragile": np.random.choice(["YES", "NO"], p=[0.15, 0.85]),
                "warranty_months": int(np.random.choice([0, 6, 12, 24], p=[0.3, 0.3, 0.25, 0.15])),
            })
            pid += 1
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "products.csv"), index=False)
    return df


def generate_transactions(vendors_df, products_df):
    rows = []
    tid = 1
    for month in MONTHS:
        month_start = datetime.strptime(month, "%Y-%m")
        for _, v in vendors_df.iterrows():
            vendor_products = products_df[products_df["vendor_id"] == v["vendor_id"]]
            n_txns = np.random.randint(15, 45)
            for _ in range(n_txns):
                prod = vendor_products.sample(1).iloc[0]
                day = np.random.randint(1, 28)
                txn_date = month_start.replace(day=day)
                qty = int(np.random.randint(1, 10))
                unit_price = round(prod["base_price"] * np.random.uniform(0.85, 1.15), 2)
                status = np.random.choice(STATUSES, p=[0.78, 0.10, 0.07, 0.05])
                shipping = round(np.random.choice([0.0, 3.99, 5.99, 8.99, 12.99]), 2)
                discount_pct = 0.0
                promo_id = ""
                if np.random.random() < 0.25:
                    discount_pct = round(np.random.choice([0.05, 0.10, 0.15, 0.20, 0.25]), 2)
                    promo_id = f"PR{np.random.randint(1, 30):03d}"

                rating_val = ""
                if status == "COMPLETED" and np.random.random() < 0.70:
                    rating_val = int(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.08, 0.17, 0.35, 0.35]))
                elif status == "RETURNED" and np.random.random() < 0.40:
                    rating_val = int(np.random.choice([1, 2, 3], p=[0.50, 0.30, 0.20]))

                row = {
                    "transaction_id": f"T{str(tid).zfill(6)}",
                    "vendor_id": v["vendor_id"],
                    "product_id": prod["product_id"],
                    "transaction_date": txn_date.strftime("%Y-%m-%d"),
                    "quantity": qty,
                    "unit_price": unit_price,
                    "shipping_fee": shipping,
                    "discount_pct": discount_pct,
                    "status": status,
                    "customer_region": np.random.choice(REGIONS),
                    "payment_method": np.random.choice(["CREDIT", "DEBIT", "WALLET", "COD"]),
                    "promo_id": promo_id,
                    "rating": rating_val,
                }
                rows.append(row)
                tid += 1

    nullify_indices = np.random.choice(len(rows), size=int(len(rows) * 0.03), replace=False)
    for idx in nullify_indices:
        col = np.random.choice(["shipping_fee", "rating", "discount_pct"])
        rows[idx][col] = ""

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "transactions.csv"), index=False)
    return df


def generate_commission_rates():
    rows = []
    for tier in TIERS:
        for seg in SEGMENTS:
            base = {"PLATINUM": 0.06, "GOLD": 0.08, "SILVER": 0.10, "BRONZE": 0.12}[tier]
            seg_adj = {"ELECTRONICS": 0.02, "APPAREL": 0.01, "HOME_GOODS": 0.0, "GROCERY": -0.01}[seg]
            rate1 = round(base + seg_adj + np.random.uniform(-0.005, 0.005), 4)
            rate2 = round(rate1 + np.random.uniform(-0.01, 0.01), 4)
            rows.append({
                "tier": tier,
                "segment": seg,
                "commission_rate": rate1,
                "effective_from": "2024-01-01",
                "effective_to": "2024-03-31",
            })
            rows.append({
                "tier": tier,
                "segment": seg,
                "commission_rate": rate2,
                "effective_from": "2024-04-01",
                "effective_to": "2024-06-30",
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "commission_rates.csv"), index=False)
    return df


def generate_seasonal_factors():
    rows = []
    for seg in SEGMENTS:
        for month in MONTHS:
            factor = round(np.random.uniform(0.80, 1.25), 3)
            if seg == "APPAREL" and month in ("2024-03", "2024-04"):
                factor = round(factor * 1.1, 3)
            if seg == "ELECTRONICS" and month in ("2024-01",):
                factor = round(factor * 1.15, 3)
            rows.append({
                "segment": seg,
                "month": month,
                "demand_factor": factor,
                "return_adjustment": round(np.random.uniform(0.90, 1.10), 3),
                "shipping_weight": round(np.random.uniform(0.8, 1.2), 3),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "seasonal_factors.csv"), index=False)
    return df


def generate_complaints(transactions_df):
    rows = []
    cid = 1
    candidates = transactions_df[transactions_df["status"].isin(["COMPLETED", "RETURNED"])].sample(frac=0.12, random_state=42)
    for _, txn in candidates.iterrows():
        txn_date = datetime.strptime(txn["transaction_date"], "%Y-%m-%d")
        days_after = np.random.randint(1, 45)
        complaint_date = txn_date + timedelta(days=days_after)
        severity = np.random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"], p=[0.40, 0.30, 0.20, 0.10])
        resolved = np.random.choice(["YES", "NO", "PENDING"], p=[0.60, 0.15, 0.25])
        resolution_days = ""
        if resolved == "YES":
            resolution_days = int(np.random.randint(1, 30))
        refund_amount = ""
        if resolved == "YES" and np.random.random() < 0.40:
            refund_amount = round(txn["unit_price"] * txn["quantity"] * np.random.uniform(0.1, 1.0), 2)
        rows.append({
            "complaint_id": f"C{str(cid).zfill(5)}",
            "transaction_id": txn["transaction_id"],
            "vendor_id": txn["vendor_id"],
            "complaint_date": complaint_date.strftime("%Y-%m-%d"),
            "complaint_type": np.random.choice(COMPLAINT_TYPES),
            "severity": severity,
            "resolved": resolved,
            "resolution_days": resolution_days,
            "refund_amount": refund_amount,
        })
        cid += 1
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "complaints.csv"), index=False)
    return df


def generate_promotions():
    rows = []
    for i in range(1, 30):
        promo_type = np.random.choice(PROMO_TYPES)
        start_m = np.random.randint(0, 5)
        dur = np.random.randint(1, 3)
        start_date = datetime(2024, 1 + start_m, 1)
        end_date = datetime(2024, min(1 + start_m + dur, 6), 28)
        min_qty = int(np.random.choice([1, 2, 3, 5]))
        max_discount = round(np.random.uniform(0.05, 0.30), 2)
        budget = round(np.random.uniform(500, 5000), 2)
        vendor_ids = np.random.choice(VENDORS, size=np.random.randint(2, 8), replace=False)
        for vid in vendor_ids:
            rows.append({
                "promo_id": f"PR{i:03d}",
                "vendor_id": vid,
                "promo_type": promo_type,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "min_quantity": min_qty,
                "max_discount_pct": max_discount,
                "budget_cap": budget,
                "stackable": np.random.choice(["YES", "NO"], p=[0.3, 0.7]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "promotions.csv"), index=False)
    return df


def generate_market_segments():
    rows = []
    for seg in SEGMENTS:
        rows.append({
            "segment": seg,
            "min_vendor_count": np.random.choice([3, 4, 5]),
            "concentration_threshold": round(np.random.uniform(0.30, 0.50), 2),
            "entropy_floor": round(np.random.uniform(1.0, 2.0), 3),
            "vci_ceiling": round(np.random.uniform(2.0, 4.0), 3),
            "staleness_alpha": 0.40,
            "staleness_beta": 0.35,
            "staleness_gamma": 0.25,
            "health_decay_rate": round(np.random.uniform(0.90, 0.98), 3),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "market_segments.csv"), index=False)
    return df


if __name__ == "__main__":
    vendors = generate_vendors()
    products = generate_products(vendors)
    transactions = generate_transactions(vendors, products)
    commissions = generate_commission_rates()
    seasonal = generate_seasonal_factors()
    complaints = generate_complaints(transactions)
    promotions = generate_promotions()
    segments = generate_market_segments()

    print(f"vendors.csv: {len(vendors)} rows")
    print(f"products.csv: {len(products)} rows")
    print(f"transactions.csv: {len(transactions)} rows")
    print(f"commission_rates.csv: {len(commissions)} rows")
    print(f"seasonal_factors.csv: {len(seasonal)} rows")
    print(f"complaints.csv: {len(complaints)} rows")
    print(f"promotions.csv: {len(promotions)} rows")
    print(f"market_segments.csv: {len(segments)} rows")

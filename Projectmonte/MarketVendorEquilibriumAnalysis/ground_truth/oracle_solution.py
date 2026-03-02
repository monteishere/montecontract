import pandas as pd
import numpy as np
import math
from pathlib import Path


def load_data(task_dir):
    vendors = pd.read_csv(task_dir / "vendors.csv")
    products = pd.read_csv(task_dir / "products.csv")
    transactions = pd.read_csv(task_dir / "transactions.csv")
    commission_rates = pd.read_csv(task_dir / "commission_rates.csv")
    seasonal_factors = pd.read_csv(task_dir / "seasonal_factors.csv")
    complaints = pd.read_csv(task_dir / "complaints.csv")
    promotions = pd.read_csv(task_dir / "promotions.csv")
    market_segments = pd.read_csv(task_dir / "market_segments.csv")
    return vendors, products, transactions, commission_rates, seasonal_factors, complaints, promotions, market_segments


def clean_transactions(transactions):
    transactions = transactions.copy()
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])
    transactions["month"] = transactions["transaction_date"].dt.to_period("M").astype(str)
    transactions["shipping_fee"] = pd.to_numeric(transactions["shipping_fee"], errors="coerce").fillna(0.0)
    transactions["discount_pct"] = pd.to_numeric(transactions["discount_pct"], errors="coerce").fillna(0.0)
    transactions["rating"] = pd.to_numeric(transactions["rating"], errors="coerce")
    transactions["quantity"] = pd.to_numeric(transactions["quantity"], errors="coerce")
    transactions["unit_price"] = pd.to_numeric(transactions["unit_price"], errors="coerce")
    return transactions


def clean_complaints(complaints):
    complaints = complaints.copy()
    complaints["complaint_date"] = pd.to_datetime(complaints["complaint_date"])
    complaints["resolution_days"] = pd.to_numeric(complaints["resolution_days"], errors="coerce")
    complaints["refund_amount"] = pd.to_numeric(complaints["refund_amount"], errors="coerce").fillna(0.0)
    return complaints


def get_commission_rate(tier, segment, txn_date, commission_rates):
    commission_rates = commission_rates.copy()
    commission_rates["effective_from"] = pd.to_datetime(commission_rates["effective_from"])
    commission_rates["effective_to"] = pd.to_datetime(commission_rates["effective_to"])
    matching = commission_rates[
        (commission_rates["tier"] == tier) &
        (commission_rates["segment"] == segment) &
        (commission_rates["effective_from"] <= txn_date) &
        (commission_rates["effective_to"] >= txn_date)
    ]
    if len(matching) > 0:
        return matching.iloc[0]["commission_rate"]
    return 0.0


def compute_line_revenue(row):
    gross = row["quantity"] * row["unit_price"]
    discount = gross * row["discount_pct"]
    shipping = row["shipping_fee"]
    return round(gross - discount + shipping, 2)


def compute_vendor_monthly(transactions, vendors, commission_rates, seasonal_factors, complaints):
    months_list = sorted(transactions["month"].unique())
    vendor_ids = sorted(vendors["vendor_id"].unique())

    commission_rates_clean = commission_rates.copy()
    commission_rates_clean["effective_from"] = pd.to_datetime(commission_rates_clean["effective_from"])
    commission_rates_clean["effective_to"] = pd.to_datetime(commission_rates_clean["effective_to"])

    results = []
    prev_vsi = {}

    for month in months_list:
        for vid in vendor_ids:
            vendor_info = vendors[vendors["vendor_id"] == vid].iloc[0]
            segment = vendor_info["segment"]
            tier = vendor_info["tier"]
            max_return_pct = vendor_info["max_return_pct"]

            month_txns = transactions[
                (transactions["vendor_id"] == vid) &
                (transactions["month"] == month)
            ]

            revenue_txns = month_txns[month_txns["status"].isin(["COMPLETED", "RETURNED"])]

            gross_revenue = 0.0
            for _, txn in revenue_txns.iterrows():
                gross_revenue += compute_line_revenue(txn)
            gross_revenue = round(gross_revenue, 2)

            returned_txns = month_txns[month_txns["status"] == "RETURNED"]
            returned_revenue = 0.0
            for _, txn in returned_txns.iterrows():
                returned_revenue += compute_line_revenue(txn)
            returned_revenue = round(returned_revenue, 2)

            net_revenue = round(gross_revenue - returned_revenue, 2)

            month_start = pd.Timestamp(month + "-01")
            month_mid = month_start + pd.Timedelta(days=14)
            rate = get_commission_rate(tier, segment, month_mid, commission_rates_clean)
            commission_amount = round(net_revenue * rate, 2)

            sf_row = seasonal_factors[
                (seasonal_factors["segment"] == segment) &
                (seasonal_factors["month"] == month)
            ]
            demand_factor = 1.0
            shipping_weight = 1.0
            return_adjustment = 1.0
            if len(sf_row) > 0:
                demand_factor = sf_row.iloc[0]["demand_factor"]
                shipping_weight = sf_row.iloc[0]["shipping_weight"]
                return_adjustment = sf_row.iloc[0]["return_adjustment"]

            seasonal_adjusted_revenue = round(net_revenue * demand_factor, 2)

            total_txns = len(month_txns[month_txns["status"].isin(["COMPLETED", "RETURNED"])])
            returned_count = len(returned_txns)
            return_rate = round(returned_count / total_txns, 4) if total_txns > 0 else 0.0

            adjusted_return_rate = round(return_rate * return_adjustment, 4)

            rated_txns = month_txns[month_txns["rating"].notna()]
            avg_rating = round(rated_txns["rating"].mean(), 2) if len(rated_txns) > 0 else 0.0

            complaint_month = complaints[
                (complaints["vendor_id"] == vid) &
                (complaints["complaint_date"].dt.to_period("M").astype(str) == month)
            ]
            complaint_count = len(complaint_month)

            resolved_complaints = complaint_month[complaint_month["resolution_days"].notna()]
            avg_resolution_days = round(resolved_complaints["resolution_days"].mean(), 2) if len(resolved_complaints) > 0 else 0.0

            critical_complaints = len(complaint_month[complaint_month["severity"] == "CRITICAL"])

            weighted_shipping = 0.0
            for _, txn in revenue_txns.iterrows():
                weighted_shipping += txn["shipping_fee"] * shipping_weight
            weighted_shipping = round(weighted_shipping, 2)

            if vid not in prev_vsi:
                base_score = avg_rating * 20.0 if avg_rating > 0 else 50.0
                complaint_penalty = critical_complaints * 5.0 + (complaint_count - critical_complaints) * 1.5
                vsi = round(max(0.0, min(100.0, base_score - complaint_penalty)), 2)
            else:
                p_vsi = prev_vsi[vid]
                if adjusted_return_rate < max_return_pct:
                    raw = avg_rating * 20.0 if avg_rating > 0 else 50.0
                    complaint_penalty = critical_complaints * 5.0 + (complaint_count - critical_complaints) * 1.5
                    current_score = max(0.0, min(100.0, raw - complaint_penalty))
                    vsi = round(0.7 * p_vsi + 0.3 * current_score, 2)
                else:
                    decay = 0.85 if adjusted_return_rate < max_return_pct * 1.5 else 0.6
                    vsi = round(p_vsi * decay, 2)
                vsi = round(max(0.0, min(100.0, vsi)), 2)

            prev_vsi[vid] = vsi

            results.append({
                "vendor_id": vid,
                "month": month,
                "gross_revenue": gross_revenue,
                "net_revenue": net_revenue,
                "commission_amount": commission_amount,
                "seasonal_adjusted_revenue": seasonal_adjusted_revenue,
                "adjusted_return_rate": adjusted_return_rate,
                "avg_rating": avg_rating,
                "complaint_count": complaint_count,
                "critical_complaints": critical_complaints,
                "avg_resolution_days": avg_resolution_days,
                "weighted_shipping": weighted_shipping,
                "vsi": vsi,
            })

    return pd.DataFrame(results)


def compute_segment_equilibrium(vendor_monthly, market_segments, vendors):
    months_list = sorted(vendor_monthly["month"].unique())
    segments = sorted(market_segments["segment"].unique())

    vendor_segment_map = dict(zip(vendors["vendor_id"], vendors["segment"]))
    vm_with_seg = vendor_monthly.copy()
    vm_with_seg["segment"] = vm_with_seg["vendor_id"].map(vendor_segment_map)

    prev_shares = {}
    prev_health_streaks = {}
    results = []

    for month in months_list:
        for segment in segments:
            seg_config = market_segments[market_segments["segment"] == segment].iloc[0]
            entropy_floor = seg_config["entropy_floor"]
            vci_ceiling = seg_config["vci_ceiling"]
            alpha = seg_config["staleness_alpha"]
            beta = seg_config["staleness_beta"]
            gamma = seg_config["staleness_gamma"]
            concentration_threshold = seg_config["concentration_threshold"]

            seg_month = vm_with_seg[
                (vm_with_seg["month"] == month) &
                (vm_with_seg["segment"] == segment)
            ]

            segment_vendor_revenues = {}
            for _, row in seg_month.iterrows():
                segment_vendor_revenues[row["vendor_id"]] = row["net_revenue"]

            seg_total = sum(segment_vendor_revenues.values())
            if seg_total <= 0:
                seg_total = 1.0

            shares = {}
            for vid, rev in segment_vendor_revenues.items():
                shares[vid] = rev / seg_total

            vendor_count = len([v for v, s in shares.items() if s > 0])

            hhi = 0.0
            for vid, share in shares.items():
                hhi += share * share
            hhi = round(hhi, 6)

            entropy = 0.0
            for vid, share in shares.items():
                if share > 0:
                    entropy -= share * math.log2(share)
            entropy = round(entropy, 4)

            key = (segment, month)
            prev_key = None
            month_idx = months_list.index(month)
            if month_idx > 0:
                prev_key = (segment, months_list[month_idx - 1])

            if prev_key and prev_key in prev_shares:
                prev_s = prev_shares[prev_key]
                common_vendors = set(shares.keys()) & set(prev_s.keys())
                if len(common_vendors) >= 2:
                    curr_vals = [shares.get(v, 0) for v in common_vendors]
                    prev_vals = [prev_s.get(v, 0) for v in common_vendors]
                    var_prev = np.var(prev_vals)
                    var_curr = np.var(curr_vals)
                    if var_curr > 0:
                        vci = round(var_prev / var_curr, 4)
                    else:
                        vci = round(99.0, 4)
                else:
                    vci = round(1.0, 4)
            else:
                vci = round(1.0, 4)

            prev_shares[key] = shares

            concentrated_count = len([v for v, s in shares.items() if s > concentration_threshold])

            comp_a = round(alpha * (1.0 / (entropy + 0.01)), 4)
            comp_b = round(beta * vci, 4)
            comp_c = round(gamma * (1.0 / (concentrated_count + 1)), 4)
            msi = round(comp_a + comp_b + comp_c, 4)

            if entropy < entropy_floor and vci > vci_ceiling:
                health_class = "CRITICAL"
            elif entropy < entropy_floor or (vci > vci_ceiling and msi > 5.0):
                health_class = "DETERIORATING"
            elif msi > 3.0 and concentrated_count <= 1:
                health_class = "STAGNANT"
            elif abs(vci - 1.0) > 0.5:
                health_class = "TRANSITIONING"
            elif entropy >= entropy_floor and msi <= 2.0:
                health_class = "HEALTHY"
            else:
                health_class = "STABLE"

            streak_key = segment
            if streak_key not in prev_health_streaks:
                prev_health_streaks[streak_key] = 0

            if health_class in ("HEALTHY", "STABLE"):
                prev_health_streaks[streak_key] = 0
            else:
                prev_health_streaks[streak_key] += 1

            results.append({
                "segment": segment,
                "month": month,
                "vendor_count": vendor_count,
                "total_revenue": round(seg_total, 2),
                "hhi": hhi,
                "shannon_entropy": entropy,
                "vci": vci,
                "msi": msi,
                "concentrated_vendors": concentrated_count,
                "health_class": health_class,
                "unhealthy_streak": prev_health_streaks[streak_key],
            })

    return pd.DataFrame(results)


def compute_vendor_risk_profile(vendor_monthly, vendors, complaints):
    vendor_ids = sorted(vendors["vendor_id"].unique())
    results = []

    for vid in vendor_ids:
        v_info = vendors[vendors["vendor_id"] == vid].iloc[0]
        vm = vendor_monthly[vendor_monthly["vendor_id"] == vid].sort_values("month")

        if len(vm) == 0:
            continue

        avg_vsi = round(vm["vsi"].mean(), 2)
        min_vsi = round(vm["vsi"].min(), 2)
        max_vsi = round(vm["vsi"].max(), 2)

        v_complaints = complaints[complaints["vendor_id"] == vid]
        total_complaints = len(v_complaints)
        critical_complaint_count = len(v_complaints[v_complaints["severity"] == "CRITICAL"])

        avg_monthly_revenue = round(vm["net_revenue"].mean(), 2)
        rev_std = vm["net_revenue"].std()
        rev_mean = vm["net_revenue"].mean()
        revenue_volatility = round(rev_std / rev_mean, 4) if rev_mean != 0 else 0.0

        vsi_values = vm["vsi"].tolist()
        max_consecutive_decline = 0
        current_decline = 0
        for i in range(1, len(vsi_values)):
            if vsi_values[i] < vsi_values[i - 1]:
                current_decline += 1
                max_consecutive_decline = max(max_consecutive_decline, current_decline)
            else:
                current_decline = 0

        total_return_rate = round(vm["adjusted_return_rate"].mean(), 4)

        if avg_vsi < 40 or max_consecutive_decline >= 4:
            risk_category = "HIGH_RISK"
        elif avg_vsi < 60 or total_complaints > 30 or revenue_volatility > 0.5:
            risk_category = "MODERATE_RISK"
        elif avg_vsi < 75 and critical_complaint_count > 0:
            risk_category = "WATCH"
        else:
            risk_category = "LOW_RISK"

        worst_month = vm.loc[vm["vsi"].idxmin(), "month"]

        results.append({
            "vendor_id": vid,
            "vendor_name": v_info["vendor_name"],
            "segment": v_info["segment"],
            "tier": v_info["tier"],
            "avg_vsi": avg_vsi,
            "min_vsi": min_vsi,
            "max_vsi": max_vsi,
            "total_complaints": total_complaints,
            "critical_complaint_count": critical_complaint_count,
            "avg_monthly_revenue": avg_monthly_revenue,
            "revenue_volatility": revenue_volatility,
            "total_return_rate": total_return_rate,
            "consecutive_decline_months": max_consecutive_decline,
            "risk_category": risk_category,
            "worst_month": worst_month,
        })

    return pd.DataFrame(results)


def compute_promotion_effectiveness(transactions, promotions, vendors):
    promo_vendor_pairs = promotions[["promo_id", "vendor_id"]].drop_duplicates()
    results = []

    for _, pv in promo_vendor_pairs.iterrows():
        pid = pv["promo_id"]
        vid = pv["vendor_id"]

        promo_info = promotions[
            (promotions["promo_id"] == pid) &
            (promotions["vendor_id"] == vid)
        ]
        if len(promo_info) == 0:
            continue
        promo_info = promo_info.iloc[0]

        promo_start = pd.to_datetime(promo_info["start_date"])
        promo_end = pd.to_datetime(promo_info["end_date"])
        budget_cap = promo_info["budget_cap"]
        promo_type = promo_info["promo_type"]
        min_quantity = promo_info["min_quantity"]
        max_discount_pct = promo_info["max_discount_pct"]

        promo_txns = transactions[
            (transactions["vendor_id"] == vid) &
            (transactions["promo_id"] == pid) &
            (transactions["status"] == "COMPLETED") &
            (transactions["transaction_date"] >= promo_start) &
            (transactions["transaction_date"] <= promo_end)
        ]

        if len(promo_txns) == 0:
            continue

        total_promo_revenue = 0.0
        for _, txn in promo_txns.iterrows():
            total_promo_revenue += compute_line_revenue(txn)
        total_promo_revenue = round(total_promo_revenue, 2)

        total_promo_transactions = len(promo_txns)

        avg_discount = round(promo_txns["discount_pct"].mean(), 4)

        effective_discount = min(avg_discount, max_discount_pct)
        effective_discount = round(effective_discount, 4)

        roi = round((total_promo_revenue - budget_cap) / budget_cap, 4) if budget_cap > 0 else 0.0

        non_promo_txns = transactions[
            (transactions["vendor_id"] == vid) &
            (transactions["status"] == "COMPLETED") &
            (
                (transactions["promo_id"] != pid) |
                (transactions["promo_id"].isna()) |
                (transactions["promo_id"] == "")
            ) &
            (transactions["transaction_date"] >= promo_start) &
            (transactions["transaction_date"] <= promo_end)
        ]

        if len(non_promo_txns) > 0:
            non_promo_avg = 0.0
            for _, txn in non_promo_txns.iterrows():
                non_promo_avg += compute_line_revenue(txn)
            non_promo_avg = non_promo_avg / len(non_promo_txns)

            promo_avg = total_promo_revenue / total_promo_transactions
            lift_ratio = round(promo_avg / non_promo_avg, 4) if non_promo_avg > 0 else 0.0
        else:
            lift_ratio = 0.0

        results.append({
            "promo_id": pid,
            "vendor_id": vid,
            "promo_type": promo_type,
            "start_date": promo_start.strftime("%Y-%m-%d"),
            "end_date": promo_end.strftime("%Y-%m-%d"),
            "total_promo_revenue": total_promo_revenue,
            "total_promo_transactions": total_promo_transactions,
            "avg_discount_applied": avg_discount,
            "effective_discount": effective_discount,
            "roi": roi,
            "lift_ratio": lift_ratio,
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values(["promo_id", "vendor_id"]).reset_index(drop=True)
    return df


def compute_market_health_ledger(segment_equilibrium, market_segments):
    months_list = sorted(segment_equilibrium["month"].unique())
    segments = sorted(market_segments["segment"].unique())

    cumulative_penalties = {}
    results = []

    for month in months_list:
        for segment in segments:
            eq_row = segment_equilibrium[
                (segment_equilibrium["segment"] == segment) &
                (segment_equilibrium["month"] == month)
            ]
            if len(eq_row) == 0:
                continue
            eq_row = eq_row.iloc[0]

            health_class = eq_row["health_class"]
            if health_class in ("HEALTHY", "STABLE"):
                if segment in cumulative_penalties:
                    cumulative_penalties[segment] = 0.0
                continue

            seg_config = market_segments[market_segments["segment"] == segment].iloc[0]
            entropy_floor = seg_config["entropy_floor"]
            vci_ceiling = seg_config["vci_ceiling"]
            health_decay_rate = seg_config["health_decay_rate"]
            min_vendor_count = seg_config["min_vendor_count"]

            entropy = eq_row["shannon_entropy"]
            vci = eq_row["vci"]
            vendor_count = eq_row["vendor_count"]
            unhealthy_streak = eq_row["unhealthy_streak"]

            entropy_gap = round(max(0.0, entropy_floor - entropy), 4)
            vci_excess = round(max(0.0, vci - vci_ceiling), 4)

            penalty_score = round(entropy_gap * 10.0 + vci_excess * 5.0, 2)

            decay_factor = round(health_decay_rate ** unhealthy_streak, 4)

            period_penalty = round(penalty_score * decay_factor, 2)

            if segment not in cumulative_penalties:
                cumulative_penalties[segment] = 0.0
            cumulative_penalties[segment] = round(cumulative_penalties[segment] + period_penalty, 2)

            credit_amount = 0.0
            if vendor_count >= min_vendor_count:
                credit_amount = round(penalty_score * 0.1, 2)

            net_penalty = round(cumulative_penalties[segment] - credit_amount, 2)

            results.append({
                "segment": segment,
                "month": month,
                "health_class": health_class,
                "entropy_gap": entropy_gap,
                "vci_excess": vci_excess,
                "penalty_score": penalty_score,
                "decay_factor": decay_factor,
                "period_penalty": period_penalty,
                "cumulative_penalty": cumulative_penalties[segment],
                "credit_amount": credit_amount,
                "net_penalty": net_penalty,
            })

    return pd.DataFrame(results)


def main():
    base_dir = Path(__file__).parent
    task_dir = base_dir.parent / "task"

    vendors, products, transactions, commission_rates, seasonal_factors, complaints, promotions, market_segments = load_data(task_dir)

    transactions = clean_transactions(transactions)
    complaints = clean_complaints(complaints)

    vendor_monthly = compute_vendor_monthly(transactions, vendors, commission_rates, seasonal_factors, complaints)
    vendor_monthly.to_csv(task_dir / "vendor_monthly.csv", index=False)

    segment_equilibrium = compute_segment_equilibrium(vendor_monthly, market_segments, vendors)
    segment_equilibrium.to_csv(task_dir / "segment_equilibrium.csv", index=False)

    vendor_risk_profile = compute_vendor_risk_profile(vendor_monthly, vendors, complaints)
    vendor_risk_profile.to_csv(task_dir / "vendor_risk_profile.csv", index=False)

    promotion_effectiveness = compute_promotion_effectiveness(transactions, promotions, vendors)
    promotion_effectiveness.to_csv(task_dir / "promotion_effectiveness.csv", index=False)

    market_health_ledger = compute_market_health_ledger(segment_equilibrium, market_segments)
    market_health_ledger.to_csv(task_dir / "market_health_ledger.csv", index=False)

    print("vendor_monthly.csv:", len(vendor_monthly), "rows")
    print("segment_equilibrium.csv:", len(segment_equilibrium), "rows")
    print("vendor_risk_profile.csv:", len(vendor_risk_profile), "rows")
    print("promotion_effectiveness.csv:", len(promotion_effectiveness), "rows")
    print("market_health_ledger.csv:", len(market_health_ledger), "rows")


if __name__ == "__main__":
    main()

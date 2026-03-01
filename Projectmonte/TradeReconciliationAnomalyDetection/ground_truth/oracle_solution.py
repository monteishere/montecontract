import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def load_data(task_dir):
    market = pd.read_csv(task_dir / 'market_data.csv')
    signals = pd.read_csv(task_dir / 'signals.csv')
    trades = pd.read_csv(task_dir / 'executed_trades.csv')
    positions = pd.read_csv(task_dir / 'position_reports.csv')
    return market, signals, trades, positions


def time_to_minutes(time_str):
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])


def detect_unmatched_trades(trades, signals):
    anomalies = []
    valid_signal_ids = set(signals['signal_id'].unique())

    for _, trade in trades.iterrows():
        sig_id = trade['signal_id']
        if sig_id == 'NONE' or sig_id == 'REVERSAL' or sig_id not in valid_signal_ids:
            severity = 'HIGH' if trade['quantity'] > 1000 else 'LOW'
            anomalies.append({
                'anomaly_type': 'UNMATCHED_TRADE',
                'identifier': trade['trade_id'],
                'date': trade['date'],
                'symbol': trade['symbol'],
                'details': f"signal_id={sig_id} qty={trade['quantity']}",
                'severity': severity,
                'trade_ids': [trade['trade_id']]
            })

    return anomalies


def detect_duplicate_trades(trades):
    anomalies = []
    seen = {}
    duplicate_trade_ids = set()

    for _, trade in trades.iterrows():
        key = (trade['date'], trade['symbol'], trade['side'], trade['quantity'], trade['price'])
        if key in seen:
            duplicate_trade_ids.add(trade['trade_id'])
            anomalies.append({
                'anomaly_type': 'DUPLICATE_TRADE',
                'identifier': trade['trade_id'],
                'date': trade['date'],
                'symbol': trade['symbol'],
                'details': f"duplicate_of={seen[key]} qty={trade['quantity']}",
                'severity': 'MEDIUM',
                'trade_ids': [trade['trade_id']]
            })
        else:
            seen[key] = trade['trade_id']

    return anomalies, duplicate_trade_ids


def detect_timing_violations(trades, signals):
    anomalies = []
    sig_dict = signals.set_index('signal_id').to_dict('index')

    for _, trade in trades.iterrows():
        sig_id = trade['signal_id']
        if sig_id not in sig_dict:
            continue

        sig = sig_dict[sig_id]
        trade_minutes = time_to_minutes(trade['time'])
        valid_from = time_to_minutes(sig['valid_from'])
        valid_until = time_to_minutes(sig['valid_until'])

        if trade_minutes < valid_from or trade_minutes > valid_until:
            trade_hour = int(trade['time'].split(':')[0])
            trade_minute = int(trade['time'].split(':')[1])

            minutes_outside = 0
            if trade_minutes < valid_from:
                minutes_outside = valid_from - trade_minutes
            else:
                minutes_outside = trade_minutes - valid_until

            if minutes_outside > 120:
                severity = 'HIGH'
            elif trade_hour == 7 or trade_hour == 8 or (trade_hour == 9 and trade_minute < 30):
                severity = 'MEDIUM'
            else:
                severity = 'LOW'

            anomalies.append({
                'anomaly_type': 'TIMING_VIOLATION',
                'identifier': trade['trade_id'],
                'date': trade['date'],
                'symbol': trade['symbol'],
                'details': f"exec_time={trade['time']} valid={sig['valid_from']}-{sig['valid_until']}",
                'severity': severity,
                'trade_ids': [trade['trade_id']]
            })

    return anomalies


def detect_price_deviations(trades, signals):
    anomalies = []
    sig_dict = signals.set_index('signal_id').to_dict('index')

    for _, trade in trades.iterrows():
        sig_id = trade['signal_id']
        if sig_id not in sig_dict:
            continue

        sig = sig_dict[sig_id]
        limit_price = sig['limit_price']
        exec_price = trade['price']

        deviation_pct = abs(exec_price - limit_price) / limit_price * 100

        if deviation_pct > 3.0:
            severity = 'HIGH' if deviation_pct > 10.0 else 'LOW'
            anomalies.append({
                'anomaly_type': 'PRICE_DEVIATION',
                'identifier': trade['trade_id'],
                'date': trade['date'],
                'symbol': trade['symbol'],
                'details': f"exec={exec_price} limit={limit_price} dev={deviation_pct:.1f}%",
                'severity': severity,
                'trade_ids': [trade['trade_id']]
            })

    return anomalies


def detect_position_mismatches(trades, positions, duplicate_trade_ids):
    anomalies = []

    for _, pos in positions.iterrows():
        report_date = pos['report_date']
        symbol = pos['symbol']
        reported = pos['reported_position']

        sym_trades = trades[(trades['symbol'] == symbol) & (trades['date'] <= report_date)]

        computed = 0
        for _, t in sym_trades.iterrows():
            if t['side'] == 'BUY':
                computed += t['quantity']
            else:
                computed -= t['quantity']

        diff = abs(computed - reported)

        if diff > 100:
            severity = 'HIGH' if diff > 500 else 'LOW'
            anomalies.append({
                'anomaly_type': 'POSITION_MISMATCH',
                'identifier': f"{symbol}_{report_date}",
                'date': report_date,
                'symbol': symbol,
                'details': f"computed={computed} reported={reported} diff={diff}",
                'severity': severity,
                'trade_ids': []
            })

    return anomalies


def detect_same_day_reversals(trades):
    anomalies = []

    grouped = trades.groupby(['date', 'symbol'])

    for (date, symbol), group in grouped:
        buys = group[group['side'] == 'BUY']['quantity'].tolist()
        sells = group[group['side'] == 'SELL']['quantity'].tolist()

        for buy_qty in buys:
            if buy_qty in sells:
                matching_trades = group[(group['quantity'] == buy_qty)]
                trade_ids = matching_trades['trade_id'].tolist()[:2]
                anomalies.append({
                    'anomaly_type': 'SAME_DAY_REVERSAL',
                    'identifier': ','.join(trade_ids),
                    'date': date,
                    'symbol': symbol,
                    'details': f"qty={buy_qty} buy_and_sell_same_day",
                    'severity': 'MEDIUM',
                    'trade_ids': trade_ids
                })
                break

    return anomalies


def generate_summary(all_anomalies, trades):
    summary = []

    type_counts = defaultdict(int)
    affected_symbols = set()
    all_trade_ids = set()

    for a in all_anomalies:
        type_counts[a['anomaly_type']] += 1
        affected_symbols.add(a['symbol'])
        for tid in a.get('trade_ids', []):
            all_trade_ids.add(tid)

    trade_lookup = trades.set_index('trade_id').to_dict('index')
    total_value = 0
    for tid in all_trade_ids:
        if tid in trade_lookup:
            t = trade_lookup[tid]
            total_value += t['quantity'] * t['price']

    summary.append({'metric': 'total_anomalies', 'value': len(all_anomalies)})
    summary.append({'metric': 'total_affected_value', 'value': round(total_value, 2)})
    summary.append({'metric': 'affected_symbols', 'value': ','.join(sorted(affected_symbols))})

    for atype in sorted(type_counts.keys()):
        summary.append({'metric': f'count_{atype}', 'value': type_counts[atype]})

    severity_counts = defaultdict(int)
    for a in all_anomalies:
        severity_counts[a['severity']] += 1

    for sev in ['HIGH', 'MEDIUM', 'LOW']:
        summary.append({'metric': f'severity_{sev}', 'value': severity_counts.get(sev, 0)})

    return summary


def main():
    task_dir = Path(__file__).parent.parent / 'task'

    market, signals, trades, positions = load_data(task_dir)

    all_anomalies = []

    all_anomalies.extend(detect_unmatched_trades(trades, signals))
    dup_anomalies, duplicate_trade_ids = detect_duplicate_trades(trades)
    all_anomalies.extend(dup_anomalies)
    all_anomalies.extend(detect_timing_violations(trades, signals))
    all_anomalies.extend(detect_price_deviations(trades, signals))
    all_anomalies.extend(detect_position_mismatches(trades, positions, duplicate_trade_ids))
    all_anomalies.extend(detect_same_day_reversals(trades))

    report_df = pd.DataFrame(all_anomalies)
    report_df = report_df[['anomaly_type', 'identifier', 'date', 'symbol', 'details', 'severity']]
    report_df = report_df.sort_values(['date', 'anomaly_type']).reset_index(drop=True)
    report_df.to_csv(task_dir / 'anomaly_report.csv', index=False)

    summary = generate_summary(all_anomalies, trades)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(task_dir / 'anomaly_summary.csv', index=False)

    print(f"Found {len(all_anomalies)} anomalies")
    print(f"Report saved to {task_dir / 'anomaly_report.csv'}")
    print(f"Summary saved to {task_dir / 'anomaly_summary.csv'}")


if __name__ == '__main__':
    main()

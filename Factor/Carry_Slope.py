# ==============================================================================
# Factor Category: Carry
# Factor Name: Carry_Slope
# Description: rolling slope of carry term-structure signal
# ==============================================================================

import os
import re
import numpy as np
import pandas as pd


BACKTEST_START_DATE = '20100101'
N_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]
EXCLUDED_SECTORS = ['Bond', 'StockIndex', 'Other', 'Others', 'Financial']
EXCLUDED_SYMBOLS = []
CUSTOM_GROUPS = [
    {
        'name': 'CustomSymbols',
        'symbols': []
    }
]


RUN_ALL = True
SECTOR_LIST = []
RUN_CUSTOM_SYMBOLS = True


def _factor_name():
    return os.path.splitext(os.path.basename(__file__))[0]


def _safe_name(text):
    return ''.join(ch for ch in str(text) if ch.isalnum() or ch in ['_', '-', '.'])


def _param_str(n_value):
    return f"N{n_value}"


def _extract_delivery_month_start(ts_code):
    if not isinstance(ts_code, str):
        return pd.NaT

    match = re.match(r'^[A-Za-z]+(\d{4})\.\w+$', ts_code)
    if not match:
        return pd.NaT

    yymm = match.group(1)
    year = 2000 + int(yymm[:2])
    month = int(yymm[2:])
    if month < 1 or month > 12:
        return pd.NaT

    return pd.Timestamp(year=year, month=month, day=1)


def _calc_base_carry(df):
    main_close = pd.to_numeric(df.get('close', np.nan), errors='coerce')
    sub_close = pd.to_numeric(df.get('sub_close', np.nan), errors='coerce')

    main_contract = df.get('mapping_ts_code', pd.Series(index=df.index, dtype='object'))
    sub_contract = df.get('sub_mapping_ts_code', pd.Series(index=df.index, dtype='object'))

    main_delivery = main_contract.apply(_extract_delivery_month_start)
    sub_delivery = sub_contract.apply(_extract_delivery_month_start)

    trade_date = pd.to_datetime(df.index.astype(str), format='%Y%m%d', errors='coerce')
    main_days_to_expiry = (main_delivery - trade_date).dt.days
    sub_days_to_expiry = (sub_delivery - trade_date).dt.days
    tenor_days_diff = sub_days_to_expiry - main_days_to_expiry

    price_spread = (main_close / (sub_close + 1e-8)) - 1.0
    annualization = 365.0 / tenor_days_diff.replace(0, np.nan)

    carry = price_spread * annualization
    return carry.replace([np.inf, -np.inf], np.nan)


def _slope_normalized(arr):
    if np.isnan(arr).any():
        return np.nan
    x = np.arange(len(arr), dtype=float)
    x = x - x.mean()
    denom = (x ** 2).sum() + 1e-12
    slope = ((arr - arr.mean()) * x).sum() / denom
    return slope / (np.abs(arr).mean() + 1e-8)


def _calc_signal(df, n_value):
    carry = _calc_base_carry(df)
    signal = carry.rolling(window=n_value).apply(_slope_normalized, raw=True)
    return signal.replace([np.inf, -np.inf], np.nan).fillna(0)


def _load_symbol_signal(ts_code, market_data_path, n_value):
    filepath = os.path.join(market_data_path, f"{ts_code}.csv")
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath)
    if 'trade_date' not in df.columns:
        return None

    df['trade_date'] = df['trade_date'].astype(str)
    df.set_index('trade_date', inplace=True)
    return _calc_signal(df, n_value).rename(ts_code)


def _run_and_save(target_name, symbols, market_data_path, n_value):
    position_series = {}
    for ts_code in symbols:
        signal = _load_symbol_signal(ts_code, market_data_path, n_value)
        if signal is not None:
            position_series[ts_code] = signal

    if not position_series:
        print(f"Skip {target_name}: no valid data")
        return

    signals = pd.DataFrame(position_series).sort_index().fillna(0).astype(float)
    signals = signals[signals.index >= str(BACKTEST_START_DATE)]

    raw_output_name = f"{_factor_name()}_{_safe_name(target_name)}_{_param_str(n_value)}_RAW_Position.csv"
    raw_output_path = os.path.join('./Result', raw_output_name)
    signals.to_csv(raw_output_path, encoding='utf-8-sig')

    print(f"Factor {_factor_name()} output saved: {raw_output_path}")


def main():
    market_data_path = './main_contract/'
    info_path = './Info.csv'

    info = pd.read_csv(info_path, encoding='utf-8-sig')
    valid_info = info[~info['sector'].str.lower().isin([s.lower() for s in EXCLUDED_SECTORS])]
    if EXCLUDED_SYMBOLS:
        valid_info = valid_info[~valid_info['ts_code'].isin(EXCLUDED_SYMBOLS)]
    valid_symbol_set = set(valid_info['ts_code'].tolist())

    print(f"Loading {_factor_name()} data...")

    for n_value in N_LIST:
        if RUN_ALL:
            _run_and_save('ALL', sorted(valid_symbol_set), market_data_path, n_value)
        if SECTOR_LIST:
            for sector in SECTOR_LIST:
                sector_symbols = valid_info[valid_info['sector'] == sector]['ts_code'].tolist()
                if sector_symbols:
                    _run_and_save(sector, sector_symbols, market_data_path, n_value)

        if RUN_CUSTOM_SYMBOLS:
            for group in CUSTOM_GROUPS:
                raw_symbols = group.get('symbols', [])
                if isinstance(raw_symbols, str):
                    raw_symbols = [raw_symbols]
                group_symbols = [s for s in raw_symbols if s in valid_symbol_set]
                group_symbols = sorted(set(group_symbols))
                merged_name = '_'.join(group_symbols) if group_symbols else group.get('name', 'CustomGroup')
                _run_and_save(merged_name, group_symbols, market_data_path, n_value)


if __name__ == '__main__':
    main()

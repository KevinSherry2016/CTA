# ==============================================================================
# Factor Category: TrendMomentum
# Factor Name: TrendMomentum_MACD
# Description: factor signal definition
# ==============================================================================

import pandas as pd
import numpy as np
import os

FACTOR_NAME = 'TrendMomentum_MACD'
PARAM_LIST = [
    {'fast_n': 6, 'slow_n': 13, 'signal_n': 5},
    {'fast_n': 8, 'slow_n': 17, 'signal_n': 9},
    {'fast_n': 12, 'slow_n': 26, 'signal_n': 9},
    {'fast_n': 19, 'slow_n': 39, 'signal_n': 9},
    {'fast_n': 24, 'slow_n': 52, 'signal_n': 18},
]

EXCLUDED_SECTORS = ['Bond', 'StockIndex', 'Other', 'Others', 'Financial']
EXCLUDED_SYMBOLS = []
CUSTOM_GROUPS = [
    {
        'name': 'CustomSymbols',
        'symbols': []
    }
]


RUN_ALL = True
SECTOR_LIST = []  # Empty means skip by-sector
RUN_CUSTOM_SYMBOLS = True

def _safe_name(text):
    return ''.join(ch for ch in str(text) if ch.isalnum() or ch in ['_', '-', '.'])


def _param_str(param):
    return f"F{param['fast_n']}_S{param['slow_n']}_SIG{param['signal_n']}"


def _calc_signal(df, param):
    close = df['adj_close']
    dif = close.ewm(span=param['fast_n'], adjust=False).mean() - close.ewm(span=param['slow_n'], adjust=False).mean()
    dea = dif.ewm(span=param['signal_n'], adjust=False).mean()
    return (dif - dea) * 2


def _load_symbol_signal(ts_code, market_data_path, param):
    filepath = os.path.join(market_data_path, f"{ts_code}.csv")
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath)
    if 'trade_date' not in df.columns:
        return None

    df['trade_date'] = df['trade_date'].astype(str)
    df.set_index('trade_date', inplace=True)
    return _calc_signal(df, param).rename(ts_code)


def _run_and_save(target_name, symbols, market_data_path, param):
    position_series = {}
    for ts_code in symbols:
        signal = _load_symbol_signal(ts_code, market_data_path, param)
        if signal is not None:
            position_series[ts_code] = signal

    if not position_series:
        print(f"Skip {target_name}: no valid data")
        return

    signals = pd.DataFrame(position_series).sort_index().fillna(0).astype(float)

    raw_output_name = f"{FACTOR_NAME}_{_safe_name(target_name)}_{_param_str(param)}_RAW_Position.csv"
    raw_output_path = os.path.join('./Result', raw_output_name)
    signals.to_csv(raw_output_path, encoding='utf-8-sig')


    print(f"Factor {FACTOR_NAME} output saved: {raw_output_path}")


def main():
    market_data_path = './main_contract/'
    info_path = './Info.csv'

    info = pd.read_csv(info_path, encoding='utf-8-sig')
    valid_info = info[~info['sector'].str.lower().isin([s.lower() for s in EXCLUDED_SECTORS])]
    if EXCLUDED_SYMBOLS:
        valid_info = valid_info[~valid_info['ts_code'].isin(EXCLUDED_SYMBOLS)]
    valid_symbol_set = set(valid_info['ts_code'].tolist())

    print(f"Loading {FACTOR_NAME} data...")

    for param in PARAM_LIST:
        if RUN_ALL:
            _run_and_save("ALL", sorted(valid_symbol_set), market_data_path, param)
        if SECTOR_LIST:
            for sector in SECTOR_LIST:
                sector_symbols = valid_info[valid_info['sector'] == sector]['ts_code'].tolist()
                if sector_symbols:
                    _run_and_save(sector, sector_symbols, market_data_path, param)

        if RUN_CUSTOM_SYMBOLS:
            for group in CUSTOM_GROUPS:
                raw_symbols = group.get('symbols', [])
                if isinstance(raw_symbols, str):
                    raw_symbols = [raw_symbols]
                group_symbols = [s for s in raw_symbols if s in valid_symbol_set]
                group_symbols = sorted(set(group_symbols))
                merged_name = '_'.join(group_symbols) if group_symbols else group.get('name', 'CustomGroup')
                _run_and_save(merged_name, group_symbols, market_data_path, param)

if __name__ == "__main__":
    main()








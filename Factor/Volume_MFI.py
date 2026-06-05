# ============================================================================== 
# 因子类别：Volume
# 因子名称：Volume_MFI
# 代表意义：资金流量指标（Money Flow Index）。结合典型价格与成交量，衡量资金流入流出强度。
# ============================================================================== 

import pandas as pd
import numpy as np
import os


FACTOR_NAME = 'Volume_MFI'
N_LIST = [10, 20, 30, 40]

EXCLUDED_SECTORS = ['Bond', 'StockIndex','Other']
EXCLUDED_SYMBOLS = []
CUSTOM_GROUPS = [
    {
        'name': 'CustomSymbols',
        'symbols': []
    }
]


def _safe_name(text):
    return ''.join(ch for ch in str(text) if ch.isalnum() or ch in ['_', '-', '.'])


def _param_str(n_value):
    return f"N{n_value}"


def _calc_signal(df, n_value):
    close = df['adj_close']
    high = df['adj_high']
    low = df['adj_low']
    volume = df.get('vol', df.get('Volume', pd.Series(dtype=float)))

    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume

    price_change = typical_price.diff()
    positive_flow = raw_money_flow.where(price_change > 0, 0.0)
    negative_flow = raw_money_flow.where(price_change < 0, 0.0)

    pos_sum = positive_flow.rolling(window=n_value, min_periods=1).sum()
    neg_sum = negative_flow.rolling(window=n_value, min_periods=1).sum()
    money_flow_ratio = pos_sum / (neg_sum + 1e-8)
    mfi = 100 - (100 / (1 + money_flow_ratio))

    return (mfi - 50) / 50


def _load_symbol_signal(ts_code, market_data_path, n_value):
    filepath = os.path.join(market_data_path, f"{ts_code}.csv")
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath)
    if 'trade_date' not in df.columns:
        return None

    df['trade_date'] = df['trade_date'].astype(str)
    df.set_index('trade_date', inplace=True)
    signal = _calc_signal(df, n_value)
    return signal.rename(ts_code)


def _run_and_save(target_name, symbols, market_data_path, n_value):
    position_series = {}
    for ts_code in symbols:
        print(f"开始计算 {FACTOR_NAME} 因子信号... symbol={ts_code}")
        signal = _load_symbol_signal(ts_code, market_data_path, n_value)
        if signal is not None:
            position_series[ts_code] = signal

    if not position_series:
        print(f"跳过 {target_name}: 无有效数据")
        return

    signals = pd.DataFrame(position_series).sort_index().fillna(0).astype(float)
    state_machine = np.sign(signals).fillna(0).astype(float)

    raw_output_name = f"{FACTOR_NAME}_{_safe_name(target_name)}_{_param_str(n_value)}_RAW_Position.csv"
    raw_output_path = os.path.join('./Result', raw_output_name)
    signals.to_csv(raw_output_path, encoding='utf-8-sig')

    sm_output_name = f"{FACTOR_NAME}_{_safe_name(target_name)}_{_param_str(n_value)}_STATE_MACHINE_Position.csv"
    sm_output_path = os.path.join('./Result', sm_output_name)
    state_machine.to_csv(sm_output_path, encoding='utf-8-sig')

    print(f"因子 {FACTOR_NAME} 输出完成: {raw_output_path}")
    print(f"因子 {FACTOR_NAME} 输出完成: {sm_output_path}")


def main():
    marketDataPath = './main_contract/'
    infoPath = './Info.csv'

    info = pd.read_csv(infoPath, encoding='utf-8-sig')
    valid_info = info[~info['sector'].str.lower().isin([s.lower() for s in EXCLUDED_SECTORS])]
    if EXCLUDED_SYMBOLS:
        valid_info = valid_info[~valid_info['ts_code'].isin(EXCLUDED_SYMBOLS)]
    valid_symbol_set = set(valid_info['ts_code'].tolist())

    print(f"正在加载 {FACTOR_NAME} 数据...")

    for n_value in N_LIST:
        for symbol in sorted(valid_symbol_set):
            _run_and_save(symbol, [symbol], marketDataPath, n_value)

        for sector in sorted(valid_info['sector'].dropna().unique().tolist()):
            sector_symbols = valid_info[valid_info['sector'] == sector]['ts_code'].tolist()
            _run_and_save(sector, sector_symbols, marketDataPath, n_value)

        for group in CUSTOM_GROUPS:
            raw_symbols = group.get('symbols', [])
            if isinstance(raw_symbols, str):
                raw_symbols = [raw_symbols]
            group_symbols = [s for s in raw_symbols if s in valid_symbol_set]
            group_symbols = sorted(set(group_symbols))
            merged_name = '_'.join(group_symbols) if group_symbols else group.get('name', 'CustomGroup')
            _run_and_save(merged_name, group_symbols, marketDataPath, n_value)


if __name__ == "__main__":
    main()

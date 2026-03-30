import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './total/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Strategy'

M_LIST = [15, 20, 25, 30, 35]
Z_OPEN_LIST = [1.2, 1.4, 1.6]
Z_CLOSE_LIST = [0.2, 0.4, 0.6]
MIN_HOLD_LIST = [3, 5]
MAX_HOLD_LIST = [10, 15, 20]
COOLDOWN_LIST = [1, 3, 5]
SIGNAL_MODE = 'trend'
ZSCORE_METHOD = 'price_ratio_over_vol'


def load_market_data():
    info = pd.read_csv(INFO_PATH, encoding='utf-8-sig')
    data = {}
    trading_days = set()

    print('正在加载数据')
    for ts_code in info['ts_code'].tolist():
        filepath = os.path.join(MARKET_DATA_PATH, f'{ts_code}.csv')
        if not os.path.exists(filepath):
            print(f'文件不存在: {filepath}')
            continue

        df = pd.read_csv(filepath)
        df['trade_date'] = df['trade_date'].astype(str)
        df.set_index('trade_date', inplace=True)
        df['adj_close'] = pd.to_numeric(df['adj_close'], errors='coerce')
        df.sort_index(inplace=True)
        data[ts_code] = df
        trading_days.update(df.index.tolist())

    return info, data, sorted(trading_days)


def compute_zscore(df, window):
    close = df['adj_close']
    ma = close.rolling(window).mean()
    return_vol = close.pct_change(fill_method=None).rolling(window).std().replace(0, np.nan)
    return ((close / ma) - 1.0) / return_vol


def build_position_from_zscore(zscore, z_open, z_close, min_hold, max_hold, cooldown_days, signal_mode):
    positions = []
    current_position = 0
    holding_days = 0
    cooldown_left = 0

    for value in zscore.to_numpy(dtype=float):
        if np.isnan(value):
            current_position = 0
            holding_days = 0
            cooldown_left = 0
            positions.append(0.0)
            continue

        if cooldown_left > 0 and current_position == 0:
            cooldown_left -= 1

        if current_position != 0:
            holding_days += 1
            can_close = holding_days >= min_hold
            should_close = abs(value) < z_close or holding_days >= max_hold
            if can_close and should_close:
                current_position = 0
                holding_days = 0
                cooldown_left = cooldown_days

        if current_position == 0 and cooldown_left == 0:
            if signal_mode == 'trend':
                if value > z_open:
                    current_position = 1
                    holding_days = 1
                elif value < -z_open:
                    current_position = -1
                    holding_days = 1
            else:
                if value > z_open:
                    current_position = -1
                    holding_days = 1
                elif value < -z_open:
                    current_position = 1
                    holding_days = 1

        positions.append(float(current_position))

    return pd.Series(positions, index=zscore.index, dtype='float64')


info, data, trading_days = load_market_data()

for window in M_LIST:
    zscore_series = {
        ts_code: compute_zscore(df, window)
        for ts_code, df in data.items()
    }

    for z_open in Z_OPEN_LIST:
        valid_z_close_list = [z_close for z_close in Z_CLOSE_LIST if z_close < z_open]
        for z_close in valid_z_close_list:
            for min_hold in MIN_HOLD_LIST:
                for max_hold in MAX_HOLD_LIST:
                    if min_hold > max_hold:
                        continue
                    for cooldown_days in COOLDOWN_LIST:
                        position_series = {}
                        for ts_code, zscore in zscore_series.items():
                            position_series[ts_code] = build_position_from_zscore(
                                zscore,
                                z_open,
                                z_close,
                                min_hold,
                                max_hold,
                                cooldown_days,
                                SIGNAL_MODE,
                            )

                        signals = pd.DataFrame(position_series, index=trading_days).fillna(0.0).astype(float)
                        output_name = (
                            f'MovingAverageV3_4_M_{window}_ZO_{z_open}_ZC_{z_close}_'
                            f'MINH_{min_hold}_H_{max_hold}_CD_{cooldown_days}_{SIGNAL_MODE}.csv'
                        )
                        output_path = os.path.join(OUTPUT_DIR, output_name)
                        signals.to_csv(output_path, encoding='utf-8-sig')
                        print(f'信号输出完成: {output_path}')
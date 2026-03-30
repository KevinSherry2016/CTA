import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './total/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Strategy'

M_LIST = [15, 20, 25, 30, 35, 40]
Z_OPEN_LIST = [1.2, 1.4, 1.6]
Z_CLOSE_LIST = [0.2, 0.4, 0.6]
MAX_HOLD_LIST = [10, 15, 20]
SIGNAL_MODE = 'trend'
ZSCORE_METHODS = [
    'price_minus_ma_over_vol',
    'price_ratio_over_vol',
    'price_minus_ma_over_atr',
]


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
        for column in ['adj_close', 'adj_high', 'adj_low']:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')
        df.sort_index(inplace=True)
        data[ts_code] = df
        trading_days.update(df.index.tolist())

    return info, data, sorted(trading_days)


def build_position_from_zscore(zscore, z_open, z_close, max_hold, signal_mode):
    positions = []
    current_position = 0
    holding_days = 0

    for value in zscore.to_numpy(dtype=float):
        if np.isnan(value):
            current_position = 0
            holding_days = 0
            positions.append(0.0)
            continue

        if current_position != 0:
            holding_days += 1
            if abs(value) < z_close or holding_days >= max_hold:
                current_position = 0
                holding_days = 0

        if current_position == 0:
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


def rolling_return_vol(close, window):
    return close.pct_change(fill_method=None).rolling(window).std().replace(0, np.nan)


def compute_atr(df, window):
    high = df['adj_high'] if 'adj_high' in df.columns else df['adj_close']
    low = df['adj_low'] if 'adj_low' in df.columns else df['adj_close']
    prev_close = df['adj_close'].shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean().replace(0, np.nan)


def compute_zscore(df, window, method):
    close = df['adj_close']
    ma = close.rolling(window).mean()

    if method == 'price_minus_ma_over_vol':
        return_vol = rolling_return_vol(close, window)
        price_vol = (ma.abs() * return_vol).replace(0, np.nan)
        return (close - ma) / price_vol

    if method == 'price_ratio_over_vol':
        return_vol = rolling_return_vol(close, window)
        return ((close / ma) - 1.0) / return_vol

    atr = compute_atr(df, window)
    return (close - ma) / atr


info, data, trading_days = load_market_data()

for zscore_method in ZSCORE_METHODS:
    for window in M_LIST:
        zscore_series = {
            ts_code: compute_zscore(df, window, zscore_method)
            for ts_code, df in data.items()
        }

        for z_open in Z_OPEN_LIST:
            valid_z_close_list = [z_close for z_close in Z_CLOSE_LIST if z_close < z_open]
            for z_close in valid_z_close_list:
                for max_hold in MAX_HOLD_LIST:
                    position_series = {}
                    for ts_code, zscore in zscore_series.items():
                        position_series[ts_code] = build_position_from_zscore(
                            zscore,
                            z_open,
                            z_close,
                            max_hold,
                            SIGNAL_MODE,
                        )

                    signals = pd.DataFrame(position_series, index=trading_days).fillna(0.0).astype(float)
                    output_name = (
                        f'MovingAverageV3_2_M_{window}_ZO_{z_open}_ZC_{z_close}_'
                        f'H_{max_hold}_{SIGNAL_MODE}_{zscore_method}.csv'
                    )
                    output_path = os.path.join(OUTPUT_DIR, output_name)
                    signals.to_csv(output_path, encoding='utf-8-sig')
                    print(f'信号输出完成: {output_path}')
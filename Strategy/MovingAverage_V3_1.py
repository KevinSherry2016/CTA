import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './total/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Strategy'

SECTOR_PARAMS = {
    'Agriculture': {'window': 20, 'z_open': 1.2, 'z_close': 0.4, 'max_hold': 12, 'signal_mode': 'mean_reversion'},
    'Energy': {'window': 35, 'z_open': 1.4, 'z_close': 0.6, 'max_hold': 20, 'signal_mode': 'trend'},
    'Ferrous': {'window': 30, 'z_open': 1.4, 'z_close': 0.5, 'max_hold': 18, 'signal_mode': 'trend'},
    'NonFerrous': {'window': 25, 'z_open': 1.3, 'z_close': 0.4, 'max_hold': 15, 'signal_mode': 'mean_reversion'},
    'Precious': {'window': 40, 'z_open': 1.6, 'z_close': 0.6, 'max_hold': 25, 'signal_mode': 'trend'},
    'Other': {'window': 20, 'z_open': 1.2, 'z_close': 0.4, 'max_hold': 10, 'signal_mode': 'mean_reversion'},
    'Financial': {'window': 30, 'z_open': 1.5, 'z_close': 0.5, 'max_hold': 15, 'signal_mode': 'trend'},
}
DEFAULT_PARAMS = {'window': 25, 'z_open': 1.4, 'z_close': 0.4, 'max_hold': 15, 'signal_mode': 'trend'}


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


def compute_price_level_zscore(df, window):
    rolling_stats = df['adj_close'].rolling(window)
    rolling_std = rolling_stats.std().replace(0, np.nan)
    return (df['adj_close'] - rolling_stats.mean()) / rolling_std


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


info, data, trading_days = load_market_data()
sector_map = info.set_index('ts_code')['sector'].to_dict()

position_series = {}
for ts_code, df in data.items():
    params = SECTOR_PARAMS.get(sector_map.get(ts_code), DEFAULT_PARAMS)
    zscore = compute_price_level_zscore(df, params['window'])
    position_series[ts_code] = build_position_from_zscore(
        zscore,
        params['z_open'],
        params['z_close'],
        params['max_hold'],
        params['signal_mode'],
    )

signals = pd.DataFrame(position_series, index=trading_days).fillna(0.0).astype(float)
output_path = os.path.join(OUTPUT_DIR, 'MovingAverageV3_1_sector_params.csv')
signals.to_csv(output_path, encoding='utf-8-sig')
print(f'信号输出完成: {output_path}')
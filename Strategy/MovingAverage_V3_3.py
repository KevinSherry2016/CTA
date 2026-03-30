import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './total/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Strategy'
ZSCORE_METHOD = 'price_ratio_over_vol'

TREND_PARAM_GRID = {
    'window': [20, 30, 40, 50],
    'z_open': [1.2, 1.4, 1.6],
    'z_close': [0.4, 0.6, 0.8],
    'max_hold': [15, 20, 25, 30],
}
MEAN_REVERSION_PARAM_GRID = {
    'window': [10, 15, 20, 25],
    'z_open': [1.4, 1.6, 1.8],
    'z_close': [0.0, 0.2, 0.4],
    'max_hold': [5, 10, 15],
}


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


def build_trend_position(zscore, z_open, z_close, max_hold):
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
            if current_position > 0 and (value < z_close or holding_days >= max_hold):
                current_position = 0
                holding_days = 0
            elif current_position < 0 and (value > -z_close or holding_days >= max_hold):
                current_position = 0
                holding_days = 0

        if current_position == 0:
            if value > z_open:
                current_position = 1
                holding_days = 1
            elif value < -z_open:
                current_position = -1
                holding_days = 1

        positions.append(float(current_position))

    return pd.Series(positions, index=zscore.index, dtype='float64')


def build_mean_reversion_position(zscore, z_open, z_close, max_hold):
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
            if value > z_open:
                current_position = -1
                holding_days = 1
            elif value < -z_open:
                current_position = 1
                holding_days = 1

        positions.append(float(current_position))

    return pd.Series(positions, index=zscore.index, dtype='float64')


def build_param_grid(param_grid):
    params = []
    for window in param_grid['window']:
        for z_open in param_grid['z_open']:
            for z_close in param_grid['z_close']:
                for max_hold in param_grid['max_hold']:
                    params.append({
                        'window': window,
                        'z_open': z_open,
                        'z_close': z_close,
                        'max_hold': max_hold,
                    })
    return params


info, data, trading_days = load_market_data()

strategy_builders = {
    'trend': (TREND_PARAM_GRID, build_trend_position),
    'mean_reversion': (MEAN_REVERSION_PARAM_GRID, build_mean_reversion_position),
}

for strategy_name, (param_grid, builder) in strategy_builders.items():
    for params in build_param_grid(param_grid):
        if params['z_close'] >= params['z_open']:
            continue

        position_series = {}
        for ts_code, df in data.items():
            zscore = compute_zscore(df, params['window'])
            position_series[ts_code] = builder(
                zscore,
                params['z_open'],
                params['z_close'],
                params['max_hold'],
            )

        signals = pd.DataFrame(position_series, index=trading_days).fillna(0.0).astype(float)
        output_name = (
            f'MovingAverageV3_3_{strategy_name}_M_{params["window"]}_ZO_{params["z_open"]}_'
            f'ZC_{params["z_close"]}_H_{params["max_hold"]}_{ZSCORE_METHOD}.csv'
        )
        output_path = os.path.join(OUTPUT_DIR, output_name)
        signals.to_csv(output_path, encoding='utf-8-sig')
        print(f'信号输出完成: {output_path}')
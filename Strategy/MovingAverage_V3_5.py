import math
import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './total/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Strategy'


PARAM_GRID = {
    'window': [15, 20, 25, 30],
    'z_open': [1.2, 1.4, 1.6],
    'z_close': [0.2, 0.4, 0.6],
    'max_hold': [10, 15, 20],
}
ZSCORE_METHOD = 'price_ratio_over_vol'
SIGNAL_MODE = 'trend'
TRAIN_DAYS = 750
TEST_DAYS = 250
STEP_DAYS = 250
TOP_PERCENTILE = 0.2
MIN_TOP20_HITS = 2


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


def compute_daily_pnl(position_df, close_df):
    returns = close_df.pct_change(fill_method=None).fillna(0.0)
    return position_df.shift(1).fillna(0.0).mul(returns).sum(axis=1)


def compute_sharpe_ratio(daily_pnl):
    pnl = daily_pnl.dropna()
    std = pnl.std()
    if std == 0 or pd.isna(std):
        return float('nan')
    return pnl.mean() / std * np.sqrt(250)


info, data, trading_days = load_market_data()
close_df = pd.DataFrame(
    {ts_code: df['adj_close'] for ts_code, df in data.items()},
    index=trading_days,
    dtype='float64',
).ffill()

param_candidates = []
zscore_cache = {}
for window in PARAM_GRID['window']:
    zscore_cache[window] = {
        ts_code: compute_zscore(df, window)
        for ts_code, df in data.items()
    }

for params in build_param_grid(PARAM_GRID):
    if params['z_close'] >= params['z_open']:
        continue

    position_series = {}
    for ts_code, zscore in zscore_cache[params['window']].items():
        position_series[ts_code] = build_position_from_zscore(
            zscore,
            params['z_open'],
            params['z_close'],
            params['max_hold'],
            SIGNAL_MODE,
        )

    position_df = pd.DataFrame(position_series, index=trading_days).fillna(0.0).astype(float)
    param_candidates.append({
        'params': params,
        'position_df': position_df,
        'key': (
            params['window'],
            params['z_open'],
            params['z_close'],
            params['max_hold'],
        ),
    })

oos_position_df = pd.DataFrame(0.0, index=trading_days, columns=close_df.columns)
top20_hit_count = {candidate['key']: 0 for candidate in param_candidates}
selection_records = []

for test_start in range(TRAIN_DAYS, len(trading_days) - TEST_DAYS + 1, STEP_DAYS):
    train_days = trading_days[test_start - TRAIN_DAYS:test_start]
    test_days = trading_days[test_start:test_start + TEST_DAYS]

    window_scores = []
    for candidate in param_candidates:
        train_position_df = candidate['position_df'].loc[train_days]
        train_close_df = close_df.loc[train_days]
        daily_pnl = compute_daily_pnl(train_position_df, train_close_df)
        sharpe = compute_sharpe_ratio(daily_pnl)
        window_scores.append({'candidate': candidate, 'sharpe': sharpe})

    window_scores.sort(key=lambda item: item['sharpe'], reverse=True)
    top_count = max(1, math.ceil(len(window_scores) * TOP_PERCENTILE))
    for item in window_scores[:top_count]:
        top20_hit_count[item['candidate']['key']] += 1

    eligible_scores = [
        item for item in window_scores
        if top20_hit_count[item['candidate']['key']] >= MIN_TOP20_HITS
    ]
    selected_item = eligible_scores[0] if eligible_scores else window_scores[0]
    selected_candidate = selected_item['candidate']

    oos_position_df.loc[test_days] = selected_candidate['position_df'].loc[test_days]
    selection_records.append({
        'train_start': train_days[0],
        'train_end': train_days[-1],
        'test_start': test_days[0],
        'test_end': test_days[-1],
        'window': selected_candidate['params']['window'],
        'z_open': selected_candidate['params']['z_open'],
        'z_close': selected_candidate['params']['z_close'],
        'max_hold': selected_candidate['params']['max_hold'],
        'train_sharpe': selected_item['sharpe'],
        'top20_hits': top20_hit_count[selected_candidate['key']],
    })

signal_output_path = os.path.join(OUTPUT_DIR, 'MovingAverageV3_5_walk_forward_signal.csv')
oos_position_df.to_csv(signal_output_path, encoding='utf-8-sig')
print(f'信号输出完成: {signal_output_path}')

selection_output_path = os.path.join(OUTPUT_DIR, 'MovingAverageV3_5_walk_forward_params.csv')
pd.DataFrame(selection_records).to_csv(selection_output_path, index=False, encoding='utf-8-sig')
print(f'参数日志输出完成: {selection_output_path}')
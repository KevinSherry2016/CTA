import math
import os

import pandas as pd

from moving_average_v3_utils import (
    OUTPUT_DIR,
    align_series_dict,
    build_basic_state_machine_position,
    build_param_grid,
    compute_daily_pnl_from_positions,
    compute_sharpe_ratio,
    compute_zscore,
    get_close_df,
    load_market_data,
)


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


info, data, trading_days = load_market_data()
close_df = get_close_df(data, trading_days)

param_candidates = []
zscore_cache = {}
for window in PARAM_GRID['window']:
    zscore_cache[window] = {
        ts_code: compute_zscore(df, window=window, method=ZSCORE_METHOD)
        for ts_code, df in data.items()
    }

for params in build_param_grid(PARAM_GRID):
    if params['z_close'] >= params['z_open']:
        continue

    position_series = {}
    for ts_code, zscore in zscore_cache[params['window']].items():
        position_series[ts_code] = build_basic_state_machine_position(
            zscore=zscore,
            z_open=params['z_open'],
            z_close=params['z_close'],
            max_hold=params['max_hold'],
            signal_mode=SIGNAL_MODE,
        )

    position_df = align_series_dict(position_series, trading_days)
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
        daily_pnl = compute_daily_pnl_from_positions(train_position_df, train_close_df)
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
from moving_average_v3_utils import (
    OUTPUT_DIR,
    align_series_dict,
    build_mean_reversion_position,
    build_param_grid,
    build_trend_position,
    compute_zscore,
    load_market_data,
)

import os


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
ZSCORE_METHOD = 'price_ratio_over_vol'


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
            zscore = compute_zscore(df, window=params['window'], method=ZSCORE_METHOD)
            position_series[ts_code] = builder(
                zscore=zscore,
                z_open=params['z_open'],
                z_close=params['z_close'],
                max_hold=params['max_hold'],
            )

        signals = align_series_dict(position_series, trading_days)
        output_name = (
            f'MovingAverageV3_3_{strategy_name}_M_{params["window"]}_ZO_{params["z_open"]}_'
            f'ZC_{params["z_close"]}_H_{params["max_hold"]}_{ZSCORE_METHOD}.csv'
        )
        output_path = os.path.join(OUTPUT_DIR, output_name)
        signals.to_csv(output_path, encoding='utf-8-sig')
        print(f'信号输出完成: {output_path}')
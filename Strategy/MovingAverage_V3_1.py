from moving_average_v3_utils import (
    OUTPUT_DIR,
    align_series_dict,
    build_basic_state_machine_position,
    compute_zscore,
    load_market_data,
)

import os


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


info, data, trading_days = load_market_data()
sector_map = info.set_index('ts_code')['sector'].to_dict()

position_series = {}
for ts_code, df in data.items():
    params = SECTOR_PARAMS.get(sector_map.get(ts_code), DEFAULT_PARAMS)
    zscore = compute_zscore(df, window=params['window'], method='price_level_zscore')
    position_series[ts_code] = build_basic_state_machine_position(
        zscore=zscore,
        z_open=params['z_open'],
        z_close=params['z_close'],
        max_hold=params['max_hold'],
        signal_mode=params['signal_mode'],
    )

signals = align_series_dict(position_series, trading_days)
output_path = os.path.join(OUTPUT_DIR, 'MovingAverageV3_1_sector_params.csv')
signals.to_csv(output_path, encoding='utf-8-sig')
print(f'信号输出完成: {output_path}')
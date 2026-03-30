from moving_average_v3_utils import (
    OUTPUT_DIR,
    align_series_dict,
    build_holding_controlled_position,
    compute_zscore,
    load_market_data,
)

import os


M_LIST = [15, 20, 25, 30, 35]
Z_OPEN_LIST = [1.2, 1.4, 1.6]
Z_CLOSE_LIST = [0.2, 0.4, 0.6]
MIN_HOLD_LIST = [3, 5]
MAX_HOLD_LIST = [10, 15, 20]
COOLDOWN_LIST = [1, 3, 5]
SIGNAL_MODE = 'trend'
ZSCORE_METHOD = 'price_ratio_over_vol'


info, data, trading_days = load_market_data()

for window in M_LIST:
    zscore_series = {
        ts_code: compute_zscore(df, window=window, method=ZSCORE_METHOD)
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
                            position_series[ts_code] = build_holding_controlled_position(
                                zscore=zscore,
                                z_open=z_open,
                                z_close=z_close,
                                min_hold=min_hold,
                                max_hold=max_hold,
                                cooldown_days=cooldown_days,
                                signal_mode=SIGNAL_MODE,
                            )

                        signals = align_series_dict(position_series, trading_days)
                        output_name = (
                            f'MovingAverageV3_4_M_{window}_ZO_{z_open}_ZC_{z_close}_'
                            f'MINH_{min_hold}_H_{max_hold}_CD_{cooldown_days}_{SIGNAL_MODE}.csv'
                        )
                        output_path = os.path.join(OUTPUT_DIR, output_name)
                        signals.to_csv(output_path, encoding='utf-8-sig')
                        print(f'信号输出完成: {output_path}')
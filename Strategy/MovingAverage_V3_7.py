import os

import numpy as np

from moving_average_v3_utils import (
    OUTPUT_DIR,
    build_basic_state_machine_position,
    compute_zscore,
    equalize_sector_gross_exposure,
    get_close_df,
    inverse_vol_scale,
    load_market_data,
    row_rank_to_unit_interval,
    row_zscore,
)


WINDOW = 25
Z_OPEN = 1.4
Z_CLOSE = 0.4
MAX_HOLD = 15
SIGNAL_MODE = 'trend'
ZSCORE_METHOD = 'price_ratio_over_vol'
VOL_WINDOW = 20
CROSS_SECTION_METHODS = ['rank', 'zscore']


def build_signal_strength(df, zscore):
    state = build_basic_state_machine_position(
        zscore=zscore,
        z_open=Z_OPEN,
        z_close=Z_CLOSE,
        max_hold=MAX_HOLD,
        signal_mode=SIGNAL_MODE,
    )
    strength = state * zscore.abs()
    strength = strength.where(state != 0, 0.0)
    return strength


info, data, trading_days = load_market_data()
close_df = get_close_df(data, trading_days)

raw_strength_dict = {}
for ts_code, df in data.items():
    zscore = compute_zscore(df, window=WINDOW, method=ZSCORE_METHOD)
    raw_strength_dict[ts_code] = build_signal_strength(df, zscore)

raw_strength_df = close_df.copy() * np.nan
for ts_code, signal_strength in raw_strength_dict.items():
    raw_strength_df[ts_code] = signal_strength
raw_strength_df = raw_strength_df.reindex(trading_days).fillna(0.0)

for cross_section_method in CROSS_SECTION_METHODS:
    if cross_section_method == 'rank':
        normalized_signal_df = row_rank_to_unit_interval(raw_strength_df)
    else:
        normalized_signal_df = row_zscore(raw_strength_df)

    normalized_signal_df = normalized_signal_df.fillna(0.0)
    scaled_signal_df = inverse_vol_scale(normalized_signal_df, close_df, VOL_WINDOW).fillna(0.0)
    sector_balanced_signal_df = equalize_sector_gross_exposure(scaled_signal_df, info).fillna(0.0)

    output_path = os.path.join(
        OUTPUT_DIR,
        f'MovingAverageV3_7_cross_section_{cross_section_method}.csv',
    )
    sector_balanced_signal_df.to_csv(output_path, encoding='utf-8-sig')
    print(f'信号输出完成: {output_path}')
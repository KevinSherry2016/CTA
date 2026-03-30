import os

import pandas as pd

from moving_average_v3_utils import (
    OUTPUT_DIR,
    align_series_dict,
    build_basic_state_machine_position,
    compute_zscore,
    load_market_data,
    rolling_return_vol,
)


WINDOW = 25
Z_OPEN = 1.4
Z_CLOSE = 0.4
MAX_HOLD = 15
SIGNAL_MODE = 'trend'
ZSCORE_METHOD = 'price_ratio_over_vol'
TREND_FILTER_WINDOW = 120
VOL_FILTER_WINDOW = 60
VOL_QUANTILE = 0.4
LIQUIDITY_WINDOW = 60
LIQUIDITY_QUANTILE = 0.3


def apply_regime_filters(df: pd.DataFrame, raw_position: pd.Series) -> pd.Series:
    close = df['adj_close']
    trend_ma = close.rolling(TREND_FILTER_WINDOW).mean()
    long_allowed = close >= trend_ma
    short_allowed = close <= trend_ma

    realized_vol = rolling_return_vol(close, VOL_FILTER_WINDOW)
    vol_threshold = realized_vol.rolling(VOL_FILTER_WINDOW).quantile(VOL_QUANTILE)
    high_enough_vol = realized_vol >= vol_threshold

    liquidity_base = df['amount'] if 'amount' in df.columns else df['vol']
    liquidity = liquidity_base.rolling(LIQUIDITY_WINDOW).mean()
    liquidity_threshold = liquidity.rolling(LIQUIDITY_WINDOW).quantile(LIQUIDITY_QUANTILE)
    liquid_enough = liquidity >= liquidity_threshold

    filtered_position = raw_position.copy()
    filtered_position[(filtered_position > 0) & (~long_allowed)] = 0.0
    filtered_position[(filtered_position < 0) & (~short_allowed)] = 0.0
    filtered_position[~high_enough_vol.fillna(False)] = 0.0
    filtered_position[~liquid_enough.fillna(False)] = 0.0
    return filtered_position


info, data, trading_days = load_market_data()

position_series = {}
for ts_code, df in data.items():
    zscore = compute_zscore(df, window=WINDOW, method=ZSCORE_METHOD)
    raw_position = build_basic_state_machine_position(
        zscore=zscore,
        z_open=Z_OPEN,
        z_close=Z_CLOSE,
        max_hold=MAX_HOLD,
        signal_mode=SIGNAL_MODE,
    )
    position_series[ts_code] = apply_regime_filters(df, raw_position)

signals = align_series_dict(position_series, trading_days)
output_path = os.path.join(OUTPUT_DIR, 'MovingAverageV3_6_filtered_signal.csv')
signals.to_csv(output_path, encoding='utf-8-sig')
print(f'信号输出完成: {output_path}')
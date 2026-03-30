import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './total/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Strategy'


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
        for column in ['adj_close', 'amount', 'vol']:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')
        df.sort_index(inplace=True)
        data[ts_code] = df
        trading_days.update(df.index.tolist())

    return info, data, sorted(trading_days)


def rolling_return_vol(close, window):
    return close.pct_change(fill_method=None).rolling(window).std().replace(0, np.nan)


def compute_zscore(df, window):
    close = df['adj_close']
    ma = close.rolling(window).mean()
    return_vol = rolling_return_vol(close, window)
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
    zscore = compute_zscore(df, WINDOW)
    raw_position = build_position_from_zscore(
        zscore,
        Z_OPEN,
        Z_CLOSE,
        MAX_HOLD,
        SIGNAL_MODE,
    )
    position_series[ts_code] = apply_regime_filters(df, raw_position)

signals = pd.DataFrame(position_series, index=trading_days).fillna(0.0).astype(float)
output_path = os.path.join(OUTPUT_DIR, 'MovingAverageV3_6_filtered_signal.csv')
signals.to_csv(output_path, encoding='utf-8-sig')
print(f'信号输出完成: {output_path}')
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
VOL_WINDOW = 20
CROSS_SECTION_METHODS = ['rank', 'zscore']


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


def row_rank_to_unit_interval(df):
    def _rank_row(row):
        valid = row.dropna()
        if valid.empty:
            return row * np.nan
        ranks = valid.rank(method='average')
        if len(valid) == 1:
            scaled = pd.Series(0.0, index=valid.index)
        else:
            scaled = (ranks - 1) / (len(valid) - 1)
            scaled = scaled * 2.0 - 1.0
        return scaled.reindex(row.index)

    return df.apply(_rank_row, axis=1)


def row_zscore(df):
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1).replace(0, np.nan)
    return df.sub(row_mean, axis=0).div(row_std, axis=0)


def inverse_vol_scale(signal_df, close_df, vol_window):
    vol_df = close_df.pct_change(fill_method=None).rolling(vol_window).std().replace(0, np.nan)
    scaled = signal_df.div(vol_df)
    return scaled.replace([np.inf, -np.inf], np.nan)


def equalize_sector_gross_exposure(signal_df, info):
    result = signal_df.copy()
    sector_map = info.set_index('ts_code')['sector'].to_dict()

    for sector in sorted(info['sector'].dropna().unique().tolist()):
        sector_columns = [column for column in result.columns if sector_map.get(column) == sector]
        if not sector_columns:
            continue
        sector_abs_sum = result[sector_columns].abs().sum(axis=1).replace(0, np.nan)
        result[sector_columns] = result[sector_columns].div(sector_abs_sum, axis=0)

    gross_sum = result.abs().sum(axis=1).replace(0, np.nan)
    result = result.div(gross_sum, axis=0)
    return result.replace([np.inf, -np.inf], np.nan)


def build_signal_strength(zscore):
    state = build_position_from_zscore(
        zscore,
        Z_OPEN,
        Z_CLOSE,
        MAX_HOLD,
        SIGNAL_MODE,
    )
    strength = state * zscore.abs()
    strength = strength.where(state != 0, 0.0)
    return strength


info, data, trading_days = load_market_data()
close_df = pd.DataFrame(
    {ts_code: df['adj_close'] for ts_code, df in data.items()},
    index=trading_days,
    dtype='float64',
).ffill()

raw_strength_dict = {}
for ts_code, df in data.items():
    zscore = compute_zscore(df, WINDOW)
    raw_strength_dict[ts_code] = build_signal_strength(zscore)

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
import os
from itertools import product

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './total/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Strategy'
TRADING_DAYS_PER_YEAR = 250


def load_info(info_path: str = INFO_PATH) -> pd.DataFrame:
    return pd.read_csv(info_path, encoding='utf-8-sig')


def load_market_data(
    market_data_path: str = MARKET_DATA_PATH,
    info_path: str = INFO_PATH,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    info = load_info(info_path)
    data: dict[str, pd.DataFrame] = {}

    numeric_columns = [
        'adj_close', 'adj_open', 'adj_high', 'adj_low',
        'close', 'open', 'high', 'low', 'vol', 'amount', 'oi',
    ]

    for ts_code in info['ts_code'].tolist():
        filepath = os.path.join(market_data_path, f'{ts_code}.csv')
        if not os.path.exists(filepath):
            continue

        df = pd.read_csv(filepath)
        if 'trade_date' not in df.columns:
            continue

        df['trade_date'] = df['trade_date'].astype(str)
        df.set_index('trade_date', inplace=True)
        df.sort_index(inplace=True)

        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')

        data[ts_code] = df

    trading_days = sorted({trade_date for df in data.values() for trade_date in df.index.tolist()})
    return info, data, trading_days


def align_series_dict(series_dict: dict[str, pd.Series], trading_days: list[str]) -> pd.DataFrame:
    return pd.DataFrame(series_dict, index=trading_days).fillna(0.0).astype(float)


def build_param_grid(param_dict: dict[str, list]) -> list[dict[str, object]]:
    keys = list(param_dict.keys())
    values = [param_dict[key] for key in keys]
    return [dict(zip(keys, value_tuple)) for value_tuple in product(*values)]


def get_close_df(data: dict[str, pd.DataFrame], trading_days: list[str]) -> pd.DataFrame:
    close_dict = {}
    for ts_code, df in data.items():
        if 'adj_close' in df.columns:
            close_dict[ts_code] = df['adj_close']
    return pd.DataFrame(close_dict, index=trading_days, dtype='float64').ffill()


def rolling_return_vol(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change(fill_method=None).rolling(window).std().replace(0, np.nan)


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    high = df['adj_high'] if 'adj_high' in df.columns else df['adj_close']
    low = df['adj_low'] if 'adj_low' in df.columns else df['adj_close']
    close = df['adj_close']
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(window).mean().replace(0, np.nan)


def compute_zscore(df: pd.DataFrame, window: int, method: str = 'price_level_zscore') -> pd.Series:
    close = df['adj_close']
    ma = close.rolling(window).mean()

    if method == 'price_level_zscore':
        price_std = close.rolling(window).std().replace(0, np.nan)
        return (close - ma) / price_std

    if method == 'price_minus_ma_over_vol':
        return_vol = rolling_return_vol(close, window)
        price_vol = (ma.abs() * return_vol).replace(0, np.nan)
        return (close - ma) / price_vol

    if method == 'price_ratio_over_vol':
        return_vol = rolling_return_vol(close, window)
        return ((close / ma) - 1.0) / return_vol

    if method == 'price_minus_ma_over_atr':
        atr = compute_atr(df, window)
        return (close - ma) / atr

    raise ValueError(f'Unsupported z-score method: {method}')


def build_basic_state_machine_position(
    zscore: pd.Series,
    z_open: float,
    z_close: float,
    max_hold: int,
    signal_mode: str,
) -> pd.Series:
    if signal_mode not in {'trend', 'mean_reversion'}:
        raise ValueError("signal_mode must be 'trend' or 'mean_reversion'.")

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
            should_close = abs(value) < z_close or holding_days >= max_hold
            if should_close:
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


def build_trend_position(
    zscore: pd.Series,
    z_open: float,
    z_close: float,
    max_hold: int,
) -> pd.Series:
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
            should_close = False
            if current_position > 0:
                should_close = value < z_close
            else:
                should_close = value > -z_close

            if should_close or holding_days >= max_hold:
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


def build_mean_reversion_position(
    zscore: pd.Series,
    z_open: float,
    z_close: float,
    max_hold: int,
) -> pd.Series:
    return build_basic_state_machine_position(
        zscore=zscore,
        z_open=z_open,
        z_close=z_close,
        max_hold=max_hold,
        signal_mode='mean_reversion',
    )


def build_holding_controlled_position(
    zscore: pd.Series,
    z_open: float,
    z_close: float,
    min_hold: int,
    max_hold: int,
    cooldown_days: int,
    signal_mode: str,
) -> pd.Series:
    if signal_mode not in {'trend', 'mean_reversion'}:
        raise ValueError("signal_mode must be 'trend' or 'mean_reversion'.")
    if min_hold > max_hold:
        raise ValueError('min_hold must be less than or equal to max_hold.')

    positions = []
    current_position = 0
    holding_days = 0
    cooldown_left = 0

    for value in zscore.to_numpy(dtype=float):
        if np.isnan(value):
            current_position = 0
            holding_days = 0
            cooldown_left = 0
            positions.append(0.0)
            continue

        if cooldown_left > 0 and current_position == 0:
            cooldown_left -= 1

        if current_position != 0:
            holding_days += 1
            can_close = holding_days >= min_hold
            should_close = abs(value) < z_close or holding_days >= max_hold
            if can_close and should_close:
                current_position = 0
                holding_days = 0
                cooldown_left = cooldown_days

        if current_position == 0 and cooldown_left == 0:
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


def compute_daily_pnl_from_positions(position_df: pd.DataFrame, close_df: pd.DataFrame) -> pd.Series:
    aligned_positions = position_df.reindex(columns=close_df.columns).fillna(0.0)
    returns = close_df.pct_change(fill_method=None).fillna(0.0)
    return aligned_positions.shift(1).fillna(0.0).mul(returns).sum(axis=1)


def compute_sharpe_ratio(daily_pnl: pd.Series, trading_days_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    pnl = daily_pnl.dropna()
    std = pnl.std()
    if std == 0 or pd.isna(std):
        return float('nan')
    return pnl.mean() / std * np.sqrt(trading_days_per_year)


def row_rank_to_unit_interval(df: pd.DataFrame) -> pd.DataFrame:
    def _rank_row(row: pd.Series) -> pd.Series:
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


def row_zscore(df: pd.DataFrame) -> pd.DataFrame:
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1).replace(0, np.nan)
    return df.sub(row_mean, axis=0).div(row_std, axis=0)


def inverse_vol_scale(signal_df: pd.DataFrame, close_df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    vol_df = close_df.pct_change(fill_method=None).rolling(vol_window).std().replace(0, np.nan)
    scaled = signal_df.div(vol_df)
    return scaled.replace([np.inf, -np.inf], np.nan)


def equalize_sector_gross_exposure(signal_df: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
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
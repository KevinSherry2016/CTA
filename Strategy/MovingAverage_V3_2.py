import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './total/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Strategy'
SUMMARY_CSV = 'MovingAverage_V3_2_summary.csv'

M_LIST = [10, 15, 20, 25, 30, 35, 40, 45, 50]   
Z_OPEN_LIST = [1.2, 1.4, 1.6, 1.8, 2.0] 
Z_CLOSE_LIST = [0.0, 0.2, 0.4, 0.6, 0.8]
MAX_HOLD_LIST = [5, 10, 15, 20, 25, 30] 
SIGNAL_MODE = 'trend'
ZSCORE_METHODS = [
    'price_minus_ma_over_vol',
    'price_ratio_over_vol',
    'price_minus_ma_over_atr',
]


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
        for column in ['adj_close', 'adj_high', 'adj_low']:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')
        if 'mapping_ts_code' in df.columns:
            df['mapping_ts_code'] = df['mapping_ts_code'].astype(str)
        df.sort_index(inplace=True)
        data[ts_code] = df
        trading_days.update(df.index.tolist())

    return info, data, sorted(trading_days)


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


def rolling_return_vol(close, window):
    return close.pct_change(fill_method=None).rolling(window).std().replace(0, np.nan)


def compute_atr(df, window):
    high = df['adj_high'] if 'adj_high' in df.columns else df['adj_close']
    low = df['adj_low'] if 'adj_low' in df.columns else df['adj_close']
    prev_close = df['adj_close'].shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean().replace(0, np.nan)


def compute_zscore(df, window, method):
    close = df['adj_close']
    ma = close.rolling(window).mean()

    if method == 'price_minus_ma_over_vol':
        return_vol = rolling_return_vol(close, window)
        price_vol = (ma.abs() * return_vol).replace(0, np.nan)
        return (close - ma) / price_vol

    if method == 'price_ratio_over_vol':
        return_vol = rolling_return_vol(close, window)
        return ((close / ma) - 1.0) / return_vol

    atr = compute_atr(df, window)
    return (close - ma) / atr


def build_close_df(data, trading_days):
    close_data = {ts_code: df['adj_close'] for ts_code, df in data.items()}
    return pd.DataFrame(close_data, index=trading_days, dtype='float64').ffill()


def build_main_contract_df(data, trading_days):
    contract_data = {}
    for ts_code, df in data.items():
        if 'mapping_ts_code' in df.columns:
            contract_data[ts_code] = df['mapping_ts_code']
    return pd.DataFrame(contract_data).reindex(trading_days)


def calc_normalized(positions, close_df):
    ret_df = close_df.pct_change(fill_method=None)
    pos_df = positions.reindex(columns=close_df.columns).fillna(0.0).shift(1).fillna(0.0)
    pnl_per_asset = pos_df * ret_df
    daily_pnl = pnl_per_asset.sum(axis=1)

    scale = daily_pnl.std()
    if scale == 0 or pd.isna(scale):
        scale = 1.0

    norm_pos_df = pos_df / scale
    norm_daily_pnl = daily_pnl / scale
    return norm_pos_df, norm_daily_pnl


def rollover_adjusted_turnover(pos_df, main_contract_df):
    if main_contract_df.empty:
        return pos_df.diff().fillna(0.0).abs().sum(axis=1)

    prev_contract = main_contract_df.shift(1)
    is_rollover = (
        main_contract_df.ne(prev_contract)
        & main_contract_df.notna()
        & prev_contract.notna()
    ).reindex(columns=pos_df.columns, fill_value=False)

    prev_pos = pos_df.shift(1).fillna(0.0)
    normal_turnover = pos_df.diff().fillna(0.0).abs()
    rollover_turnover = prev_pos.abs() + pos_df.abs()
    turnover_df = normal_turnover.where(~is_rollover, rollover_turnover)
    return turnover_df.sum(axis=1)


def calc_metrics(norm_daily_pnl, norm_pos_df, main_contract_df):
    pnl = norm_daily_pnl.dropna()
    std = pnl.std()
    sharpe = pnl.mean() / std * np.sqrt(250) if std != 0 else float('nan')

    cumulative = pnl.cumsum()
    rolling_max = cumulative.cummax()
    drawdown = cumulative - rolling_max
    max_drawdown = drawdown.min()

    max_drawdown_days = 0
    peak_idx = 0
    for idx in range(len(cumulative)):
        if cumulative.iloc[idx] >= rolling_max.iloc[idx]:
            peak_idx = idx
        else:
            max_drawdown_days = max(max_drawdown_days, idx - peak_idx)

    gmv = norm_pos_df.abs().sum(axis=1)
    turnover = rollover_adjusted_turnover(norm_pos_df, main_contract_df)
    total_turnover = turnover.sum()
    holding_period = (gmv.sum() / total_turnover) * 2 if total_turnover != 0 else float('nan')
    pot = (pnl.sum() / total_turnover) * 10000 if total_turnover != 0 else float('nan')

    return {
        'sharpeRatio': sharpe,
        'maxDrawdown': max_drawdown,
        'maxDrawdownDays': max_drawdown_days,
        'holdingPeriod': holding_period,
        'pot': pot,
    }


def main():
    info, data, trading_days = load_market_data()
    close_df = build_close_df(data, trading_days)
    main_contract_df = build_main_contract_df(data, trading_days)
    summary_rows = []

    for zscore_method in ZSCORE_METHODS:
        print(f'开始遍历 zscore_method: {zscore_method}')
        for window in M_LIST:
            zscore_series = {
                ts_code: compute_zscore(df, window, zscore_method)
                for ts_code, df in data.items()
            }

            for z_open in Z_OPEN_LIST:
                valid_z_close_list = [z_close for z_close in Z_CLOSE_LIST if z_close < z_open]
                for z_close in valid_z_close_list:
                    for max_hold in MAX_HOLD_LIST:
                        position_series = {}
                        for ts_code, zscore in zscore_series.items():
                            position_series[ts_code] = build_position_from_zscore(
                                zscore,
                                z_open,
                                z_close,
                                max_hold,
                                SIGNAL_MODE,
                            )

                        positions = pd.DataFrame(position_series, index=trading_days).fillna(0.0).astype(float)
                        norm_pos_df, norm_daily_pnl = calc_normalized(positions, close_df)
                        metrics = calc_metrics(norm_daily_pnl, norm_pos_df, main_contract_df)

                        strategy_file = (
                            f'MovingAverageV3_2_M_{window}_ZO_{z_open}_ZC_{z_close}_'
                            f'H_{max_hold}_{SIGNAL_MODE}_{zscore_method}.csv'
                        )
                        row = {
                            'strategyFile': strategy_file,
                            'zscore_method': zscore_method,
                            'window': window,
                            'z_open': z_open,
                            'z_close': z_close,
                            'max_hold': max_hold,
                            'signal_mode': SIGNAL_MODE,
                        }
                        row.update(metrics)
                        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(['zscore_method', 'sharpeRatio', 'pot'], ascending=[True, False, False], inplace=True)
    output_path = os.path.join(OUTPUT_DIR, SUMMARY_CSV)
    summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'汇总输出完成: {output_path}')


if __name__ == '__main__':
    main()
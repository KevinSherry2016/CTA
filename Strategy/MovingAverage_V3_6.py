import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './total/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Strategy'
SUMMARY_CSV = 'MovingAverage_V3_6_summary.csv'


WINDOW_LIST = [15, 20, 25, 30, 35]
Z_OPEN_LIST = [1.2, 1.4, 1.6]
Z_CLOSE_LIST = [0.2, 0.4, 0.6]
MAX_HOLD_LIST = [10, 15, 20]
SIGNAL_MODE_LIST = ['trend']
ZSCORE_METHOD = 'price_ratio_over_vol'
TREND_FILTER_WINDOW_LIST = [60, 120, 180]
VOL_FILTER_WINDOW_LIST = [40, 60]
VOL_QUANTILE_LIST = [0.3, 0.4, 0.5]
LIQUIDITY_WINDOW_LIST = [40, 60]
LIQUIDITY_QUANTILE_LIST = [0.2, 0.3, 0.4]


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
        if 'mapping_ts_code' in df.columns:
            df['mapping_ts_code'] = df['mapping_ts_code'].astype(str)
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


def apply_regime_filters(
    df: pd.DataFrame,
    raw_position: pd.Series,
    trend_filter_window: int,
    vol_filter_window: int,
    vol_quantile: float,
    liquidity_window: int,
    liquidity_quantile: float,
) -> pd.Series:
    close = df['adj_close']
    trend_ma = close.rolling(trend_filter_window).mean()
    long_allowed = close >= trend_ma
    short_allowed = close <= trend_ma

    realized_vol = rolling_return_vol(close, vol_filter_window)
    vol_threshold = realized_vol.rolling(vol_filter_window).quantile(vol_quantile)
    high_enough_vol = realized_vol >= vol_threshold

    liquidity_base = df['amount'] if 'amount' in df.columns else df['vol']
    liquidity = liquidity_base.rolling(liquidity_window).mean()
    liquidity_threshold = liquidity.rolling(liquidity_window).quantile(liquidity_quantile)
    liquid_enough = liquidity >= liquidity_threshold

    filtered_position = raw_position.copy()
    filtered_position[(filtered_position > 0) & (~long_allowed)] = 0.0
    filtered_position[(filtered_position < 0) & (~short_allowed)] = 0.0
    filtered_position[~high_enough_vol.fillna(False)] = 0.0
    filtered_position[~liquid_enough.fillna(False)] = 0.0
    return filtered_position


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
    return pos_df / scale, daily_pnl / scale


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
    return normal_turnover.where(~is_rollover, rollover_turnover).sum(axis=1)


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

    total_valid_z_close_count = sum(len([z_close for z_close in Z_CLOSE_LIST if z_close < z_open]) for z_open in Z_OPEN_LIST)
    total_param_count = (
        len(WINDOW_LIST)
        * total_valid_z_close_count
        * len(MAX_HOLD_LIST)
        * len(SIGNAL_MODE_LIST)
        * len(TREND_FILTER_WINDOW_LIST)
        * len(VOL_FILTER_WINDOW_LIST)
        * len(VOL_QUANTILE_LIST)
        * len(LIQUIDITY_WINDOW_LIST)
        * len(LIQUIDITY_QUANTILE_LIST)
    )
    param_index = 0

    print(f'开始遍历 V3_6 参数，总组合数: {total_param_count}')

    zscore_cache = {}
    for window in WINDOW_LIST:
        zscore_cache[window] = {
            ts_code: compute_zscore(df, window)
            for ts_code, df in data.items()
        }

    for window in WINDOW_LIST:
        for z_open in Z_OPEN_LIST:
            valid_z_close_list = [z_close for z_close in Z_CLOSE_LIST if z_close < z_open]
            for z_close in valid_z_close_list:
                for max_hold in MAX_HOLD_LIST:
                    for signal_mode in SIGNAL_MODE_LIST:
                        raw_position_series = {}
                        for ts_code, zscore in zscore_cache[window].items():
                            raw_position_series[ts_code] = build_position_from_zscore(
                                zscore,
                                z_open,
                                z_close,
                                max_hold,
                                signal_mode,
                            )

                        for trend_filter_window in TREND_FILTER_WINDOW_LIST:
                            for vol_filter_window in VOL_FILTER_WINDOW_LIST:
                                for vol_quantile in VOL_QUANTILE_LIST:
                                    for liquidity_window in LIQUIDITY_WINDOW_LIST:
                                        for liquidity_quantile in LIQUIDITY_QUANTILE_LIST:
                                            param_index += 1
                                            print(
                                                f'[进度 {param_index}/{total_param_count}] '
                                                f'window={window}, z_open={z_open}, z_close={z_close}, '
                                                f'max_hold={max_hold}, signal_mode={signal_mode}, '
                                                f'trend_filter_window={trend_filter_window}, '
                                                f'vol_filter_window={vol_filter_window}, vol_quantile={vol_quantile}, '
                                                f'liquidity_window={liquidity_window}, liquidity_quantile={liquidity_quantile}'
                                            )

                                            position_series = {}
                                            for ts_code, df in data.items():
                                                position_series[ts_code] = apply_regime_filters(
                                                    df,
                                                    raw_position_series[ts_code],
                                                    trend_filter_window,
                                                    vol_filter_window,
                                                    vol_quantile,
                                                    liquidity_window,
                                                    liquidity_quantile,
                                                )

                                            positions = pd.DataFrame(position_series, index=trading_days).fillna(0.0).astype(float)
                                            norm_pos_df, norm_daily_pnl = calc_normalized(positions, close_df)
                                            metrics = calc_metrics(norm_daily_pnl, norm_pos_df, main_contract_df)

                                            strategy_file = (
                                                f'MovingAverageV3_6_M_{window}_ZO_{z_open}_ZC_{z_close}_H_{max_hold}_'
                                                f'{signal_mode}_TFW_{trend_filter_window}_VFW_{vol_filter_window}_'
                                                f'VQ_{vol_quantile}_LQW_{liquidity_window}_LQQ_{liquidity_quantile}.csv'
                                            )
                                            summary_row = {
                                                'strategyFile': strategy_file,
                                                'window': window,
                                                'z_open': z_open,
                                                'z_close': z_close,
                                                'max_hold': max_hold,
                                                'signal_mode': signal_mode,
                                                'zscore_method': ZSCORE_METHOD,
                                                'trend_filter_window': trend_filter_window,
                                                'vol_filter_window': vol_filter_window,
                                                'vol_quantile': vol_quantile,
                                                'liquidity_window': liquidity_window,
                                                'liquidity_quantile': liquidity_quantile,
                                            }
                                            summary_row.update(metrics)
                                            summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(['sharpeRatio', 'pot'], ascending=[False, False], inplace=True)
    output_path = os.path.join(OUTPUT_DIR, SUMMARY_CSV)
    summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'汇总输出完成: {output_path}')


if __name__ == '__main__':
    main()
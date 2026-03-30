import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './total/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Strategy'
ZSCORE_METHOD = 'price_ratio_over_vol'
SUMMARY_CSV = 'MovingAverage_V3_3_summary.csv'

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
        if 'mapping_ts_code' in df.columns:
            df['mapping_ts_code'] = df['mapping_ts_code'].astype(str)
        df.sort_index(inplace=True)
        data[ts_code] = df
        trading_days.update(df.index.tolist())

    return info, data, sorted(trading_days)


def compute_zscore(df, window):
    close = df['adj_close']
    ma = close.rolling(window).mean()
    return_vol = close.pct_change(fill_method=None).rolling(window).std().replace(0, np.nan)
    return ((close / ma) - 1.0) / return_vol


def build_trend_position(zscore, z_open, z_close, max_hold):
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
            if current_position > 0 and (value < z_close or holding_days >= max_hold):
                current_position = 0
                holding_days = 0
            elif current_position < 0 and (value > -z_close or holding_days >= max_hold):
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


def build_mean_reversion_position(zscore, z_open, z_close, max_hold):
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
            if value > z_open:
                current_position = -1
                holding_days = 1
            elif value < -z_open:
                current_position = 1
                holding_days = 1

        positions.append(float(current_position))

    return pd.Series(positions, index=zscore.index, dtype='float64')


def build_param_grid(param_grid):
    params = []
    for window in param_grid['window']:
        for z_open in param_grid['z_open']:
            for z_close in param_grid['z_close']:
                for max_hold in param_grid['max_hold']:
                    params.append({
                        'window': window,
                        'z_open': z_open,
                        'z_close': z_close,
                        'max_hold': max_hold,
                    })
    return params


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
                zscore = compute_zscore(df, params['window'])
                position_series[ts_code] = builder(
                    zscore,
                    params['z_open'],
                    params['z_close'],
                    params['max_hold'],
                )

            positions = pd.DataFrame(position_series, index=trading_days).fillna(0.0).astype(float)
            norm_pos_df, norm_daily_pnl = calc_normalized(positions, close_df)
            metrics = calc_metrics(norm_daily_pnl, norm_pos_df, main_contract_df)

            strategy_file = (
                f'MovingAverageV3_3_{strategy_name}_M_{params["window"]}_ZO_{params["z_open"]}_'
                f'ZC_{params["z_close"]}_H_{params["max_hold"]}_{ZSCORE_METHOD}.csv'
            )
            row = {
                'strategyFile': strategy_file,
                'strategy_name': strategy_name,
                'window': params['window'],
                'z_open': params['z_open'],
                'z_close': params['z_close'],
                'max_hold': params['max_hold'],
                'zscore_method': ZSCORE_METHOD,
            }
            row.update(metrics)
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(['strategy_name', 'sharpeRatio', 'pot'], ascending=[True, False, False], inplace=True)
    output_path = os.path.join(OUTPUT_DIR, SUMMARY_CSV)
    summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'汇总输出完成: {output_path}')


if __name__ == '__main__':
    main()
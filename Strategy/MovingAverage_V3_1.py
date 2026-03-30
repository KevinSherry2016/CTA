import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './total/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Strategy'
SUMMARY_CSV = 'MovingAverage_V3_1_summary.csv'

WINDOW_LIST = [10, 15, 20, 25, 30, 35, 40, 45, 50]
Z_OPEN_LIST = [1.2, 1.4, 1.6, 1.8, 2.0]
Z_CLOSE_LIST = [0.0, 0.2, 0.4, 0.6, 0.8]
MAX_HOLD_LIST = [5, 10, 15, 20, 25, 30]
SIGNAL_MODE_LIST = ['trend', 'mean_reversion']


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


def compute_price_level_zscore(df, window):
    rolling_stats = df['adj_close'].rolling(window)
    rolling_std = rolling_stats.std().replace(0, np.nan)
    return (df['adj_close'] - rolling_stats.mean()) / rolling_std


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


def build_close_df(data, symbols, trading_days):
    close_data = {symbol: data[symbol]['adj_close'] for symbol in symbols if symbol in data}
    return pd.DataFrame(close_data, index=trading_days, dtype='float64').ffill()


def build_main_contract_df(data, symbols, trading_days):
    contract_data = {}
    for symbol in symbols:
        if symbol not in data:
            continue
        if 'mapping_ts_code' in data[symbol].columns:
            contract_data[symbol] = data[symbol]['mapping_ts_code']
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
    sector_groups = info.groupby('sector')['ts_code'].apply(list).to_dict()
    summary_rows = []
    total_valid_z_close_count = sum(len([z_close for z_close in Z_CLOSE_LIST if z_close < z_open]) for z_open in Z_OPEN_LIST)
    total_param_count_per_sector = (
        len(WINDOW_LIST)
        * total_valid_z_close_count
        * len(MAX_HOLD_LIST)
        * len(SIGNAL_MODE_LIST)
    )

    for sector, symbols in sector_groups.items():
        sector_symbols = [symbol for symbol in symbols if symbol in data]
        if not sector_symbols:
            print(f'跳过空 sector: {sector}')
            continue

        print(
            f'开始遍历 sector: {sector}，品种数: {len(sector_symbols)}，'
            f'参数组合数: {total_param_count_per_sector}'
        )
        close_df = build_close_df(data, sector_symbols, trading_days)
        main_contract_df = build_main_contract_df(data, sector_symbols, trading_days)

        zscore_cache = {}
        for window in WINDOW_LIST:
            zscore_cache[window] = {
                symbol: compute_price_level_zscore(data[symbol], window)
                for symbol in sector_symbols
            }

        sector_param_index = 0
        for window in WINDOW_LIST:
            for z_open in Z_OPEN_LIST:
                valid_z_close_list = [z_close for z_close in Z_CLOSE_LIST if z_close < z_open]
                for z_close in valid_z_close_list:
                    for max_hold in MAX_HOLD_LIST:
                        for signal_mode in SIGNAL_MODE_LIST:
                            sector_param_index += 1
                            print(
                                f'[{sector}] 进度 {sector_param_index}/{total_param_count_per_sector} '
                                f'| window={window}, z_open={z_open}, z_close={z_close}, '
                                f'max_hold={max_hold}, signal_mode={signal_mode}'
                            )
                            position_series = {}
                            for symbol in sector_symbols:
                                position_series[symbol] = build_position_from_zscore(
                                    zscore_cache[window][symbol],
                                    z_open,
                                    z_close,
                                    max_hold,
                                    signal_mode,
                                )

                            positions = pd.DataFrame(position_series, index=trading_days).fillna(0.0).astype(float)
                            norm_pos_df, norm_daily_pnl = calc_normalized(positions, close_df)
                            metrics = calc_metrics(norm_daily_pnl, norm_pos_df, main_contract_df)

                            strategy_file = (
                                f'MovingAverageV3_1_sector_{sector}_M_{window}_ZO_{z_open}_'
                                f'ZC_{z_close}_H_{max_hold}_{signal_mode}.csv'
                            )
                            row = {
                                'sector': sector,
                                'strategyFile': strategy_file,
                                'window': window,
                                'z_open': z_open,
                                'z_close': z_close,
                                'max_hold': max_hold,
                                'signal_mode': signal_mode,
                            }
                            row.update(metrics)
                            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(['sector', 'sharpeRatio', 'pot'], ascending=[True, False, False], inplace=True)
    output_path = os.path.join(OUTPUT_DIR, SUMMARY_CSV)
    summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'汇总输出完成: {output_path}')


if __name__ == '__main__':
    main()

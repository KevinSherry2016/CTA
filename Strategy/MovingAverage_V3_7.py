import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './total/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Strategy'
SUMMARY_CSV = 'MovingAverage_V3_7_summary.csv'


WINDOW_LIST = [15, 20, 25, 30, 35]
Z_OPEN_LIST = [1.2, 1.4, 1.6]
Z_CLOSE_LIST = [0.2, 0.4, 0.6]
MAX_HOLD_LIST = [10, 15, 20]
SIGNAL_MODE_LIST = ['trend', 'mean_reversion']
ZSCORE_METHOD = 'price_ratio_over_vol'
VOL_WINDOW_LIST = [10, 20, 30]
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


def build_signal_strength(zscore, z_open, z_close, max_hold, signal_mode):
    state = build_position_from_zscore(
        zscore,
        z_open,
        z_close,
        max_hold,
        signal_mode,
    )
    strength = state * zscore.abs()
    strength = strength.where(state != 0, 0.0)
    return strength


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
    close_df = pd.DataFrame(
        {ts_code: df['adj_close'] for ts_code, df in data.items()},
        index=trading_days,
        dtype='float64',
    ).ffill()
    main_contract_df = build_main_contract_df(data, trading_days)

    total_valid_z_close_count = sum(len([z_close for z_close in Z_CLOSE_LIST if z_close < z_open]) for z_open in Z_OPEN_LIST)
    total_param_count = (
        len(WINDOW_LIST)
        * total_valid_z_close_count
        * len(MAX_HOLD_LIST)
        * len(SIGNAL_MODE_LIST)
        * len(VOL_WINDOW_LIST)
        * len(CROSS_SECTION_METHODS)
    )
    param_index = 0
    print(f'开始遍历 V3_7 参数，总组合数: {total_param_count}')

    zscore_cache = {}
    for window in WINDOW_LIST:
        zscore_cache[window] = {
            ts_code: compute_zscore(df, window)
            for ts_code, df in data.items()
        }

    summary_rows = []
    for window in WINDOW_LIST:
        for z_open in Z_OPEN_LIST:
            valid_z_close_list = [z_close for z_close in Z_CLOSE_LIST if z_close < z_open]
            for z_close in valid_z_close_list:
                for max_hold in MAX_HOLD_LIST:
                    for signal_mode in SIGNAL_MODE_LIST:
                        raw_strength_dict = {}
                        for ts_code, zscore in zscore_cache[window].items():
                            raw_strength_dict[ts_code] = build_signal_strength(
                                zscore,
                                z_open,
                                z_close,
                                max_hold,
                                signal_mode,
                            )

                        raw_strength_df = close_df.copy() * np.nan
                        for ts_code, signal_strength in raw_strength_dict.items():
                            raw_strength_df[ts_code] = signal_strength
                        raw_strength_df = raw_strength_df.reindex(trading_days).fillna(0.0)

                        for vol_window in VOL_WINDOW_LIST:
                            for cross_section_method in CROSS_SECTION_METHODS:
                                param_index += 1
                                print(
                                    f'[进度 {param_index}/{total_param_count}] '
                                    f'window={window}, z_open={z_open}, z_close={z_close}, '
                                    f'max_hold={max_hold}, signal_mode={signal_mode}, '
                                    f'vol_window={vol_window}, cross_section_method={cross_section_method}'
                                )

                                if cross_section_method == 'rank':
                                    normalized_signal_df = row_rank_to_unit_interval(raw_strength_df)
                                else:
                                    normalized_signal_df = row_zscore(raw_strength_df)

                                normalized_signal_df = normalized_signal_df.fillna(0.0)
                                scaled_signal_df = inverse_vol_scale(normalized_signal_df, close_df, vol_window).fillna(0.0)
                                sector_balanced_signal_df = equalize_sector_gross_exposure(scaled_signal_df, info).fillna(0.0)

                                norm_pos_df, norm_daily_pnl = calc_normalized(sector_balanced_signal_df, close_df)
                                metrics = calc_metrics(norm_daily_pnl, norm_pos_df, main_contract_df)
                                row = {
                                    'strategyFile': (
                                        f'MovingAverageV3_7_M_{window}_ZO_{z_open}_ZC_{z_close}_'
                                        f'H_{max_hold}_{signal_mode}_VW_{vol_window}_{cross_section_method}.csv'
                                    ),
                                    'cross_section_method': cross_section_method,
                                    'window': window,
                                    'z_open': z_open,
                                    'z_close': z_close,
                                    'max_hold': max_hold,
                                    'signal_mode': signal_mode,
                                    'zscore_method': ZSCORE_METHOD,
                                    'vol_window': vol_window,
                                }
                                row.update(metrics)
                                summary_rows.append(row)

    output_path = os.path.join(OUTPUT_DIR, SUMMARY_CSV)
    pd.DataFrame(summary_rows).to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'汇总输出完成: {output_path}')


if __name__ == '__main__':
    main()

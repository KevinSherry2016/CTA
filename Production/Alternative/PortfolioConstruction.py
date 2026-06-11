import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, 'Result')
NORM_START_DATE = '20200101'
NORM_END_DATE = '20251231'


def _rollover_adjusted_turnover(pos_df, main_contract_df):
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


def calc_metrics(norm_daily_pnl, norm_pos_df, main_contract_df, trading_days_per_year=250):
    pnl = norm_daily_pnl.dropna()
    std = pnl.std()
    sharpe = pnl.mean() / std * (trading_days_per_year ** 0.5) if std != 0 else float('nan')
    gmv = norm_pos_df.abs().sum(axis=1)
    turnover = _rollover_adjusted_turnover(norm_pos_df, main_contract_df)
    total_turnover = turnover.sum()
    holding_period = (gmv.sum() / total_turnover) * 2 if total_turnover != 0 else float('nan')
    pot = (pnl.sum() / total_turnover) * 10000 if total_turnover != 0 else float('nan')
    return sharpe, pot, holding_period


def do_merge_and_normalize(output_name, file_weights):
    os.makedirs(RESULT_DIR, exist_ok=True)
    position_dfs = []
    for fname, weight in file_weights.items():
        path = os.path.join(RESULT_DIR, fname)
        if not os.path.exists(path):
            print(f'Warning: {path} not found. Skipping.')
            continue
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.astype(str)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        position_dfs.append((fname, float(weight), df))

    if not position_dfs:
        print(f'No valid position files for {output_name}')
        return

    all_index = sorted({idx for _, _, df in position_dfs for idx in df.index})
    all_columns = sorted({col for _, _, df in position_dfs for col in df.columns})
    merged_pos = pd.DataFrame(0.0, index=all_index, columns=all_columns, dtype=float)
    total_weight = 0.0
    for fname, weight, df in position_dfs:
        aligned = df.reindex(index=all_index, columns=all_columns).fillna(0.0)
        merged_pos = merged_pos + aligned * weight
        total_weight += abs(weight)
        print(f'Merged {fname} with weight {weight}')
    if total_weight != 0:
        merged_pos = merged_pos / total_weight

    market_data_path = os.path.join(BASE_DIR, '..', '..', 'main_contract')
    data = {}
    main_contracts = {}
    for col in merged_pos.columns:
        fp = os.path.join(market_data_path, f'{col}.csv')
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            if 'trade_date' in df.columns:
                df['trade_date'] = df['trade_date'].astype(str)
                df.set_index('trade_date', inplace=True)
                data[col] = pd.to_numeric(df.get('adj_close'), errors='coerce')
                if 'mapping_ts_code' in df.columns:
                    main_contracts[col] = df['mapping_ts_code']

    close_df = pd.DataFrame(data).reindex(index=merged_pos.index).ffill()
    main_contract_df = pd.DataFrame(main_contracts).reindex(index=merged_pos.index, columns=merged_pos.columns)
    ret_df = close_df.pct_change(fill_method=None)
    daily_pnl = (merged_pos.shift(1).fillna(0.0) * ret_df).sum(axis=1).fillna(0.0)

    pnl_for_scale = daily_pnl.copy()
    pnl_for_scale = pnl_for_scale[(pnl_for_scale.index >= NORM_START_DATE) & (pnl_for_scale.index <= NORM_END_DATE)]
    scale = pnl_for_scale.std()
    if scale == 0 or pd.isna(scale):
        scale = 1.0

    norm_pos_df = merged_pos / scale
    norm_daily_pnl = daily_pnl / scale

    out_pos = os.path.join(RESULT_DIR, f'{output_name}_norm_Position.csv')
    out_pnl = os.path.join(RESULT_DIR, f'{output_name}_norm_PnL.csv')
    out_gmv = os.path.join(RESULT_DIR, f'{output_name}_GMV.csv')
    out_png = os.path.join(RESULT_DIR, f'{output_name}_cPnL.png')
    norm_pos_df.to_csv(out_pos, encoding='utf-8-sig')
    norm_daily_pnl.to_frame(name='PnL').to_csv(out_pnl, encoding='utf-8-sig')
    norm_pos_df.abs().sum(axis=1).to_frame(name='GMV').to_csv(out_gmv, encoding='utf-8-sig')

    sharpe, pot, hold = calc_metrics(norm_daily_pnl, norm_pos_df, main_contract_df)
    trade_dates = pd.to_datetime(norm_daily_pnl.index, format='%Y%m%d')
    cumulative_pnl = norm_daily_pnl.cumsum()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(trade_dates, cumulative_pnl, linewidth=2, color='#1f77b4')
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
    ax.set_title(f'{output_name} | Sharpe: {sharpe:.2f} | POT: {pot:.2f} | Hold: {hold:.1f}d')
    ax.set_ylabel('Cumulative PnL')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(left=trade_dates[0])
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f'Generated normalized files for {output_name}; scale={scale}')


def main():
    tasks = [
        {
            'output_name': 'L1_Sector_Agriculture',
            'file_weights': {
                'Volume_OiVolumeResonance_Agriculture_N40_STATE_MACHINE_norm_Position.csv': 1.0,
                'CrossSectional_TailReturnAsymmetry_Agriculture_N70_TANH_norm_Position.csv': 1.0,
            },
        },
        {
            'output_name': 'L1_Sector_Energy',
            'file_weights': {
                'Volume_MFI_Energy_N30_ZSCORE_norm_Position.csv': 1.0,
                'CrossSectional_TailReturnAsymmetry_Energy_N70_ZSCORE_norm_Position.csv': 1.0,
            },
        },
        {
            'output_name': 'L1_Sector_Ferrous',
            'file_weights': {
                'Volume_PriceVolumeCorrelation_Ferrous_N40_ZSCORE_norm_Position.csv': 1.0,
                'CrossSectional_Skewness_Ferrous_N80_ZSCORE_norm_Position.csv': 1.0,
            },
        },
        {
            'output_name': 'L1_Sector_NonFerrous',
            'file_weights': {
                'Volume_OiVolumeResonance_NonFerrous_N20_STATE_MACHINE_norm_Position.csv': 1.0,
                'Volume_OIPriceFlow_NonFerrous_N70_TANH_norm_Position.csv': 1.0,
            },
        },
        {
            'output_name': 'L1_Sector_Precious',
            'file_weights': {
                'Microstructure_WickImbalance_Precious_N40_STATE_MACHINE_norm_Position.csv': 1.0,
                'Microstructure_WickImbalance_Precious_N60_STATE_MACHINE_norm_Position.csv': 1.0,
            },
        },
        {
            'output_name': 'L2_Sector_Merge_All',
            'file_weights': {
                'L1_Sector_Agriculture_norm_Position.csv': 1.0,
                'L1_Sector_Energy_norm_Position.csv': 1.0,
                'L1_Sector_Ferrous_norm_Position.csv': 1.0,
                'L1_Sector_NonFerrous_norm_Position.csv': 1.0,
                'L1_Sector_Precious_norm_Position.csv': 0.25,
            },
        },
    ]

    for task in tasks:
        do_merge_and_normalize(task['output_name'], task['file_weights'])


if __name__ == '__main__':
    main()

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NORM_START_DATE = '20150101'
NORM_END_DATE = '20201231'

def _rollover_adjusted_turnover(pos_df, main_contract_df):
    prev_contract = main_contract_df.shift(1)
    is_rollover = (
        main_contract_df.ne(prev_contract)
        & main_contract_df.notna()
        & prev_contract.notna()
    ).reindex(columns=pos_df.columns, fill_value=False)
    prev_pos = pos_df.shift(1).fillna(0.0)
    normal_turnover   = pos_df.diff().fillna(0.0).abs()
    rollover_turnover = prev_pos.abs() + pos_df.abs()
    turnover_df = normal_turnover.where(~is_rollover, rollover_turnover)
    return turnover_df.sum(axis=1)

def calc_metrics(norm_daily_pnl, norm_pos_df, main_contract_df, trading_days_per_year=250):
    pnl = norm_daily_pnl.dropna()
    std = pnl.std()
    sharpe = pnl.mean() / std * (trading_days_per_year ** 0.5) if std != 0 else float('nan')
    gmv      = norm_pos_df.abs().sum(axis=1)
    turnover = _rollover_adjusted_turnover(norm_pos_df, main_contract_df)
    total_turnover = turnover.sum()
    holding_period = (gmv.sum() / total_turnover) * 2 if total_turnover != 0 else float('nan')
    pot = (pnl.sum() / total_turnover) * 10000 if total_turnover != 0 else float('nan')
    return sharpe, pot, holding_period

def do_merge_and_normalize(OUTPUT_NAME, FILE_WEIGHTS, RESULT_DIR, marketDataPath):
    OUTPUT_POS = os.path.join(RESULT_DIR, f'{OUTPUT_NAME}_norm_Position.csv')
    OUTPUT_PNL = os.path.join(RESULT_DIR, f'{OUTPUT_NAME}_norm_PnL.csv')
    OUTPUT_PNG = os.path.join(RESULT_DIR, f'{OUTPUT_NAME}_cPnL.png')
    
    # Optional Global input glob if files not specified
    INPUT_GLOB = '*_norm_Position.csv'
    
    if not FILE_WEIGHTS:
        files = glob.glob(os.path.join(RESULT_DIR, INPUT_GLOB))
        for f in files:
            name = os.path.basename(f)
            FILE_WEIGHTS[name] = 1.0
            
    print(f"Merging {len(FILE_WEIGHTS)} files for {OUTPUT_NAME}...")
    
    position_dfs = []
    for fname, weight in FILE_WEIGHTS.items():
        path = os.path.join(RESULT_DIR, fname)
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping.")
            continue
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.astype(str)
        df = df.astype(float)
        position_dfs.append((fname, weight, df))
        
    if not position_dfs:
        print(f"No valid position files found to merge for {OUTPUT_NAME}.\n")
        return
        
    all_index = sorted({idx for _, _, df in position_dfs for idx in df.index})
    all_columns = sorted({col for _, _, df in position_dfs for col in df.columns})

    merged_pos = pd.DataFrame(0.0, index=all_index, columns=all_columns, dtype=float)
    total_weight = 0.0

    for fname, weight, df in position_dfs:
        aligned = df.reindex(index=all_index, columns=all_columns).fillna(0.0)
        merged_pos = merged_pos + aligned * weight
        total_weight += weight
        print(f"Merged {fname} with weight {weight}")

    merged_pos.sort_index(inplace=True)
    
    # ==== Volatility Normalization ====
    print("Loading Market Data for Normalization...")
    
    data = {}
    main_contracts = {}
    for col in merged_pos.columns:
        fp = os.path.join(marketDataPath, f'{col}.csv')
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            if 'trade_date' in df.columns:
                df['trade_date'] = df['trade_date'].astype(str)
                df.set_index('trade_date', inplace=True)
                df['adj_close'] = pd.to_numeric(df.get('adj_close'), errors='coerce')
                data[col] = df['adj_close']
                if 'mapping_ts_code' in df.columns:
                    main_contracts[col] = df['mapping_ts_code']

    close_df = pd.DataFrame(data).reindex(index=merged_pos.index).ffill()
    main_contract_df = pd.DataFrame(main_contracts).reindex(index=merged_pos.index, columns=merged_pos.columns)
    
    ret_df = close_df.pct_change(fill_method=None)
    
    pos_shifted = merged_pos.shift(1).fillna(0.0)
    pnl_per_asset = pos_shifted * ret_df
    daily_pnl = pnl_per_asset.sum(axis=1)
    
    pnl_for_scale = daily_pnl.copy()
    if NORM_START_DATE:
        pnl_for_scale = pnl_for_scale[pnl_for_scale.index >= NORM_START_DATE]
    if NORM_END_DATE:
        pnl_for_scale = pnl_for_scale[pnl_for_scale.index <= NORM_END_DATE]
        
    scale = pnl_for_scale.std()
    if scale == 0 or pd.isna(scale):
        scale = 1.0
        
    print(f"Calculated standard deviation for scaling: {scale}")
    
    norm_pos_df = merged_pos / scale
    norm_daily_pnl = daily_pnl / scale
    
    norm_pos_df.to_csv(OUTPUT_POS, encoding='utf-8-sig')
    norm_daily_pnl.to_frame(name='PnL').to_csv(OUTPUT_PNL, encoding='utf-8-sig')
    
    # ==== Plotting ====
    _mc_df = main_contract_df.reindex(index=norm_pos_df.index, columns=norm_pos_df.columns)
    sharpe, pot, hold = calc_metrics(norm_daily_pnl, norm_pos_df, _mc_df)
    
    plot_title = f"{OUTPUT_NAME} | Sharpe: {sharpe:.2f} | POT: {pot:.2f} | Hold: {hold:.1f}d"
    
    trade_dates = pd.to_datetime(norm_daily_pnl.index, format='%Y%m%d')
    cumulative_pnl = norm_daily_pnl.cumsum()
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(trade_dates, cumulative_pnl, linewidth=2, color='#1f77b4')
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
    ax.set_title(plot_title)
    ax.set_ylabel('Cumulative PnL')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(left=trade_dates[0])
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)
    
    print(f"Generated normalized files for {OUTPUT_NAME} in Result directory.\n")

def main():
    # ==== Configurations ====
    RESULT_DIR = os.path.join(BASE_DIR, 'Result')
    os.makedirs(RESULT_DIR, exist_ok=True)
    marketDataPath = os.path.join(BASE_DIR, '../../main_contract')

    # 支持多次合并任务。按照列表顺序依次执行。
    # 前一个任务输出的文件将自动存在 Result 目录中，如果下一个任务声明了其名称，即可直接作为输入合并。
    MERGE_TASKS = [
        {
            'output_name': 'L1_Sector_Merge_Agriculture',
            'file_weights': {
                'AgricultureOils_Volume_CMF_norm_Position.csv': 2.0,
                'AgricultureSofts_TrendMomentum_DonchianChannel_norm_Position.csv': 1.0,
            }
        },
        {
            'output_name': 'L1_Sector_Merge_Energy',
            'file_weights': {
                'Energy_TrendMomentum_MovingAverageBias_norm_Position.csv': 1.0,
                'Energy_Volume_CMF_norm_Position.csv': 1.0,
            }
        },
        {
            'output_name': 'L1_Sector_Merge_Precious',
            'file_weights': {
                'Precious_Microstructure_BuyingSellingPressure_norm_Position.csv': 1.0,
                'Precious_TrendMomentum_DualMACrossover_Short_norm_Position.csv': 2.0,
                'Precious_TrendMomentum_DualMACrossover_Long_norm_Position.csv': 2.0,
            }
        },
        {
            'output_name': 'L1_Sector_Merge_Ferrous',
            'file_weights': {
                'Ferrous_CrossSectional_OvernightVsIntraday_norm_Position.csv': 1.0,
                'Ferrous_TrendMomentum_MovingAverageBias_norm_Position.csv': 1.0,
                'Ferrous_Volume_VolumeMomentum_norm_Position.csv':0.5,
            }
        },
        {
            'output_name': 'L1_Sector_Merge_NonFerrous_CuAl',
            'file_weights': {
                'NonFerrousCuAl_TrendMomentum_MACD_norm_Position.csv': 1.0,
                'NonFerrousCuAL_TrendMomentum_MovingAverageBias_norm_Position.csv': 1.0,
            }
        },
        {
            'output_name': 'L1_Sector_Merge_NonFerrous_Others',
            'file_weights': {
                'NonFerrousOthers_Microstructure_BuyingSellingPressure_norm_Position.csv': 1.0,
                'NonFerrousOthers_TrendMomentum_DualMACrossover_norm_Position.csv': 1.0,
            }
        },
        {
            'output_name': 'L2_Sector_Merge_NonFerrous',
            'file_weights': {
                'L1_Sector_Merge_NonFerrous_CuAl_norm_Position.csv': 2.0,
                'L1_Sector_Merge_NonFerrous_Others_norm_Position.csv': 1.0,
            }
        },

        {
            'output_name': 'L3_Sector_Merge_All',
            'file_weights': {
                'L2_Sector_Merge_NonFerrous_norm_Position.csv': 25,
                'L1_Sector_Merge_Precious_norm_Position.csv': 5,
                'L1_Sector_Merge_Energy_norm_Position.csv': 25,
                'L1_Sector_Merge_Agriculture_norm_Position.csv': 20,
                'L1_Sector_Merge_Ferrous_norm_Position.csv': 25,
            }
        },
        
    ]

    for task in MERGE_TASKS:
        do_merge_and_normalize(
            OUTPUT_NAME=task['output_name'],
            FILE_WEIGHTS=task['file_weights'],
            RESULT_DIR=RESULT_DIR,
            marketDataPath=marketDataPath
        )

if __name__ == "__main__":
    main()

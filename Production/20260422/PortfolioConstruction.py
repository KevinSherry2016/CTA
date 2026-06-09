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

def convert_simple_to_compound(file_path, output_path, start_date=None, target_volatility=0.16, divisor=100.0):
    """
    将单利PnL转换为复利净值。
    
    参数:
    file_path: 输入的单利PnL csv文件路径
    output_path: 输出的复合净值 csv文件路径
    start_date: 开始计算的日期，格式为字符串 'YYYYMMDD'，如 '20150101'。如果为None，则从第一天开始。
    target_volatility: 目标的年化波动率参数，默认 0.16（即 16%）。若设为 0.20，日收益会扩大 20%/16% 倍。
    divisor: 如果PnL是百分比（如 4.06 代表 4.06%），则使用 100。
             如果PnL已经是小数形式（如 0.0406 代表 4.06%），则使用 1。
    """
    df = pd.read_csv(file_path, index_col=0)
    
    if start_date is not None:
        df = df[df.index.astype(str) >= str(start_date)]
        if df.empty:
            print("警告: 给定的起始日期之后没有数据，请检查起止日期。")
            return
            
    pnl_col = df.columns[0]
    volatility_scale = target_volatility / 0.16
    daily_return = (df[pnl_col] / divisor) * volatility_scale
    
    net_values, positions, compound_pnls = [], [], []
    drawdowns, max_drawdowns = [], []
    
    current_net_value, max_net_value, current_position = 1.0, 1.0, 1.0
    running_max_dd = 0.0
    
    for i, ret in enumerate(daily_return):
        positions.append(current_position)
        actual_ret = ret * current_position
        compound_pnls.append(actual_ret)
        
        current_net_value *= (1 + actual_ret)
        net_values.append(current_net_value)
        
        if current_net_value > max_net_value:
            max_net_value = current_net_value
            
        drawdown = 1.0 - (current_net_value / max_net_value)
        drawdowns.append(drawdown)
        
        if drawdown > running_max_dd:
            running_max_dd = drawdown
        max_drawdowns.append(running_max_dd)
        
        # if drawdown <= 0.05:
        #     current_position = 1.0
        # elif drawdown <= 0.10:
        #     current_position = 0.8
        # elif drawdown <= 0.15:
        #     current_position = 0.5
        # else:
        #     current_position = 0.3

        if drawdown <= 0.03:
            current_position = 1.0
        elif drawdown <= 0.05:
            current_position = 0.8
        elif drawdown <= 0.08:
            current_position = 0.5
        else:
            current_position = 0.3
            
    result_df = pd.DataFrame({
        'Simple_PnL': df[pnl_col],
        'Position_Factor': positions,
        'Compound_PnL_Rate': compound_pnls,
        'Compound_Net_Value': net_values,
        'Drawdown': drawdowns,
        'Max_Drawdown': max_drawdowns
    }, index=df.index)
    
    result_df.to_csv(output_path)
    print(f"Risk control metrics generated and saved to: {output_path}")

    mean_return = np.mean(compound_pnls)
    std_return = np.std(compound_pnls)
    sharpe_ratio = np.sqrt(252) * mean_return / std_return if std_return != 0 else 0
    mdd = max(max_drawdowns)
    
    plot_title = f"Compound Net Value | Sharpe: {sharpe_ratio:.2f} | MDD: {mdd:.2%}"
    trade_dates = pd.to_datetime(result_df.index, format='%Y%m%d')
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(trade_dates, result_df['Compound_Net_Value'], linewidth=2, color='#1f77b4')
    ax.axhline(1.0, color='black', linewidth=0.6, linestyle='--')
    ax.set_title(plot_title)
    ax.set_ylabel('Net Value')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(left=trade_dates[0])
    fig.autofmt_xdate()
    plt.tight_layout()
    
    pic_path = output_path.replace('.csv', '.png')
    fig.savefig(pic_path, dpi=150)
    plt.close(fig)
    print(f"Risk control plot saved to: {pic_path}\n")

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

def do_merge_and_normalize(OUTPUT_NAME, FILE_WEIGHTS, RESULT_DIR, marketDataPath, TASK_TYPE='merge'):
    OUTPUT_POS = os.path.join(RESULT_DIR, f'{OUTPUT_NAME}_norm_Position.csv')
    OUTPUT_PNL = os.path.join(RESULT_DIR, f'{OUTPUT_NAME}_norm_PnL.csv')
    OUTPUT_PNG = os.path.join(RESULT_DIR, f'{OUTPUT_NAME}_cPnL.png')
    OUTPUT_GMV = os.path.join(RESULT_DIR, f'{OUTPUT_NAME}_GMV.csv')
    
    # Optional Global input glob if files not specified
    INPUT_GLOB = '*_norm_Position.csv'
    
    # 支持 delta_neutral 时只输入列表或字符串
    if isinstance(FILE_WEIGHTS, list):
        FILE_WEIGHTS = {f: 1.0 for f in FILE_WEIGHTS}
    elif isinstance(FILE_WEIGHTS, str):
        FILE_WEIGHTS = {FILE_WEIGHTS: 1.0}
        
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
    
    # ==== 仓位中性化 (Delta Neutral) ====
    if TASK_TYPE == 'delta_neutral':
        print(f"Applying delta neutral transformation for {OUTPUT_NAME}...")
        # 每天各品种仓位 = 仓位 - 总仓位均值
        merged_pos = merged_pos.sub(merged_pos.mean(axis=1), axis=0)
    
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
    
    # ==== GMV ====
    gmv = norm_pos_df.abs().sum(axis=1)
    gmv.to_frame(name='GMV').to_csv(OUTPUT_GMV, encoding='utf-8-sig')
    
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

        ##将各个品类的策略合并成一个品类的组合策略，权重可以根据历史表现调整，或者简单平均
        {
            'output_name': 'L1_Sector_Merge_Agriculture',
            'type': 'merge',
            'file_weights': {
                'AgricultureOils_Volume_CMF_norm_Position.csv': 2.0,
                'AgricultureSofts_TrendMomentum_DonchianChannel_norm_Position.csv': 1.0,
            }
        },
        {
            'output_name': 'L1_Sector_Merge_Energy',
            'type': 'merge',
            'file_weights': {
                'Energy_TrendMomentum_MovingAverageBias_norm_Position.csv': 1.0,
                'Energy_Volume_CMF_norm_Position.csv': 1.0,
            }
        },
        {
            'output_name': 'L1_Sector_Merge_Precious',
            'type': 'merge',
            'file_weights': {
                'Precious_Microstructure_BuyingSellingPressure_norm_Position.csv': 1.0,
                'Precious_TrendMomentum_DualMACrossover_Short_norm_Position.csv': 2.0,
                'Precious_TrendMomentum_DualMACrossover_Long_norm_Position.csv': 2.0,
            }
        },
        {
            'output_name': 'L1_Sector_Merge_Ferrous',
            'type': 'merge',
            'file_weights': {
                'Ferrous_CrossSectional_OvernightVsIntraday_norm_Position.csv': 1.0,
                'Ferrous_TrendMomentum_MovingAverageBias_norm_Position.csv': 1.0,
            }
        },
        {
            'output_name': 'L1_Sector_Merge_NonFerrous_CuAl',
            'type': 'merge',
            'file_weights': {
                'NonFerrousCuAl_TrendMomentum_MACD_norm_Position.csv': 1.0,
                'NonFerrousCuAL_TrendMomentum_MovingAverageBias_norm_Position.csv': 1.0,
            }
        },
        {
            'output_name': 'L1_Sector_Merge_NonFerrous_Others',
            'type': 'merge',
            'file_weights': {
                'NonFerrousOthers_Microstructure_BuyingSellingPressure_norm_Position.csv': 1.0,
                'NonFerrousOthers_TrendMomentum_DualMACrossover_norm_Position.csv': 1.0,
            }
        },
        {
            'output_name': 'L2_Sector_Merge_NonFerrous',
            'type': 'merge',
            'file_weights': {
                'L1_Sector_Merge_NonFerrous_CuAl_norm_Position.csv': 2.0,
                'L1_Sector_Merge_NonFerrous_Others_norm_Position.csv': 1.0,
            }
        },

        {
            'output_name': 'L3_Sector_Merge_All',
            'type': 'merge',
            'file_weights': {
                'L2_Sector_Merge_NonFerrous_norm_Position.csv': 25,
                'L1_Sector_Merge_Precious_norm_Position.csv': 5,
                'L1_Sector_Merge_Energy_norm_Position.csv': 25,
                'L1_Sector_Merge_Agriculture_norm_Position.csv': 20,
                'L1_Sector_Merge_Ferrous_norm_Position.csv': 25,
            }
        },

        {
            'output_name': 'L3_Sector_Merge_All_DeltaNeutral',
            'type': 'delta_neutral',
            'file_weights': ['L3_Sector_Merge_All_norm_Position.csv']
        },
        
        {
            'output_name': 'L3_Final',
            'type': 'merge',
            'file_weights': {
                'L3_Sector_Merge_All_norm_Position.csv': 50,
                'L3_Sector_Merge_All_DeltaNeutral_norm_Position.csv': 100,
            }
        },
    ]

    for task in MERGE_TASKS:
        do_merge_and_normalize(
            OUTPUT_NAME=task['output_name'],
            FILE_WEIGHTS=task['file_weights'],
            RESULT_DIR=RESULT_DIR,
            marketDataPath=marketDataPath,
            TASK_TYPE=task.get('type', 'merge')
        )
        
    print("=" * 40)
    print("Running Risk Control on Final Portfolio...")
    last_task = MERGE_TASKS[-1]
    final_output_name = last_task['output_name']
    
    input_file = os.path.join(RESULT_DIR, f"{final_output_name}_norm_PnL.csv")
    output_file = os.path.join(RESULT_DIR, f"{final_output_name}_compound_net_value.csv")
    
    target_volatility = 0.20 # 默认使用20%年化波动计算风险指标
    convert_simple_to_compound(input_file, output_file, start_date='20200101', target_volatility=target_volatility, divisor=100.0)

if __name__ == "__main__":
    main()

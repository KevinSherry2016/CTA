# ==============================================================================
# 策略类型：Unknown
# 策略名称：TrendMomentum_DualMACrossover
# 代表意义：Unknown
# 适用板块：Ferrous
# ==============================================================================

import pandas as pd
import numpy as np
import os

def main():
    # 兼容在不同目录层级运行
    marketDataPath = '../main_contract/' if not os.path.exists('./main_contract/') else './main_contract/'
    infoPath = '../Info.csv' if not os.path.exists('./Info.csv') else './Info.csv'
    FINAL_VOL_WINDOW = 20

    # 1. 过滤品种与划分板块
    info = pd.read_csv(infoPath, encoding='utf-8-sig')
    EXCLUDE_SECTORS = ['stockindex', 'bond', 'other', 'others', 'financial']
    TARGET_SECTORS = ['Ferrous']  # 只需要测试Ferrous板块
    EXCLUDE_SYMBOLS = ['SF.ZCE','SM.ZCE','SS.SHF']  # 增加特定品种过滤

    valid_info = info[~info['sector'].astype(str).str.lower().isin(EXCLUDE_SECTORS)].copy()
    if EXCLUDE_SYMBOLS:
        valid_info = valid_info[~valid_info['ts_code'].isin(EXCLUDE_SYMBOLS)]
    sector_map = valid_info.groupby('sector')['ts_code'].apply(list).to_dict()

    # 2. 提取日度行情数据
    data = {}
    print(f"正在加载 {__file__} 数据...")
    tradingDayList = []
    
    for ts_code in valid_info['ts_code']:
        filepath = os.path.join(marketDataPath, f"{ts_code}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, dtype={'trade_date': str})
            df.set_index('trade_date', inplace=True)
            if len(tradingDayList) == 0:
                tradingDayList = df.index.tolist()
            for col in ['open', 'high', 'low', 'close', 'adj_close', 'adj_open', 'adj_high', 'adj_low', 'vol', 'oi']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            data[ts_code] = df

    N_LIST = [20, 30, 40, 50, 60, 70]

    
    POSITION_SMOOTH_DAYS = 10

    print("开始进行板块回测与参数评估...")
    
    results = [] 
    for FACTOR_MODE in ['RAW', 'STATE_MACHINE']:
        best_positions = {}

        sector_best_pos = {}
        
        for sector, ts_codes in sector_map.items():
            if pd.isna(sector) or str(sector).lower() in EXCLUDE_SECTORS: continue
            if TARGET_SECTORS and sector not in TARGET_SECTORS: continue
                
            valid_symbols = [c for c in ts_codes if c in data]
            if not valid_symbols: continue

            max_metric = -999
            
            for N in N_LIST:

                pnl_sm = []
                turnover_sm = []
                temp_pos_sm = {}
                
                for ts_code in valid_symbols:
                    df = data[ts_code]
                    close = df['adj_close']
                    open_p = df['adj_open']
                    high = df['adj_high']
                    low = df['adj_low']
                    volume = df.get('vol', df.get('Volume', pd.Series(dtype=float)))
                    oi = df.get('oi', pd.Series(dtype=float))
                    daily_ret = close.pct_change(fill_method=None)
                    vol_series = daily_ret.rolling(FINAL_VOL_WINDOW, min_periods=1).std().replace(0, np.nan)

                    short_ma = close.rolling(window=fast_n).mean()
                    long_ma = close.rolling(window=slow_n).mean()
                    raw_sig = (short_ma / long_ma - 1).apply(lambda x: np.sign(x) if pd.notna(x) else np.nan)

                    if FACTOR_MODE == 'RAW':
                        # RAW模式：直接使用相对连续值
                        states = raw_sig.copy().fillna(0)
                    else:
                        # 状态机映射 [-1, 0, 1]
                        states = pd.Series(0.0, index=close.index)
                        states[raw_sig > 0] = 1.0
                        states[raw_sig < 0] = -1.0
                    
                    # 目标仓位等于状态值除以波动率
                    pos_sm = states / vol_series
                    if POSITION_SMOOTH_DAYS > 1:
                        pos_sm = pos_sm.ewm(span=POSITION_SMOOTH_DAYS, adjust=False).mean()
                    
                    temp_pos_sm[ts_code] = pos_sm
                    
                    # PnL Calculation
                    position_yesterday = pos_sm.shift(1).fillna(0)
                    pnl_sm.append(position_yesterday * daily_ret.fillna(0))
                    
                    # Turnover Calculation
                    turnover_sm.append(pos_sm.diff().abs().fillna(0))

                if pnl_sm:
                    df_sm = pd.concat(pnl_sm, axis=1).sum(axis=1) # Daily PnL
                    sharpe = df_sm.mean() / df_sm.std() * np.sqrt(252) if df_sm.std() > 0 else 0
                    
                    pot = 0
                    if turnover_sm:
                        tv_series = pd.concat(turnover_sm, axis=1).sum(axis=1)
                        total_turnover = tv_series.sum()
                        total_pnl_sm = df_sm.sum()
                        if total_turnover > 0:
                            pot = total_pnl_sm / total_turnover * 10000

                    record = {
                        'Mode': FACTOR_MODE,
                        'Sector': sector,
                        'N': N,

                        'Sharpe': round(sharpe, 4),
                        'POT': round(pot, 4)
                    }
                    results.append(record)
                    
                    if sharpe > max_metric:
                        max_metric = sharpe
                        sector_best_pos = temp_pos_sm

            if sector_best_pos: best_positions.update(sector_best_pos)

        # -------- 数据保存与输出 --------
        output_dir = './Result' if os.path.exists('./Result') else '../Result'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if best_positions:
            df_pos_sm = pd.DataFrame(best_positions, index=tradingDayList).fillna(0)
            df_pos_sm = df_pos_sm.reindex(tradingDayList) # Ensure index order
            pos_path = os.path.join(output_dir, f"TrendMomentum_DualMACrossover_{FACTOR_MODE}_Ferrous_Position.csv")
            df_pos_sm.to_csv(pos_path, encoding='utf-8-sig')

    # -------- 数据保存与输出（回测结果） --------
    if results:
        res_df = pd.DataFrame(results)
        raw_res = res_df[res_df['Mode'] == 'RAW'].copy()
        state_res = res_df[res_df['Mode'] == 'STATE_MACHINE'].copy()

        raw_path = os.path.join(output_dir, f"TrendMomentum_DualMACrossover_RAW_Ferrous_BacktestResult.csv")
        state_path = os.path.join(output_dir, f"TrendMomentum_DualMACrossover_STATE_MACHINE_Ferrous_BacktestResult.csv")
        
        raw_res.to_csv(raw_path, encoding='utf-8-sig', index=False)
        state_res.to_csv(state_path, encoding='utf-8-sig', index=False)

if __name__ == "__main__":
    main()

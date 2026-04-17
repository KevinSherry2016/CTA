# ==============================================================================
# 策略类型：微观结构单因子策略
# 策略名称：BuyingSellingPressure
# 代表意义：基于买卖压力(收盘价在日内高低点的位置)的单因子策略。使用状态机模式直接将信号转换为[-1, 1]。
# 适用板块：Ferrous
# ==============================================================================

import pandas as pd
import numpy as np
import os

def main():
    marketDataPath = '../main_contract/' if not os.path.exists('./main_contract/') else './main_contract/'
    infoPath = '../Info.csv' if not os.path.exists('./Info.csv') else './Info.csv'
    FINAL_VOL_WINDOW = 20

    info = pd.read_csv(infoPath, encoding='utf-8-sig')
    EXCLUDE_SECTORS = ['stockindex', 'bond', 'other', 'others', 'financial']
    TARGET_SECTORS = ['Ferrous']  
    EXCLUDE_SYMBOLS = ['SF.ZCE','SM.ZCE','SS.SHF']  

    valid_info = info[~info['sector'].astype(str).str.lower().isin(EXCLUDE_SECTORS)].copy()
    if EXCLUDE_SYMBOLS:
        valid_info = valid_info[~valid_info['ts_code'].isin(EXCLUDE_SYMBOLS)]
    sector_map = valid_info.groupby('sector')['ts_code'].apply(list).to_dict()

    data = {}
    print("正在加载数据...")
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

    N_LIST = [5, 10, 15, 20, 25, 30, 40, 50, 60]
    
    POSITION_SMOOTH_DAYS = 10

    print("开始进行板块回测与参数评估...")
    
    results = [] 
    best_positions = {}

    for sector, ts_codes in sector_map.items():
        if pd.isna(sector) or str(sector).lower() in EXCLUDE_SECTORS: continue
        if TARGET_SECTORS and sector not in TARGET_SECTORS: continue
            
        valid_symbols = [c for c in ts_codes if c in data]
        if not valid_symbols: continue

        best_params = None
        max_metric = -999
        sector_best_pos = {}
        
        for n in N_LIST:
            pnl_sm = []
            turnover_sm = []
            temp_pos_sm = {}
            
            for ts_code in valid_symbols:
                df = data[ts_code]
                close = df['adj_close']
                high = df['adj_high']
                low = df['adj_low']
                
                daily_ret = close.pct_change(fill_method=None)
                vol = daily_ret.rolling(FINAL_VOL_WINDOW, min_periods=1).std().replace(0, np.nan)

                # --- 因子计算逻辑 ---
                pressure = (close - low) / (high - low + 1e-9)
                pressure_ma = pressure.rolling(window=n).mean()
                raw_sig = (pressure_ma - 0.5) * 2
                
                # 状态机映射 (-1 或 1)
                states = pd.Series(np.sign(raw_sig), index=close.index).fillna(0)
                
                # 目标仓位等于状态值除以波动率
                pos_sm = states / vol
                if POSITION_SMOOTH_DAYS > 1:
                    pos_sm = pos_sm.ewm(span=POSITION_SMOOTH_DAYS, adjust=False).mean()
                
                temp_pos_sm[ts_code] = pos_sm
                pnl_sm.append(pos_sm.shift(1) * daily_ret.fillna(0))
                turnover_sm.append(pos_sm.diff().abs().fillna(0))

            if pnl_sm:
                df_sm = pd.concat(pnl_sm, axis=1).sum(axis=1)
                sharpe = df_sm.mean() / df_sm.std() * np.sqrt(252) if df_sm.std() > 0 else 0
                
                tv_sm = pd.concat(turnover_sm, axis=1).sum(axis=1)
                pot = df_sm.sum() / tv_sm.sum() * 10000 if tv_sm.sum() > 0 else 0

                record = {
                    'Sector': sector,
                    'N': n,
                    'Sharpe': round(sharpe, 4),
                    'POT': round(pot, 4)
                }
                print(f"Computed Record: N={n}, Sharpe={sharpe:.4f}")
                results.append(record)
                
                if sharpe > max_metric:
                    max_metric = sharpe
                    best_params = n
                    sector_best_pos = temp_pos_sm

        if best_params is not None:
            if sector_best_pos: best_positions.update(sector_best_pos)

    # -------- 数据保存与输出 --------
    output_dir = './Result' if os.path.exists('./Result') else '../Result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if results:
        res_df = pd.DataFrame(results)
        report_path = os.path.join(output_dir, "BuyingSellingPressure_Ferrous_BacktestResult.csv")
        res_df.to_csv(report_path, encoding='utf-8-sig', index=False)

    if best_positions:
        df_pos_sm = pd.DataFrame(best_positions, index=tradingDayList).fillna(0)
        pos_path = os.path.join(output_dir, "BuyingSellingPressure_Position_Ferrous.csv")
        df_pos_sm.to_csv(pos_path, encoding='utf-8-sig')

if __name__ == "__main__":
    main()
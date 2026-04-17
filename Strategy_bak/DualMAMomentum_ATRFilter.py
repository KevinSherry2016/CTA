# ==============================================================================
# 策略类型：趋势动量复合策略 + ATR波动率过滤与增强（参照提供框架）
# 策略名称：DualMAMomentum_ATRFilter
# 代表意义：在基础的双均线+动量框架（DualMAMomentum）长，引入ATR的短期和长期均线关系来管理仓位：
#          当短期波动率收窄（缩波），减轻头寸仓位；当短期波动率放大（扩波），放大头寸仓位乘数。
# 适用板块：Ferrous
# ==============================================================================

import pandas as pd
import numpy as np
import os
import itertools

def main():
    marketDataPath = '../main_contract/' if not os.path.exists('./main_contract/') else './main_contract/'
    infoPath = '../Info.csv' if not os.path.exists('./Info.csv') else './Info.csv'
    FINAL_VOL_WINDOW = 20

    # 1. 过滤品种与划分板块
    info = pd.read_csv(infoPath, encoding='utf-8-sig')
    EXCLUDE_SECTORS = ['stockindex', 'bond', 'other', 'others', 'financial']
    TARGET_SECTORS = ['Ferrous']
    EXCLUDE_SYMBOLS = ['SF.ZCE','SM.ZCE','SS.SHF']

    valid_info = info[~info['sector'].astype(str).str.lower().isin(EXCLUDE_SECTORS)].copy()
    if EXCLUDE_SYMBOLS:
        valid_info = valid_info[~valid_info['ts_code'].isin(EXCLUDE_SYMBOLS)]
    sector_map = valid_info.groupby('sector')['ts_code'].apply(list).to_dict()

    # 2. 提取日度行情数据
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

    # 参数列表 
    FAST_MA_LIST = [5, 10, 20]
    SLOW_MA_LIST = [20, 40, 60]
    MOM_N_LIST = [3, 5, 10]
    
    # ATR过滤参考系数
    USE_ATR_FILTER = True
    ATR_SHORT = 5
    ATR_LONG = 20
    ATR_SOFT_WEIGHT = 0.5
    ATR_ENHANCE_WEIGHT = 1.5
    WEIGHT_SMOOTH_DAYS = 10
    
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
        
        param_combinations = list(itertools.product(FAST_MA_LIST, SLOW_MA_LIST, MOM_N_LIST))
        
        for fast_ma, slow_ma, mom_n in param_combinations:
            if fast_ma >= slow_ma or mom_n >= fast_ma:
                continue
                
            pnl_sm = []
            turnover_sm = []
            temp_pos_sm = {}
            
            for ts_code in valid_symbols:
                df = data[ts_code]
                close = df['adj_close']
                daily_ret = close.pct_change(fill_method=None)
                
                # 资产本身日后收益率维度的倒数波动率因子
                vol = daily_ret.rolling(FINAL_VOL_WINDOW, min_periods=1).std().replace(0, np.nan)

                # --- 因子计算逻辑 ---
                ma_fast = close.rolling(fast_ma).mean()
                ma_slow = close.rolling(slow_ma).mean()
                t_ma = pd.Series(0, index=close.index)
                t_ma[ma_fast > ma_slow] = 1
                t_ma[ma_fast < ma_slow] = -1
                
                mom_sig = close.pct_change(mom_n).fillna(0)
                t_mom = pd.Series(0, index=close.index)
                t_mom[mom_sig > 0] = 1
                t_mom[mom_sig < 0] = -1
                
                # 原始的基础状态机映射
                states = pd.Series(0.0, index=close.index)
                states[(t_ma == 1) & (t_mom == 1)] = 1.0
                states[(t_ma == 1) & (t_mom == -1)] = 0.5
                states[(t_ma == -1) & (t_mom == -1)] = -1.0
                states[(t_ma == -1) & (t_mom == 1)] = -0.5
                
                # --------------------
                # 引入ATR来管理权重
                weight = pd.Series(1.0, index=close.index)
                if USE_ATR_FILTER:
                    adj_high = df['adj_high']
                    adj_low = df['adj_low']
                    adj_pre_close = df['adj_close'].shift(1)
                    
                    tr1 = adj_high - adj_low
                    tr2 = (adj_high - adj_pre_close).abs()
                    tr3 = (adj_low - adj_pre_close).abs()
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    
                    atr_short = tr.rolling(ATR_SHORT).mean()
                    atr_long = tr.rolling(ATR_LONG).mean()
                    
                    # 当短期波幅小于长波幅（缩波期），减仓至0.5
                    weight.loc[atr_short < atr_long] = ATR_SOFT_WEIGHT
                    # 当短期波幅激增（扩波期爆发），乘数放大至1.5
                    weight.loc[atr_short > atr_long] = ATR_ENHANCE_WEIGHT

                # 对ATR调节的波动率过滤离散权重进行指数平滑
                if WEIGHT_SMOOTH_DAYS > 1:
                    weight = weight.ewm(span=WEIGHT_SMOOTH_DAYS, adjust=False).mean()
                # --------------------
                
                # 目标仓位等于状态值除以日度收益率标准差，并引入ATR管理的仓位乘数
                pos_sm = (states / vol) * weight
                if POSITION_SMOOTH_DAYS > 1:
                    pos_sm = pos_sm.ewm(span=POSITION_SMOOTH_DAYS, adjust=False).mean()
                
                temp_pos_sm[ts_code] = pos_sm
                pnl_sm.append(pos_sm.shift(1) * daily_ret)
                turnover_sm.append(pos_sm.diff().abs().fillna(0))

            if pnl_sm:
                df_sm = pd.concat(pnl_sm, axis=1).sum(axis=1)
                sharpe = df_sm.mean() / df_sm.std() * np.sqrt(252) if df_sm.std() > 0 else 0
                
                tv_sm = pd.concat(turnover_sm, axis=1).sum(axis=1)
                pot = df_sm.sum() / tv_sm.sum() * 10000 if tv_sm.sum() > 0 else 0

                record = {
                    'Sector': sector,
                    'FastMA': fast_ma,
                    'SlowMA': slow_ma,
                    'MomN': mom_n,
                    'Sharpe': round(sharpe, 4),
                    'POT': round(pot, 4)
                }
                results.append(record)
                
                if sharpe > max_metric:
                    max_metric = sharpe
                    best_params = (fast_ma, slow_ma, mom_n)
                    sector_best_pos = temp_pos_sm

        if best_params is not None:
            if sector_best_pos: best_positions.update(sector_best_pos)

    # -------- 数据保存与输出 --------
    output_dir = './Result' if os.path.exists('./Result') else '../Result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if results:
        res_df = pd.DataFrame(results)
        report_path = os.path.join(output_dir, "DualMAMomentum_ATRFilter_Ferrous_BacktestResult.csv")
        res_df.to_csv(report_path, encoding='utf-8-sig', index=False)
        print(f"回测结果报表已保存至: {report_path}")

    if best_positions:
        df_pos_sm = pd.DataFrame(best_positions, index=tradingDayList).fillna(0)
        pos_path = os.path.join(output_dir, f"DualMAMomentum_ATRFilter_Position_Ferrous.csv")
        df_pos_sm.to_csv(pos_path, encoding='utf-8-sig')

if __name__ == "__main__":
    main()
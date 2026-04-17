# ==============================================================================
# 策略类型：趋势动量量仓复合策略
# 策略名称：DualMAMomentumVolOI
# 代表意义：以双均线因子为趋势主逻辑，动量辅助。加入量仓（成交量+持仓量）共振确认。
# 如果成交量和持仓量同时放大说明趋势动能极强，放大仓位；同时缩量则缩小仓位；发生背离则保持基础仓位。同时规避换月期的跳空影响。
# 适用板块：Ferrous
# ==============================================================================

import pandas as pd
import numpy as np
import os
import itertools

def main():
    # 兼容在不同目录层级运行
    marketDataPath = '../main_contract/' if not os.path.exists('./main_contract/') else './main_contract/'
    infoPath = '../Info.csv' if not os.path.exists('./Info.csv') else './Info.csv'
    FINAL_VOL_WINDOW = 20

    # 1. 过滤品种与划分板块
    info = pd.read_csv(infoPath, encoding='utf-8-sig')
    EXCLUDE_SECTORS = ['stockindex', 'bond', 'other', 'others', 'financial']
    TARGET_SECTORS = ['Ferrous']  # 只需要测试Ferrous板块
    EXCLUDE_SYMBOLS = ['SF.ZCE','SM.ZCE','SS.SHF']  # 过滤表现不佳的品种

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
    CONFIRM_MA_LIST = [5, 10, 20] # 量仓指标的移动平均天数
    
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
        
        param_combinations = list(itertools.product(FAST_MA_LIST, SLOW_MA_LIST, MOM_N_LIST, CONFIRM_MA_LIST))
        
        for fast_ma, slow_ma, mom_n, confirm_n in param_combinations:
            if fast_ma >= slow_ma or mom_n >= fast_ma:
                continue
                
            pnl_sm = []
            turnover_sm = []
            temp_pos_sm = {}
            
            for ts_code in valid_symbols:
                df = data[ts_code]
                close = df['adj_close']
                volume = df['vol']
                oi = df['oi']
                daily_ret = close.pct_change(fill_method=None)
                vol_metric = daily_ret.rolling(FINAL_VOL_WINDOW, min_periods=1).std().replace(0, np.nan)

                # --- 因子计算逻辑 ---
                # 1. 主趋势：双均线 
                ma_fast = close.rolling(fast_ma).mean()
                ma_slow = close.rolling(slow_ma).mean()
                t_ma = pd.Series(0, index=close.index)
                t_ma[ma_fast > ma_slow] = 1
                t_ma[ma_fast < ma_slow] = -1
                
                # 2. 辅助趋势：动量因子 N日收益率
                mom_sig = close.pct_change(mom_n).fillna(0)
                t_mom = pd.Series(0, index=close.index)
                t_mom[mom_sig > 0] = 1
                t_mom[mom_sig < 0] = -1
                
                # 3. 趋势确认：量仓共振
                vol_ma = volume.rolling(confirm_n).mean()
                oi_ma = oi.rolling(confirm_n).mean()
                
                t_vol = volume > vol_ma
                t_oi = oi > oi_ma
                
                # 换月掩码：如果在计算窗口（confirm_n）内发生过换月，屏蔽量仓信号
                if 'mapping_ts_code' in df.columns:
                    is_roll = df['mapping_ts_code'] != df['mapping_ts_code'].shift(1)
                    is_roll.iloc[0] = False # 首日不算换月
                    is_roll_period = is_roll.rolling(confirm_n, min_periods=1).max() > 0
                else:
                    is_roll_period = pd.Series(False, index=close.index)
                
                # 状态机映射
                states = pd.Series(0.0, index=close.index)
                
                # A. 基础双均线与动量状态
                states[(t_ma == 1) & (t_mom == 1)] = 1.0
                states[(t_ma == 1) & (t_mom == -1)] = 0.5
                states[(t_ma == -1) & (t_mom == -1)] = -1.0
                states[(t_ma == -1) & (t_mom == 1)] = -0.5
                
                # B. 根据量仓同向调整：
                # (成交量均线上方 & 持仓量均线上方) -> 齐升放大（*1.2）
                both_up = t_vol & t_oi
                # (成交量均线下方 & 持仓量均线下方) -> 齐跌缩小（*0.8）
                both_down = (~t_vol) & (~t_oi)
                
                # 一升一降（背离）的情况下，保持原状态（*1.0）不处理
                states[both_up] = states[both_up] * 1.2
                states[both_down] = states[both_down] * 0.8
                
                # C. 换月期间处理：保持之前的仓位情况不变
                states.loc[is_roll_period] = np.nan
                states = states.ffill().fillna(0.0)
                
                # 目标仓位等于状态值除以波动率
                pos_sm = states / vol_metric
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
                    'ConfirmMA': confirm_n,
                    'Sharpe': round(sharpe, 4),
                    'POT': round(pot, 4)
                }
                results.append(record)
                
                if sharpe > max_metric:
                    max_metric = sharpe
                    best_params = (fast_ma, slow_ma, mom_n, confirm_n)
                    sector_best_pos = temp_pos_sm

        if best_params is not None:
            if sector_best_pos: best_positions.update(sector_best_pos)

    # -------- 数据保存与输出 --------
    output_dir = './Result' if os.path.exists('./Result') else '../Result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if results:
        res_df = pd.DataFrame(results)
        report_path = os.path.join(output_dir, "DualMAMomentumVolOI_Ferrous_BacktestResult.csv")
        res_df.to_csv(report_path, encoding='utf-8-sig', index=False)
        print(f"回测结果报表已保存至: {report_path}")
        print(res_df)

    if best_positions:
        df_pos_sm = pd.DataFrame(best_positions, index=tradingDayList).fillna(0)
        pos_path = os.path.join(output_dir, f"DualMAMomentumVolOI_Position_Ferrous.csv")
        df_pos_sm.to_csv(pos_path, encoding='utf-8-sig')
        print(f"最优仓位文件已保存至: {pos_path}")

if __name__ == "__main__":
    main()
# ==============================================================================
# 策略类型：趋势量仓综合策略
# 策略名称：DualMAVolOI
# 代表意义：在纯双均线的基础上，增加持仓量和成交量的共振确认。
# 快线超过慢线开多，跌破开空。开仓后如果成交量持仓量双双放大则加仓（*1.2），双双缩小则减仓（*0.8）。
# 同时加入针对主力合约换月时的跳空数据屏蔽与仓位冻结逻辑。
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
    EXCLUDE_SYMBOLS = ['SF.ZCE','SM.ZCE','SS.SHF']  # 过滤换月/流动性异常品种

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
    CONFIRM_MA_LIST = [5, 10, 20] # 量仓指标的移动平均天数
    
    POSITION_SMOOTH_DAYS = 1

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
        
        param_combinations = list(itertools.product(FAST_MA_LIST, SLOW_MA_LIST, CONFIRM_MA_LIST))
        
        for fast_ma, slow_ma, confirm_n in param_combinations:
            if fast_ma >= slow_ma:
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
                # 主趋势：双均线 
                ma_fast = close.rolling(fast_ma).mean()
                ma_slow = close.rolling(slow_ma).mean()
                
                # 状态机映射 [-1, 0, 1]
                # 快线大于慢线开多，小于开空
                states = pd.Series(0.0, index=close.index)
                states[ma_fast > ma_slow] = 1.0
                states[ma_fast < ma_slow] = -1.0
                
                # --- 量仓共振逻辑 ---
                vol_ma = volume.rolling(confirm_n).mean()
                oi_ma = oi.rolling(confirm_n).mean()
                
                t_vol = volume > vol_ma
                t_oi = oi > oi_ma
                
                # 量仓同向上升，放大状态量1.2倍
                both_up = t_vol & t_oi
                # 量仓同向下降，缩小状态量0.8倍
                both_down = (~t_vol) & (~t_oi)
                
                states[both_up] = states[both_up] * 1.2
                states[both_down] = states[both_down] * 0.8
                
                # --- 换月期处理 ---
                # 构建换月掩码：如果在计算窗口内发生过主力合的切换，屏蔽量仓突变引起的逻辑异常，保持持仓比例
                if 'mapping_ts_code' in df.columns:
                    # 判断当天是否为换月日
                    is_roll = df['mapping_ts_code'] != df['mapping_ts_code'].shift(1)
                    is_roll.iloc[0] = False # 首日默认非换月
                    # 在计算窗口（比如confirm_n）内发生过换月，均被标记为清洗期
                    is_roll_period = is_roll.rolling(confirm_n, min_periods=1).max() > 0
                else:
                    is_roll_period = pd.Series(False, index=close.index)
                
                # 如果处于换月清洗期，强制产生NaN并向下填充（冻结上一次的有效状态）
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
                    'ConfirmMA': confirm_n,
                    'Sharpe': round(sharpe, 4),
                    'POT': round(pot, 4)
                }
                results.append(record)
                
                if sharpe > max_metric:
                    max_metric = sharpe
                    best_params = (fast_ma, slow_ma, confirm_n)
                    sector_best_pos = temp_pos_sm

        if best_params is not None:
            if sector_best_pos: best_positions.update(sector_best_pos)

    # -------- 数据保存与输出 --------
    output_dir = './Result' if os.path.exists('./Result') else '../Result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if results:
        res_df = pd.DataFrame(results)
        report_path = os.path.join(output_dir, "DualMAVolOI_Ferrous_BacktestResult.csv")
        res_df.to_csv(report_path, encoding='utf-8-sig', index=False)
        print(f"回测结果报表已保存至: {report_path}")
        print(res_df)

    if best_positions:
        df_pos_sm = pd.DataFrame(best_positions, index=tradingDayList).fillna(0)
        pos_path = os.path.join(output_dir, f"DualMAVolOI_Position_Ferrous.csv")

if __name__ == "__main__":
    main()
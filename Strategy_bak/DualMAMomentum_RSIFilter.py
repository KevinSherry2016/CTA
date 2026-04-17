# ==============================================================================
# 策略类型：趋势动量复合策略 + RSI 相对强弱指标过滤与增强
# 策略名称：DualMAMomentum_RSIFilter
# 代表意义：在基础的双均线+动量框架长，引入RSI共振确认趋势
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

    info = pd.read_csv(infoPath, encoding='utf-8-sig')
    EXCLUDE_SECTORS = ['stockindex', 'bond', 'other', 'others', 'financial']
    TARGET_SECTORS = ['Ferrous']
    EXCLUDE_SYMBOLS = ['SF.ZCE','SM.ZCE','SS.SHF']

    valid_info = info[~info['sector'].astype(str).str.lower().isin(EXCLUDE_SECTORS)].copy()
    if EXCLUDE_SYMBOLS:
        valid_info = valid_info[~valid_info['ts_code'].isin(EXCLUDE_SYMBOLS)]
    sector_map = valid_info.groupby('sector')['ts_code'].apply(list).to_dict()

    data = {}
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

    FAST_MA_LIST = [5, 10, 20]
    SLOW_MA_LIST = [20, 40, 60]
    MOM_N_LIST = [3, 5, 10]
    
    USE_RSI_FILTER = True
    RSI_PERIOD = 14
    RSI_SOFT_WEIGHT = 0.5
    RSI_ENHANCE_WEIGHT = 1.5
    WEIGHT_SMOOTH_DAYS = 10
    POSITION_SMOOTH_DAYS = 10
    
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
            if fast_ma >= slow_ma or mom_n >= fast_ma: continue
                
            pnl_sm = []
            turnover_sm = []
            temp_pos_sm = {}
            
            for ts_code in valid_symbols:
                df = data[ts_code]
                close = df['adj_close']
                daily_ret = close.pct_change(fill_method=None)
                vol = daily_ret.rolling(FINAL_VOL_WINDOW, min_periods=1).std().replace(0, np.nan)

                ma_fast = close.rolling(fast_ma).mean()
                ma_slow = close.rolling(slow_ma).mean()
                t_ma = pd.Series(0, index=close.index)
                t_ma[ma_fast > ma_slow] = 1
                t_ma[ma_fast < ma_slow] = -1
                
                mom_sig = close.pct_change(mom_n).fillna(0)
                t_mom = pd.Series(0, index=close.index)
                t_mom[mom_sig > 0] = 1
                t_mom[mom_sig < 0] = -1
                
                states = pd.Series(0.0, index=close.index)
                states[(t_ma == 1) & (t_mom == 1)] = 1.0
                states[(t_ma == 1) & (t_mom == -1)] = 0.5
                states[(t_ma == -1) & (t_mom == -1)] = -1.0
                states[(t_ma == -1) & (t_mom == 1)] = -0.5
                
                # RSI过滤逻辑 (同向共振增强)
                weight = pd.Series(1.0, index=close.index)
                if USE_RSI_FILTER:
                    delta = close.diff()
                    up = delta.clip(lower=0)
                    down = -1 * delta.clip(upper=0)
                    ema_up = up.ewm(com=RSI_PERIOD-1, adjust=False).mean()
                    ema_down = down.ewm(com=RSI_PERIOD-1, adjust=False).mean()
                    rs = ema_up / ema_down
                    rsi = 100 - (100 / (1 + rs))
                    
                    # 共振确认，RSI>50多增强，RSI<50空增强
                    weight.loc[(states > 0) & (rsi > 50)] = RSI_ENHANCE_WEIGHT
                    weight.loc[(states < 0) & (rsi < 50)] = RSI_ENHANCE_WEIGHT
                    weight.loc[(states > 0) & (rsi <= 50)] = RSI_SOFT_WEIGHT
                    weight.loc[(states < 0) & (rsi >= 50)] = RSI_SOFT_WEIGHT

                if WEIGHT_SMOOTH_DAYS > 1:
                    weight = weight.ewm(span=WEIGHT_SMOOTH_DAYS, adjust=False).mean()
                
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
                results.append({'Sector': sector, 'FastMA': fast_ma, 'SlowMA': slow_ma, 'MomN': mom_n, 'Sharpe': round(sharpe, 4), 'POT': round(pot, 4)})
                
                if sharpe > max_metric:
                    max_metric = sharpe
                    best_params = (fast_ma, slow_ma, mom_n)
                    sector_best_pos = temp_pos_sm

        if best_params is not None:
            if sector_best_pos: best_positions.update(sector_best_pos)

    output_dir = './Result' if os.path.exists('./Result') else '../Result'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
        
    if results:
        pd.DataFrame(results).to_csv(os.path.join(output_dir, "DualMAMomentum_RSIFilter_BacktestResult.csv"), encoding='utf-8-sig', index=False)
    if best_positions:
        pd.DataFrame(best_positions, index=tradingDayList).fillna(0).to_csv(os.path.join(output_dir, f"DualMAMomentum_RSIFilter_Position.csv"), encoding='utf-8-sig')

if __name__ == "__main__":
    main()
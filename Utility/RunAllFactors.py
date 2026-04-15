import os
import numpy as np
import pandas as pd

MARKET_DATA_PATH = './main_contract/'
INFO_PATH = './Info.csv'
FINAL_VOL_WINDOW = 20

info = pd.read_csv(INFO_PATH, encoding='utf-8-sig')
sector_map = info.groupby('sector')['ts_code'].apply(list).to_dict()
EXCLUDE_SECTORS = ['StockIndex', 'Bond', 'Other', 'Others']

data = {}
for ts_code in info['ts_code']:
    fp = os.path.join(MARKET_DATA_PATH, f'{ts_code}.csv')
    if not os.path.exists(fp): continue
    df = pd.read_csv(fp, dtype={'trade_date': str}).set_index('trade_date')
    for col in ['open', 'high', 'low', 'close', 'adj_close', 'adj_open', 'adj_high', 'adj_low', 'vol', 'oi']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    data[ts_code] = df

def rolling_zscore(series, window=60):
    min_p = max(window // 2, 1)
    mu = series.rolling(window, min_periods=min_p).mean()
    sigma = series.rolling(window, min_periods=min_p).std().replace(0, np.nan)
    return (series - mu) / sigma

def MACD_signal(close, N):
    short = max(3, int(N/2))
    dif = close.ewm(span=short).mean() - close.ewm(span=N).mean()
    dea = dif.ewm(span=max(3, int(short/2))).mean()
    return (dif - dea) * 2

def RSI_signal(close, N):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=N, min_periods=max(1, N//2)).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=N, min_periods=max(1, N//2)).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return (rsi - 50) / 100

def get_signal(df, factor_name, N):
    close = df['adj_close']
    open_ = df['adj_open']
    high = df['adj_high']
    low = df['adj_low']
    vol = df.get('vol', pd.Series(1, index=close.index))
    oi = df.get('oi', pd.Series(1, index=close.index))
    pre_close = close.shift(1)
    
    # 1. 收益率因子
    if factor_name == 'TimeSeriesMomentum': return close.pct_change(N).fillna(0)
    # 2.价格均线偏离率
    elif factor_name == 'MovingAverageBias': return (close / close.rolling(N).mean() - 1).fillna(0)
    # 3. 双均线交叉
    elif factor_name == 'DualMACrossover': return (close.rolling(max(1, int(N/2))).mean() / close.rolling(N).mean() - 1).fillna(0)
    # 4. MACD
    elif factor_name == 'MACD': return MACD_signal(close, N).fillna(0)
    # 5. 唐奇安通道位置
    elif factor_name == 'DonchianChannel':
        c_min = low.rolling(N).min()
        c_max = high.rolling(N).max()
        return ((close - c_min) / (c_max - c_min + 1e-9) - 0.5).fillna(0)
    # 6. 布林带突破
    elif factor_name == 'BollingerBands':
        mu = close.rolling(N).mean()
        std = close.rolling(N).std().replace(0, 1e-9)
        return ((close - mu) / std).fillna(0)
    # 7. 相对强弱指数RSI
    elif factor_name == 'RSI': return RSI_signal(close, N).fillna(0)
    
    # 波动率与风险类
    # 8. 真实波动幅度
    elif factor_name == 'ATR':
        tr1 = high - low
        tr2 = (high - pre_close).abs()
        tr3 = (low - pre_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_ratio = tr.rolling(N).mean() / (close+1e-9)
        return -rolling_zscore(atr_ratio, 50).fillna(0)
    # 9. 历史收益率波动率
    elif factor_name == 'HistoricalVolatility': 
        hv = close.pct_change().rolling(N).std()
        return -rolling_zscore(hv, 50).fillna(0)
    # 10. 日内振幅
    elif factor_name == 'IntradayAmplitude': 
        amp = (high - low) / (open_ + 1e-9)
        return -rolling_zscore(amp.rolling(N).mean(), 50).fillna(0)
    # 11. 上下行波动率倾向
    elif factor_name == 'DownsideUpsideVolatility':
        ret = close.pct_change()
        up_v = ret.where(ret > 0, 0).rolling(N).std()
        dn_v = ret.where(ret < 0, 0).rolling(N).std()
        return ((up_v - dn_v) / (up_v + dn_v + 1e-9)).fillna(0)
        
    # 成交量与持仓量类
    # 12. 成交量/持仓量动量
    elif factor_name == 'VolumeMomentum': return (vol / (vol.rolling(N).mean() + 1e-9) - 1).fillna(0)
    # 13. 量价相关性
    elif factor_name == 'PriceVolumeCorrelation': return close.pct_change().rolling(N).corr(vol.pct_change()).fillna(0)
    # 14. 能量潮指标
    elif factor_name == 'OBV':
        obv = (np.sign(close.diff()) * vol).cumsum()
        return (obv / (obv.rolling(N).mean() + 1e-9) - 1).fillna(0)
    # 15. 持仓量变化率
    elif factor_name == 'OpenInterestROC': return (oi / oi.shift(N) - 1).fillna(0)
    
    # 截面与非对称性因子
    # 16. 收益率偏度
    elif factor_name == 'Skewness': return close.pct_change().rolling(N).skew().fillna(0)
    # 17. 收益率峰度
    elif factor_name == 'Kurtosis': return -close.pct_change().rolling(N).kurt().fillna(0)
    # 18. 隔夜与日内收益率差异
    elif factor_name == 'OvernightVsIntraday':
        intraday = close / (open_ + 1e-9) - 1
        overnight = open_ / (pre_close + 1e-9) - 1
        return (intraday - overnight).rolling(N).mean().fillna(0)
        
    # 微观结构演化
    # 19. amihud 缺乏流动性指标
    elif factor_name == 'AmihudIlliquidity': 
        amihud = close.pct_change().abs() / (vol + 1e-9)
        return -rolling_zscore(amihud, N).fillna(0)
    # 20. 买卖压力
    elif factor_name == 'BuyingSellingPressure':
        return (((close - low) / (high - low + 1e-9) - 0.5) * 2).rolling(N).mean().fillna(0)
        
    return pd.Series(0, index=close.index)

FACTORS = [
    'TimeSeriesMomentum', 'MovingAverageBias', 'DualMACrossover', 'MACD', 'DonchianChannel', 'BollingerBands', 'RSI',
    'ATR', 'HistoricalVolatility', 'IntradayAmplitude', 'DownsideUpsideVolatility',
    'VolumeMomentum', 'PriceVolumeCorrelation', 'OBV', 'OpenInterestROC',
    'Skewness', 'Kurtosis', 'OvernightVsIntraday',
    'AmihudIlliquidity', 'BuyingSellingPressure'
]

N_LIST = [5, 10, 20, 40, 60]

results = []

for factor in FACTORS:
    print(f"Testing factor: {factor}")
    for sector, ts_codes in sector_map.items():
        if sector in EXCLUDE_SECTORS: continue
        valid_symbols = [c for c in ts_codes if c in data]
        if not valid_symbols: continue
            
        best_n = None
        best_sharpes = {'No zscore': -999, 'zscore': -999, 'State Machine': -999}
        max_metric = -999
        
        for N in N_LIST:
            pnl_noz, pnl_z, pnl_sm = [], [], []
            for ts_code in valid_symbols:
                df = data[ts_code]
                daily_ret = df['adj_close'].pct_change(fill_method=None)
                vol = daily_ret.rolling(FINAL_VOL_WINDOW, min_periods=1).std().replace(0, np.nan)
                
                raw_sig = get_signal(df, factor, N)
                
                pos_noz = raw_sig / vol
                z_sig = rolling_zscore(raw_sig, 60)
                pos_z = z_sig / vol
                sm_sig = np.sign(raw_sig)
                pos_sm = sm_sig / vol
                
                pnl_noz.append(pos_noz.shift(1) * daily_ret)
                pnl_z.append(pos_z.shift(1) * daily_ret)
                pnl_sm.append(pos_sm.shift(1) * daily_ret)
                
            if pnl_noz:
                df_noz = pd.concat(pnl_noz, axis=1).sum(axis=1)
                df_z = pd.concat(pnl_z, axis=1).sum(axis=1)
                df_sm = pd.concat(pnl_sm, axis=1).sum(axis=1)
                
                sr_noz = df_noz.mean() / df_noz.std() * np.sqrt(252) if df_noz.std() > 0 else 0
                sr_z = df_z.mean() / df_z.std() * np.sqrt(252) if df_z.std() > 0 else 0
                sr_sm = df_sm.mean() / df_sm.std() * np.sqrt(252) if df_sm.std() > 0 else 0
                
                # Check parameter robustness (max parameter using 'No zscore' sr_noz, or default)
                # To be fair we can evaluate 'State Machine' or 'zscore' too, but let's stick to max(sr_noz).
                if sr_noz > max_metric:
                    max_metric = sr_noz
                    best_n = N
                    best_sharpes = {'No zscore': sr_noz, 'zscore': sr_z, 'State Machine': sr_sm}
                
        if best_n is not None:
            results.append({
                'factor': factor,
                'sector': sector,
                'zscore': round(best_sharpes['zscore'], 4),
                'No zscore': round(best_sharpes['No zscore'], 4),
                'State Machine': round(best_sharpes['State Machine'], 4),
                'parameters': f'N={best_n}'
            })

res_df = pd.DataFrame(results)
res_df.to_csv('AllFactors_Result.csv', index=False)
print(res_df.head(20))

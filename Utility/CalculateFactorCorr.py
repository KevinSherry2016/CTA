import os
import numpy as np
import pandas as pd

MARKET_DATA_PATH = './main_contract/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Result/'

info = pd.read_csv(INFO_PATH, encoding='utf-8-sig')
sector_map = info.groupby('sector')['ts_code'].apply(list).to_dict()
EXCLUDE_SECTORS = ['StockIndex', 'Bond', 'Other', 'Others']

# Load data
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

def get_signal(df, factor_name, N=20):
    close = df['adj_close']
    open_ = df['adj_open']
    high = df['adj_high']
    low = df['adj_low']
    vol = df.get('vol', pd.Series(1, index=close.index))
    oi = df.get('oi', pd.Series(1, index=close.index))
    pre_close = close.shift(1)
    
    if factor_name == 'TimeSeriesMomentum': return close.pct_change(N).fillna(0)
    elif factor_name == 'MovingAverageBias': return (close / close.rolling(N).mean() - 1).fillna(0)
    elif factor_name == 'DualMACrossover': return (close.rolling(max(1, int(N/2))).mean() / close.rolling(N).mean() - 1).fillna(0)
    elif factor_name == 'MACD': return MACD_signal(close, N).fillna(0)
    elif factor_name == 'DonchianChannel':
        c_min = low.rolling(N).min()
        c_max = high.rolling(N).max()
        return ((close - c_min) / (c_max - c_min + 1e-9) - 0.5).fillna(0)
    elif factor_name == 'BollingerBands':
        mu = close.rolling(N).mean()
        std = close.rolling(N).std().replace(0, 1e-9)
        return ((close - mu) / std).fillna(0)
    elif factor_name == 'RSI': return RSI_signal(close, N).fillna(0)
    elif factor_name == 'ATR':
        tr1 = high - low
        tr2 = (high - pre_close).abs()
        tr3 = (low - pre_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_ratio = tr.rolling(N).mean() / (close+1e-9)
        return -rolling_zscore(atr_ratio, 50).fillna(0)
    elif factor_name == 'HistoricalVolatility': 
        hv = close.pct_change().rolling(N).std()
        return -rolling_zscore(hv, 50).fillna(0)
    elif factor_name == 'IntradayAmplitude': 
        amp = (high - low) / (open_ + 1e-9)
        return -rolling_zscore(amp.rolling(N).mean(), 50).fillna(0)
    elif factor_name == 'DownsideUpsideVolatility':
        ret = close.pct_change()
        up_v = ret.where(ret > 0, 0).rolling(N).std()
        dn_v = ret.where(ret < 0, 0).rolling(N).std()
        return ((up_v - dn_v) / (up_v + dn_v + 1e-9)).fillna(0)
    elif factor_name == 'VolumeMomentum': return (vol / (vol.rolling(N).mean() + 1e-9) - 1).fillna(0)
    elif factor_name == 'PriceVolumeCorrelation': return close.pct_change().rolling(N).corr(vol.pct_change()).fillna(0)
    elif factor_name == 'OBV':
        obv = (np.sign(close.diff()) * vol).cumsum()
        return (obv / (obv.rolling(N).mean() + 1e-9) - 1).fillna(0)
    elif factor_name == 'OpenInterestROC': return (oi / oi.shift(N) - 1).fillna(0)
    elif factor_name == 'Skewness': return close.pct_change().rolling(N).skew().fillna(0)
    elif factor_name == 'Kurtosis': return -close.pct_change().rolling(N).kurt().fillna(0)
    elif factor_name == 'OvernightVsIntraday':
        intraday = close / (open_ + 1e-9) - 1
        overnight = open_ / (pre_close + 1e-9) - 1
        return (intraday - overnight).rolling(N).mean().fillna(0)
    elif factor_name == 'AmihudIlliquidity': 
        amihud = close.pct_change().abs() / (vol + 1e-9)
        return -rolling_zscore(amihud, N).fillna(0)
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

os.makedirs(OUTPUT_DIR, exist_ok=True)
N = 20 # Using standard param 20 for baseline correlation
FINAL_VOL_WINDOW = 20

for sector, ts_codes in sector_map.items():
    if sector in EXCLUDE_SECTORS: continue
    valid_symbols = [c for c in ts_codes if c in data]
    if not valid_symbols: continue

    print(f"Calculating correlation for sector: {sector}")
    factor_pnls = {factor: [] for factor in FACTORS}

    for ts_code in valid_symbols:
        df = data[ts_code]
        daily_ret = df['adj_close'].pct_change(fill_method=None)
        vol = daily_ret.rolling(FINAL_VOL_WINDOW, min_periods=1).std().replace(0, np.nan)
        
        for factor in FACTORS:
            raw_sig = get_signal(df, factor, N)
            pos = raw_sig / vol
            pnl = pos.shift(1) * daily_ret
            factor_pnls[factor].append(pnl)
        
    if factor_pnls[FACTORS[0]]:
        sector_pnl_df = pd.DataFrame({
            factor: pd.concat(pnls, axis=1).sum(axis=1)
            for factor, pnls in factor_pnls.items()
        })
        corr_matrix = sector_pnl_df.corr(method='pearson')
        corr_matrix.to_csv(os.path.join(OUTPUT_DIR, f'Factor_Correlation_{sector}.csv'))
        print(f"Saved {sector} correlation matrix")

print('Finished calculating correlations.')

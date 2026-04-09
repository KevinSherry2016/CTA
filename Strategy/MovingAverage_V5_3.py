"""
MovingAverage_V5_3  —  信号: price_minus_slow_ma_over_atr
    信号 = (收盘价 - slow_ma) / ATR
"""
import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './main_contract/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Result'
FINAL_VOL_WINDOW = 20
SECTOR_FILTER = None

SIGNAL_DEF = 'price_minus_slow_ma_over_atr'

SLOW_WINDOW_LIST    = [20, 40, 60, 80, 100, 120]
ATR_WINDOW_LIST     = [15]
Z_SCORE_WINDOW_LIST = [60]
USE_ZSCORE          = True   # False 则跳过 z-score，直接用原始信号
SMOOTH_T_LIST       = [1]
SIGNAL_MODE_LIST    = ['trend']

# fast_w 对 p-sma/atr 本身无影响（信号不用 fast_ma），
# 但保留以便文件名与其他 V5 系列一致，固定为 5。
FAST_WINDOW = 5


# ── 单品种计算函数 ─────────────────────────────────────────────────────────────

def calc_atr(df, window):
    high = df['adj_high']
    low  = df['adj_low']
    close = df['adj_close']
    prev_close = close.shift(1)
    tr = np.maximum(high - low,
                    np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return tr.rolling(window).mean().replace(0, np.nan)


def calc_strength(df, slow_w, atr_w):
    close = df['adj_close']
    slow_ma = close.rolling(slow_w).mean()
    atr = calc_atr(df, atr_w)
    return (close - slow_ma) / atr


def rolling_zscore(series, window):
    min_p = max(window // 2, 1)
    mu    = series.rolling(window, min_periods=min_p).mean()
    sigma = series.rolling(window, min_periods=min_p).std().replace(0, np.nan)
    return (series - mu) / sigma


def calc_ret_vol(df, window):
    ret = df['adj_close'].pct_change(fill_method=None)
    return ret.rolling(window, min_periods=1).std().replace(0, np.nan)


# ── 加载数据 ───────────────────────────────────────────────────────────────────
info = pd.read_csv(INFO_PATH, encoding='utf-8-sig')
sector_map = info.groupby('sector')['ts_code'].apply(list).to_dict()

data = {}
print('正在加载数据')
for ts_code in info['ts_code']:
    fp = os.path.join(MARKET_DATA_PATH, f'{ts_code}.csv')
    if not os.path.exists(fp):
        print(f'文件不存在: {fp}')
        continue
    df = pd.read_csv(fp, dtype={'trade_date': str}).set_index('trade_date')
    df['adj_close'] = pd.to_numeric(df['adj_close'], errors='coerce')
    df['adj_high']  = pd.to_numeric(df['adj_high'],  errors='coerce')
    df['adj_low']   = pd.to_numeric(df['adj_low'],   errors='coerce')
    data[ts_code] = df

trading_days = pd.read_csv(
    os.path.join(MARKET_DATA_PATH, 'CU.SHF.csv'), dtype={'trade_date': str}
)['trade_date'].tolist()

sector_map = {s: [c for c in codes if c in data] for s, codes in sector_map.items()}
sector_map = {s: codes for s, codes in sector_map.items() if codes}
sector_map['All'] = sorted(data.keys())
if SECTOR_FILTER:
    sector_map = {s: v for s, v in sector_map.items() if s in SECTOR_FILTER}

# ── 主循环 ─────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'涉及 sector: {list(sector_map.keys())}')

for sector, ts_codes in sector_map.items():
    print(f'\n===== Sector: {sector} ({len(ts_codes)} 个品种) =====')

    for atr_w in ATR_WINDOW_LIST:
        for slow_w in SLOW_WINDOW_LIST:

            strength_dict = {}
            ret_vol_dict  = {}
            for ts_code in ts_codes:
                strength_dict[ts_code] = calc_strength(data[ts_code], slow_w, atr_w)
                ret_vol_dict[ts_code]  = calc_ret_vol(data[ts_code], FINAL_VOL_WINDOW)

            zw_values = Z_SCORE_WINDOW_LIST if USE_ZSCORE else [0]
            for zw in zw_values:
                if USE_ZSCORE:
                    zscore_dict = {}
                    for ts_code in ts_codes:
                        zscore_dict[ts_code] = rolling_zscore(strength_dict[ts_code], zw)
                else:
                    zscore_dict = strength_dict

                for smooth_t in SMOOTH_T_LIST:
                    for mode in SIGNAL_MODE_LIST:
                        print(f'  {SIGNAL_DEF} S={slow_w} ATR={atr_w} ZW={zw if USE_ZSCORE else "noZ"} T={smooth_t} {mode}')

                        smoothed_dict = {}
                        for ts_code in ts_codes:
                            s = zscore_dict[ts_code].copy()
                            if mode == 'mean_reversion':
                                s = -s
                            if smooth_t > 1:
                                s = s.rolling(smooth_t, min_periods=1).mean()
                            smoothed_dict[ts_code] = s

                        rows = []
                        for trade_date in trading_days:
                            daily_signal = {}
                            for ts_code in ts_codes:
                                if trade_date in smoothed_dict[ts_code].index:
                                    pos = smoothed_dict[ts_code].loc[trade_date]
                                else:
                                    pos = np.nan
                                if trade_date in ret_vol_dict[ts_code].index:
                                    rv = ret_vol_dict[ts_code].loc[trade_date]
                                else:
                                    rv = np.nan
                                if pd.notna(pos) and pd.notna(rv) and rv != 0:
                                    daily_signal[ts_code] = pos / rv
                                else:
                                    daily_signal[ts_code] = 0.0
                            rows.append(daily_signal)

                        signals = pd.DataFrame(rows, index=trading_days)

                        zw_tag = f'ZW_{zw}_' if USE_ZSCORE else 'noZ_'
                        name = (
                            f'MovingAverageV5_3_{sector}_{SIGNAL_DEF}_'
                            f'F_{FAST_WINDOW}_S_{slow_w}_ATR_{atr_w}_'
                            f'{zw_tag}T_{smooth_t}_VOL_{FINAL_VOL_WINDOW}_{mode}.csv'
                        )
                        signals.to_csv(os.path.join(OUTPUT_DIR, name), encoding='utf-8-sig')

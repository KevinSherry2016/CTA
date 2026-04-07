import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './main_contract/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Result'
FINAL_VOL_WINDOW = 20
SECTOR_FILTER = None

# ── 参数网格 ───────────────────────────────────────────────────────────────────
SIGNAL_DEF_LIST = [
    'fast_slow_gap_over_vol',
    'fast_slow_gap_over_atr',
    'price_minus_slow_ma_over_atr',
    'slow_ma_slope_over_atr',
]
FAST_WINDOW_LIST = [5, 10, 15, 20, 25, 30]
SLOW_WINDOW_LIST = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
ATR_WINDOW_LIST = [10, 15, 20]
SLOPE_LOOKBACK_LIST = [3, 5, 10]
Z_SCORE_WINDOW_LIST = [60, 120, 250]
SMOOTH_T_LIST = [1, 5, 10]
SIGNAL_MODE_LIST = ['trend']


# ── 单品种计算函数 ─────────────────────────────────────────────────────────────

def calc_atr(df, window):
    """计算单品种 ATR。"""
    high = df['adj_high']
    low = df['adj_low']
    close = df['adj_close']
    prev_close = close.shift(1)
    tr = np.maximum(high - low,
                    np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return tr.rolling(window).mean().replace(0, np.nan)


def calc_strength(df, sig_def, fast_w, slow_w, atr_w, slp_k):
    """计算单品种趋势强度序列。"""
    close = df['adj_close']
    fast_ma = close.rolling(fast_w).mean()
    slow_ma = close.rolling(slow_w).mean()

    if sig_def == 'fast_slow_gap_over_vol':
        vol = close.rolling(slow_w).std().replace(0, np.nan)
        return (fast_ma - slow_ma) / vol

    atr = calc_atr(df, atr_w)
    if sig_def == 'fast_slow_gap_over_atr':
        return (fast_ma - slow_ma) / atr
    if sig_def == 'price_minus_slow_ma_over_atr':
        return (close - slow_ma) / atr
    if sig_def == 'slow_ma_slope_over_atr':
        return (slow_ma - slow_ma.shift(slp_k)) / atr

    raise ValueError(f'unsupported signal_def: {sig_def}')


def rolling_zscore(series, window):
    """滚动 z-score 标准化。"""
    min_p = max(window // 2, 1)
    mu = series.rolling(window, min_periods=min_p).mean()
    sigma = series.rolling(window, min_periods=min_p).std().replace(0, np.nan)
    return (series - mu) / sigma


def calc_ret_vol(df, window):
    """计算单品种收益波动率。"""
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
    df['adj_high'] = pd.to_numeric(df['adj_high'], errors='coerce')
    df['adj_low'] = pd.to_numeric(df['adj_low'], errors='coerce')
    data[ts_code] = df

# 交易日列表
trading_days = pd.read_csv(
    os.path.join(MARKET_DATA_PATH, 'CU.SHF.csv'), dtype={'trade_date': str}
)['trade_date'].tolist()

# 按已加载品种过滤 sector_map
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

    for sig_def in SIGNAL_DEF_LIST:
        for slow_w in SLOW_WINDOW_LIST:
            for fast_w in FAST_WINDOW_LIST:
                if fast_w >= slow_w:
                    continue

                atr_ws = ATR_WINDOW_LIST if 'over_atr' in sig_def else [ATR_WINDOW_LIST[0]]
                slp_ks = SLOPE_LOOKBACK_LIST if 'slope' in sig_def else [SLOPE_LOOKBACK_LIST[0]]

                for atr_w in atr_ws:
                    for slp_k in slp_ks:
                        # 逐品种预计算趋势强度和收益波动率
                        strength_dict = {}
                        ret_vol_dict = {}
                        for ts_code in ts_codes:
                            strength_dict[ts_code] = calc_strength(
                                data[ts_code], sig_def, fast_w, slow_w, atr_w, slp_k
                            )
                            ret_vol_dict[ts_code] = calc_ret_vol(data[ts_code], FINAL_VOL_WINDOW)

                        for zw in Z_SCORE_WINDOW_LIST:
                            # 逐品种预计算 z-score
                            zscore_dict = {}
                            for ts_code in ts_codes:
                                zscore_dict[ts_code] = rolling_zscore(strength_dict[ts_code], zw)

                            for smooth_t in SMOOTH_T_LIST:
                                for mode in SIGNAL_MODE_LIST:
                                    print(
                                        f'  {sig_def} F={fast_w} S={slow_w} '
                                        f'ATR={atr_w} SLP={slp_k} ZW={zw} T={smooth_t} {mode}'
                                    )

                                    # 逐品种：方向 → 平滑
                                    smoothed_dict = {}
                                    for ts_code in ts_codes:
                                        s = zscore_dict[ts_code].copy()
                                        if mode == 'mean_reversion':
                                            s = -s
                                        if smooth_t > 1:
                                            s = s.rolling(smooth_t, min_periods=1).mean()
                                        smoothed_dict[ts_code] = s

                                    # 遍历每个交易日，计算当天所有品种的信号
                                    rows = []
                                    for trade_date in trading_days:
                                        daily_signal = {}
                                        for ts_code in ts_codes:
                                            # 取当天的 z-score 仓位
                                            if trade_date in smoothed_dict[ts_code].index:
                                                pos = smoothed_dict[ts_code].loc[trade_date]
                                            else:
                                                pos = np.nan

                                            # 除以收益波动率做缩放
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

                                    name = (
                                        f'MovingAverageV5_{sector}_{sig_def}_'
                                        f'F_{fast_w}_S_{slow_w}_ATR_{atr_w}_SLP_{slp_k}_'
                                        f'ZW_{zw}_T_{smooth_t}_VOL_{FINAL_VOL_WINDOW}_{mode}.csv'
                                    )
                                    signals.to_csv(os.path.join(OUTPUT_DIR, name), encoding='utf-8-sig')

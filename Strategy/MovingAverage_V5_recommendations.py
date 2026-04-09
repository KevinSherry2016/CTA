"""
MovingAverage_V5_recommendations

Read sector-level recommendations from analysis_tables/task5_sector_signal_recommendations.csv
and generate signals only for recommended signal definitions, z-score setting, and
parameter ranges.
"""

import os
from itertools import product

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './main_contract/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Result'
FINAL_VOL_WINDOW = 20
SECTOR_FILTER = None
SIGNAL_MODE_LIST = ['trend']
DEFAULT_Z_WINDOW = 60

# Discrete search grids used in V5 backtests.
FAST_WINDOW_GRID = [5, 10, 15, 20]
SLOW_WINDOW_GRID = [20, 40, 60, 80, 100, 120]
ATR_WINDOW_GRID = [15]
SLOPE_LOOKBACK_GRID = [3]
SMOOTH_T_GRID = [1, 5, 10]

SIGNAL_DEF_MAP = {
    'V5_1': 'fast_slow_gap_over_vol',
    'V5_2': 'fast_slow_gap_over_atr',
    'V5_3': 'price_minus_slow_ma_over_atr',
    'V5_4': 'slow_ma_slope_over_atr',
}

# Hardcoded from task5_sector_signal_recommendations.csv.
RECOMMENDATIONS = [
    {'sector': 'Agriculture', 'signal': 'V5_3', 'uses_zscore': 'No', 'param_range': 'F=5, S=60, ATR=15.0, T=5'},
    {'sector': 'Bond', 'signal': 'V5_1', 'uses_zscore': 'No', 'param_range': 'F=10, S=100, T=5'},
    {'sector': 'Energy', 'signal': 'V5_1', 'uses_zscore': 'Yes', 'param_range': 'F=10, S=80, T=5'},
    {'sector': 'Ferrous', 'signal': 'V5_3', 'uses_zscore': 'No', 'param_range': 'F=5, S=50, ATR=15.0, T=5'},
    {'sector': 'NonFerrous', 'signal': 'V5_2', 'uses_zscore': 'Yes', 'param_range': 'F=10, S=100, ATR=15.0, T=5'},
    {'sector': 'Other', 'signal': 'V5_3', 'uses_zscore': 'Yes', 'param_range': 'F=5, S=30, ATR=15.0, T=5'},
    {'sector': 'Precious', 'signal': 'V5_3', 'uses_zscore': 'No', 'param_range': 'F=5, S=100, ATR=15.0, T=5'},
    {'sector': 'StockIndex', 'signal': 'V5_3', 'uses_zscore': 'Yes', 'param_range': 'F=5, S=30, ATR=15.0, T=5'},
]


def parse_int_range(text):
    """Parse '5' or '5~20' into (low, high)."""
    if pd.isna(text):
        return None
    s = str(text).strip()
    if not s:
        return None
    if '~' in s:
        a, b = s.split('~', 1)
        return int(float(a)), int(float(b))
    v = int(float(s))
    return v, v


def filter_grid_by_range(grid, r):
    """Filter predefined discrete grid by parsed range tuple."""
    if r is None:
        return list(grid)
    lo, hi = r
    vals = [x for x in grid if lo <= x <= hi]
    # Fallback to nearest in-grid value if range misses all grid points.
    if not vals:
        nearest = min(grid, key=lambda x: abs(x - lo))
        vals = [nearest]
    return vals


def parse_param_range(param_text):
    """Parse 'F=5~20, S=60~100, ATR=15.0, T=1' into dict of ranges."""
    out = {}
    if pd.isna(param_text):
        return out
    s = str(param_text).strip()
    if not s:
        return out

    parts = [p.strip() for p in s.split(',') if p.strip()]
    for part in parts:
        if '=' not in part:
            continue
        k, v = part.split('=', 1)
        out[k.strip()] = parse_int_range(v.strip())
    return out


def calc_atr(df, window):
    """Calculate ATR for one instrument."""
    high = df['adj_high']
    low = df['adj_low']
    close = df['adj_close']
    prev_close = close.shift(1)
    tr = np.maximum(high - low,
                    np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return tr.rolling(window).mean().replace(0, np.nan)


def calc_strength(df, sig_def, fast_w, slow_w, atr_w, slp_k):
    """Calculate trend strength for one instrument."""
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
    """Rolling z-score standardization."""
    min_p = max(window // 2, 1)
    mu = series.rolling(window, min_periods=min_p).mean()
    sigma = series.rolling(window, min_periods=min_p).std().replace(0, np.nan)
    return (series - mu) / sigma


def calc_ret_vol(df, window):
    """Calculate return volatility for position scaling."""
    ret = df['adj_close'].pct_change(fill_method=None)
    return ret.rolling(window, min_periods=1).std().replace(0, np.nan)


def load_recommendation_jobs():
    """Build deduplicated run jobs from recommendation CSV."""
    rec = pd.DataFrame(RECOMMENDATIONS)
    jobs = []
    seen = set()

    for _, row in rec.iterrows():
        sector = str(row['sector']).strip()
        signal_key = str(row['signal']).strip()
        sig_def = SIGNAL_DEF_MAP.get(signal_key)
        if sig_def is None:
            continue

        use_zscore = str(row['uses_zscore']).strip().lower() == 'yes'
        params = parse_param_range(row.get('param_range', ''))

        fast_list = filter_grid_by_range(FAST_WINDOW_GRID, params.get('F'))
        slow_list = filter_grid_by_range(SLOW_WINDOW_GRID, params.get('S'))

        atr_list = ATR_WINDOW_GRID
        if 'over_atr' in sig_def:
            atr_list = filter_grid_by_range(ATR_WINDOW_GRID, params.get('ATR'))

        slp_list = [SLOPE_LOOKBACK_GRID[0]]
        if 'slope' in sig_def:
            slp_list = filter_grid_by_range(SLOPE_LOOKBACK_GRID, params.get('SLP'))

        smooth_list = filter_grid_by_range(SMOOTH_T_GRID, params.get('T'))

        for f, s, atr_w, slp_k, t in product(fast_list, slow_list, atr_list, slp_list, smooth_list):
            if f >= s:
                continue
            key = (sector, signal_key, sig_def, use_zscore, DEFAULT_Z_WINDOW, f, s, atr_w, slp_k, t)
            if key in seen:
                continue
            seen.add(key)
            jobs.append({
                'sector': sector,
                'signal_key': signal_key,
                'sig_def': sig_def,
                'use_zscore': use_zscore,
                'z_window': DEFAULT_Z_WINDOW,
                'fast_w': f,
                'slow_w': s,
                'atr_w': atr_w,
                'slp_k': slp_k,
                'smooth_t': t,
            })

    return jobs


def main():
    info = pd.read_csv(INFO_PATH, encoding='utf-8-sig')
    sector_map = info.groupby('sector')['ts_code'].apply(list).to_dict()

    data = {}
    print('Loading market data...')
    for ts_code in info['ts_code']:
        fp = os.path.join(MARKET_DATA_PATH, f'{ts_code}.csv')
        if not os.path.exists(fp):
            print(f'Missing: {fp}')
            continue
        df = pd.read_csv(fp, dtype={'trade_date': str}).set_index('trade_date')
        for c in ['adj_close', 'adj_high', 'adj_low']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        data[ts_code] = df

    trading_days = pd.read_csv(
        os.path.join(MARKET_DATA_PATH, 'CU.SHF.csv'), dtype={'trade_date': str}
    )['trade_date'].tolist()

    sector_map = {s: [c for c in codes if c in data] for s, codes in sector_map.items()}
    sector_map = {s: codes for s, codes in sector_map.items() if codes}
    sector_map['All'] = sorted(data.keys())
    if SECTOR_FILTER:
        sector_map = {s: v for s, v in sector_map.items() if s in SECTOR_FILTER}

    jobs = load_recommendation_jobs()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'Loaded jobs: {len(jobs)}')
    for i, job in enumerate(jobs, start=1):
        sector = job['sector']
        if sector not in sector_map:
            continue
        ts_codes = sector_map[sector]
        if not ts_codes:
            continue

        sig_def = job['sig_def']
        fast_w = job['fast_w']
        slow_w = job['slow_w']
        atr_w = job['atr_w']
        slp_k = job['slp_k']
        smooth_t = job['smooth_t']
        use_zscore = job['use_zscore']
        z_window = job['z_window']

        print(
            f"[{i}/{len(jobs)}] {sector} {job['signal_key']} {sig_def} "
            f"F={fast_w} S={slow_w} ATR={atr_w} SLP={slp_k} "
            f"{'ZW_60' if use_zscore else 'noZ'} T={smooth_t}"
        )

        strength_dict = {}
        ret_vol_dict = {}
        for ts_code in ts_codes:
            strength_dict[ts_code] = calc_strength(data[ts_code], sig_def, fast_w, slow_w, atr_w, slp_k)
            ret_vol_dict[ts_code] = calc_ret_vol(data[ts_code], FINAL_VOL_WINDOW)

        if use_zscore:
            signal_dict = {k: rolling_zscore(v, z_window) for k, v in strength_dict.items()}
            zw_tag = f'ZW_{z_window}_'
        else:
            signal_dict = strength_dict
            zw_tag = 'noZ_'

        for mode in SIGNAL_MODE_LIST:
            smoothed_dict = {}
            for ts_code in ts_codes:
                s = signal_dict[ts_code].copy()
                if mode == 'mean_reversion':
                    s = -s
                if smooth_t > 1:
                    s = s.ewm(span=smooth_t, min_periods=1).mean()
                smoothed_dict[ts_code] = s

            rows = []
            for trade_date in trading_days:
                daily_signal = {}
                for ts_code in ts_codes:
                    pos = smoothed_dict[ts_code].loc[trade_date] if trade_date in smoothed_dict[ts_code].index else np.nan
                    rv = ret_vol_dict[ts_code].loc[trade_date] if trade_date in ret_vol_dict[ts_code].index else np.nan
                    if pd.notna(pos) and pd.notna(rv) and rv != 0:
                        daily_signal[ts_code] = pos / rv
                    else:
                        daily_signal[ts_code] = 0.0
                rows.append(daily_signal)

            signals = pd.DataFrame(rows, index=trading_days)

            name = (
                f"MovingAverageV5_REC_{sector}_{sig_def}_"
                f"F_{fast_w}_S_{slow_w}_ATR_{atr_w}_SLP_{slp_k}_"
                f"{zw_tag}T_{smooth_t}_VOL_{FINAL_VOL_WINDOW}_{mode}.csv"
            )
            signals.to_csv(os.path.join(OUTPUT_DIR, name), encoding='utf-8-sig')


if __name__ == '__main__':
    main()

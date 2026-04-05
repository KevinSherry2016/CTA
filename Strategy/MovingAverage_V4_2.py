import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './total/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Result'

# V4_2 直接采用 V4_1 推荐表中的优选参数。
# 当前推荐结果基于 ATR=15、SLP=5，因此这里固定，不再扩展无效维度。
DEFAULT_ATR_WINDOW = 15
DEFAULT_SLOPE_LOOKBACK = 5
FINAL_VOL_WINDOW = 20
SIGNAL_MODE_LIST = ['trend']

# 按 V4_1 汇总分析写死的各 sector 推荐参数。
SECTOR_RECOMMENDED_CONFIG: dict[str, dict] = {
    'Agriculture': {
        # 农业单独扩宽参数：
        # 1. 保留表现最好的 slope 信号
        # 2. 补充更容易触发仓位的 gap / price 类信号
        # 3. 加入 30/50 慢窗、0.4 平仓阈值、T=1/10 以提高覆盖度
        'signal_defs': [
            'slow_ma_slope_over_atr',
        ],
        'fast_windows': [8],
        'slow_windows': [40],
        'z_open_list': [0.6],
        'z_close_list': [0.2],
        'smooth_t_list': [5],
        'atr_window': DEFAULT_ATR_WINDOW,
        'slope_lookback': DEFAULT_SLOPE_LOOKBACK,
        'signal_modes': SIGNAL_MODE_LIST,
    },
    'All': {
        'signal_defs': ['slow_ma_slope_over_atr'],
        'fast_windows': [8],
        'slow_windows': [40],
        'z_open_list': [0.6],
        'z_close_list': [0.2],
        'smooth_t_list': [5],
        'atr_window': DEFAULT_ATR_WINDOW,
        'slope_lookback': DEFAULT_SLOPE_LOOKBACK,
        'signal_modes': SIGNAL_MODE_LIST,
    },
    'Bond': {
        'signal_defs': ['slow_ma_slope_over_atr'],
        'fast_windows': [8],
        'slow_windows': [40],
        'z_open_list': [0.6],
        'z_close_list': [0.2],
        'smooth_t_list': [5],
        'atr_window': DEFAULT_ATR_WINDOW,
        'slope_lookback': DEFAULT_SLOPE_LOOKBACK,
        'signal_modes': SIGNAL_MODE_LIST,
    },
    'Energy': {
        'signal_defs': ['fast_slow_gap_over_atr'],
        'fast_windows': [8],
        'slow_windows': [40],
        'z_open_list': [0.6],
        'z_close_list': [0.2],
        'smooth_t_list': [5],
        'atr_window': DEFAULT_ATR_WINDOW,
        'slope_lookback': DEFAULT_SLOPE_LOOKBACK,
        'signal_modes': SIGNAL_MODE_LIST,
    },
    'Ferrous': {
        'signal_defs': ['price_minus_slow_ma_over_atr'],
        'fast_windows': [8],
        'slow_windows': [40],
        'z_open_list': [0.6],
        'z_close_list': [0.2],
        'smooth_t_list': [5],
        'atr_window': DEFAULT_ATR_WINDOW,
        'slope_lookback': DEFAULT_SLOPE_LOOKBACK,
        'signal_modes': SIGNAL_MODE_LIST,
    },
    'NonFerrous': {
        'signal_defs': ['fast_slow_gap_over_vol'],
        'fast_windows': [8],
        'slow_windows': [40],
        'z_open_list': [0.6],
        'z_close_list': [0.2],
        'smooth_t_list': [5],
        'atr_window': DEFAULT_ATR_WINDOW,
        'slope_lookback': DEFAULT_SLOPE_LOOKBACK,
        'signal_modes': SIGNAL_MODE_LIST,
    },
    'Precious': {
        'signal_defs': ['fast_slow_gap_over_vol'],
        'fast_windows': [8],
        'slow_windows': [40],
        'z_open_list': [0.6],
        'z_close_list': [0.2],
        'smooth_t_list': [5],
        'atr_window': DEFAULT_ATR_WINDOW,
        'slope_lookback': DEFAULT_SLOPE_LOOKBACK,
        'signal_modes': SIGNAL_MODE_LIST,
    },
    'StockIndex': {
        'signal_defs': ['slow_ma_slope_over_atr'],
        'fast_windows': [8],
        'slow_windows': [40],
        'z_open_list': [0.6],
        'z_close_list': [0.2],
        'smooth_t_list': [5],
        'atr_window': DEFAULT_ATR_WINDOW,
        'slope_lookback': DEFAULT_SLOPE_LOOKBACK,
        'signal_modes': SIGNAL_MODE_LIST,
    },
}


def smooth_position_series(position: pd.Series, smooth_t: int) -> pd.Series:
    """对最终仓位做 T 日滚动平滑。"""
    if smooth_t <= 1:
        return position.astype('float64')
    return position.rolling(window=smooth_t, min_periods=1).mean().astype('float64')


def calc_rolling_return_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """计算近 N 天收益波动率，用于最终仓位缩放。"""
    if window <= 1:
        return pd.Series(1.0, index=df.index, dtype='float64')

    returns = pd.to_numeric(df['adj_close'], errors='coerce').pct_change(fill_method=None)
    return returns.rolling(window=window, min_periods=1).std().replace(0, np.nan)


def build_position_from_strength(
    strength: pd.Series,
    z_open: float,
    z_close: float,
    signal_mode: str,
) -> pd.Series:
    """根据强度信号与双阈值生成固定仓位状态机信号。"""
    if signal_mode not in {'trend', 'mean_reversion'}:
        raise ValueError("signal_mode must be 'trend' or 'mean_reversion'.")

    positions = []
    current_position = 0

    for value in strength.to_numpy(dtype=float):
        if np.isnan(value):
            current_position = 0
            positions.append(0.0)
            continue

        if current_position != 0 and abs(value) < z_close:
            current_position = 0

        if current_position == 0:
            if signal_mode == 'trend':
                if value > z_open:
                    current_position = 1
                elif value < -z_open:
                    current_position = -1
            else:
                if value > z_open:
                    current_position = -1
                elif value < -z_open:
                    current_position = 1

        positions.append(float(current_position))

    return pd.Series(positions, index=strength.index, dtype='float64')


def calc_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """计算 ATR（Average True Range）。"""
    high = pd.to_numeric(df['adj_high'], errors='coerce')
    low = pd.to_numeric(df['adj_low'], errors='coerce')
    close = pd.to_numeric(df['adj_close'], errors='coerce')
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(window).mean().replace(0, np.nan)


def build_strength_series(
    df: pd.DataFrame,
    fast_window: int,
    slow_window: int,
    signal_def: str,
    atr_window: int,
    slope_lookback: int,
) -> pd.Series:
    """根据 signal_def 构造趋势强度序列。"""
    close = pd.to_numeric(df['adj_close'], errors='coerce')
    fast_ma = close.rolling(fast_window).mean()
    slow_ma = close.rolling(slow_window).mean()
    rolling_vol = close.rolling(slow_window).std().replace(0, np.nan)
    atr = calc_atr(df, atr_window)

    if signal_def == 'fast_slow_gap_over_vol':
        return (fast_ma - slow_ma) / rolling_vol

    if signal_def == 'fast_slow_gap_over_atr':
        return (fast_ma - slow_ma) / atr

    if signal_def == 'price_minus_slow_ma_over_atr':
        return (close - slow_ma) / atr

    if signal_def == 'slow_ma_slope_over_atr':
        return (slow_ma - slow_ma.shift(slope_lookback)) / atr

    raise ValueError(f'unsupported signal_def: {signal_def}')


def estimate_sector_combo_count(config: dict) -> int:
    """估算单个 sector 的输出组合数。"""
    valid_fs = sum(
        1
        for slow_window in config['slow_windows']
        for fast_window in config['fast_windows']
        if fast_window < slow_window
    )
    valid_z = sum(
        1
        for z_open in config['z_open_list']
        for z_close in config['z_close_list']
        if z_close < z_open
    )
    return (
        len(config['signal_defs'])
        * valid_fs
        * valid_z
        * len(config['smooth_t_list'])
        * len(config['signal_modes'])
    )


info = pd.read_csv(INFO_PATH, encoding='utf-8-sig')
recommendation_map = SECTOR_RECOMMENDED_CONFIG
ts_code_list = info['ts_code'].tolist()

base_sector_map: dict[str, list[str]] = (
    info.groupby('sector')['ts_code']
    .apply(list)
    .to_dict()
)

temp = pd.read_csv(os.path.join(MARKET_DATA_PATH, 'CU.SHF.csv'))
trading_day_list = temp['trade_date'].astype(str).tolist()

data = {}
print('正在加载数据')
for ts_code in ts_code_list:
    filepath = os.path.join(MARKET_DATA_PATH, f'{ts_code}.csv')
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['trade_date'] = df['trade_date'].astype(str)
        df.set_index('trade_date', inplace=True)
        df['adj_close'] = pd.to_numeric(df['adj_close'], errors='coerce')
        df['adj_high'] = pd.to_numeric(df['adj_high'], errors='coerce')
        df['adj_low'] = pd.to_numeric(df['adj_low'], errors='coerce')
        data[ts_code] = df
    else:
        print(f'文件不存在: {filepath}')

loaded_ts_codes = list(data.keys())
sector_map: dict[str, list[str]] = {
    sector: [ts_code for ts_code in sector_ts_codes if ts_code in data]
    for sector, sector_ts_codes in base_sector_map.items()
}
sector_map = {k: v for k, v in sector_map.items() if v}
sector_map['All'] = loaded_ts_codes

sector_map = {
    sector: ts_codes
    for sector, ts_codes in sector_map.items()
    if sector in recommendation_map
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

total_estimated_count = sum(
    estimate_sector_combo_count(recommendation_map[sector])
    for sector in sector_map
)
print(f'读取推荐分组数: {len(sector_map)}  {list(sector_map.keys())}')
print(f'预计输出文件数: {total_estimated_count}')

print('开始生成参数组合信号 (V4_2 按 sector 推荐配置)')
for sector, sector_ts_codes in sector_map.items():
    sector_config = recommendation_map[sector]
    sector_data = {ts_code: data[ts_code] for ts_code in sector_ts_codes if ts_code in data}
    if not sector_data:
        print(f'[{sector}] 无数据，跳过')
        continue

    print(f'\n===== Sector: {sector} ({len(sector_data)} 个品种) =====')
    print(f'[{sector}] 推荐信号: {sector_config["signal_defs"]}')
    print(
        f'[{sector}] 推荐参数: '
        f'F={sector_config["fast_windows"]}, '
        f'S={sector_config["slow_windows"]}, '
        f'ZO={sector_config["z_open_list"]}, '
        f'ZC={sector_config["z_close_list"]}, '
        f'T={sector_config["smooth_t_list"]}'
    )

    for signal_def in sector_config['signal_defs']:
        for slow_window in sector_config['slow_windows']:
            for fast_window in sector_config['fast_windows']:
                if fast_window >= slow_window:
                    continue

                strength_series = {}
                for ts_code, df in sector_data.items():
                    strength_series[ts_code] = build_strength_series(
                        df=df,
                        fast_window=fast_window,
                        slow_window=slow_window,
                        signal_def=signal_def,
                        atr_window=sector_config['atr_window'],
                        slope_lookback=sector_config['slope_lookback'],
                    )

                for z_open in sector_config['z_open_list']:
                    valid_z_close_list = [
                        z_close for z_close in sector_config['z_close_list'] if z_close < z_open
                    ]

                    for z_close in valid_z_close_list:
                        for smooth_t in sector_config['smooth_t_list']:
                            for signal_mode in sector_config['signal_modes']:
                                print(
                                    f'[{sector}] 正在计算信号: '
                                    f'signal_def={signal_def}, '
                                    f'F={fast_window}, S={slow_window}, '
                                    f'ATR={sector_config["atr_window"]}, '
                                    f'SLP={sector_config["slope_lookback"]}, '
                                    f'z_open={z_open}, z_close={z_close}, '
                                    f'T={smooth_t}, mode={signal_mode}'
                                )

                                position_series = {}
                                for ts_code, strength in strength_series.items():
                                    raw_position = build_position_from_strength(
                                        strength=strength,
                                        z_open=z_open,
                                        z_close=z_close,
                                        signal_mode=signal_mode,
                                    )
                                    position_series[ts_code] = smooth_position_series(
                                        raw_position,
                                        smooth_t=smooth_t,
                                    )

                                signals = (
                                    pd.DataFrame(position_series, index=trading_day_list)
                                    .fillna(0.0)
                                    .astype(float)
                                )

                                vol_series = {}
                                for ts_code, df in sector_data.items():
                                    vol_series[ts_code] = calc_rolling_return_vol(
                                        df,
                                        window=FINAL_VOL_WINDOW,
                                    ).reindex(trading_day_list)

                                vol_df = (
                                    pd.DataFrame(vol_series, index=trading_day_list)
                                    .replace(0, np.nan)
                                )
                                signals = signals.div(vol_df).replace([np.inf, -np.inf], np.nan).fillna(0.0)

                                output_name = (
                                    f'MovingAverageV4_2_{sector}_{signal_def}_'
                                    f'F_{fast_window}_S_{slow_window}_'
                                    f'ATR_{sector_config["atr_window"]}_'
                                    f'SLP_{sector_config["slope_lookback"]}_'
                                    f'ZO_{z_open}_ZC_{z_close}_T_{smooth_t}_'
                                    f'{signal_mode}.csv'
                                )
                                output_path = os.path.join(OUTPUT_DIR, output_name)
                                signals.to_csv(output_path, encoding='utf-8-sig')
                                print(f'信号输出完成: {output_path}')
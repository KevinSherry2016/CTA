import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './total/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Result'

# 运行档位：
# core -> 先跑核心参数，快速验证信号质量
# full -> 全参数网格，做完整扫描
RUN_PROFILE = 'core'

# ── 参数网格 ───────────────────────────────────────────────────────────────────
# fast / slow 分别表示短周期和长周期均线窗口。
# 这里根据 V4_summary 的回测结果做过一轮筛选：
# 1. 过近的快慢均线组合（例如 8/10、12/15、15/20）经常完全不触发或触发极少
# 2. 更有效的组合通常对应“更明确的长短周期分离”
# 因此这里去掉了问题最明显的 15，并把 slow_window 整体右移到 25 以上。
FAST_M_LIST = [5, 8, 10, 12]
SLOW_M_LIST = [25, 30, 35, 40, 45, 50]

# z_open 用于开仓阈值，z_close 用于平仓阈值。
# 根据 V4_summary 统计：
# 1. z_open 在 0.6 ~ 1.1 的参数组合大多仍有信号触发
# 2. z_open >= 1.2 后，空信号组合明显增多，因此先移除 1.2 和 1.3
# 3. z_close 对“是否触发”影响不大，所以先保留原有候选集合
Z_OPEN_LIST = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
Z_CLOSE_LIST = [0.1, 0.2, 0.3, 0.4, 0.5]

# 交易模式：
# trend          -> 信号为正做多，信号为负做空
# mean_reversion -> 信号为正做空，信号为负做多
SIGNAL_MODE_LIST = ['trend']

# 仓位平滑窗口（T 日）：
# T=1 表示不平滑；T>1 表示对最终仓位做滚动均值平滑。
POSITION_SMOOTH_T_LIST = [1, 5, 10]

# SIGNAL_DEF_LIST 表示“信号定义”的集合。
# 这一步是本版本的核心升级：不再把趋势强度写死为单一公式，而是允许同一套
# 状态机去测试多种候选信号，先比较“信号本身”的质量，再谈参数微调。
#
# 各信号定义的直观含义：
# 1. fast_slow_gap_over_vol
#    看“快均线和慢均线拉开了多少”，再除以价格自身波动率。
#    适合衡量双均线之间的相对距离，是最直接的双均线趋势强度。
#
# 2. fast_slow_gap_over_atr
#    同样看快慢均线距离，但改用 ATR 归一化。
#    相比 rolling_std，它对期货合约跳空、振幅、日内波动更敏感，通常更贴近交易尺度。
#
# 3. price_minus_slow_ma_over_atr
#    不再看“快线相对慢线”，而是直接看“当前价格偏离慢均线有多远”。
#    它更像“价格已经脱离中枢多少”，通常比双均线差值更敏感、更容易触发。
#
# 4. slow_ma_slope_over_atr
#    不看价格离均线有多远，而看“慢均线自己在往哪个方向走、走得多快”。
#    它关注的是趋势推进速度，通常比均线距离类信号更稳，但也可能更滞后。
SIGNAL_DEF_LIST = [
    'fast_slow_gap_over_vol',
    'fast_slow_gap_over_atr',
    'price_minus_slow_ma_over_atr',
    'slow_ma_slope_over_atr',
]

# ATR_WINDOW_LIST 用于 *_over_atr 类信号的归一化窗口。
# SLOPE_LOOKBACK_LIST 用于 slow_ma_slope_over_atr 中的斜率回看窗口。
ATR_WINDOW_LIST = [10, 15, 20]
SLOPE_LOOKBACK_LIST = [3, 5, 10]


def get_param_grid(profile: str) -> dict:
    """根据运行档位返回参数网格。"""
    if profile == 'core':
        # 核心组合：先保证覆盖主要信号与中间参数，避免首轮就跑 2w+ 组合。
        return {
            'signal_defs': [
                'fast_slow_gap_over_vol',
                'fast_slow_gap_over_atr',
                'price_minus_slow_ma_over_atr',
                'slow_ma_slope_over_atr',
            ],
            'fast_windows': [8, 10],
            'slow_windows': [30, 40, 50],
            'z_open_list': [0.6, 0.8, 1.0],
            'z_close_list': [0.2, 0.4],
            'atr_windows': [15],
            'slope_lookbacks': [5],
            'smooth_t_list': [1, 5, 10],
            'signal_modes': SIGNAL_MODE_LIST,
        }

    if profile == 'full':
        return {
            'signal_defs': SIGNAL_DEF_LIST,
            'fast_windows': FAST_M_LIST,
            'slow_windows': SLOW_M_LIST,
            'z_open_list': Z_OPEN_LIST,
            'z_close_list': Z_CLOSE_LIST,
            'atr_windows': ATR_WINDOW_LIST,
            'slope_lookbacks': SLOPE_LOOKBACK_LIST,
            'smooth_t_list': POSITION_SMOOTH_T_LIST,
            'signal_modes': SIGNAL_MODE_LIST,
        }

    raise ValueError("RUN_PROFILE must be either 'core' or 'full'.")


def estimate_combo_count(grid: dict) -> int:
    """估算将要输出的组合数量。"""
    valid_fs = sum(
        1
        for slow_window in grid['slow_windows']
        for fast_window in grid['fast_windows']
        if fast_window < slow_window
    )
    valid_z = sum(
        1
        for z_open in grid['z_open_list']
        for z_close in grid['z_close_list']
        if z_close < z_open
    )

    total = 0
    for signal_def in grid['signal_defs']:
        atr_windows = grid['atr_windows'] if 'over_atr' in signal_def else [15]
        slope_lookbacks = grid['slope_lookbacks'] if 'slope' in signal_def else [5]
        total += (
            valid_fs
            * len(atr_windows)
            * len(slope_lookbacks)
            * valid_z
            * len(grid['smooth_t_list'])
        )
    return total * len(grid['signal_modes'])


def smooth_position_series(position: pd.Series, smooth_t: int) -> pd.Series:
    """对最终仓位做 T 日滚动平滑。"""
    if smooth_t <= 1:
        return position.astype('float64')
    return position.rolling(window=smooth_t, min_periods=1).mean().astype('float64')


def build_position_from_strength(
    strength: pd.Series,
    z_open: float,
    z_close: float,
    signal_mode: str,
) -> pd.Series:
    """根据强度信号与双阈值生成固定仓位状态机信号。

    状态机逻辑：
    1. 空仓时，只有 |strength| 超过 z_open 才允许开仓
    2. 持仓后，只有 |strength| 回落到 z_close 以内才平仓
    3. z_open 和 z_close 之间形成“滞后区间”，减少边界附近来回开平

    返回值：
    - 1.0 -> 持有多头
    - 0.0 -> 空仓
    - -1.0 -> 持有空头
    """
    if signal_mode not in {'trend', 'mean_reversion'}:
        raise ValueError("signal_mode must be 'trend' or 'mean_reversion'.")

    positions = []
    current_position = 0

    for value in strength.to_numpy(dtype=float):
        # 信号缺失时直接空仓，避免使用不完整窗口产生的假信号。
        if np.isnan(value):
            current_position = 0
            positions.append(0.0)
            continue

        # 已持仓时，只有当强度衰减回 z_close 以内才平仓。
        if current_position != 0 and abs(value) < z_close:
            current_position = 0

        # 仅在空仓状态下检查是否重新开仓。
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
    """计算 ATR（Average True Range）。

    这里使用 adj_high / adj_low / adj_close 来计算真实波动幅度，
    这样不同品种、不同价格尺度下的趋势信号可以更可比。
    """
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
    """根据 signal_def 构造趋势强度序列。

    可选信号定义：
    1. fast_slow_gap_over_vol
       (fast_ma - slow_ma) / rolling_std(close, slow_window)
         含义：快均线比慢均线高多少，再除以价格波动率。
         strength > 0 表示短期价格结构强于长期趋势；绝对值越大，表示两条均线拉得越开。

    2. fast_slow_gap_over_atr
       (fast_ma - slow_ma) / ATR
         含义：快慢均线差值相对于 ATR 有多大。
         更适合期货，因为 ATR 反映的是更接近交易真实感受的波动尺度。

    3. price_minus_slow_ma_over_atr
       (price - slow_ma) / ATR
         含义：当前价格相对慢均线偏离了多少个 ATR。
         本质上是在测量“价格脱离长期中枢的程度”，对趋势启动往往更敏感。

    4. slow_ma_slope_over_atr
       (slow_ma_t - slow_ma_{t-k}) / ATR
         含义：慢均线在过去 k 天一共抬升/下移了多少，再除以 ATR。
         本质上是在测量长期趋势线本身的斜率，而不是价格与均线的瞬时距离。

    这些定义都输出一个“可正可负”的连续强度值，后续统一交给状态机处理。
    """
    close = pd.to_numeric(df['adj_close'], errors='coerce')
    fast_ma = close.rolling(fast_window).mean()
    slow_ma = close.rolling(slow_window).mean()

    # 基于价格水平的滚动波动率，适合做最简单的归一化。
    rolling_vol = close.rolling(slow_window).std().replace(0, np.nan)
    atr = calc_atr(df, atr_window)

    if signal_def == 'fast_slow_gap_over_vol':
        # 快线在慢线上方越多，strength 越大；反之越小。
        # 这是最标准的双均线距离型趋势信号。
        return (fast_ma - slow_ma) / rolling_vol

    if signal_def == 'fast_slow_gap_over_atr':
        # 与上面相同，只是把归一化尺度从价格波动率换成 ATR。
        # 这样更强调“可交易波动”而不是纯统计波动。
        return (fast_ma - slow_ma) / atr

    if signal_def == 'price_minus_slow_ma_over_atr':
        # 价格高于慢均线越多，说明价格离长期趋势中枢越远。
        # 适合抓趋势加速或突破后的延续阶段。
        return (close - slow_ma) / atr

    if signal_def == 'slow_ma_slope_over_atr':
        # 如果慢均线持续上行，则该值为正；持续下行则为负。
        # 适合过滤掉“价格一时偏离，但长期趋势线本身并未转向”的情况。
        return (slow_ma - slow_ma.shift(slope_lookback)) / atr

    raise ValueError(f'unsupported signal_def: {signal_def}')


info = pd.read_csv(INFO_PATH, encoding='utf-8-sig')
ts_code_list = info['ts_code'].tolist()

# Info.csv 原始 sector 分组；后续会结合已加载数据再生成最终 sector_map。
base_sector_map: dict[str, list[str]] = (
    info.groupby('sector')['ts_code']
    .apply(list)
    .to_dict()
)

# 如果只想跑某几个分组，在这里指定；设为 None 则跑全部。
# 注意：支持新增分组 'All'（全部品种）。
SECTOR_FILTER: list[str] | None = None

temp = pd.read_csv(os.path.join(MARKET_DATA_PATH, 'CU.SHF.csv'))
trading_day_list = temp['trade_date'].astype(str).tolist()

data = {}
print('正在加载数据')
for ts_code in ts_code_list:
    filepath = os.path.join(MARKET_DATA_PATH, f'{ts_code}.csv')
    if os.path.exists(filepath):
        # V4_1 需要 ATR，因此除了 adj_close，还需要 adj_high / adj_low。
        df = pd.read_csv(filepath)
        df['trade_date'] = df['trade_date'].astype(str)
        df.set_index('trade_date', inplace=True)
        df['adj_close'] = pd.to_numeric(df['adj_close'], errors='coerce')
        df['adj_high'] = pd.to_numeric(df['adj_high'], errors='coerce')
        df['adj_low'] = pd.to_numeric(df['adj_low'], errors='coerce')
        data[ts_code] = df
    else:
        print(f'文件不存在: {filepath}')

# 基于“实际已加载数据”构建最终分组，避免空文件导致的无效品种。
loaded_ts_codes = list(data.keys())
sector_map: dict[str, list[str]] = {
    sector: [ts_code for ts_code in sector_ts_codes if ts_code in data]
    for sector, sector_ts_codes in base_sector_map.items()
}
sector_map = {k: v for k, v in sector_map.items() if v}

# 新增 All 分组：包含全部已加载品种。
sector_map['All'] = loaded_ts_codes

if SECTOR_FILTER is not None:
    sector_map = {k: v for k, v in sector_map.items() if k in SECTOR_FILTER}

os.makedirs(OUTPUT_DIR, exist_ok=True)

param_grid = get_param_grid(RUN_PROFILE)
estimated_count = estimate_combo_count(param_grid)
print(f'运行档位: {RUN_PROFILE}')
print(f'涉及 sector 数: {len(sector_map)}  {list(sector_map.keys())}')
print(f'预计输出文件数 (单 sector): {estimated_count}  合计: {estimated_count * len(sector_map)}')

print('开始生成参数组合信号 (V4_1 分 sector 多信号定义)')
for sector, sector_ts_codes in sector_map.items():
    # 过滤出该 sector 中实际有数据的品种
    sector_data = {ts_code: data[ts_code] for ts_code in sector_ts_codes if ts_code in data}
    if not sector_data:
        print(f'[{sector}] 无数据，跳过')
        continue
    print(f'\n===== Sector: {sector} ({len(sector_data)} 个品种) =====')

    for signal_def in param_grid['signal_defs']:
        print(f'[{sector}] 当前信号定义: {signal_def}')

        for slow_window in param_grid['slow_windows']:
            for fast_window in param_grid['fast_windows']:
                if fast_window >= slow_window:
                    continue

                current_atr_windows = param_grid['atr_windows'] if 'over_atr' in signal_def else [15]
                current_slope_lookbacks = param_grid['slope_lookbacks'] if 'slope' in signal_def else [5]

                for atr_window in current_atr_windows:
                    for slope_lookback in current_slope_lookbacks:
                        strength_series = {}
                        for ts_code, df in sector_data.items():
                            strength_series[ts_code] = build_strength_series(
                                df=df,
                                fast_window=fast_window,
                                slow_window=slow_window,
                                signal_def=signal_def,
                                atr_window=atr_window,
                                slope_lookback=slope_lookback,
                            )

                        for z_open in param_grid['z_open_list']:
                            valid_z_close_list = [
                                z_close for z_close in param_grid['z_close_list'] if z_close < z_open
                            ]

                            for z_close in valid_z_close_list:
                                for smooth_t in param_grid['smooth_t_list']:
                                    for signal_mode in param_grid['signal_modes']:
                                        print(
                                            f'[{sector}] 正在计算信号: '
                                            f'signal_def={signal_def}, '
                                            f'F={fast_window}, S={slow_window}, '
                                            f'ATR={atr_window}, SLP={slope_lookback}, '
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

                                        output_name = (
                                            f'MovingAverageV4_1_{sector}_{signal_def}_'
                                            f'F_{fast_window}_S_{slow_window}_'
                                            f'ATR_{atr_window}_SLP_{slope_lookback}_'
                                            f'ZO_{z_open}_ZC_{z_close}_T_{smooth_t}_{signal_mode}.csv'
                                        )
                                        output_path = os.path.join(OUTPUT_DIR, output_name)
                                        signals.to_csv(output_path, encoding='utf-8-sig')
                                        print(f'信号输出完成: {output_path}')
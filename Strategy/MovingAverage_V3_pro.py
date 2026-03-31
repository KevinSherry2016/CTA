import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './total/'
INFO_PATH        = './Info.csv'
OUTPUT_DIR       = './Result'

# ── 参数网格 ───────────────────────────────────────────────────────────────────
M_LIST       = [10, 15, 20, 25, 30, 35, 40, 45, 50]   # z-score 回望窗口
Z_OPEN_LIST  = [1.2, 1.4, 1.6, 1.8, 2.0]              # 入场阈值 / tanh 缩放锚点
Z_CLOSE_LIST = [0.0, 0.2, 0.4, 0.6, 0.8]              # 出场/死区阈值，须 < z_open
MAX_HOLD_LIST = [5, 10, 15, 20, 25, 30]               # 最长持仓天数（仅 state_proportional）
SIGNAL_MODE  = 'trend'                                 # 'trend' 或 'mean_reversion'

# 连续信号度量方式（可同时启用多种）：
#   'tanh'               : signal = tanh(z / z_open)，纯连续，无状态机
#   'linear'             : signal = clip(z / z_open, −2, 2)，纯连续，无状态机
#   'state_proportional' : 与 V3 状态机相同，但仓位大小 = clip(|z| / z_open, 0, 2)
SCALE_METHOD_LIST = ['tanh', 'linear', 'state_proportional']

# 连续信号的绝对上限（适用于 linear 与 state_proportional）
POSITION_CAP = 2.0


# ── 连续信号构建函数 ───────────────────────────────────────────────────────────

def _direction_sign(signal_mode: str, z_value: float) -> int:
    """根据 signal_mode 返回信号方向（+1 做多 / −1 做空）。"""
    # trend 模式：z > 0 → 价格相对均值偏高 → 做多（追涨）
    # mean_reversion 模式：z > 0 → 价格偏高 → 做空（均值回归）
    if signal_mode == 'trend':
        return 1 if z_value > 0 else -1
    else:
        return -1 if z_value > 0 else 1


def build_tanh_signal(
    zscore: pd.Series,
    z_open: float,
    z_close: float,
    signal_mode: str,
) -> pd.Series:
    """纯连续信号：signal = tanh(z / z_open)。

    - z_open 作为 tanh 的缩放锚点：|z| = z_open 时仓位约为 ±0.76
    - 均值回归模式：信号反向
    - |z| < z_close：信号置零（死区）
    """
    if signal_mode not in {'trend', 'mean_reversion'}:
        raise ValueError("signal_mode must be 'trend' or 'mean_reversion'.")

    z   = zscore.to_numpy(dtype=float)
    # tanh 将 z/z_open 映射到 (−1, 1)，越偏离均值仓位越重，但有饱和上限
    # 例：z = z_open 时 tanh(1) ≈ 0.76；z = 2×z_open 时 tanh(2) ≈ 0.96
    raw = np.tanh(z / z_open)
    if signal_mode == 'mean_reversion':
        # 均值回归：信号方向与 z-score 方向相反
        raw = -raw

    # 死区：|z| < z_close 时信号归零，避免在均值附近频繁交易
    raw = np.where(np.abs(z) < z_close, 0.0, raw)
    # z-score 为 NaN（窗口不足）时同样归零
    raw = np.where(np.isnan(z),         0.0, raw)

    return pd.Series(raw, index=zscore.index, dtype='float64')


def build_linear_signal(
    zscore: pd.Series,
    z_open: float,
    z_close: float,
    signal_mode: str,
    cap: float = POSITION_CAP,
) -> pd.Series:
    """纯连续信号：signal = clip(z / z_open, −cap, cap)。

    - |z| = z_open 时仓位恰好为 ±1.0
    - 超过 z_open 后线性增大，上限由 cap 控制
    """
    if signal_mode not in {'trend', 'mean_reversion'}:
        raise ValueError("signal_mode must be 'trend' or 'mean_reversion'.")

    z   = zscore.to_numpy(dtype=float)
    # z/z_open 使 |z| = z_open 时仓位恰好为 ±1.0，超过后线性增大
    # clip 将仓位绝对值上限控制在 cap（默认 2.0），避免极端 z-score 导致仓位失控
    raw = np.clip(z / z_open, -cap, cap)
    if signal_mode == 'mean_reversion':
        # 均值回归：信号方向与 z-score 方向相反
        raw = -raw

    # 死区：|z| < z_close 时信号归零
    raw = np.where(np.abs(z) < z_close, 0.0, raw)
    # z-score 为 NaN（窗口不足）时同样归零
    raw = np.where(np.isnan(z),         0.0, raw)

    return pd.Series(raw, index=zscore.index, dtype='float64')


def build_state_proportional_signal(
    zscore: pd.Series,
    z_open: float,
    z_close: float,
    max_hold: int,
    signal_mode: str,
    cap: float = POSITION_CAP,
) -> pd.Series:
    """状态机进出场 + 连续仓位大小。

    进出场逻辑与 V3 完全相同，但仓位大小改为：
        position = direction × clip(|z_score_当日| / z_open, 0, cap)

    即：偏离越强，仓位越重；仍受 max_hold 和 z_close 控制。
    """
    if signal_mode not in {'trend', 'mean_reversion'}:
        raise ValueError("signal_mode must be 'trend' or 'mean_reversion'.")

    positions        = []
    current_direction = 0   # 当前持仓方向：+1 多头 / −1 空头 / 0 空仓
    holding_days      = 0   # 当前仓位已持有的天数

    for value in zscore.to_numpy(dtype=float):
        # z-score 为 NaN（窗口不足或价格缺失）时，强制清仓并输出 0
        if np.isnan(value):
            current_direction = 0
            holding_days      = 0
            positions.append(0.0)
            continue

        # ① 检查平仓（先于开仓判断，允许同日先平后开）
        if current_direction != 0:
            holding_days += 1
            # 平仓条件：z-score 已回到 z_close 以内（偏离收敛），或持仓达最大天数
            if abs(value) < z_close or holding_days >= max_hold:
                current_direction = 0
                holding_days      = 0

        # ② 检查开仓（仅在空仓状态下触发）
        if current_direction == 0:
            if signal_mode == 'trend':
                # 趋势模式：偏离足够大时顺向追入
                if value > z_open:
                    current_direction = 1   # z 高 → 价格强势 → 做多
                    holding_days      = 1   # 开仓当天记为第 1 天
                elif value < -z_open:
                    current_direction = -1  # z 低 → 价格弱势 → 做空
                    holding_days      = 1
            else:
                # 均值回归模式：偏离足够大时反向布局
                if value > z_open:
                    current_direction = -1  # z 高 → 价格高估 → 做空
                    holding_days      = 1
                elif value < -z_open:
                    current_direction = 1   # z 低 → 价格低估 → 做多
                    holding_days      = 1

        # ③ 输出今日仓位
        if current_direction == 0:
            positions.append(0.0)
        else:
            # 仓位大小与当日 z-score 绝对值成正比：偏离越大仓位越重
            # |z| = z_open → size = 1.0；|z| = 2×z_open → size = 2.0（被 cap 截断）
            size = min(abs(value) / z_open, cap)
            positions.append(current_direction * size)

    return pd.Series(positions, index=zscore.index, dtype='float64')


# ── 数据加载 ───────────────────────────────────────────────────────────────────

info          = pd.read_csv(INFO_PATH, encoding='utf-8-sig')
ts_code_list  = info['ts_code'].tolist()

temp             = pd.read_csv(os.path.join(MARKET_DATA_PATH, 'CU.SHF.csv'))
trading_day_list = temp['trade_date'].astype(str).tolist()

data = {}
print('正在加载数据')
for ts_code in ts_code_list:
    filepath = os.path.join(MARKET_DATA_PATH, f'{ts_code}.csv')
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['trade_date'] = df['trade_date'].astype(str)
        df.set_index('trade_date', inplace=True)
        # 价格字段转数值，无法解析的值（如空字符串）转为 NaN，后续 z-score 会同步为 NaN
        df['adj_close'] = pd.to_numeric(df['adj_close'], errors='coerce')
        data[ts_code] = df
    else:
        print(f'文件不存在: {filepath}')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 主循环 ────────────────────────────────────────────────────────────────────
print('开始生成连续仓位信号（V3_pro）')

for window in M_LIST:
    # ── 预计算 z-score ────────────────────────────────────────────────────────
    # z-score 只与窗口 window 有关，与 scale_method / z_open / z_close 无关，
    # 在最外层预计算后在内层循环复用，避免重复计算。
    zscore_series: dict[str, pd.Series] = {}
    for ts_code, df in data.items():
        rolling = df['adj_close'].rolling(window)
        # std 为 0（价格完全不变）时替换为 NaN，防止除零产生 inf
        std     = rolling.std().replace(0, np.nan)
        zscore_series[ts_code] = (df['adj_close'] - rolling.mean()) / std

    for scale_method in SCALE_METHOD_LIST:
        for z_open in Z_OPEN_LIST:
            # z_close 必须严格小于 z_open，否则死区范围会覆盖入场阈值，永远不会开仓
            valid_z_close_list = [zc for zc in Z_CLOSE_LIST if zc < z_open]

            for z_close in valid_z_close_list:
                # state_proportional 有状态机，max_hold 实际生效；
                # tanh / linear 是纯函数，不需要 max_hold，用 [0] 占位使循环结构统一
                hold_iter = MAX_HOLD_LIST if scale_method == 'state_proportional' else [0]

                for max_hold in hold_iter:
                    print(
                        f'M={window}, scale={scale_method}, '
                        f'z_open={z_open}, z_close={z_close}'
                        + (f', max_hold={max_hold}' if scale_method == 'state_proportional' else '')
                        + f', mode={SIGNAL_MODE}'
                    )

                    position_series = {}
                    for ts_code, zscore in zscore_series.items():
                        if scale_method == 'tanh':
                            pos = build_tanh_signal(zscore, z_open, z_close, SIGNAL_MODE)
                        elif scale_method == 'linear':
                            pos = build_linear_signal(zscore, z_open, z_close, SIGNAL_MODE)
                        else:  # state_proportional
                            pos = build_state_proportional_signal(
                                zscore, z_open, z_close, max_hold, SIGNAL_MODE
                            )
                        position_series[ts_code] = pos

                    # 对齐到全量交易日，缺失品种信号填 0（未上市/退市期间视为无仓位）
                    signals = (
                        pd.DataFrame(position_series, index=trading_day_list)
                        .fillna(0.0)
                        .astype(float)
                    )

                    # state_proportional 文件名含 max_hold，tanh / linear 不含
                    if scale_method == 'state_proportional':
                        output_name = (
                            f'MovingAverageV3_pro_M_{window}'
                            f'_ZO_{z_open}_ZC_{z_close}_H_{max_hold}'
                            f'_{SIGNAL_MODE}_{scale_method}.csv'
                        )
                    else:
                        output_name = (
                            f'MovingAverageV3_pro_M_{window}'
                            f'_ZO_{z_open}_ZC_{z_close}'
                            f'_{SIGNAL_MODE}_{scale_method}.csv'
                        )

                    output_path = os.path.join(OUTPUT_DIR, output_name)
                    signals.to_csv(output_path, encoding='utf-8-sig')
                    print(f'  → 输出完成: {output_name}')

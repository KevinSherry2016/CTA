import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './total/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Result'

# 参数网格
M_LIST = [10, 15, 20, 25, 30, 35, 40, 45, 50]          # z-score 回望窗口
Z_OPEN_LIST = [1.2, 1.4, 1.6, 1.8, 2.0]                # 入场阈值
Z_CLOSE_LIST = [0.0, 0.2, 0.4, 0.6, 0.8]               # 出场阈值，需严格小于 z_open
MAX_HOLD_LIST = [5, 10, 15, 20, 25, 30]                # 最长持仓天数
SIGNAL_MODE = 'trend'                                  # 'trend' 或 'mean_reversion'

def build_position_from_zscore(
    zscore: pd.Series,
    z_open: float,
    z_close: float,
    max_hold: int,
    signal_mode: str,
) -> pd.Series:
    """根据 z-score、双阈值和最长持仓天数生成固定仓位状态机信号。

    返回值含义：
    - 1.0: 持有多头
    - 0.0: 空仓
    - -1.0: 持有空头

    执行顺序分两步：
    1. 先检查已有仓位是否要平仓
    2. 如果已经空仓，再判断当天是否满足新的开仓条件

    这样写的好处是：同一天允许“先平旧仓，再按新信号开新仓”。
    """
    if signal_mode not in {'trend', 'mean_reversion'}:
        raise ValueError("signal_mode must be 'trend' or 'mean_reversion'.")

    # positions 保存每天最终输出的仓位。
    positions = []
    # current_position 只取 -1 / 0 / 1，表示当前持仓方向。
    current_position = 0
    # holding_days 记录当前这笔仓位已经持有了多少天。
    holding_days = 0

    for value in zscore.to_numpy(dtype=float):
        # z-score 为空时，不保留旧仓，直接回到空仓。
        if np.isnan(value):
            current_position = 0
            holding_days = 0
            positions.append(0.0)
            continue

        if current_position != 0:
            # 只有在已经持仓的情况下，才继续累加持仓天数。
            holding_days += 1

            # 平仓条件有两个：
            # 1. z-score 回到 z_close 区间内，说明偏离已经收敛
            # 2. 持仓时间达到 max_hold，强制离场
            should_close = abs(value) < z_close or holding_days >= max_hold
            if should_close:
                current_position = 0
                holding_days = 0

        if current_position == 0:
            # 只有在空仓状态下，才允许重新开仓。
            # trend:        z > z_open 做多，z < -z_open 做空
            # mean_reversion: z > z_open 做空，z < -z_open 做多
            if signal_mode == 'trend':
                if value > z_open:
                    current_position = 1
                    # 开仓当天记为第 1 天，便于和 max_hold 对齐。
                    holding_days = 1
                elif value < -z_open:
                    current_position = -1
                    holding_days = 1
            else:
                if value > z_open:
                    current_position = -1
                    holding_days = 1
                elif value < -z_open:
                    current_position = 1
                    holding_days = 1

        # 记录当天循环结束后的最终仓位。
        positions.append(float(current_position))

    return pd.Series(positions, index=zscore.index, dtype='float64')


info = pd.read_csv(INFO_PATH, encoding='utf-8-sig')
ts_code_list = info['ts_code'].tolist()

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
        data[ts_code] = df
    else:
        print(f'文件不存在: {filepath}')

print('开始生成参数组合信号')
for window in M_LIST:
    zscore_series = {}
    for ts_code, df in data.items():
        rolling_stats = df['adj_close'].rolling(window)
        rolling_std = rolling_stats.std().replace(0, np.nan)
        zscore = (df['adj_close'] - rolling_stats.mean()) / rolling_std
        zscore_series[ts_code] = zscore

    for z_open in Z_OPEN_LIST:
        valid_z_close_list = [z_close for z_close in Z_CLOSE_LIST if z_close < z_open]

        for z_close in valid_z_close_list:
            for max_hold in MAX_HOLD_LIST:
                print(
                    '正在计算信号: '
                    f'M={window}, z_open={z_open}, z_close={z_close}, max_hold={max_hold}, mode={SIGNAL_MODE}'
                )
                position_series = {}
                for ts_code, zscore in zscore_series.items():
                    position_series[ts_code] = build_position_from_zscore(
                        zscore=zscore,
                        z_open=z_open,
                        z_close=z_close,
                        max_hold=max_hold,
                        signal_mode=SIGNAL_MODE,
                    )

                signals = pd.DataFrame(position_series, index=trading_day_list).fillna(0.0).astype(float)
                output_name = (
                    f'MovingAverageV3_M_{window}_ZO_{z_open}_ZC_{z_close}_H_{max_hold}_{SIGNAL_MODE}.csv'
                )
                output_path = os.path.join(OUTPUT_DIR, output_name)
                signals.to_csv(output_path, encoding='utf-8-sig')
                print(f'信号输出完成: {output_path}')
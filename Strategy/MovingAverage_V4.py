import os

import numpy as np
import pandas as pd


MARKET_DATA_PATH = './total/'
INFO_PATH = './Info.csv'
OUTPUT_DIR = './Result'

# 参数网格（V4：双均线）
FAST_M_LIST = [5, 8, 10, 12, 15]                      # 快均线窗口
SLOW_M_LIST = [10, 15, 20, 25, 30, 35, 40, 45, 50]            # 慢均线窗口（需 > FAST_M）
Z_OPEN_LIST = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]                    # 入场阈值（下调，提升触发率）
Z_CLOSE_LIST = [0.1, 0.2, 0.3, 0.4, 0.5]              # 出场阈值，需严格小于 z_open
SIGNAL_MODE = 'trend'                                  # 'trend' 或 'mean_reversion'


def build_position_from_strength(
    strength: pd.Series,
    z_open: float,
    z_close: float,
    signal_mode: str,
) -> pd.Series:
    """根据双均线强度与双阈值生成固定仓位状态机信号。

    strength 定义：
        strength = (fast_ma - slow_ma) / rolling_std(adj_close, slow_window)

    返回值含义：
    - 1.0: 持有多头
    - 0.0: 空仓
    - -1.0: 持有空头
    """
    if signal_mode not in {'trend', 'mean_reversion'}:
        raise ValueError("signal_mode must be 'trend' or 'mean_reversion'.")

    positions = []
    current_position = 0

    for value in strength.to_numpy(dtype=float):
        if np.isnan(value):
            current_position = 0
            positions.append(0.0)
            continue

        if current_position != 0:
            should_close = abs(value) < z_close
            if should_close:
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

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('开始生成参数组合信号 (V4 双均线)')
for slow_window in SLOW_M_LIST:
    for fast_window in FAST_M_LIST:
        if fast_window >= slow_window:
            continue

        strength_series = {}
        for ts_code, df in data.items():
            fast_ma = df['adj_close'].rolling(fast_window).mean()
            slow_ma = df['adj_close'].rolling(slow_window).mean()

            # 用慢窗口波动率标准化快慢均线差值，避免不同价格尺度不可比
            rolling_std = df['adj_close'].rolling(slow_window).std().replace(0, np.nan)
            strength = (fast_ma - slow_ma) / rolling_std
            strength_series[ts_code] = strength

        for z_open in Z_OPEN_LIST:
            valid_z_close_list = [z_close for z_close in Z_CLOSE_LIST if z_close < z_open]

            for z_close in valid_z_close_list:
                print(
                    '正在计算信号: '
                    f'F={fast_window}, S={slow_window}, '
                    f'z_open={z_open}, z_close={z_close}, mode={SIGNAL_MODE}'
                )

                position_series = {}
                for ts_code, strength in strength_series.items():
                    position_series[ts_code] = build_position_from_strength(
                        strength=strength,
                        z_open=z_open,
                        z_close=z_close,
                        signal_mode=SIGNAL_MODE,
                    )

                signals = pd.DataFrame(position_series, index=trading_day_list).fillna(0.0).astype(float)

                output_name = (
                    f'MovingAverageV4_F_{fast_window}_S_{slow_window}_'
                    f'ZO_{z_open}_ZC_{z_close}_{SIGNAL_MODE}.csv'
                )
                output_path = os.path.join(OUTPUT_DIR, output_name)
                signals.to_csv(output_path, encoding='utf-8-sig')
                print(f'信号输出完成: {output_path}')

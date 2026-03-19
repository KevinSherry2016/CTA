import pandas as pd
import numpy as np
import os

marketDataPath = './total/'

infoPath = './Info.csv'
info = pd.read_csv(infoPath, encoding='utf-8-sig')
ts_code_list = info['ts_code'].tolist()

momentum_window = 20    # 动量回望窗口（交易日）

temp = pd.read_csv(marketDataPath + 'CU.SHF.csv')
tradingDayList = temp['trade_date'].astype(str).tolist()

data = {}
print('正在加载数据')
for ts_code in ts_code_list:
    filepath = marketDataPath + ts_code + '.csv'
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['trade_date'] = df['trade_date'].astype(str)
        df.set_index('trade_date', inplace=True)
        data[ts_code] = df
    else:
        print(f'文件不存在: {filepath}')

# ── 预计算每个品种的动量信号序列 ──────────────────────────────────────────────
# 动量定义：过去 momentum_window 个交易日的累计收益率
#   momentum = adj_close / adj_close.shift(momentum_window) - 1
#
# 信号规则：
#   momentum > 0  → +1 做多（价格处于上升趋势）
#   momentum < 0  → -1 做空（价格处于下降趋势）
#   momentum = 0  →  0 空仓
print('正在计算信号')
position_series = {}
for ts_code, df in data.items():
    momentum = df['adj_close'] / df['adj_close'].shift(momentum_window) - 1

    signal = pd.Series(
        np.where(momentum > 0, 1, np.where(momentum < 0, -1, 0)),
        index=df.index
    )
    position_series[ts_code] = signal  # 收盘后产生信号，原始序列不 shift

# ── 合并所有品种信号，对齐全部交易日后输出 CSV ────────────────────────────────
signals = pd.DataFrame(position_series, index=tradingDayList).fillna(0).astype(float)
output_path = './Strategy/Momentum_V1_signal.csv'
signals.to_csv(output_path, encoding='utf-8-sig')
print(f'信号输出完成: {output_path}')

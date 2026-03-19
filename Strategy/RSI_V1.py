import pandas as pd
import numpy as np
import os

marketDataPath = './total/'

infoPath = './Info.csv'
info = pd.read_csv(infoPath, encoding='utf-8-sig')
ts_code_list = info['ts_code'].tolist()

rsi_window  = 14    # RSI 计算周期
rsi_long_entry  = 50    # RSI 高于此值做多
rsi_long_exit   = 70    # RSI 高于此值停止做多（平仓）
rsi_short_entry = 50    # RSI 低于此值做空
rsi_short_exit  = 30    # RSI 低于此值停止做空（平仓）

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

# ── 预计算每个品种的 RSI 及信号序列 ──────────────────────────────────────────
# RSI 公式（Wilder 平滑法）：
#   delta = adj_close.diff()
#   gain  = delta.clip(lower=0).ewm(alpha=1/rsi_window, adjust=False).mean()
#   loss  = (-delta).clip(lower=0).ewm(alpha=1/rsi_window, adjust=False).mean()
#   RS    = gain / loss
#   RSI   = 100 - 100 / (1 + RS)
#
# 信号规则：
#   50 < RSI < 70  → +1 做多
#   30 < RSI < 50  → -1 做空
#   RSI >= 70      →  0 空仓（超买，停止做多）
#   RSI <= 30      →  0 空仓（超卖，停止做空）
print('正在计算信号')
position_series = {}
for ts_code, df in data.items():
    delta = df['adj_close'].diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / rsi_window, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(alpha=1 / rsi_window, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - 100 / (1 + rs)

    signal = pd.Series(
        np.where(
            (rsi > rsi_long_entry) & (rsi < rsi_long_exit),   1,   # 做多
            np.where(
                (rsi < rsi_short_entry) & (rsi > rsi_short_exit), -1,  # 做空
                0                                                        # 空仓
            )
        ),
        index=df.index
    )
    position_series[ts_code] = signal  # 收盘后产生信号，原始序列不 shift

# ── 合并所有品种信号，对齐全部交易日后输出 CSV ────────────────────────────────
signals = pd.DataFrame(position_series, index=tradingDayList).fillna(0).astype(float)
output_path = './Strategy/RSI_V1_signal.csv'
signals.to_csv(output_path, encoding='utf-8-sig')
print(f'信号输出完成: {output_path}')

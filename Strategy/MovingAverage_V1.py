import pandas as pd
import numpy as np
import os

marketDataPath = './total/'

infoPath = './Info.csv'
info = pd.read_csv(infoPath, encoding='utf-8-sig')
ts_code_list = info['ts_code'].tolist()

short_window = 10   
long_window  = 60

temp = pd.read_csv(marketDataPath + 'CU.SHF.csv')
tradingDayList = temp['trade_date'].astype(str).tolist()

data = {}
print(f'正在加载数据')
for ts_code in ts_code_list:
    filepath = marketDataPath + ts_code + '.csv'
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['trade_date'] = df['trade_date'].astype(str)
        df.set_index('trade_date', inplace=True)
        data[ts_code] = df
    else:
        print(f'文件不存在: {filepath}')

# ── 预计算每个品种的信号序列 ──────────────────────────────────────────────────
# 信号: +1 做多, -1 做空, 0 空仓
print(f'正在计算信号')
position_series = {}
for ts_code, df in data.items():
    short_ma = df['adj_close'].rolling(short_window).mean()
    long_ma  = df['adj_close'].rolling(long_window).mean()
    signal   = pd.Series(
        np.where(short_ma > long_ma, 1, np.where(short_ma < long_ma, -1, 0)),
        index=df.index
    )
    position_series[ts_code] = signal  # 收盘后产生信号，原始序列不 shift

# ── 合并所有品种信号，对齐全部交易日后输出 CSV ────────────────────────────────
signals = pd.DataFrame(position_series, index=tradingDayList).fillna(0).astype(float)
output_path = './Strategy/MA_V1_signal.csv'
signals.to_csv(output_path, encoding='utf-8-sig')
print(f'信号输出完成: {output_path}')

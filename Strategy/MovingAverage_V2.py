import pandas as pd
import numpy as np
import os

marketDataPath = './total/'

infoPath = './Info.csv'
info = pd.read_csv(infoPath, encoding='utf-8-sig')
ts_code_list = info['ts_code'].tolist()

# 参数网格
M_LIST      = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]   # z-score 回望窗口
N_LIST      = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]   # 信号 rolling 窗口
Z_OPEN_LIST = [1.2, 1.4, 1.6, 1.8, 2.0]                 # 开仓 z-score 阈值

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

print('开始生成参数组合信号')
for M in M_LIST:
    # 预计算每个品种的 z-score（与 z_open / N 无关，可复用）
    zscore_series = {}
    for ts_code, df in data.items():
        roll = df['adj_close'].rolling(M)
        z = (df['adj_close'] - roll.mean()) / roll.std()
        zscore_series[ts_code] = z

    for z_open in Z_OPEN_LIST:
        # 根据 z_open 生成原始信号
        raw_signals = {}
        for ts_code, z in zscore_series.items():
            raw = pd.Series(
                np.where(z > z_open, 1, np.where(z < -z_open, -1, 0)),
                index=z.index
            )
            raw_signals[ts_code] = raw

        for N in N_LIST:
            print(f'正在计算信号: M={M}, N={N}, z_open={z_open}')
            position_series = {}
            for ts_code, raw in raw_signals.items():
                position_series[ts_code] = raw.rolling(window=N).sum()

            signals = pd.DataFrame(position_series, index=tradingDayList).fillna(0).astype(float)
            output_name = f'MovingAverageV2_M_{M}_N_{N}_Z_{z_open}.csv'
            output_path = os.path.join('./Strategy', output_name)
            signals.to_csv(output_path, encoding='utf-8-sig')
            print(f'信号输出完成: {output_path}')

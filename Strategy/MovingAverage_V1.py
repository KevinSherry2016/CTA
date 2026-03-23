import pandas as pd
import numpy as np
import os

marketDataPath = './total/'

infoPath = './Info.csv'
info = pd.read_csv(infoPath, encoding='utf-8-sig')
ts_code_list = info['ts_code'].tolist()

# 参数网格
M_LIST = [20, 30, 40, 50, 60]   # 长周期
N_LIST = [5, 10, 15, 20]        # 短周期
T_LIST = [5, 10, 15, 20]        # 持仓周期

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

def apply_holding_period(signal: pd.Series, holding_period: int) -> pd.Series:
    """持仓周期 T 定义为信号 rolling(window=T).sum()。"""
    return signal.rolling(window=holding_period).sum()

print(f'开始生成参数组合信号')
for long_window in M_LIST:
    for short_window in N_LIST:
        if long_window <= short_window:
            continue
        for holding_period in T_LIST:
            # ── 预计算每个品种的信号序列 ──────────────────────────────────────
            # 信号: +1 做多, -1 做空, 0 空仓
            print(f'正在计算信号: M={long_window}, N={short_window}, T={holding_period}')
            position_series = {}
            for ts_code, df in data.items():
                short_ma = df['adj_close'].rolling(short_window).mean()
                long_ma = df['adj_close'].rolling(long_window).mean()
                raw_signal = pd.Series(
                    np.where(short_ma > long_ma, 1, np.where(short_ma < long_ma, -1, 0)),
                    index=df.index
                )
                signal = apply_holding_period(raw_signal, holding_period)
                position_series[ts_code] = signal

            # ── 合并所有品种信号，对齐全部交易日后输出 CSV ────────────────────
            signals = pd.DataFrame(position_series, index=tradingDayList).fillna(0).astype(float)
            output_name = f'MovingAverageV1_M_{long_window}_N_{short_window}_T_{holding_period}.csv'
            output_path = os.path.join('./Strategy', output_name)
            signals.to_csv(output_path, encoding='utf-8-sig')
            print(f'信号输出完成: {output_path}')

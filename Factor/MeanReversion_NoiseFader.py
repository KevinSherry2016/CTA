# ==============================================================================
# 因子类别：MeanReversion
# 因子名称：MeanReversion_NoiseFader
# 代表意义：基于考夫曼效率比率(KER)的震荡逆势追踪。当趋势充满噪音(低KER)时逆势高抛低吸，避开高效率的直线单边市。
# ==============================================================================

import pandas as pd
import numpy as np
import os

def main():
    marketDataPath = './main_contract/'
    infoPath = './Info.csv'
    calendarPath = './trade_calendar.csv'

    info = pd.read_csv(infoPath, encoding='utf-8-sig')
    valid_sectors = info[~info['sector'].str.lower().isin(['bond', 'stockindex', 'others', 'other', 'financial'])]
    ts_code_list = valid_sectors['ts_code'].tolist()

    data = {}
    print(f"正在加载 MeanReversion_NoiseFader 数据...")

    tradingDayList = []
    for ts_code in ts_code_list:
        filepath = os.path.join(marketDataPath, f"{ts_code}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['trade_date'] = df['trade_date'].astype(str)
            df.set_index('trade_date', inplace=True)
            if len(tradingDayList) == 0:
                tradingDayList = df.index.tolist()
            data[ts_code] = df

    N = 20
    position_series = {}

    print(f"开始计算 MeanReversion_NoiseFader 因子信号...")
    for ts_code, df in data.items():
        close = df['adj_close']
        open_p = df['adj_open']
        high = df['adj_high']
        low = df['adj_low']
        volume = df.get('vol', df.get('Volume', pd.Series(dtype=float)))
        oi = df.get('oi', df.get('OpenInterest', pd.Series(dtype=float)))

        # --- 因子计算逻辑 ---
        # 计算方向性位移和路径波动总和
        direction = (close - close.shift(N)).abs()
        volatility = close.diff().abs().rolling(window=N, min_periods=1).sum()
        
        # 考夫曼效率比率(KER)
        ker = direction / (volatility + 1e-8)
        
        # 噪音比例为1-KER，针对高噪音区间产生反方向均值回归信号
        # 如果是单边市(ker高)，信号会被大幅度压制接近0
        # 如果是震荡市(ker低)，且目前收盘高于N日前，就看空
        trend_dir = np.sign(close - close.shift(N))
        signal = -trend_dir * (1 - ker)
        signal = signal.fillna(0)
        # --------------------

        position_series[ts_code] = signal

    signals = pd.DataFrame(position_series, index=tradingDayList).fillna(0).astype(float)
    output_name = f"MeanReversion_NoiseFader.csv"
    output_path = os.path.join('./Result', output_name)
    signals.to_csv(output_path, encoding='utf-8-sig')
    print(f"因子 MeanReversion_NoiseFader 输出完成: {output_path}")

if __name__ == "__main__":
    main()

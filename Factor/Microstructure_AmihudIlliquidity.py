# ==============================================================================
# 因子类别：Microstructure
# 因子名称：Microstructure_AmihudIlliquidity
# 代表意义：Amihud缺乏流动性指标
# ==============================================================================

import pandas as pd
import numpy as np
import os

def main():
    marketDataPath = './main_contract/'
    infoPath = './Info.csv'
    calendarPath = './trade_calendar.csv'

    # 1. 过滤品种 (仅交易商品期货，不交易bond, stockIndex, Others)
    info = pd.read_csv(infoPath, encoding='utf-8-sig')
    valid_sectors = info[~info['sector'].str.lower().isin(['bond', 'stockindex', 'others', 'other', 'financial'])]
    ts_code_list = valid_sectors['ts_code'].tolist()

    # 2. 读取交易日历
    # 假设交易日历也是csv

    # 3. 提取日度行情数据
    data = {}
    print(f"正在加载 Microstructure_AmihudIlliquidity 数据...")

    tradingDayList = []
    for ts_code in ts_code_list:
        filepath = os.path.join(marketDataPath, f"{ts_code}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['trade_date'] = df['trade_date'].astype(str)
            df.set_index('trade_date', inplace=True)
            # 提取所有的交易日
            if len(tradingDayList) == 0:
                tradingDayList = df.index.tolist()
            data[ts_code] = df

    # 参数
    N = 60
    position_series = {}

    print(f"开始计算 Microstructure_AmihudIlliquidity 因子信号...")
    for ts_code, df in data.items():
        # 获取需要的数据列
        close = df['adj_close']
        open_p = df['adj_open']
        high = df['adj_high']
        low = df['adj_low']
        volume = df.get('vol', df.get('Volume', pd.Series(dtype=float)))
        oi = df.get('oi', df.get('OpenInterest', pd.Series(dtype=float)))

        # --- 因子计算逻辑 ---
        ret = close.pct_change()
        illq = ret.abs() / (volume + 1e-9)
        illq_mean = illq.rolling(window=N).mean()
        baseline_window = 120
        illq_baseline_mean = illq_mean.rolling(window=baseline_window, min_periods=30).mean()
        illq_baseline_std = illq_mean.rolling(window=baseline_window, min_periods=30).std()
        zscore = (illq_mean - illq_baseline_mean) / (illq_baseline_std + 1e-12)
        signal = (-zscore).clip(-3, 3) / 3
        # --------------------

        # 填充到position_series (NaN填为0)
        position_series[ts_code] = signal

    # 4. 合并所有品种信号，输出CSV
    signals = pd.DataFrame(position_series, index=tradingDayList).fillna(0).astype(float)
    output_name = f"Microstructure_AmihudIlliquidity.csv"
    output_path = os.path.join('./Result', output_name)
    signals.to_csv(output_path, encoding='utf-8-sig')
    print(f"因子 Microstructure_AmihudIlliquidity 输出完成: {output_path}")

if __name__ == "__main__":
    main()

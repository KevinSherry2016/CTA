# ==============================================================================
# 因子类别：MeanReversion
# 因子名称：MeanReversion_ShortTermReversal
# 代表意义：单端短期收益率反转。与动量趋势互为镜像，具有极强的负相关性。
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
    print(f"正在加载 MeanReversion_ShortTermReversal 数据...")

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

    N = 40
    position_series = {}

    print(f"开始计算 MeanReversion_ShortTermReversal 因子信号...")
    for ts_code, df in data.items():
        close = df['adj_close']
        open_p = df['adj_open']
        high = df['adj_high']
        low = df['adj_low']
        volume = df.get('vol', df.get('Volume', pd.Series(dtype=float)))
        oi = df.get('oi', df.get('OpenInterest', pd.Series(dtype=float)))

        # --- 因子计算逻辑 ---
        # 纯粹的短期收益率均值回归，最近N天跌得越多越买入
        signal = -close.pct_change(periods=N).fillna(0)
        # --------------------

        position_series[ts_code] = signal

    signals = pd.DataFrame(position_series, index=tradingDayList).fillna(0).astype(float)
    output_name = f"MeanReversion_ShortTermReversal.csv"
    output_path = os.path.join('./Result', output_name)
    signals.to_csv(output_path, encoding='utf-8-sig')
    print(f"因子 MeanReversion_ShortTermReversal 输出完成: {output_path}")

if __name__ == "__main__":
    main()

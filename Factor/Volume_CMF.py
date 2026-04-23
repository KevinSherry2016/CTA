# ==============================================================================
# 因子类别：Volume
# 因子名称：Volume_CMF
# 代表意义：蔡金资金流量指标(Chaikin Money Flow)。利用日内波动的收盘相对位置，以及放量大小计算资金潜伏厚度。可以与传统价格突破实现极佳互补。
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
    print(f"正在加载 Volume_CMF 数据...")

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

    print(f"开始计算 Volume_CMF 因子信号...")
    for ts_code, df in data.items():
        close = df['adj_close']
        open_p = df['adj_open']
        high = df['adj_high']
        low = df['adj_low']
        volume = df.get('vol', df.get('Volume', pd.Series(dtype=float)))
        oi = df.get('oi', df.get('OpenInterest', pd.Series(dtype=float)))

        # --- 因子计算逻辑 ---
        # 资金乘数：当前收盘越接近最高价其乘数愈接近1，而越接近最低价乘数愈接近-1
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low + 1e-8)
        
        # 资金流向体积
        money_flow_volume = money_flow_multiplier * volume
        
        # 过去N天的累积成交量比例作为整体主导信号
        cmf = money_flow_volume.rolling(window=N, min_periods=1).sum() / (volume.rolling(window=N, min_periods=1).sum() + 1e-8)
        
        signal = cmf.fillna(0)
        # --------------------

        position_series[ts_code] = signal

    signals = pd.DataFrame(position_series, index=tradingDayList).fillna(0).astype(float)
    output_name = f"Volume_CMF.csv"
    output_path = os.path.join('./Result', output_name)
    signals.to_csv(output_path, encoding='utf-8-sig')
    print(f"因子 Volume_CMF 输出完成: {output_path}")

if __name__ == "__main__":
    main()

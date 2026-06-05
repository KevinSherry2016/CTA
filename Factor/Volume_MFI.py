# ============================================================================== 
# 因子类别：Volume
# 因子名称：Volume_MFI
# 代表意义：资金流量指标（Money Flow Index）。结合典型价格与成交量，衡量资金流入流出强度。
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
    print(f"正在加载 Volume_MFI 数据...")

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
    N = 14
    position_series = {}

    print(f"开始计算 Volume_MFI 因子信号...")
    for ts_code, df in data.items():
        # 获取需要的数据列
        close = df['adj_close']
        open_p = df['adj_open']
        high = df['adj_high']
        low = df['adj_low']
        volume = df.get('vol', df.get('Volume', pd.Series(dtype=float)))
        oi = df.get('oi', df.get('OpenInterest', pd.Series(dtype=float)))

        # --- 因子计算逻辑 ---
        typical_price = (high + low + close) / 3.0
        raw_money_flow = typical_price * volume

        price_change = typical_price.diff()
        positive_flow = raw_money_flow.where(price_change > 0, 0.0)
        negative_flow = raw_money_flow.where(price_change < 0, 0.0)

        pos_sum = positive_flow.rolling(window=N, min_periods=1).sum()
        neg_sum = negative_flow.rolling(window=N, min_periods=1).sum()
        money_flow_ratio = pos_sum / (neg_sum + 1e-8)
        mfi = 100 - (100 / (1 + money_flow_ratio))

        signal = (mfi - 50) / 50
        # --------------------

        # 填充到position_series (NaN填为0)
        position_series[ts_code] = signal

    # 4. 合并所有品种信号，输出CSV
    signals = pd.DataFrame(position_series, index=tradingDayList).fillna(0).astype(float)
    output_name = f"Volume_MFI.csv"
    output_path = os.path.join('./Result', output_name)
    signals.to_csv(output_path, encoding='utf-8-sig')
    print(f"因子 Volume_MFI 输出完成: {output_path}")


if __name__ == "__main__":
    main()
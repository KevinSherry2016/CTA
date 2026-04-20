# ==============================================================================
# 因子类别：TrendMomentum
# 因子名称：TrendMomentum_MACD
# 代表意义：平滑异同移动平均线
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
    print(f"正在加载 TrendMomentum_MACD 数据...")

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

    # --- 参数定义 ---
    PARAM_LIST = [
        {'fast_n': 6, 'slow_n': 13, 'signal_n': 5},
        {'fast_n': 8, 'slow_n': 17, 'signal_n': 9},
        {'fast_n': 12, 'slow_n': 26, 'signal_n': 9},
        {'fast_n': 19, 'slow_n': 39, 'signal_n': 9},
        {'fast_n': 24, 'slow_n': 52, 'signal_n': 18}
    ]
    # --------------------

    print(f"开始计算 TrendMomentum_MACD 因子信号...")
    for param in PARAM_LIST:
        fast_n = param['fast_n']
        slow_n = param['slow_n']
        signal_n = param['signal_n']
        position_series = {}

        for ts_code, df in data.items():
            # 获取需要的数据列
            close = df['adj_close']

            # --- 因子计算逻辑 ---
            dif = close.ewm(span=fast_n, adjust=False).mean() - close.ewm(span=slow_n, adjust=False).mean()
            dea = dif.ewm(span=signal_n, adjust=False).mean()
            signal = (dif - dea) * 2
            # --------------------

            # 填充到position_series (NaN填为0)
            position_series[ts_code] = signal

        # 4. 合并所有品种信号，输出CSV
        signals = pd.DataFrame(position_series, index=tradingDayList).fillna(0).astype(float)
        output_name = f"TrendMomentum_MACD_{fast_n}_{slow_n}_{signal_n}.csv"
        output_path = os.path.join('./Result', output_name)
        signals.to_csv(output_path, encoding='utf-8-sig')
        print(f"因子 TrendMomentum_MACD 输出完成: {output_path}")

if __name__ == "__main__":
    main()

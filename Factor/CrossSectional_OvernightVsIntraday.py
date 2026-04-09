# ==============================================================================
# 因子类别：CrossSectional
# 因子名称：CrossSectional_OvernightVsIntraday
# 代表意义：隔夜与日内收益率差异
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
    print(f"正在加载 CrossSectional_OvernightVsIntraday 数据...")

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
    N = 25
    position_series = {}

    print(f"开始计算 CrossSectional_OvernightVsIntraday 因子信号...")
    for ts_code, df in data.items():
        # 获取需要的数据列
        close = df.get('close', df.get('Close', pd.Series(dtype=float)))
        open_p = df.get('open', df.get('Open', pd.Series(dtype=float)))
        high = df.get('high', df.get('High', pd.Series(dtype=float)))
        low = df.get('low', df.get('Low', pd.Series(dtype=float)))
        volume = df.get('vol', df.get('Volume', pd.Series(dtype=float)))
        oi = df.get('oi', df.get('OpenInterest', pd.Series(dtype=float)))

        # --- 因子计算逻辑 ---
        intraday_ret = close / (open_p + 1e-9) - 1
        overnight_ret = open_p / (close.shift(1) + 1e-9) - 1
        signal = (intraday_ret - overnight_ret).rolling(window=N).mean()
        # --------------------

        # 填充到position_series (NaN填为0)
        position_series[ts_code] = signal

    # 4. 合并所有品种信号，输出CSV
    signals = pd.DataFrame(position_series, index=tradingDayList).fillna(0).astype(float)
    output_name = f"CrossSectional_OvernightVsIntraday.csv"
    output_path = os.path.join('./Result', output_name)
    signals.to_csv(output_path, encoding='utf-8-sig')
    print(f"因子 CrossSectional_OvernightVsIntraday 输出完成: {output_path}")

if __name__ == "__main__":
    main()

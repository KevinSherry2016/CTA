# ==============================================================================
# 因子类别：Volume
# 因子名称：Volume_PriceOiDivergence
# 代表意义：量价背离与资金流向。利用持仓量变动方向鉴别行情质量。涨跌由增仓推动信号同向，由减仓推动则产生逆向反转信号。
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
    print(f"正在加载 Volume_PriceOiDivergence 数据...")

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

    print(f"开始计算 Volume_PriceOiDivergence 因子信号...")
    for ts_code, df in data.items():
        close = df['adj_close']
        open_p = df['adj_open']
        high = df['adj_high']
        low = df['adj_low']
        volume = df.get('vol', df.get('Volume', pd.Series(dtype=float)))
        oi = df.get('oi', df.get('OpenInterest', pd.Series(dtype=float)))

        # --- 因子计算逻辑 ---
        # 1. 判断换月：通过复权因子变动进行判断
        raw_close = df.get('close', close)
        adj_factor = close / raw_close.replace(0, np.nan)
        is_rollover = adj_factor.diff().abs() > 1e-5
        
        # 2. 只要过去N天内发生过换月，直接剔除并用 ffill 继承之前信号
        invalid_window = is_rollover.rolling(window=N, min_periods=1).max() > 0

        price_ret = close.pct_change(periods=N).fillna(0)
        oi_change = oi.pct_change(periods=N).fillna(0)
        
        # 增仓表明趋势成立(同向)，减仓表明趋势存疑产生衰竭(反向)
        signal = price_ret * np.sign(oi_change)
        
        signal[invalid_window] = np.nan
        signal = signal.ffill()
        # --------------------

        position_series[ts_code] = signal

    signals = pd.DataFrame(position_series, index=tradingDayList).fillna(0).astype(float)
    output_name = f"Volume_PriceOiDivergence.csv"
    output_path = os.path.join('./Result', output_name)
    signals.to_csv(output_path, encoding='utf-8-sig')
    print(f"因子 Volume_PriceOiDivergence 输出完成: {output_path}")

if __name__ == "__main__":
    main()

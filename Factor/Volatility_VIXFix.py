# ==============================================================================
# 因子类别：Volatility
# 因子名称：Volatility_VIXFix
# 代表意义：基于Williams VIX Fix的合成商品恐慌指数。捕捉市场超跌过度恐慌时的均值回归买点（左侧抄底）与乐观抛售。与传统趋势高度负相关。
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
    print(f"正在加载 Volatility_VIXFix 数据...")

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

    print(f"开始计算 Volatility_VIXFix 因子信号...")
    for ts_code, df in data.items():
        close = df['adj_close']
        open_p = df['adj_open']
        high = df['adj_high']
        low = df['adj_low']
        volume = df.get('vol', df.get('Volume', pd.Series(dtype=float)))
        oi = df.get('oi', df.get('OpenInterest', pd.Series(dtype=float)))

        # --- 因子计算逻辑 ---
        # 恐慌发生时，价格往往从高点剧烈回落
        highest_close = close.rolling(window=N, min_periods=1).max()
        vix_fix = (highest_close - low) / (highest_close + 1e-8) * 100
        
        # v_fix 飙升代表价格刚经历惨烈下跌并被极度低估，此时做多捕捉修复反转
        # 产生正向回归信号做超跌修复反弹。反之亦然
        signal = vix_fix.fillna(0)
        # --------------------

        position_series[ts_code] = signal

    signals = pd.DataFrame(position_series, index=tradingDayList).fillna(0).astype(float)
    output_name = f"Volatility_VIXFix.csv"
    output_path = os.path.join('./Result', output_name)
    signals.to_csv(output_path, encoding='utf-8-sig')
    print(f"因子 Volatility_VIXFix 输出完成: {output_path}")

if __name__ == "__main__":
    main()

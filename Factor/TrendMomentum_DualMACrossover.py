# ==============================================================================
# 因子族：TrendMomentum
# 因子名称：TrendMomentum_DualMACrossover
# 因子说明：双均线交叉信号
# ==============================================================================

import pandas as pd
import numpy as np
import os

def main():
    marketDataPath = './main_contract/'
    infoPath = './Info.csv'
    calendarPath = './trade_calendar.csv'

    # 1. 过滤品种 (由于主要做商品期货，排除掉bond, stockIndex, Others)
    info = pd.read_csv(infoPath, encoding='utf-8-sig')
    valid_sectors = info[~info['sector'].str.lower().isin(['bond', 'stockindex', 'others', 'other', 'financial'])]
    ts_code_list = valid_sectors['ts_code'].tolist()

    # 2. 读取交易日历
    # 此处如果需要也可以从csv读

    # 3. 读取日度行情数据
    data = {}
    print(f"正在加载 TrendMomentum_DualMACrossover 数据...")

    tradingDayList = []
    for ts_code in ts_code_list:
        filepath = os.path.join(marketDataPath, f"{ts_code}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['trade_date'] = df['trade_date'].astype(str)
            df.set_index('trade_date', inplace=True)
            # 获取所有的交易日
            if len(tradingDayList) == 0:
                tradingDayList = df.index.tolist()
            data[ts_code] = df

    # --- 参数定义 ---
    PARAM_LIST = [
        {'fast_n': 5, 'slow_n': 20},
        {'fast_n': 10, 'slow_n': 40},
        {'fast_n': 10, 'slow_n': 60},
        {'fast_n': 20, 'slow_n': 40},
        {'fast_n': 20, 'slow_n': 60},
        {'fast_n': 30, 'slow_n': 60},
    ]
    # --------------------

    print(f"开始计算 TrendMomentum_DualMACrossover 因子信号...")
    for param in PARAM_LIST:
        fast_n = param['fast_n']
        slow_n = param['slow_n']
        position_series = {}
        
        for ts_code, df in data.items():
            # 获取需要的数据列
            close = df['adj_close']
            open_p = df['adj_open']
            high = df['adj_high']
            low = df['adj_low']
            volume = df.get('vol', df.get('Volume', pd.Series(dtype=float)))
            oi = df.get('oi', df.get('OpenInterest', pd.Series(dtype=float)))

            # --- 因子计算逻辑 ---
            short_ma = close.rolling(window=fast_n).mean()
            long_ma = close.rolling(window=slow_n).mean()
            signal = (short_ma / long_ma - 1).apply(lambda x: np.sign(x) if pd.notna(x) else np.nan)
            # --------------------

            # 分配到position_series (NaN视为0)
            position_series[ts_code] = signal

        # 4. 合并所有品种信号并输出CSV
        signals = pd.DataFrame(position_series, index=tradingDayList).fillna(0).astype(float)
        output_name = f"TrendMomentum_DualMACrossover_{fast_n}_{slow_n}.csv"
        output_path = os.path.join('./Result', output_name)
        signals.to_csv(output_path, encoding='utf-8-sig')
        print(f"生成 TrendMomentum_DualMACrossover 成功，路径: {output_path}")

if __name__ == '__main__':
    main()

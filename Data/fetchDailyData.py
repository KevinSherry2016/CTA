import tushare as ts
import time
import pandas as pd
import os
from datetime import datetime, timedelta

pro = ts.pro_api('d5de2aa0de5bf28ad29b96416062e16d894b864c6aa6d526de10e35c')

def fetch_all_daily(start_date='20100104', end_date='20260404', save_dir='daily_data'):
    """遍历每一天（含非交易日），通过fut_daily获取全部日线行情数据"""
    os.makedirs(save_dir, exist_ok=True)

    # 生成起止日期之间的每一天
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    trade_dates = []
    cur = start
    while cur <= end:
        trade_dates.append(cur.strftime('%Y%m%d'))
        cur += timedelta(days=1)

    for i, date in enumerate(trade_dates):
        save_path = os.path.join(save_dir, f'{date}.csv')
        if os.path.exists(save_path):
            print(f'[{i+1}/{len(trade_dates)}] {date} 已存在，跳过')
            continue
        print(f'[{i+1}/{len(trade_dates)}] 正在获取 {date} 的全部日线行情数据...')
        try:
            df = pro.fut_daily(trade_date=date)
            if df is not None and not df.empty:
                df.to_csv(save_path, index=False)
                print(f'  已保存 {len(df)} 条记录至 {save_path}')
        except Exception as e:
            print(f'  获取 {date} 数据失败: {e}')
        # 控制请求频率，避免被封禁
        time.sleep(3)

    print(f'全部完成，数据已保存至 {save_dir}/')


def generate_trade_calendar(save_dir='daily_data', output='trade_calendar.csv'):
    """遍历daily_data下的csv文件，提取日期作为交易日历保存"""
    dates = [f.replace('.csv', '') for f in os.listdir(save_dir) if f.endswith('.csv')]
    dates.sort()
    cal_df = pd.DataFrame({'trade_date': dates})
    cal_df.to_csv(output, index=False)
    print(f'交易日历已保存至 {output}，共 {len(dates)} 个交易日')
    return cal_df


if __name__ == '__main__':
    fetch_all_daily()
    generate_trade_calendar()


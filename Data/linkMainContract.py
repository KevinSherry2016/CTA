import tushare as ts
import time
import pandas as pd
import os

pro = ts.pro_api('d5de2aa0de5bf28ad29b96416062e16d894b864c6aa6d526de10e35c')

main_dir = 'main_contract'


def process_main_contract():
    """为每个品种获取pre_main_close并计算复权价格"""
    files = sorted([f for f in os.listdir(main_dir) if f.endswith('.csv')])

    for f in files:
        ts_code = f.replace('.csv', '')
        file_path = os.path.join(main_dir, f)

        print(f'正在处理 {ts_code}...')
        df = pd.read_csv(file_path, dtype={'ts_code': str, 'trade_date': str})

        # 跳过已处理的品种
        if 'adj_close' in df.columns:
            print(f'  {ts_code} 已处理，跳过')
            continue

        df.sort_values(by='trade_date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 获取换月时前主力合约的收盘价
        df['pre_main_close'] = pd.NA
        for i in range(1, len(df)):
            prev_ts_code = df.loc[i - 1, 'ts_code']
            curr_ts_code = df.loc[i, 'ts_code']

            if prev_ts_code != curr_ts_code:
                trading_day = df.loc[i, 'trade_date']
                print(f'  {trading_day} 换月: {prev_ts_code} -> {curr_ts_code}，查询 {prev_ts_code} 当日收盘价...')
                try:
                    temp = pro.fut_daily(ts_code=prev_ts_code, start_date=trading_day, end_date=trading_day)
                    if temp is not None and not temp.empty:
                        close = temp.loc[0, 'close']
                        df.loc[i, 'pre_main_close'] = close
                        print(f'    pre_main_close = {close}')
                    else:
                        print(f'    *** 未查到 {prev_ts_code} 在 {trading_day} 的数据 ***')
                except Exception as e:
                    print(f'    *** 查询失败: {e} ***')
                time.sleep(3)

        # 计算复权因子和复权价格
        print(f'  正在计算 {ts_code} 的复权因子...')
        df['roll_factor'] = 1.0
        for i in range(1, len(df)):
            if pd.notna(df.loc[i, 'pre_main_close']):
                df.loc[i, 'roll_factor'] = df.loc[i, 'pre_main_close'] / df.loc[i, 'close']
        df['roll_factor'] = df['roll_factor'].cumprod()

        df['adj_close'] = df['close'] * df['roll_factor']
        df['adj_open'] = df['open'] * df['roll_factor']
        df['adj_high'] = df['high'] * df['roll_factor']
        df['adj_low'] = df['low'] * df['roll_factor']
        df['adj_settle'] = df['settle'] * df['roll_factor']

        df.to_csv(file_path, index=False)
        print(f'  {ts_code} 已保存至 {file_path}，共 {len(df)} 条')


if __name__ == '__main__':
    process_main_contract()

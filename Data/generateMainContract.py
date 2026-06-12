import tushare as ts
import time
import pandas as pd
import os
import re
from datetime import datetime, timedelta

pro = ts.pro_api('d5de2aa0de5bf28ad29b96416062e16d894b864c6aa6d526de10e35c')

info_df = pd.read_csv('Info.csv', dtype={'ts_code': str})
ts_code_list = info_df['ts_code'].tolist()


def get_contract_ym(ts_code):
    """从合约代码中提取交割年月，如CU1004.SHF -> (2010, 4)"""
    match = re.match(r'[A-Za-z]+(\d{4})\.\w+', ts_code)
    if match:
        yymm = match.group(1)
        yy = int(yymm[:2])
        mm = int(yymm[2:])
        return (2000 + yy, mm)
    return None


def calculate_main_contract(ts_code_list, start_date='20100101', end_date=datetime.now().strftime('%Y%m%d'),
                            daily_dir='daily_data', calendar_file='trade_calendar.csv',
                            save_dir='main_contract'):
    """遍历品种，从daily_data中确定每日主力合约并保存"""
    os.makedirs(save_dir, exist_ok=True)

    # 读取交易日历并筛选日期范围
    cal_df = pd.read_csv(calendar_file, dtype={'trade_date': str})
    trade_dates = cal_df[(cal_df['trade_date'] >= start_date) & (cal_df['trade_date'] <= end_date)]['trade_date'].tolist()

    for ts_code in ts_code_list:
        product = ts_code.split('.')[0]  # 如 CU
        exchange = ts_code.split('.')[1]  # 如 SHF
        print(f'正在计算 {ts_code} 的主力合约...')


        save_path = os.path.join(save_dir, f'{ts_code}.csv')
        existing_results = []
        actual_start_date = start_date
        current_main_ym = None  # current main contract ym
        current_main_code = None  # current main contract code

        if os.path.exists(save_path):
            existing_df = pd.read_csv(save_path, dtype={'ts_code': str, 'trade_date': str})
            if not existing_df.empty:
                existing_results = existing_df.to_dict('records')
                last_date = existing_df['trade_date'].max()
                last_date_dt = datetime.strptime(last_date, '%Y%m%d')
                actual_start_date = (last_date_dt + timedelta(days=1)).strftime('%Y%m%d')
                last_ts_code = existing_df.iloc[-1]['mapping_ts_code'] if 'mapping_ts_code' in existing_df.columns else existing_df.iloc[-1]['ts_code']
                current_main_code = last_ts_code
                current_main_ym = get_contract_ym(last_ts_code)
                print(f'  Found existing output, append from {actual_start_date}, current main: {current_main_code}')
        
        sub_trade_dates = cal_df[(cal_df['trade_date'] >= actual_start_date) & (cal_df['trade_date'] <= end_date)]['trade_date'].tolist()

        results = existing_results
        total = len(sub_trade_dates)


        for idx, date in enumerate(sub_trade_dates):
            daily_file = os.path.join(daily_dir, f'{date}.csv')
            if not os.path.exists(daily_file):
                continue

            df = pd.read_csv(daily_file, dtype={'ts_code': str, 'trade_date': str})

            # 筛选该品种的具体合约（排除聚合行如CU.SHF）
            pattern = re.compile(rf'^{product}\d{{4}}\.{exchange}$')
            contracts = df[df['ts_code'].apply(lambda x: bool(pattern.match(x)))].copy()

            if contracts.empty:
                continue

            # 解析每个合约的交割年月
            contracts['delivery_ym'] = contracts['ts_code'].apply(get_contract_ym)
            contracts = contracts[contracts['delivery_ym'].notna()]

            # 约束1：排除已进入交割月当月的合约（合约YYMM <= 当前月份则已过期或需强制换月）
            trade_dt = datetime.strptime(date, '%Y%m%d')
            current_ym = (trade_dt.year, trade_dt.month)
            contracts = contracts[contracts['delivery_ym'].apply(lambda ym: ym > current_ym)]

            if contracts.empty:
                continue

            # 约束2：主力合约只能往后，不能往前
            if current_main_ym is not None:
                contracts = contracts[contracts['delivery_ym'].apply(lambda ym: ym >= current_main_ym)]

            if contracts.empty:
                continue

            # 按OI降序排列，取最大的作为主力
            contracts = contracts.sort_values(by='oi', ascending=False)
            main_row = contracts.iloc[0]
            main_ts_code = main_row['ts_code']
            main_ym = main_row['delivery_ym']

            sub_mapping_ts_code = None
            sub_row_dict = {}
            # 次主力要求到期晚于主力：在OI排序后，选第一个 delivery_ym > main_ym 的合约
            sub_candidates = contracts[contracts['delivery_ym'].apply(lambda ym: ym > main_ym)]
            if not sub_candidates.empty:
                sub_row = sub_candidates.iloc[0]
                sub_mapping_ts_code = sub_row['ts_code']
                for col in sub_row.index:
                    if col not in ['ts_code', 'trade_date', 'delivery_ym']:
                        sub_row_dict[f'sub_{col}'] = sub_row[col]

            # 输出进度和换月信息
            if current_main_code != main_ts_code:
                print(f'  [{idx+1}/{total}] {date} 主力切换: {current_main_code} -> {main_ts_code} (oi={main_row["oi"]})')
                current_main_code = main_ts_code

            current_main_ym = main_ym
            new_row = main_row.drop('delivery_ym').to_dict()
            new_row['mapping_ts_code'] = new_row.pop('ts_code')
            new_row['sub_mapping_ts_code'] = sub_mapping_ts_code
            new_row.update(sub_row_dict)
            new_row['ts_code'] = ts_code
            results.append(new_row)

        if results:
            result_df = pd.DataFrame(results)
            result_df.sort_values(by='trade_date', inplace=True)
            
            save_path = os.path.join(save_dir, f'{ts_code}.csv')
            result_df.to_csv(save_path, index=False)
            print(f'  {ts_code} 主力合约已保存至 {save_path}，共 {len(result_df)} 条')
        else:
            print(f'  {ts_code} 未找到主力合约数据')

if __name__ == '__main__':
    calculate_main_contract(ts_code_list)

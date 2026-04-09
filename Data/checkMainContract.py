import pandas as pd
import os

main_dir = 'main_contract'


def check_main_contract():
    files = sorted([f for f in os.listdir(main_dir) if f.endswith('.csv')])

    contracts_by_year = {}  # {品种: {年份: [合约列表]}}
    count_by_year = {}      # {品种: {年份: 合约个数}}
    all_years = set()

    for f in files:
        ts_code = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(main_dir, f), dtype={'trade_date': str, 'mapping_ts_code': str})
        df.sort_values(by='trade_date', inplace=True)

        contracts_by_year[ts_code] = {}
        count_by_year[ts_code] = {}

        # 按时间顺序提取每年的主力合约（去重保持顺序）
        df['year'] = df['trade_date'].str[:4]
        for year, group in df.groupby('year'):
            all_years.add(year)
            seen = []
            for code in group['mapping_ts_code']:
                if not seen or seen[-1] != code:
                    seen.append(code)
            contracts_by_year[ts_code][year] = '/'.join(seen)
            count_by_year[ts_code][year] = len(seen)

    all_years = sorted(all_years)

    # 生成CSV1：每年的主力合约列表
    rows1 = []
    for ts_code in sorted(contracts_by_year):
        row = {'品种': ts_code}
        for year in all_years:
            row[year] = contracts_by_year[ts_code].get(year, '')
        rows1.append(row)
    df1 = pd.DataFrame(rows1)

    # 生成Sheet2：每年的主力合约个数
    rows2 = []
    for ts_code in sorted(count_by_year):
        row = {'品种': ts_code}
        for year in all_years:
            row[year] = count_by_year[ts_code].get(year, '')
        rows2.append(row)
    df2 = pd.DataFrame(rows2)

    # 保存到同一个Excel文件的两个Sheet
    with pd.ExcelWriter('main_contract_check.xlsx', engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='主力合约明细', index=False)
        df2.to_excel(writer, sheet_name='主力合约个数', index=False)
    print(f'已保存 main_contract_check.xlsx')


if __name__ == '__main__':
    check_main_contract()

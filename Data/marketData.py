import tushare as ts
import time
import pandas as pd

pro = ts.pro_api('d5de2aa0de5bf28ad29b96416062e16d894b864c6aa6d526de10e35c')

ts_code_list = [
                # 'CU.SHF',
                # 'AL.SHF',
                # 'ZN.SHF',
                # 'PB.SHF',
                # 'NI.SHF',
                # 'SN.SHF',
                # 'AO.SHF',
                # 'AD.SHF',
                # 'AU.SHF',
                # 'AG.SHF',
                # 'RB.SHF',
                # 'HC.SHF',
                # 'SS.SHF',
                # 'SC.SHF',
                # 'FU.SHF',
                # 'LU.SHF',
                # 'BU.SHF',
                # 'RU.SHF',
                # 'SP.SHF',
                # 'OP.SHF',
                # 'EC.SHF',
                # 'A.DCE',
                # 'B.DCE',
                # 'M.DCE',
                # 'Y.DCE',
                # 'P.DCE',
                # 'C.DCE',
                # 'CS.DCE',
                # 'JD.DCE',
                # 'LH.DCE',
                # 'LG.DCE',
                # 'JM.DCE',
                # 'J.DCE',
                # 'I.DCE',
                # 'L.DCE',
                # 'V.DCE',
                # 'PP.DCE',
                # 'EG.DCE',
                # 'EB.DCE',
                # 'PG.DCE',
                # 'BZ.DCE',
                # 'CF.ZCE',
                # 'SR.ZCE',
                # 'OI.ZCE',
                # 'RM.ZCE',
                # 'CY.ZCE',
                # 'AP.ZCE',
                # 'PTA.ZCE',
                # 合约从ME变为MA，需要额外处理
                # 'MA.ZCE',
                # 'FG.ZCE',
                # 'SF.ZCE',
                # 'SM.ZCE',
                # 'UR.ZCE',
                # 'SA.ZCE',
                # 'PF.ZCE',
                # 'PX.ZCE',
                # 'SH.ZCE',
                # 'PL.ZCE',
                # 'IF.CFX',
                # 'IC.CFX', 
                # 'IM.CFX',
                # 'T.CFX',
                # 'TS.CFX',
                # 'TF.CFX',

                #无法获取数据
                #'TL.CFX',
]

for ts_code in ts_code_list:
    print(f'正在获取{ts_code}的日线行情数据...')
    daily_df = pro.fut_daily(ts_code=ts_code, start_date='20100101', end_date='20180101')
    daily_df.sort_values(by='trade_date', inplace=True)
    daily_df.to_csv(f'{ts_code}_daily.csv', index=False)
    print(f'正在获取{ts_code}的主力合约...')
    mapping_df = pro.fut_mapping(ts_code=ts_code)
    mapping_df.sort_values(by='trade_date', inplace=True)
    if ts_code == 'BU.SHF':
        mapping_df.loc[mapping_df['trade_date'].astype(str) == '20140915', 'mapping_ts_code'] = 'BU1412.SHF'
    if ts_code == 'PB.SHF':
        mapping_df.loc[mapping_df['trade_date'].astype(str) == '20130916', 'mapping_ts_code'] = 'PB1312.SHF'
    if ts_code == 'FU.SHF':
        mapping_df.loc[mapping_df['trade_date'].astype(str) == '20111130', 'mapping_ts_code'] = 'FU1203.SHF'
        mapping_df.loc[mapping_df['trade_date'].astype(str) == '20130426', 'mapping_ts_code'] = 'FU1309.SHF'
    if ts_code == 'B.DCE':
        mapping_df.loc[mapping_df['trade_date'].astype(str) == '20140915', 'mapping_ts_code'] = 'B1505.DCE'
    if ts_code == 'SM.ZCE':
        mapping_df.loc[mapping_df['trade_date'].astype(str) == '20150116', 'mapping_ts_code'] = 'SM1505.ZCE'
    mapping_df.to_csv(f'{ts_code}_mapping.csv', index=False)
    print(f'正在合并{ts_code}成完整的日线行情数据...')
    mapping_df.drop(columns=['ts_code'], inplace=True)
    total_df = pd.merge(daily_df, mapping_df, on='trade_date', how='left')
    total_df.sort_values(by='trade_date', inplace=True)
    if ts_code == 'IF.CFX':
        total_df = total_df[total_df['trade_date']>= '20150115']
    if ts_code == 'TF.CFX':
        total_df = total_df[total_df['trade_date']>= '20150105']
    total_df =total_df.reset_index()
    total_df.to_csv(f'{ts_code}_total.csv', index=False)
    print(f'正在添加{ts_code}中换月当天前主力合约的收盘价...')
    total_df['pre_main_close'] = pd.NA
    for i in range(1, len(total_df)):
        tradingDay = total_df.loc[i, 'trade_date']
        prev_mapping_ts_code = total_df.loc[i-1, 'mapping_ts_code']
        mapping_ts_code = total_df.loc[i, 'mapping_ts_code']
        if(prev_mapping_ts_code != mapping_ts_code):
            print('querying prev close for', prev_mapping_ts_code, tradingDay)
            temp = pro.fut_daily(ts_code=prev_mapping_ts_code, start_date=str(tradingDay), end_date=str(tradingDay))
            close = temp.loc[0, 'close']
            total_df.loc[i, 'pre_main_close'] = close
            time.sleep(5)
    total_df.sort_values(by='trade_date', inplace=True)
    total_df.to_csv(f'{ts_code}_with_pre_main_close.csv', index=False)
    print(f'正在计算{ts_code}复权因子...')
    total_df = pd.read_csv(f'{ts_code}_with_pre_main_close.csv')

    total_df['roll_factor'] = 1.0
    for i in range(1, len(total_df)):
        if pd.notna(total_df.loc[i, 'pre_main_close']):
            total_df.loc[i, 'roll_factor'] = total_df.loc[i, 'pre_main_close']/total_df.loc[i, 'close']
    total_df['roll_factor'] = total_df['roll_factor'].cumprod()
    total_df['adj_close'] = total_df['close'] * total_df['roll_factor']
    total_df['adj_open'] = total_df['open'] * total_df['roll_factor']
    total_df['adj_high'] = total_df['high'] * total_df['roll_factor']
    total_df['adj_low'] = total_df['low'] * total_df['roll_factor']
    total_df['adj_settle'] = total_df['settle'] * total_df['roll_factor']
    total_df.to_csv(f'{ts_code}_with_roll_factor.csv', index=False)
    # 避免请求过快导致被封禁
    time.sleep(10)

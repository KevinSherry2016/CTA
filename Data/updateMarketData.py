import tushare as ts
import time
import pandas as pd

pro = ts.pro_api('d5de2aa0de5bf28ad29b96416062e16d894b864c6aa6d526de10e35c')

ts_code_list = [
                'CU.SHF',
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
    daily_df = pro.fut_daily(ts_code=ts_code, start_date='20260101', end_date=str(time.strftime('%Y%m%d')))
    daily_df.sort_values(by='trade_date', inplace=True)
    daily_df.to_csv(f'{ts_code}_daily.csv', index=False)
    print(f'正在获取{ts_code}的主力合约...')
    mapping_df = pro.fut_mapping(ts_code=ts_code)
    mapping_df.sort_values(by='trade_date', inplace=True)
    mapping_df.to_csv(f'{ts_code}_mapping.csv', index=False)
    
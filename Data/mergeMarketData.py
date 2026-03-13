import pandas as pd

ts_code_list = [
                'CU.SHF',
                'AL.SHF',
                'ZN.SHF',
                'PB.SHF',
                'NI.SHF',
                'SN.SHF',
                'AO.SHF',
                'AD.SHF',
                'AU.SHF',
                'AG.SHF',
                'RB.SHF',
                'HC.SHF',
                'SS.SHF',
                'SC.SHF',
                'FU.SHF',
                'LU.SHF',
                'BU.SHF',
                'RU.SHF',
                'SP.SHF',
                'OP.SHF',
                'EC.SHF',
                'A.DCE',
                'B.DCE',
                'M.DCE',
                'Y.DCE',
                'P.DCE',
                'C.DCE',
                'CS.DCE',
                'JD.DCE',
                'LH.DCE',
                'LG.DCE',
                'JM.DCE',
                'J.DCE',
                'I.DCE',
                'L.DCE',
                'V.DCE',
                'PP.DCE',
                'EG.DCE',
                'EB.DCE',
                'PG.DCE',
                'BZ.DCE',
                'CF.ZCE',
                'SR.ZCE',
                'OI.ZCE',
                'RM.ZCE',
                'CY.ZCE',
                'AP.ZCE',
                'PTA.ZCE',
                'MA.ZCE',
                'FG.ZCE',
                'SF.ZCE',
                'SM.ZCE',
                'UR.ZCE',
                'SA.ZCE',
                'PF.ZCE',
                'PX.ZCE',
                'SH.ZCE',
                'PL.ZCE',
                'IF.CFX',
                'IC.CFX', 
                'IM.CFX',
                'T.CFX',
                'TS.CFX',
                'TF.CFX',

                #无法获取数据
                #'TL.CFX',
]

for ts_code in ts_code_list:
    print(f'正在处理{ts_code}...')
    part1 = pd.read_csv(f'./part1/{ts_code}_with_pre_main_close.csv')
    part2 = pd.read_csv(f'./part2/{ts_code}_with_pre_main_close.csv')
    total_df = pd.concat([part1, part2], ignore_index=True)
    total_df.sort_values(by='trade_date', inplace=True)
    total_df.drop_duplicates(subset='trade_date', keep='first', inplace=True)
    total_df =total_df.reset_index()
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
    total_df.drop(columns=['index'], inplace=True)
    total_df.to_csv(f'{ts_code}.csv', index=False)

# ==============================================================================
# 因子类别：CrossSectional
# 因子名称：CrossSectional_OvernightVsIntraday
# 代表意义：隔夜与日内收益率差异
# ==============================================================================

import pandas as pd
import numpy as np
import os

def rolling_zscore(series, window=60):
    min_p = max(window // 2, 1)
    mu = series.rolling(window, min_periods=min_p).mean()
    sigma = series.rolling(window, min_periods=min_p).std().replace(0, np.nan)
    return (series - mu) / sigma

def main():
    marketDataPath = './main_contract/'
    infoPath = './Info.csv'
    FINAL_VOL_WINDOW = 20

    # 1. 过滤品种与划分板块
    info = pd.read_csv(infoPath, encoding='utf-8-sig')
    EXCLUDE_SECTORS = ['stockindex', 'bond', 'other', 'others', 'financial']
    sector_map = info.groupby('sector')['ts_code'].apply(list).to_dict()

    # 2. 提取日度行情数据
    data = {}
    print(f"正在加载 CrossSectional_OvernightVsIntraday 数据...")

    tradingDayList = []
    for ts_code in info['ts_code']:
        filepath = os.path.join(marketDataPath, f"{ts_code}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, dtype={'trade_date': str})
            df.set_index('trade_date', inplace=True)
            if len(tradingDayList) == 0:
                tradingDayList = df.index.tolist()
            for col in ['open', 'high', 'low', 'close', 'adj_close', 'adj_open', 'adj_high', 'adj_low', 'vol', 'oi']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            data[ts_code] = df

    # 参数列表 (默认候选)
    N_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

    # ========== 回测指定设置 ==========
    # 指定要回测的板块列表（为空代表全部有效板块，如 ['Black', 'Chemical']）
    TARGET_SECTORS = ['Precious']
    # 指定要获取计算的参数列表（为空代表默认测试 N_LIST 所有候选，如 [20, 40])
    TARGET_PARAMS = [20]
    # 指定要计算的信号处理模式（为空代表全部计算，可选：'No zscore', 'zscore', 'State Machine'）
    TARGET_MODES = ['State Machine']
    # 评判最优参数的基准模式（决定将什么参数下的品种仓位选入最终投资组合中，当该模式未计算时采用列表第一个计算的模式）
    EVAL_MODE = 'No zscore'
    # 仓位平滑天数（1代表不平滑，大于1代表进行N天滚动平均平滑）
    POSITION_SMOOTH_DAYS = 10
    # ==================================

    print(f"开始进行板块回测与参数评估...")

    test_n_list = TARGET_PARAMS if TARGET_PARAMS else N_LIST

    results = []  # 记录所有 Sector-参数 组合的表现
    best_positions_noz = {}
    best_positions_z = {}
    best_positions_sm = {}

    for sector, ts_codes in sector_map.items():
        if pd.isna(sector) or str(sector).lower() in EXCLUDE_SECTORS: continue
        # 通过 TARGET_SECTORS 过滤特定板块
        if TARGET_SECTORS and sector not in TARGET_SECTORS: continue

        valid_symbols = [c for c in ts_codes if c in data]
        if not valid_symbols: continue

        best_n = None
        max_metric = -999
        sector_best_noz, sector_best_z, sector_best_sm = {}, {}, {}

        for N in test_n_list:
            pnl_noz, pnl_z, pnl_sm = [], [], []
            temp_pos_noz, temp_pos_z, temp_pos_sm = {}, {}, {}

            for ts_code in valid_symbols:
                df = data[ts_code]
                close = df['adj_close']
                open_p = df['adj_open']
                daily_ret = close.pct_change(fill_method=None)
                vol = daily_ret.rolling(FINAL_VOL_WINDOW, min_periods=1).std().replace(0, np.nan)

                # --- 因子计算逻辑 ---
                intraday_ret = close / (open_p + 1e-9) - 1
                overnight_ret = open_p / (close.shift(1) + 1e-9) - 1
                raw_sig = (intraday_ret - overnight_ret).rolling(window=N).mean()
                # --------------------

                # 计算指定的信号模式仓位和单日盈亏
                if not TARGET_MODES or 'No zscore' in TARGET_MODES:
                    pos_noz = raw_sig / vol
                    if POSITION_SMOOTH_DAYS > 1:
                        pos_noz = pos_noz.rolling(POSITION_SMOOTH_DAYS, min_periods=1).mean()
                    temp_pos_noz[ts_code] = pos_noz
                    pnl_noz.append(pos_noz.shift(1) * daily_ret)

                if not TARGET_MODES or 'zscore' in TARGET_MODES:
                    z_sig = rolling_zscore(raw_sig, 60)
                    pos_z = z_sig / vol
                    if POSITION_SMOOTH_DAYS > 1:
                        pos_z = pos_z.rolling(POSITION_SMOOTH_DAYS, min_periods=1).mean()
                    temp_pos_z[ts_code] = pos_z
                    pnl_z.append(pos_z.shift(1) * daily_ret)

                if not TARGET_MODES or 'State Machine' in TARGET_MODES:
                    sm_sig = np.sign(raw_sig)
                    pos_sm = sm_sig / vol
                    if POSITION_SMOOTH_DAYS > 1:
                        pos_sm = pos_sm.rolling(POSITION_SMOOTH_DAYS, min_periods=1).mean()
                    temp_pos_sm[ts_code] = pos_sm
                    pnl_sm.append(pos_sm.shift(1) * daily_ret)

            # 汇总当天组合盈亏并计算夏普比率
            sharpe_dict = {}
            if pnl_noz:
                df_noz = pd.concat(pnl_noz, axis=1).sum(axis=1)
                sharpe_dict['No zscore'] = df_noz.mean() / df_noz.std() * np.sqrt(252) if df_noz.std() > 0 else 0
            if pnl_z:
                df_z = pd.concat(pnl_z, axis=1).sum(axis=1)
                sharpe_dict['zscore'] = df_z.mean() / df_z.std() * np.sqrt(252) if df_z.std() > 0 else 0
            if pnl_sm:
                df_sm = pd.concat(pnl_sm, axis=1).sum(axis=1)
                sharpe_dict['State Machine'] = df_sm.mean() / df_sm.std() * np.sqrt(252) if df_sm.std() > 0 else 0

            # 记录表格列名所需的各个sector在不同状态和参数下的sharpRatio值
            record = {
                'Sector': sector,
                'Parameter': f'N={N}'
            }
            if 'No zscore' in sharpe_dict: record['Sharpe (No zscore)'] = round(sharpe_dict['No zscore'], 4)
            if 'zscore' in sharpe_dict: record['Sharpe (zscore)'] = round(sharpe_dict['zscore'], 4)
            if 'State Machine' in sharpe_dict: record['Sharpe (State Machine)'] = round(sharpe_dict['State Machine'], 4)
            results.append(record)

            # 使用作为基准模式的夏普作为选参依据 (若指定的基准未计算，则默认兜底取计算的第一个夏普作为打分基准)
            eval_key = EVAL_MODE if EVAL_MODE in sharpe_dict else list(sharpe_dict.keys())[0] if sharpe_dict else None

            if eval_key and sharpe_dict[eval_key] > max_metric:
                max_metric = sharpe_dict[eval_key]
                best_n = N
                sector_best_noz = temp_pos_noz
                sector_best_z = temp_pos_z
                sector_best_sm = temp_pos_sm

        if best_n is not None:
            # 收集该板块当前最佳参数所对应的全部品种仓位
            if sector_best_noz: best_positions_noz.update(sector_best_noz)
            if sector_best_z: best_positions_z.update(sector_best_z)
            if sector_best_sm: best_positions_sm.update(sector_best_sm)

    # -------- 数据保存与输出 --------
    if not os.path.exists('./Result'):
        os.makedirs('./Result')

    # 构造文件名的后缀部分，把指定的筛选条件体现在文件名中
    filename_suffix = ""
    if TARGET_SECTORS:
        filename_suffix += "_" + "_".join(TARGET_SECTORS)
    if TARGET_PARAMS:
        filename_suffix += "_N" + "_".join(map(str, TARGET_PARAMS))
    if POSITION_SMOOTH_DAYS > 1:
        filename_suffix += f"_Smooth{POSITION_SMOOTH_DAYS}"

    # 1. 保存参数寻优和回测表现报表
    res_df = pd.DataFrame(results)
    report_path = os.path.join('./Result', "CrossSectional_OvernightVsIntraday_BacktestResult.csv")
    res_df.to_csv(report_path, encoding='utf-8-sig', index=False)
    print(f"回测结果报表已保存至: {report_path}")
    print(res_df)

    # 2. 合并全品种仓位并生成最终仓位文件
    print("正在合并并输出各模式的仓位文件...")

    if best_positions_noz:
        df_pos_noz = pd.DataFrame(best_positions_noz, index=tradingDayList).fillna(0)
        df_pos_noz.to_csv(os.path.join('./Result', f"CrossSectional_OvernightVsIntraday_NoZscore_Position{filename_suffix}.csv"), encoding='utf-8-sig')
    if best_positions_z:
        df_pos_z = pd.DataFrame(best_positions_z, index=tradingDayList).fillna(0)
        df_pos_z.to_csv(os.path.join('./Result', f"CrossSectional_OvernightVsIntraday_Zscore_Position{filename_suffix}.csv"), encoding='utf-8-sig')
    if best_positions_sm:
        df_pos_sm = pd.DataFrame(best_positions_sm, index=tradingDayList).fillna(0)
        df_pos_sm.to_csv(os.path.join('./Result', f"CrossSectional_OvernightVsIntraday_StateMachine_Position{filename_suffix}.csv"), encoding='utf-8-sig')

    print("仓位文件已按计算覆盖模式生成成基于不同权重的 .csv 以供后续调用验证！")

if __name__ == "__main__":
    main()

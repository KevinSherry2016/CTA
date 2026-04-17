# ==============================================================================
# 因子类别：TrendMomentum
# 因子名称：TrendMomentum_TimeSeriesMomentum_RSI1Filter
# 代表意义：时间序列动量 + 单一RSI过滤器
# ==============================================================================

import pandas as pd
import numpy as np
import os


def rolling_zscore(series, window=60):
    min_p = max(window // 2, 1)
    mu = series.rolling(window, min_periods=min_p).mean()
    sigma = series.rolling(window, min_periods=min_p).std().replace(0, np.nan)
    return (series - mu) / sigma


def calc_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.rolling(window, min_periods=max(1, window // 2)).mean()
    avg_loss = loss.rolling(window, min_periods=max(1, window // 2)).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def main():
    marketDataPath = './main_contract/'
    infoPath = './Info.csv'
    FINAL_VOL_WINDOW = 20

    # 1. 过滤品种与划分板块
    info = pd.read_csv(infoPath, encoding='utf-8-sig')
    EXCLUDE_SECTORS = ['stockindex', 'bond', 'other', 'others', 'financial']
    TARGET_SYMBOLS = []
    EXCLUDE_SYMBOLS = ['SF.ZCE', 'SM.ZCE', 'SS.SHF', 'A.DCE', 'B.DCE', 'LH.DCE', 'JD.DCE']

    valid_info = info[~info['sector'].astype(str).str.lower().isin(EXCLUDE_SECTORS)].copy()
    if TARGET_SYMBOLS:
        valid_info = valid_info[valid_info['ts_code'].isin(TARGET_SYMBOLS)]
    if EXCLUDE_SYMBOLS:
        valid_info = valid_info[~valid_info['ts_code'].isin(EXCLUDE_SYMBOLS)]
    sector_map = valid_info.groupby('sector')['ts_code'].apply(list).to_dict()

    # 2. 提取日度行情数据
    data = {}
    print("正在加载 TrendMomentum_TimeSeriesMomentum_RSI1Filter 数据...")

    tradingDayList = []
    for ts_code in valid_info['ts_code']:
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
    N_LIST = [10, 20, 30, 40, 50, 60]

    # ========== 回测指定设置 ==========
    TARGET_SECTORS = []
    TARGET_PARAMS = []
    TARGET_MODES = []
    EVAL_MODE = 'No zscore'
    POSITION_SMOOTH_DAYS = 10

    # 单一新增过滤器：RSI确认过滤，减少中性区交易
    USE_RSI_FILTER = True
    RSI_WINDOW = 14
    RSI_LONG_THRESHOLD = 55
    RSI_SHORT_THRESHOLD = 45
    RSI_SOFT_WEIGHT = 0.5
    # ==================================

    print("开始进行板块回测与参数评估...")

    test_n_list = TARGET_PARAMS if TARGET_PARAMS else N_LIST

    results = []
    best_positions_noz = {}
    best_positions_z = {}
    best_positions_sm = {}

    for sector, ts_codes in sector_map.items():
        if pd.isna(sector) or str(sector).lower() in EXCLUDE_SECTORS:
            continue
        if TARGET_SECTORS and sector not in TARGET_SECTORS:
            continue

        valid_symbols = [c for c in ts_codes if c in data]
        if not valid_symbols:
            continue

        best_n = None
        max_metric = -999
        sector_best_noz, sector_best_z, sector_best_sm = {}, {}, {}

        for N in test_n_list:
            pnl_noz, pnl_z, pnl_sm = [], [], []
            turnover_noz, turnover_z, turnover_sm = [], [], []
            temp_pos_noz, temp_pos_z, temp_pos_sm = {}, {}, {}

            for ts_code in valid_symbols:
                df = data[ts_code]
                close = df['adj_close']
                daily_ret = close.pct_change(fill_method=None)
                vol = daily_ret.rolling(FINAL_VOL_WINDOW, min_periods=1).std().replace(0, np.nan)

                # --- 因子计算逻辑 ---
                raw_sig = close.pct_change(N).fillna(0)

                if USE_RSI_FILTER:
                    rsi = calc_rsi(close, RSI_WINDOW)
                    rsi_long_ok = rsi >= RSI_LONG_THRESHOLD
                    rsi_short_ok = rsi <= RSI_SHORT_THRESHOLD
                    weight = pd.Series(1.0, index=raw_sig.index)
                    # 软过滤：不再清零，未通过确认条件时仅降低信号强度以减少换手。
                    weight.loc[(raw_sig > 0) & (~rsi_long_ok)] = RSI_SOFT_WEIGHT
                    weight.loc[(raw_sig < 0) & (~rsi_short_ok)] = RSI_SOFT_WEIGHT
                    raw_sig = raw_sig * weight
                # --------------------

                if not TARGET_MODES or 'No zscore' in TARGET_MODES:
                    pos_noz = raw_sig / vol
                    if POSITION_SMOOTH_DAYS > 1:
                        pos_noz = pos_noz.ewm(span=POSITION_SMOOTH_DAYS, adjust=False).mean()
                    temp_pos_noz[ts_code] = pos_noz
                    pnl_noz.append(pos_noz.shift(1) * daily_ret)
                    turnover_noz.append(pos_noz.diff().abs().fillna(0))

                if not TARGET_MODES or 'zscore' in TARGET_MODES:
                    z_sig = rolling_zscore(raw_sig, 60)
                    pos_z = z_sig / vol
                    if POSITION_SMOOTH_DAYS > 1:
                        pos_z = pos_z.ewm(span=POSITION_SMOOTH_DAYS, adjust=False).mean()
                    temp_pos_z[ts_code] = pos_z
                    pnl_z.append(pos_z.shift(1) * daily_ret)
                    turnover_z.append(pos_z.diff().abs().fillna(0))

                if not TARGET_MODES or 'State Machine' in TARGET_MODES:
                    sm_sig = np.sign(raw_sig)
                    pos_sm = sm_sig / vol
                    if POSITION_SMOOTH_DAYS > 1:
                        pos_sm = pos_sm.ewm(span=POSITION_SMOOTH_DAYS, adjust=False).mean()
                    temp_pos_sm[ts_code] = pos_sm
                    pnl_sm.append(pos_sm.shift(1) * daily_ret)
                    turnover_sm.append(pos_sm.diff().abs().fillna(0))

            sharpe_dict = {}
            pot_dict = {}
            if pnl_noz:
                df_noz = pd.concat(pnl_noz, axis=1).sum(axis=1)
                sharpe_dict['No zscore'] = df_noz.mean() / df_noz.std() * np.sqrt(252) if df_noz.std() > 0 else 0
                tv_noz = pd.concat(turnover_noz, axis=1).sum(axis=1)
                pot_dict['No zscore'] = df_noz.sum() / tv_noz.sum() * 10000 if tv_noz.sum() > 0 else 0
            if pnl_z:
                df_z = pd.concat(pnl_z, axis=1).sum(axis=1)
                sharpe_dict['zscore'] = df_z.mean() / df_z.std() * np.sqrt(252) if df_z.std() > 0 else 0
                tv_z = pd.concat(turnover_z, axis=1).sum(axis=1)
                pot_dict['zscore'] = df_z.sum() / tv_z.sum() * 10000 if tv_z.sum() > 0 else 0
            if pnl_sm:
                df_sm = pd.concat(pnl_sm, axis=1).sum(axis=1)
                sharpe_dict['State Machine'] = df_sm.mean() / df_sm.std() * np.sqrt(252) if df_sm.std() > 0 else 0
                tv_sm = pd.concat(turnover_sm, axis=1).sum(axis=1)
                pot_dict['State Machine'] = df_sm.sum() / tv_sm.sum() * 10000 if tv_sm.sum() > 0 else 0

            record = {
                'Sector': sector,
                'Parameter': f'N={N}'
            }
            if 'No zscore' in sharpe_dict:
                record['Sharpe (No zscore)'] = round(sharpe_dict['No zscore'], 4)
            if 'zscore' in sharpe_dict:
                record['Sharpe (zscore)'] = round(sharpe_dict['zscore'], 4)
            if 'State Machine' in sharpe_dict:
                record['Sharpe (State Machine)'] = round(sharpe_dict['State Machine'], 4)
            if 'No zscore' in pot_dict:
                record['POT (No zscore)'] = round(pot_dict['No zscore'], 4)
            if 'zscore' in pot_dict:
                record['POT (zscore)'] = round(pot_dict['zscore'], 4)
            if 'State Machine' in pot_dict:
                record['POT (State Machine)'] = round(pot_dict['State Machine'], 4)
            results.append(record)

            eval_key = EVAL_MODE if EVAL_MODE in sharpe_dict else (list(sharpe_dict.keys())[0] if sharpe_dict else None)

            if eval_key and sharpe_dict[eval_key] > max_metric:
                max_metric = sharpe_dict[eval_key]
                best_n = N
                sector_best_noz = temp_pos_noz
                sector_best_z = temp_pos_z
                sector_best_sm = temp_pos_sm

        if best_n is not None:
            if sector_best_noz:
                best_positions_noz.update(sector_best_noz)
            if sector_best_z:
                best_positions_z.update(sector_best_z)
            if sector_best_sm:
                best_positions_sm.update(sector_best_sm)

    if not os.path.exists('./Result'):
        os.makedirs('./Result')

    filename_suffix = ""
    if TARGET_SECTORS:
        filename_suffix += "_" + "_".join(TARGET_SECTORS)
    if TARGET_PARAMS:
        filename_suffix += "_N" + "_".join(map(str, TARGET_PARAMS))
    if POSITION_SMOOTH_DAYS > 1:
        filename_suffix += f"_Smooth{POSITION_SMOOTH_DAYS}"
    filename_suffix += f"_RSI{RSI_WINDOW}_{RSI_LONG_THRESHOLD}_{RSI_SHORT_THRESHOLD}"

    res_df = pd.DataFrame(results)
    report_path = os.path.join('./Result', 'TrendMomentum_TimeSeriesMomentum_RSI1Filter_BacktestResult.csv')
    res_df.to_csv(report_path, encoding='utf-8-sig', index=False)
    print(f"回测结果报表已保存至: {report_path}")
    print(res_df)

    print("正在合并并输出各模式的仓位文件...")

    if best_positions_noz:
        df_pos_noz = pd.DataFrame(best_positions_noz, index=tradingDayList).fillna(0)
        df_pos_noz.to_csv(os.path.join('./Result', f"TrendMomentum_TimeSeriesMomentum_RSI1Filter_NoZscore_Position{filename_suffix}.csv"), encoding='utf-8-sig')
    if best_positions_z:
        df_pos_z = pd.DataFrame(best_positions_z, index=tradingDayList).fillna(0)
        df_pos_z.to_csv(os.path.join('./Result', f"TrendMomentum_TimeSeriesMomentum_RSI1Filter_Zscore_Position{filename_suffix}.csv"), encoding='utf-8-sig')
    if best_positions_sm:
        df_pos_sm = pd.DataFrame(best_positions_sm, index=tradingDayList).fillna(0)
        df_pos_sm.to_csv(os.path.join('./Result', f"TrendMomentum_TimeSeriesMomentum_RSI1Filter_StateMachine_Position{filename_suffix}.csv"), encoding='utf-8-sig')

    print("仓位文件已按计算覆盖模式生成成基于不同权重的 .csv 以供后续调用验证！")


if __name__ == '__main__':
    main()

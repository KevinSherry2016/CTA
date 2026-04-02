"""
Momentum_V1.py

改进说明：将原始的简单动量脚本替换为更完备的版本，包含：
 - 更稳健的数据加载（与 MovingAverage 系列一致）
 - 时间序列动量计算、可选 z-score 标准化
 - 横截面排名转仓位或时间序列阈值转仓位
 - T 日平滑（与 README/MovingAverage_V4 中定义一致）
 - 按日归一化到目标杠杆后输出 CSV，便于后续评估

本文件尽量加注释，便于直接阅读与修改。
"""

from pathlib import Path
import typing as tp
import os

import numpy as np
import pandas as pd


# ----------------------------- 配置区（可修改） -----------------------------
MARKET_DATA_DIR = Path('./total')
INFO_PATH = Path('./Info.csv')
RESULT_DIR = Path('./Result')
STRATEGY_NAME = 'Momentum_V1'

# 策略参数
MOM_LOOKBACK = 20          # 时间序列动量窗口（天）
USE_ZSCORE = True         # 是否对动量因子做时间序列 z-score 标准化
SMOOTH_T = 1               # 将每日仓位平滑 T 天；1 表示不平滑
CROSS_SECTION_RANK = True  # 是否用横截面排名构造头寸
TOP_PERCENT = 0.4          # 横截面做多/做空百分比（top/bottom）
USE_VOL_CONTROL = True     # 是否在最终仓位上做波动率控制
VOL_WINDOW = 20            # 近 N 天波动率窗口（天）
MOM_LOOKBACK_LIST = [5, 10, 20]
TOP_PERCENT_LIST = [0.2, 0.4]

OUTPUT_FILENAME = f"{STRATEGY_NAME}_momentum_{MOM_LOOKBACK}_T_{SMOOTH_T}_position.csv"


# ----------------------------- 数据加载 -------------------------------------

def load_universe(info_path: Path = INFO_PATH, market_dir: Path = MARKET_DATA_DIR) -> tp.Tuple[list[str], dict]:
    """加载品种列表与行情数据。

    - 从 `Info.csv` 读取 `ts_code` 列作为品种列表。
    - 每个品种的行情文件应为 `./total/{ts_code}.csv`，并包含 `trade_date` 与 `adj_close`。
    - 返回 (trading_days_list, data_dict)
    """
    info = pd.read_csv(info_path, encoding='utf-8-sig')
    ts_codes = info['ts_code'].tolist()

    # 取第一个可用文件作为交易日基准
    sample_path = None
    for ts in ts_codes:
        p = market_dir / f"{ts}.csv"
        if p.exists():
            sample_path = p
            break
    if sample_path is None:
        # 若没有文件，尝试列出目录下任意 csv
        files = sorted(market_dir.glob('*.csv'))
        if not files:
            raise FileNotFoundError(f'未在 {market_dir} 找到任何 CSV 文件')
        sample_path = files[0]

    sample = pd.read_csv(sample_path)
    trading_days = sample['trade_date'].astype(str).tolist()

    data: dict[str, pd.DataFrame] = {}
    print('正在加载数据')
    for ts in ts_codes:
        path = market_dir / f"{ts}.csv"
        if not path.exists():
            print(f'警告: 未找到 {path}，跳过该品种')
            continue
        df = pd.read_csv(path)
        df['trade_date'] = df['trade_date'].astype(str)
        df.set_index('trade_date', inplace=True)
        # 强制要求存在 adj_close，不能用 close 代替
        if 'adj_close' not in df.columns:
            raise KeyError(f'{path} 中缺少 adj_close 列；不能使用 close 代替')
        data[ts] = df

    return trading_days, data


# ----------------------------- 因子计算 -------------------------------------

def compute_time_series_momentum(df: pd.DataFrame, lookback: int = MOM_LOOKBACK) -> pd.Series:
    """时间序列动量：P_t / P_{t-k} - 1。返回与 df.index 对齐的 Series。"""
    return df['adj_close'].pct_change(periods=lookback)


def ts_zscore(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """简单的时间序列 z-score 标准化： (x - mean) / std 。"""
    if not window or window <= 1:
        return series
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std().replace(0, np.nan)
    return (series - mean) / std


# ----------------------------- 因子到仓位 ---------------------------------

def make_positions_from_factor(factor_df: pd.DataFrame,
                               cross_section_rank: bool = CROSS_SECTION_RANK,
                               top_percent: float = TOP_PERCENT,
                               top_k: tp.Optional[int] = None) -> pd.DataFrame:
    """将因子矩阵（日期 x 品种）转换为连续仓位矩阵。

    - 如果 cross_section_rank=True：对每日进行横截面排名，做多 top_percent，做空 bottom 同样比例，
      并按排名强弱分配连续权重。
    - 否则直接使用因子值的相对强弱生成连续仓位。
    """
    positions = pd.DataFrame(index=factor_df.index, columns=factor_df.columns, dtype=float)

    if cross_section_rank:
        for date, row in factor_df.iterrows():
            vals = row.dropna()
            if vals.empty or len(vals) < 2:
                positions.loc[date] = 0.0
                continue
            ranks = vals.rank(method='first')
            n = len(vals)
            k = top_k if (top_k is not None and top_k > 0) else max(1, int(np.floor(n * top_percent)))
            k = min(k, n // 2)
            if k <= 0:
                positions.loc[date] = 0.0
                continue

            long_idx = ranks[ranks > n - k].index
            short_idx = ranks[ranks <= k].index
            long_strength = ranks.loc[long_idx] - (n - k)
            short_strength = (k + 1) - ranks.loc[short_idx]

            pos_row = pd.Series(0.0, index=factor_df.columns)
            pos_row.loc[long_idx] = 0.5 * long_strength / long_strength.sum()
            pos_row.loc[short_idx] = -0.5 * short_strength / short_strength.sum()
            positions.loc[date] = pos_row
    else:
        gross = factor_df.abs().sum(axis=1).replace(0, np.nan)
        positions = factor_df.div(gross, axis=0)

    return positions.fillna(0.0)


def smooth_positions(pos_df: pd.DataFrame, window: int = SMOOTH_T) -> pd.DataFrame:
    """对仓位做 T 日平滑（rolling mean）。window<=1 时返回原序列。"""
    if window <= 1:
        return pos_df
    return pos_df.rolling(window=window, min_periods=1).mean()


def compute_rolling_volatility(df: pd.DataFrame, window: int) -> pd.Series:
    """计算单品种近 N 天收益波动率。"""
    if not window or window <= 1:
        return pd.Series(1.0, index=df.index)
    returns = df['adj_close'].pct_change(fill_method=None)
    return returns.rolling(window=window, min_periods=1).std().replace(0, np.nan)


def format_top_percent(top_pct: float) -> str:
    """将 top_percent 转为适合文件名的字符串。"""
    return str(top_pct).replace('.', 'p')


# ----------------------------- 主流程 -------------------------------------

def run(universe: tp.Optional[list[str]] = None,
    lookback: int = MOM_LOOKBACK,
    use_zscore: bool = USE_ZSCORE,
    use_vol_control: bool = USE_VOL_CONTROL,
    vol_window: int = VOL_WINDOW,
    smooth_t: int = SMOOTH_T,
    cross_rank: bool = CROSS_SECTION_RANK,
    top_pct: float = TOP_PERCENT,
    top_k: tp.Optional[int] = None) -> Path:
    """生成并保存标准化仓位 CSV，返回输出路径。

    步骤：加载数据 -> 计算因子 -> 可选 z-score（窗口默认等于 lookback） -> 因子映射为仓位 -> 平滑 -> 可选波动率控制 -> 输出
    """
    trading_days, data = load_universe()
    if universe:
        data = {k: v for k, v in data.items() if k in universe}

    factor_df = pd.DataFrame(index=trading_days, columns=list(data.keys()), dtype=float)
    for ts, df in data.items():
        factor = compute_time_series_momentum(df, lookback=lookback)
        factor_df[ts] = factor.reindex(trading_days)

    if use_zscore and lookback > 1:
        for ts in factor_df.columns:
            factor_df[ts] = ts_zscore(factor_df[ts], window=lookback)

    raw_positions = make_positions_from_factor(factor_df, cross_section_rank=cross_rank, top_percent=top_pct, top_k=top_k)
    smoothed = smooth_positions(raw_positions, window=smooth_t)
    final_positions = smoothed.copy()

    if use_vol_control:
        effective_vol_window = vol_window if vol_window and vol_window > 1 else lookback
        vol_df = pd.DataFrame(index=trading_days, columns=list(data.keys()), dtype=float)
        for ts, df in data.items():
            vol_df[ts] = compute_rolling_volatility(df, window=effective_vol_window).reindex(trading_days)
        final_positions = final_positions.div(vol_df).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 不进行每日归一化，直接输出平滑后的仓位矩阵
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    vol_tag = f"_vol_{vol_window}" if use_vol_control else "_vol_off"
    out_filename = f"{STRATEGY_NAME}_momentum_{lookback}_top_{format_top_percent(top_pct)}_T_{smooth_t}{vol_tag}_position.csv"
    outpath = RESULT_DIR / out_filename
    final_positions.to_csv(outpath, encoding='utf-8-sig')
    return outpath


def run_grid(lookbacks: tp.Sequence[int],
             top_percents: tp.Sequence[float],
             universe: tp.Optional[list[str]] = None,
             use_zscore: bool = USE_ZSCORE,
             use_vol_control: bool = USE_VOL_CONTROL,
             vol_window: int = VOL_WINDOW,
             smooth_t: int = SMOOTH_T,
             cross_rank: bool = CROSS_SECTION_RANK,
             top_k: tp.Optional[int] = None) -> list[Path]:
    """对一组回望窗口和 top_percent 循环执行 `run`，并返回生成的文件路径列表。"""
    results: list[Path] = []
    total_runs = len(lookbacks) * len(top_percents)
    current_run = 0
    print(f'开始批量生成仓位，共 {total_runs} 组参数')
    for lb in lookbacks:
        for top_pct in top_percents:
            current_run += 1
            print(f'[{current_run}/{total_runs}] 正在运行: lookback={lb}, top_percent={top_pct}, use_zscore={use_zscore}, use_vol_control={use_vol_control}, vol_window={vol_window}, smooth_t={smooth_t}')
            p = run(universe=universe,
                    lookback=lb,
                    use_zscore=use_zscore,
                use_vol_control=use_vol_control,
                vol_window=vol_window,
                    smooth_t=smooth_t,
                    cross_rank=cross_rank,
                    top_pct=top_pct,
                    top_k=top_k)
            print(f'[{current_run}/{total_runs}] 已输出: {p}')
            results.append(p)
    print(f'批量生成完成，共输出 {len(results)} 个仓位文件')
    return results


if __name__ == '__main__':
    run_grid(MOM_LOOKBACK_LIST, TOP_PERCENT_LIST)

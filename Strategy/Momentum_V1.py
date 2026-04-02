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
ZSCORE_WINDOW = 20         # 时间序列 z-score 窗口（0 或 None 表示不做 z-score）
SMOOTH_T = 5               # 将每日仓位平滑 T 天；1 表示不平滑
CROSS_SECTION_RANK = True  # 是否用横截面排名构造头寸
TOP_PERCENT = 0.2          # 横截面做多/做空百分比（top/bottom）
TARGET_LEVERAGE = 1.0      # 归一化后每日绝对仓位和

OUTPUT_FILENAME = f"{STRATEGY_NAME}_momentum_{MOM_LOOKBACK}_T_{SMOOTH_T}_position_normalized.csv"


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
        # 兼容性：若不存在 adj_close，则用 close 代替
        if 'adj_close' not in df.columns:
            if 'close' in df.columns:
                df['adj_close'] = df['close']
            else:
                raise KeyError(f'{path} 中缺少 adj_close/close 列')
        data[ts] = df

    return trading_days, data


# ----------------------------- 因子计算 -------------------------------------

def compute_time_series_momentum(df: pd.DataFrame, lookback: int = MOM_LOOKBACK) -> pd.Series:
    """时间序列动量：P_t / P_{t-k} - 1。返回与 df.index 对齐的 Series。"""
    return df['adj_close'].pct_change(periods=lookback)


def ts_zscore(series: pd.Series, window: int = ZSCORE_WINDOW, min_periods: int = 1) -> pd.Series:
    """简单的时间序列 z-score 标准化： (x - mean) / std 。"""
    if not window or window <= 1:
        return series
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std().replace(0, np.nan)
    return (series - mean) / std


# ----------------------------- 因子到仓位 ---------------------------------

def make_positions_from_factor(factor_df: pd.DataFrame,
                               cross_section_rank: bool = CROSS_SECTION_RANK,
                               top_percent: float = TOP_PERCENT) -> pd.DataFrame:
    """将因子矩阵（日期 x 品种）转换为仓位矩阵（-1/0/1）

    - 如果 cross_section_rank=True：对每日进行横截面排名，做多 top_percent，做空 bottom 同样比例。
    - 否则按时间序列阈值（>0 做多，<0 做空）生成仓位。
    """
    positions = pd.DataFrame(index=factor_df.index, columns=factor_df.columns, dtype=float)

    if cross_section_rank:
        for date, row in factor_df.iterrows():
            vals = row.dropna()
            if vals.empty:
                positions.loc[date] = 0.0
                continue
            ranks = vals.rank(method='first')
            n = len(vals)
            k = max(1, int(np.floor(n * top_percent)))

            long_idx = ranks[ranks > n - k].index
            short_idx = ranks[ranks <= k].index

            pos_row = pd.Series(0.0, index=factor_df.columns)
            pos_row.loc[long_idx] = 1.0
            pos_row.loc[short_idx] = -1.0
            positions.loc[date] = pos_row
    else:
        positions = factor_df.applymap(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))

    return positions.fillna(0.0)


def smooth_positions(pos_df: pd.DataFrame, window: int = SMOOTH_T) -> pd.DataFrame:
    """对仓位做 T 日平滑（rolling mean）。window<=1 时返回原序列。"""
    if window <= 1:
        return pos_df
    return pos_df.rolling(window=window, min_periods=1).mean()


def normalize_positions(pos_df: pd.DataFrame, target_leverage: float = TARGET_LEVERAGE) -> pd.DataFrame:
    """按每日绝对值之和缩放仓位，使每日 abs sum == target_leverage。

    - 若当日绝对和为 0（无仓），保持 0
    - 此处使用按日缩放，适合低频组合层面统一杠杆
    """
    abs_sum = pos_df.abs().sum(axis=1).replace(0, np.nan)
    scale = target_leverage / abs_sum
    normed = pos_df.mul(scale, axis=0).fillna(0.0)
    return normed


# ----------------------------- 主流程 -------------------------------------

def run(universe: tp.Optional[list[str]] = None,
        lookback: int = MOM_LOOKBACK,
        zscore_window: int = ZSCORE_WINDOW,
        smooth_t: int = SMOOTH_T,
        cross_rank: bool = CROSS_SECTION_RANK,
        top_pct: float = TOP_PERCENT,
        target_lev: float = TARGET_LEVERAGE) -> Path:
    """生成并保存标准化仓位 CSV，返回输出路径。

    步骤：加载数据 -> 计算因子 -> 可选 z-score -> 因子映射为仓位 -> 平滑 -> 归一化 -> 输出
    """
    trading_days, data = load_universe()
    if universe:
        data = {k: v for k, v in data.items() if k in universe}

    factor_df = pd.DataFrame(index=trading_days, columns=list(data.keys()), dtype=float)
    for ts, df in data.items():
        factor = compute_time_series_momentum(df, lookback=lookback)
        factor_df[ts] = factor.reindex(trading_days)

    if zscore_window and zscore_window > 1:
        for ts in factor_df.columns:
            factor_df[ts] = ts_zscore(factor_df[ts], window=zscore_window)

    raw_positions = make_positions_from_factor(factor_df, cross_section_rank=cross_rank, top_percent=top_pct)
    smoothed = smooth_positions(raw_positions, window=smooth_t)
    normed = normalize_positions(smoothed, target_leverage=target_lev)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = RESULT_DIR / OUTPUT_FILENAME
    normed.to_csv(outpath, encoding='utf-8-sig')
    print(f'输出标准化仓位: {outpath}  （shape={normed.shape}）')
    return outpath


if __name__ == '__main__':
    run()

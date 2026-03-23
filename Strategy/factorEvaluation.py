import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


POSITION_CSV     = './Strategy/position.csv'
MARKET_DATA_PATH = './total/'
INFO_PATH        = './Info.csv'
OUTPUT_PREFIX    = None     # None → 与 POSITION_CSV 同目录同文件名前缀
NORM_START_DATE  = None     # 标准化计算的起始日期，格式 'YYYYMMDD'，None 表示不限
NORM_END_DATE    = None     # 标准化计算的结束日期，格式 'YYYYMMDD', None 表示不限


# ── 数据加载 ───────────────────────────────────────────────────────────────────

def load_positions(position_csv_path):
    positions = pd.read_csv(position_csv_path, index_col=0)
    positions.index = positions.index.astype(str)
    return positions.astype(float)


def load_close_data(symbols, trading_days, market_data_path):
    close_data = {}
    for ts_code in symbols:
        csv_path = os.path.join(market_data_path, f'{ts_code}.csv')
        if not os.path.exists(csv_path):
            print(f'文件不存在，跳过: {csv_path}')
            continue
        df = pd.read_csv(csv_path)
        if 'trade_date' not in df.columns or 'adj_close' not in df.columns:
            print(f'字段缺失，跳过: {csv_path}')
            continue
        df['trade_date'] = df['trade_date'].astype(str)
        df['adj_close']  = pd.to_numeric(df['adj_close'], errors='coerce')
        df.set_index('trade_date', inplace=True)
        close_data[ts_code] = df['adj_close']

    if not close_data:
        raise ValueError('未加载到任何行情数据，请检查 market_data_path 或持仓列名。')

    return pd.DataFrame(close_data, dtype='float64').reindex(trading_days).ffill()


def load_main_contract(symbols, trading_days, market_data_path):
    """加载每个品种每日的主力合约代码（mapping_ts_code 列）。"""
    data = {}
    for ts_code in symbols:
        csv_path = os.path.join(market_data_path, f'{ts_code}.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, usecols=['trade_date', 'mapping_ts_code'])
        df['trade_date'] = df['trade_date'].astype(str)
        df.set_index('trade_date', inplace=True)
        data[ts_code] = df['mapping_ts_code']
    return pd.DataFrame(data).reindex(trading_days)


# ── 计算标准化仓位与 PnL ────────────────────────────────────────────────────────

def calc_normalized(positions, close_df, info,
                    norm_start=None, norm_end=None):
    """
    返回:
        norm_pos_df          : 标准化后的仓位 DataFrame（行=交易日，列=品种）
        norm_daily_pnl       : 标准化后的每日策略总 PnL Series
        norm_sector_daily    : 标准化后的各 sector 每日 PnL DataFrame（列=sector名）

    norm_start / norm_end : 用于计算标准化因子的日期区间（字符串 'YYYYMMDD'），
                            None 表示使用全部时间段。
    """
    ret_df  = close_df.pct_change(fill_method=None)
    # 昨日收盘信号 → 今日持仓（shift 前先对齐列）
    pos_df  = positions.reindex(columns=close_df.columns).fillna(0.0).shift(1).fillna(0.0)

    pnl_per_asset = pos_df * ret_df                 # 各品种每日 PnL
    daily_pnl     = pnl_per_asset.sum(axis=1)       # 策略每日总 PnL

    # 标准化因子：在指定日期区间内计算 daily_pnl 的标准差
    pnl_for_scale = daily_pnl
    if norm_start is not None:
        pnl_for_scale = pnl_for_scale[pnl_for_scale.index >= norm_start]
    if norm_end is not None:
        pnl_for_scale = pnl_for_scale[pnl_for_scale.index <= norm_end]
    scale = pnl_for_scale.std()
    if scale == 0 or pd.isna(scale):
        scale = 1.0

    norm_pos_df        = pos_df / scale
    norm_daily_pnl     = daily_pnl / scale

    # 各 sector 每日标准化 PnL
    asset_cols = list(pos_df.columns)
    norm_sector_daily = pd.DataFrame(index=daily_pnl.index)
    for sector, group in info.groupby('sector'):
        cols = [c for c in group['ts_code'].tolist() if c in asset_cols]
        if not cols:
            continue
        norm_sector_daily[sector] = pnl_per_asset[cols].sum(axis=1) / scale

    return norm_pos_df, norm_daily_pnl, norm_sector_daily


# ── 指标计算 ───────────────────────────────────────────────────────────────────

def _rollover_adjusted_turnover(pos_df, main_contract_df):
    """
    计算考虑换月的每日换手量。

    换月识别规则：若品种在 T 日的主力合约代码（mapping_ts_code）与 T-1 日不同，
    则 T 日为换月日。此时：
      - 平旧合约（T-1 EOD）+ 开新合约（T SOD）的换手合并计入 T 日：
        turnover[T] = |pos[T-1]|（平旧） + |pos[T]|（开新）
      - 正常日：turnover[T] = |pos[T] - pos[T-1]|
    """
    # 换月日：T 日主力合约 != T-1 日主力合约（两日均有合约信息）
    prev_contract = main_contract_df.shift(1)
    is_rollover = (
        main_contract_df.ne(prev_contract)
        & main_contract_df.notna()
        & prev_contract.notna()
    ).reindex(columns=pos_df.columns, fill_value=False)

    prev_pos = pos_df.shift(1).fillna(0.0)

    normal_turnover   = pos_df.diff().fillna(0.0).abs()
    rollover_turnover = prev_pos.abs() + pos_df.abs()

    turnover_df = normal_turnover.where(~is_rollover, rollover_turnover)
    return turnover_df.sum(axis=1)


def calc_metrics(norm_daily_pnl, norm_pos_df, main_contract_df, trading_days_per_year=250):
    """
    计算策略绩效指标。

    参数:
        norm_daily_pnl       : 标准化后的每日 PnL Series
        norm_pos_df          : 标准化后的仓位 DataFrame（行=交易日，列=品种）
        main_contract_df     : 每日主力合约代码 DataFrame（来自 mapping_ts_code）
        trading_days_per_year: 年化交易日数（默认 250）
    返回:
        dict，包含各指标：
            sharpeRatio        : 年化 Sharpe Ratio
            maxDrawdown        : 最大回撤幅度（单位：标准差，即标准化 PnL 的累计跌幅）
            maxDrawdownDays    : 从回撤最深点往前追溯到前高所经历的天数（最久回撤持续天数）
            holdingPeriod      : 平均持仓天数，公式 = (sum(GMV) / sum(Turnover)) * 2
            pot                : 万分之收益换手比，公式 = (sum(pnl) / sum(Turnover)) * 10000
    """
    pnl = norm_daily_pnl.dropna()
    std = pnl.std()
    sharpe = pnl.mean() / std * (trading_days_per_year ** 0.5) if std != 0 else float('nan')

    # 最大回撤
    cum = pnl.cumsum()
    rolling_max = cum.cummax()
    drawdown = cum - rolling_max          # 始终 <= 0

    max_drawdown = drawdown.min()         # 最大回撤幅度（负数，单位同 norm_daily_pnl）

    # 最久回撤天数：对每个回撤低点，找其前方最近的前高点，计算天数差
    # 遍历所有处于回撤中的位置，找最长的一段
    max_dd_days = 0
    peak_idx = 0
    for i in range(len(cum)):
        if cum.iloc[i] >= rolling_max.iloc[i]:
            peak_idx = i          # 创新高，更新前高位置
        else:
            days = i - peak_idx
            if days > max_dd_days:
                max_dd_days = days

    # 持仓周期：(sum(GMV) / sum(Turnover)) * 2，换手考虑换月
    gmv      = norm_pos_df.abs().sum(axis=1)
    turnover = _rollover_adjusted_turnover(norm_pos_df, main_contract_df)
    total_turnover = turnover.sum()
    holding_period = (gmv.sum() / total_turnover) * 2 if total_turnover != 0 else float('nan')
    pot = (pnl.sum() / total_turnover) * 10000 if total_turnover != 0 else float('nan')

    return {
        'sharpeRatio':     sharpe,
        'maxDrawdown':     max_drawdown,
        'maxDrawdownDays': max_dd_days,
        'holdingPeriod':   holding_period,
        'pot':             pot,
    }


# ── 绘图函数 ───────────────────────────────────────────────────────────────────

def plot_sector_pnl(trade_dates, norm_daily_pnl, norm_sector_daily, output_png):
    """各 sector 累计 PnL + 总累计 PnL，同一张图。"""
    total_cum = norm_daily_pnl.cumsum()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(trade_dates, total_cum, linewidth=2, label='Total')
    for sector in norm_sector_daily.columns:
        ax.plot(trade_dates, norm_sector_daily[sector].cumsum(),
                linewidth=1, alpha=0.9, label=sector)

    ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
    ax.set_title('Cumulative PnL by Sector')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.set_xlim(left=trade_dates[0])
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def plot_weekday_distribution(trade_dates, norm_daily_pnl, output_png):
    """周一到周五的标准化 PnL 柱状图。"""
    weekday_order  = [0, 1, 2, 3, 4]
    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

    s = norm_daily_pnl.copy()
    s.index = trade_dates
    dist = s.groupby(s.index.weekday).sum().reindex(weekday_order, fill_value=0.0)

    bar_colors = ['#4C78A8' if v >= 0 else '#E45756' for v in dist.values]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(weekday_labels, dist.values, color=bar_colors)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Weekday PnL Distribution')
    ax.set_ylabel('PnL')
    plt.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def plot_cumulative_pnl(trade_dates, norm_daily_pnl, metrics, output_png):
    """累计 PnL 曲线图，标题展示核心绩效指标。"""
    cumulative_pnl = norm_daily_pnl.cumsum()
    title = (
        f'Sharpe Ratio: {metrics["sharpeRatio"]:.2f} | '
        f'POT: {metrics["pot"]:.2f} | '
        f'Holding Period: {metrics["holdingPeriod"]:.0f} days'
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(trade_dates, cumulative_pnl, linewidth=2, color='#1f77b4')
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
    ax.set_title(title)
    ax.set_ylabel('Cumulative PnL')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(left=trade_dates[0])
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main():
    info = pd.read_csv(INFO_PATH, encoding='utf-8-sig')
    if 'ts_code' not in info.columns or 'sector' not in info.columns:
        raise ValueError('Info.csv 必须包含 ts_code 和 sector 两列。')

    output_dir = Path('Strategy')

    positions    = load_positions(POSITION_CSV)
    trading_days = positions.index.tolist()
    close_df         = load_close_data(positions.columns.tolist(), trading_days, MARKET_DATA_PATH)
    main_contract_df = load_main_contract(positions.columns.tolist(), trading_days, MARKET_DATA_PATH)

    norm_pos_df, norm_daily_pnl, norm_sector_daily = calc_normalized(
        positions, close_df, info,
        norm_start=NORM_START_DATE, norm_end=NORM_END_DATE
    )
    trade_dates = pd.to_datetime(norm_daily_pnl.index, format='%Y%m%d')

    # 指标计算
    metrics = calc_metrics(norm_daily_pnl, norm_pos_df, main_contract_df)
    print(f'Sharpe Ratio      : {metrics["sharpeRatio"]:.2f}')
    print(f'Max Drawdown      : {metrics["maxDrawdown"]:.4f}  (标准差单位)')
    print(f'Max Drawdown Days : {metrics["maxDrawdownDays"]} 天')
    print(f'Holding Period    : {metrics["holdingPeriod"]:.0f} 天')
    print(f'POT               : {metrics["pot"]:.2f}  (万分之)')

    # 1. 标准化仓位 CSV
    position_csv_path = output_dir / 'position_normalized.csv'
    norm_pos_df.to_csv(position_csv_path, encoding='utf-8-sig')

    # 2. 每日标准化 PnL CSV（总 PnL + 各 sector PnL + 累计 PnL）
    daily_pnl_df = norm_sector_daily.copy()
    daily_pnl_df.insert(0, 'dailyPnl', norm_daily_pnl)
    daily_pnl_df['cumulativePnl'] = norm_daily_pnl.cumsum()
    daily_pnl_csv_path = output_dir / 'dailyPnl_normalized.csv'
    daily_pnl_df.to_csv(daily_pnl_csv_path, encoding='utf-8-sig')

    # 3. 各 sector 累计 PnL 图
    sector_png_path = output_dir / 'sectorPnl.png'
    plot_sector_pnl(trade_dates, norm_daily_pnl, norm_sector_daily, sector_png_path)

    # 4. 工作日 PnL 分布图
    weekday_png_path = output_dir / 'weekdayPnl.png'
    plot_weekday_distribution(trade_dates, norm_daily_pnl, weekday_png_path)

    # 5. 累计 PnL 图（标题展示核心指标）
    cumulative_pnl_png_path = output_dir / 'cumulativePnl.png'
    plot_cumulative_pnl(trade_dates, norm_daily_pnl, metrics, cumulative_pnl_png_path)

    print(f'标准化仓位 CSV : {Path(position_csv_path).name}')
    print(f'每日 PnL  CSV : {Path(daily_pnl_csv_path).name}')
    print(f'Sector PnL 图 : {Path(sector_png_path).name}')
    print(f'工作日分布图   : {Path(weekday_png_path).name}')
    print(f'累计 PnL 图   : {Path(cumulative_pnl_png_path).name}')

if __name__ == '__main__':
    main()

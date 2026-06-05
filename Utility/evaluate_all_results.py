import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR.parent / 'Result'
OUTPUT_DIR = BASE_DIR.parent / 'Evaluate'
MARKET_DATA_PATH = BASE_DIR.parent / 'main_contract'
INFO_PATH = BASE_DIR.parent / 'Info.csv'
NORM_START_DATE = '20150101'
NORM_END_DATE = '20201231'
SUMMARY_METRICS_CSV = 'all_metrics_summary.csv'
CLOSE_CACHE = {}
MAIN_CONTRACT_CACHE = {}


def load_positions(position_csv_path):
    positions = pd.read_csv(position_csv_path, index_col=0)
    positions.index = positions.index.astype(str)
    return positions.astype(float)


def load_close_data(symbols, trading_days):
    close_data = {}
    for ts_code in symbols:
        if ts_code not in CLOSE_CACHE:
            csv_path = MARKET_DATA_PATH / f'{ts_code}.csv'
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if 'trade_date' not in df.columns or 'adj_close' not in df.columns:
                continue
            df['trade_date'] = df['trade_date'].astype(str)
            df['adj_close'] = pd.to_numeric(df['adj_close'], errors='coerce')
            df.set_index('trade_date', inplace=True)
            CLOSE_CACHE[ts_code] = df['adj_close']
        close_data[ts_code] = CLOSE_CACHE[ts_code]

    if not close_data:
        raise ValueError('未加载到任何行情数据。')

    return pd.DataFrame(close_data, dtype='float64').reindex(trading_days).ffill()


def load_main_contract(symbols, trading_days):
    data = {}
    for ts_code in symbols:
        if ts_code not in MAIN_CONTRACT_CACHE:
            csv_path = MARKET_DATA_PATH / f'{ts_code}.csv'
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path, usecols=['trade_date', 'mapping_ts_code'])
            df['trade_date'] = df['trade_date'].astype(str)
            df.set_index('trade_date', inplace=True)
            MAIN_CONTRACT_CACHE[ts_code] = df['mapping_ts_code']
        data[ts_code] = MAIN_CONTRACT_CACHE[ts_code]
    return pd.DataFrame(data).reindex(trading_days)


def calc_raw_pnl(positions, close_df):
    ret_df = close_df.pct_change(fill_method=None)
    pos_df = positions.reindex(columns=close_df.columns).fillna(0.0).shift(1).fillna(0.0)
    pnl_per_asset = pos_df * ret_df
    daily_pnl = pnl_per_asset.sum(axis=1)
    return pos_df, pnl_per_asset, daily_pnl


def calc_normalized(pos_df, pnl_per_asset, daily_pnl, info, norm_start=None, norm_end=None):
    pnl_for_scale = daily_pnl
    if norm_start is not None:
        pnl_for_scale = pnl_for_scale[pnl_for_scale.index >= norm_start]
    if norm_end is not None:
        pnl_for_scale = pnl_for_scale[pnl_for_scale.index <= norm_end]
    scale = pnl_for_scale.std()
    if scale == 0 or pd.isna(scale):
        scale = 1.0

    norm_pos_df = pos_df / scale
    norm_pnl_per_asset = pnl_per_asset / scale
    norm_daily_pnl = daily_pnl / scale

    norm_sector_daily = pd.DataFrame(index=daily_pnl.index)
    asset_cols = list(pos_df.columns)
    for sector, group in info.groupby('sector'):
        cols = [c for c in group['ts_code'].tolist() if c in asset_cols]
        if cols:
            norm_sector_daily[sector] = pnl_per_asset[cols].sum(axis=1) / scale

    return norm_pos_df, norm_pnl_per_asset, norm_daily_pnl, norm_sector_daily, scale


def _rollover_adjusted_turnover(pos_df, main_contract_df):
    prev_contract = main_contract_df.shift(1)
    is_rollover = (
        main_contract_df.ne(prev_contract)
        & main_contract_df.notna()
        & prev_contract.notna()
    ).reindex(columns=pos_df.columns, fill_value=False)

    prev_pos = pos_df.shift(1).fillna(0.0)
    normal_turnover = pos_df.diff().fillna(0.0).abs()
    rollover_turnover = prev_pos.abs() + pos_df.abs()
    turnover_df = normal_turnover.where(~is_rollover, rollover_turnover)
    return turnover_df.sum(axis=1)


def calc_metrics(norm_daily_pnl, norm_pos_df, main_contract_df, trading_days_per_year=250):
    pnl = norm_daily_pnl.dropna()
    std = pnl.std()
    sharpe = pnl.mean() / std * (trading_days_per_year ** 0.5) if std != 0 else float('nan')

    cum = pnl.cumsum()
    rolling_max = cum.cummax()
    drawdown = cum - rolling_max
    max_drawdown = drawdown.min()

    max_dd_days = 0
    peak_idx = 0
    for i in range(len(cum)):
        if cum.iloc[i] >= rolling_max.iloc[i]:
            peak_idx = i
        else:
            max_dd_days = max(max_dd_days, i - peak_idx)

    gmv = norm_pos_df.abs().sum(axis=1)
    turnover = _rollover_adjusted_turnover(norm_pos_df, main_contract_df)
    total_turnover = turnover.sum()
    holding_period = (gmv.sum() / total_turnover) * 2 if total_turnover != 0 else float('nan')
    pot = (pnl.sum() / total_turnover) * 10000 if total_turnover != 0 else float('nan')

    return {
        'sharpeRatio': sharpe,
        'pot': pot,
        'holdingPeriod': holding_period,
        'maxDrawdown': max_drawdown,
        'maxDrawdownDays': max_dd_days,
    }


def plot_pnl(trade_dates, norm_daily_pnl, norm_pnl_per_asset, metrics, output_png):
    title = (
        f"Sharpe: {metrics['sharpeRatio']:.2f} | "
        f"POT: {metrics['pot']:.2f} | "
        f"Hold: {metrics['holdingPeriod']:.0f}d | "
        f"MDD: {metrics['maxDrawdown']:.2f} | "
        f"MDDDays: {metrics['maxDrawdownDays']}"
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    if norm_pnl_per_asset.shape[1] > 1:
        for ts_code in norm_pnl_per_asset.columns:
            ax.plot(trade_dates, norm_pnl_per_asset[ts_code].cumsum(), linewidth=1, alpha=0.8, label=ts_code)
        ax.plot(trade_dates, norm_daily_pnl.cumsum(), linewidth=2, color='#1f77b4', label='Total')
        ax.legend(loc='best', fontsize=7, ncol=3)
    else:
        ax.plot(trade_dates, norm_daily_pnl.cumsum(), linewidth=2, color='#1f77b4')

    ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
    ax.set_title(title)
    ax.set_ylabel('Cumulative PnL')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    if len(trade_dates) > 0:
        ax.set_xlim(left=trade_dates[0])
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def collect_position_csv_files(result_dir):
    return sorted(result_dir.glob('*_Position.csv'))


def evaluate_one_position_file(position_csv_path, info):
    positions = load_positions(position_csv_path)
    trading_days = positions.index.tolist()
    symbols = positions.columns.tolist()
    close_df = load_close_data(symbols, trading_days)
    main_contract_df = load_main_contract(symbols, trading_days)

    pos_df, pnl_per_asset, daily_pnl = calc_raw_pnl(positions, close_df)
    norm_pos_df, norm_pnl_per_asset, norm_daily_pnl, norm_sector_daily, scale = calc_normalized(
        pos_df, pnl_per_asset, daily_pnl, info, NORM_START_DATE, NORM_END_DATE
    )
    metrics = calc_metrics(norm_daily_pnl, norm_pos_df, main_contract_df)
    trade_dates = pd.to_datetime(norm_daily_pnl.index, format='%Y%m%d')

    output_prefix = position_csv_path.stem

    raw_pnl_df = pnl_per_asset.copy()
    raw_pnl_df.insert(0, 'dailyPnl', daily_pnl)
    raw_pnl_df['cumulativePnl'] = daily_pnl.cumsum()
    raw_pnl_path = OUTPUT_DIR / f'{output_prefix}_dailyPnl.csv'
    raw_pnl_df.to_csv(raw_pnl_path, encoding='utf-8-sig')

    norm_pos_path = OUTPUT_DIR / f'{output_prefix}_position_normalized.csv'
    norm_pos_df.to_csv(norm_pos_path, encoding='utf-8-sig')

    norm_pnl_df = norm_pnl_per_asset.copy()
    norm_pnl_df.insert(0, 'dailyPnl', norm_daily_pnl)
    norm_pnl_df['cumulativePnl'] = norm_daily_pnl.cumsum()
    norm_pnl_path = OUTPUT_DIR / f'{output_prefix}_dailyPnl_normalized.csv'
    norm_pnl_df.to_csv(norm_pnl_path, encoding='utf-8-sig')

    plot_path = OUTPUT_DIR / f'{output_prefix}_cumulativePnl.png'
    plot_pnl(trade_dates, norm_daily_pnl, norm_pnl_per_asset, metrics, plot_path)

    metrics_row = {
        'strategyFile': position_csv_path.name,
        'scale': scale,
        'sharpeRatio': metrics['sharpeRatio'],
        'pot': metrics['pot'],
        'holdingPeriod': metrics['holdingPeriod'],
        'maxDrawdown': metrics['maxDrawdown'],
        'maxDrawdownDays': metrics['maxDrawdownDays'],
    }
    return metrics_row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    info = pd.read_csv(INFO_PATH, encoding='utf-8-sig')
    if 'ts_code' not in info.columns or 'sector' not in info.columns:
        raise ValueError('Info.csv 必须包含 ts_code 和 sector 列。')

    position_files = collect_position_csv_files(RESULT_DIR)
    if not position_files:
        raise ValueError(f'未在 {RESULT_DIR} 下找到 *_Position.csv 文件。')

    print(f'待评估文件数量: {len(position_files)}')
    all_metrics = []
    for index, position_csv_path in enumerate(position_files, 1):
        try:
            metrics_row = evaluate_one_position_file(position_csv_path, info)
            all_metrics.append(metrics_row)
            if index % 20 == 0 or index == len(position_files):
                print(f'评估进度: {index}/{len(position_files)}')
        except Exception as exc:
            print(f'评估失败，跳过 {position_csv_path.name}: {exc}')

    if not all_metrics:
        raise ValueError('所有文件评估均失败。')

    summary_path = OUTPUT_DIR / SUMMARY_METRICS_CSV
    pd.DataFrame(all_metrics).to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f'汇总指标输出完成: {summary_path}')


if __name__ == '__main__':
    main()

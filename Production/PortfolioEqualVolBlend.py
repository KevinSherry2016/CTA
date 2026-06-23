import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, 'Result')
OUTPUT_NAME = 'L2_EqualVol_Trend_Reversion_Alternative'
NORM_START_DATE = '20200101'
NORM_END_DATE = '20251231'


def _read_norm_pnl(strategy_name):
    path = os.path.join(BASE_DIR, strategy_name, 'Result', 'L2_Sector_Merge_All_norm_PnL.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'PnL file not found: {path}')
    df = pd.read_csv(path, index_col=0)
    series = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(0.0)
    series.index = series.index.astype(str)
    return series


def _read_norm_position(strategy_name):
    path = os.path.join(BASE_DIR, strategy_name, 'Result', 'L2_Sector_Merge_All_norm_Position.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Position file not found: {path}')
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df.apply(pd.to_numeric, errors='coerce').fillna(0.0)


def _calc_sharpe(pnl, trading_days_per_year=250):
    pnl = pnl.dropna()
    std = pnl.std()
    return pnl.mean() / std * (trading_days_per_year ** 0.5) if std != 0 else float('nan')


def _calc_max_drawdown(cum_pnl):
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    return drawdown.min()


def _build_equal_vol_weights(pnl_map):
    all_dates = None
    for s in pnl_map.values():
        all_dates = s.index if all_dates is None else all_dates.intersection(s.index)

    aligned = {}
    for name, s in pnl_map.items():
        aligned[name] = s.reindex(all_dates).fillna(0.0)

    vol_window = pd.DataFrame(aligned)
    vol_window = vol_window[(vol_window.index >= NORM_START_DATE) & (vol_window.index <= NORM_END_DATE)]

    vols = vol_window.std(axis=0)
    vols = vols.replace(0.0, np.nan)
    inv = 1.0 / vols
    weights = inv / inv.sum()
    return weights.astype(float), pd.DataFrame(aligned)


def _merge_positions(position_map, weights):
    all_index = sorted({idx for df in position_map.values() for idx in df.index})
    all_columns = sorted({col for df in position_map.values() for col in df.columns})
    out = pd.DataFrame(0.0, index=all_index, columns=all_columns, dtype=float)

    for name, df in position_map.items():
        w = float(weights[name])
        aligned = df.reindex(index=all_index, columns=all_columns).fillna(0.0)
        out = out + aligned * w
    return out


def _calc_norm_scale(pnl):
    pnl_for_scale = pnl.copy()
    pnl_for_scale = pnl_for_scale[(pnl_for_scale.index >= NORM_START_DATE) & (pnl_for_scale.index <= NORM_END_DATE)]
    scale = pnl_for_scale.std()
    if scale == 0 or pd.isna(scale):
        return 1.0
    return float(scale)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    strategies = ['Trend', 'Reversion', 'Alternative']
    pnl_map = {s: _read_norm_pnl(s) for s in strategies}
    pos_map = {s: _read_norm_position(s) for s in strategies}

    weights, pnl_df = _build_equal_vol_weights(pnl_map)
    blend_pnl = (pnl_df * weights).sum(axis=1)
    blend_pos = _merge_positions(pos_map, weights)

    # Normalize final portfolio so daily PnL std is 1 in the configured window.
    scale = _calc_norm_scale(blend_pnl)
    blend_pnl = blend_pnl / scale
    blend_pos = blend_pos / scale
    blend_gmv = blend_pos.abs().sum(axis=1)

    sharpe = _calc_sharpe(blend_pnl)
    cum_pnl = blend_pnl.cumsum()
    max_dd = _calc_max_drawdown(cum_pnl)

    out_weights = os.path.join(RESULT_DIR, f'{OUTPUT_NAME}_weights.csv')
    out_pos = os.path.join(RESULT_DIR, f'{OUTPUT_NAME}_norm_Position.csv')
    out_pnl = os.path.join(RESULT_DIR, f'{OUTPUT_NAME}_norm_PnL.csv')
    out_gmv = os.path.join(RESULT_DIR, f'{OUTPUT_NAME}_GMV.csv')
    out_png = os.path.join(RESULT_DIR, f'{OUTPUT_NAME}_cPnL.png')

    weights.rename('weight').to_frame().to_csv(out_weights, encoding='utf-8-sig')
    blend_pos.to_csv(out_pos, encoding='utf-8-sig')
    blend_pnl.to_frame(name='PnL').to_csv(out_pnl, encoding='utf-8-sig')
    blend_gmv.to_frame(name='GMV').to_csv(out_gmv, encoding='utf-8-sig')

    trade_dates = pd.to_datetime(cum_pnl.index, format='%Y%m%d')
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(trade_dates, cum_pnl, linewidth=2, color='#1f77b4')
    ax.axhline(0.0, color='black', linewidth=0.6, linestyle='--')
    ax.set_title(
        f'{OUTPUT_NAME} | Sharpe: {sharpe:.2f} | CumPnL: {cum_pnl.iloc[-1]:.2f} | MaxDD: {max_dd:.2f}'
    )
    ax.set_ylabel('Cumulative PnL')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(left=trade_dates[0])
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print('Equal-vol blend generated successfully:')
    print(f'  Trend weight: {weights["Trend"]:.4f}')
    print(f'  Reversion weight: {weights["Reversion"]:.4f}')
    print(f'  Alternative weight: {weights["Alternative"]:.4f}')
    print(f'  Final normalization scale: {scale:.6f}')
    print(f'  Final PnL std (norm window): {blend_pnl[(blend_pnl.index >= NORM_START_DATE) & (blend_pnl.index <= NORM_END_DATE)].std():.6f}')
    print(f'  Sharpe: {sharpe:.4f}')
    print(f'  CumPnL: {cum_pnl.iloc[-1]:.2f}')
    print(f'  MaxDD: {max_dd:.2f}')
    print(f'  Output dir: {RESULT_DIR}')


if __name__ == '__main__':
    main()
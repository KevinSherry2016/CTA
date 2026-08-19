from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'Data'
RESULT_DIR = BASE_DIR / 'Result'
ASSETS = ('BTC', 'ETH', 'SOL')
VOL_WINDOW = 20
TRADING_DAYS_PER_YEAR = 365


def _read_close(asset):
    path = DATA_DIR / f'{asset}.csv'
    if not path.exists():
        raise FileNotFoundError(f'Price file not found: {path}')

    data = pd.read_csv(path)
    lower_columns = {column.lower(): column for column in data.columns}
    date_column = next(
        (lower_columns[name] for name in ('date', 'trade_date', 'datetime', 'timestamp') if name in lower_columns),
        data.columns[0],
    )
    close_column = next(
        (lower_columns[name] for name in ('close', 'adj_close', 'price') if name in lower_columns),
        None,
    )
    if close_column is None:
        raise ValueError(f'{path} must contain one of: close, adj_close, price')

    close = pd.to_numeric(data[close_column], errors='coerce')
    dates = pd.to_datetime(data[date_column], errors='coerce')
    series = pd.Series(close.to_numpy(), index=dates, name=asset).dropna()
    series = series[~series.index.duplicated(keep='last')].sort_index()
    if series.empty:
        raise ValueError(f'No valid price observations found in {path}')
    return series


def build_inverse_volatility_portfolio(price_df):
    returns = price_df.pct_change(fill_method=None)
    volatility = returns.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    inverse_volatility = 1.0 / volatility.replace(0.0, np.nan)
    target_weights = inverse_volatility.div(inverse_volatility.sum(axis=1), axis=0).fillna(0.0)
    weights = target_weights.shift(1).fillna(0.0)
    portfolio_returns = (weights * returns.fillna(0.0)).sum(axis=1)
    return returns, volatility, weights, portfolio_returns


def main():
    RESULT_DIR.mkdir(exist_ok=True)

    prices = pd.concat([_read_close(asset) for asset in ASSETS], axis=1).sort_index().ffill()
    returns, volatility, weights, portfolio_returns = build_inverse_volatility_portfolio(prices)
    net_value = (1.0 + portfolio_returns).cumprod()

    result = pd.DataFrame(
        {
            'portfolio_return': portfolio_returns,
            'net_value': net_value,
        }
    )
    result.index.name = 'date'
    weights.index.name = 'date'
    volatility.index.name = 'date'

    weights.to_csv(RESULT_DIR / 'BTC_ETH_SOL_inverse_vol_weights.csv', encoding='utf-8-sig')
    volatility.to_csv(RESULT_DIR / 'BTC_ETH_SOL_rolling_volatility.csv', encoding='utf-8-sig')
    result.to_csv(RESULT_DIR / 'BTC_ETH_SOL_inverse_vol_portfolio.csv', encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(net_value.index, net_value, linewidth=2, color='#1f77b4')
    ax.set_title(f'BTC / ETH / SOL Inverse Volatility Portfolio ({VOL_WINDOW}-day)')
    ax.set_ylabel('Net Value')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / 'BTC_ETH_SOL_inverse_vol_net_value.png', dpi=150)
    plt.close(fig)

    annualized_volatility = portfolio_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    print('BTC / ETH / SOL inverse-volatility portfolio generated.')
    print(f'Annualized portfolio volatility: {annualized_volatility:.2%}')
    print(f'Results: {RESULT_DIR}')


if __name__ == '__main__':
    main()
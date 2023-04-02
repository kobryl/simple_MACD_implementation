from typing import Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    path = "data.csv"
    macd, signal = calculate_macd(path)
    buy, sell = calculate_buy_sell(macd, signal)
    closing_prices = []
    with open(path, 'r') as f:
        data = pd.read_csv(f)
        closing_prices = data.loc[:, 'Close']
    cash = 1000
    print("Initial value: " + str(cash))
    profit, end_value = calculate_profit(closing_prices, buy, sell, cash)
    print("Final value: " + str(end_value))
    print("Profit: " + str(profit))
    plot_macd(macd, signal, buy, sell)
    plot_stock(closing_prices, buy, sell)


def plot_stock(prices: pd.Series, buy: pd.Series, sell: pd.Series) -> None:
    """
    This function plots the stock prices
    :param prices: closing prices of the data
    :param buy: buy signals
    :param sell: sell signals
    :return: None
    """
    transformed_buy, transformed_sell = transform_buy_sell(prices, buy, sell)
    plt.plot(range(len(prices)), prices)
    plt.plot(range(len(prices)), transformed_buy, 'x', color='green', label='BUY', markersize=10)
    plt.plot(range(len(prices)), transformed_sell, 'x', color='red', label='SELL', markersize=10)
    plt.legend(['Stock Price', 'BUY', 'SELL'])
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('S&P500 closing prices for 1000 days from 27-03-2019')
    plt.savefig('stock_prices.png')
    plt.show()


def transform_buy_sell(prices: pd.Series, buy: pd.Series, sell: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    This function transforms the buy and sell signals to be plotted with the stock prices
    :param prices: closing prices of the data
    :param buy: buy signals
    :param sell: sell signals
    :return: transformed buy and sell signals
    """
    transformed_buy = [np.nan] * len(buy)
    transformed_sell = [np.nan] * len(sell)
    for i in range(len(buy)):
        transformed_buy[i] = buy.iat[i] if np.isnan(buy.iat[i]) else prices.iat[i]
        transformed_sell[i] = sell.iat[i] if np.isnan(sell.iat[i]) else prices.iat[i]
    return pd.Series(transformed_buy), pd.Series(transformed_sell)


def calculate_profit(prices: pd.Series, buy: pd.Series, sell: pd.Series, initial_cash: float) -> tuple[float, float]:
    """
    This function calculates the profit of the given buy and sell signals
    :param prices: closing prices of the data
    :param buy: buy signals
    :param sell: sell signals
    :param initial_cash: initial cash
    :return: profit
    """
    value = 0
    holdings = 0
    cash = initial_cash
    for i in range(1, len(prices)):
        if not np.isnan(buy.iat[i]):
            if cash > 0:
                holdings += cash / prices.iat[i]
                cash = 0
        elif not np.isnan(sell.iat[i]):
            if holdings > 0:
                cash += holdings * prices.iat[i]
                holdings = 0
        value = holdings * prices.iat[i]
    profit = value + cash - initial_cash
    return profit, value + cash


def plot_macd(macd: pd.Series, signal: pd.Series, buy: pd.Series, sell: pd.Series) -> None:
    """
    This function plots the MACD and SIGNAL lines with buy and sell indicators
    :param macd: MACD line values
    :param signal: SIGNAL line values
    :param buy: buy signals
    :param sell: sell signals
    :return: None
    """
    plt.plot(range(len(macd)), signal)
    plt.plot(range(len(macd)), macd)
    plt.plot(range(len(macd)), buy, 'x', color='green', label='BUY', markersize=10)
    plt.plot(range(len(macd)), sell, 'x', color='red', label='SELL', markersize=10)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('MACD')
    plt.legend(['SIGNAL', 'MACD', 'BUY', 'SELL'])
    plt.savefig('macd.png')
    plt.show()


def calculate_buy_sell(macd: pd.Series, signal: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    This function calculates the buy and sell signals for the given MACD and SIGNAL lines
    :param macd: MACD line values
    :param signal: SIGNAL line values
    :return: buy and sell signals (nan if no signal, MACD value if signal)
    """
    buy = [np.nan]
    sell = [np.nan]
    for i in range(1, len(macd)):
        if macd[i - 1] < signal[i - 1] and macd[i] >= signal[i]:
            buy.append(macd[i])
            sell.append(np.nan)
        elif macd[i - 1] > signal[i - 1] and macd[i] <= signal[i]:
            buy.append(np.nan)
            sell.append(macd[i])
        else:
            buy.append(np.nan)
            sell.append(np.nan)
    return pd.Series(buy), pd.Series(sell)


def calculate_macd(path: str) -> tuple[pd.Series, pd.Series]:
    """
    This function calculates the MACD and SIGNAL lines for the given data and plots the graph
    :param path: path to the data file
    :type path: str
    :return: None
    """
    with open(path, 'r') as f:
        data = pd.read_csv(f)
        dates = data.loc[:, 'Date']
        closing_prices = data.loc[:, 'Close']
        signal = []
        macd = []
        for date, price in zip(dates, closing_prices):
            slow = calculate_slow_ema(date, dates, closing_prices)
            fast = calculate_fast_ema(date, dates, closing_prices)
            macd.append(fast - slow)
            if len(macd) < 9:
                signal.append(calculate_ema(dates[:len(macd)], pd.Series(macd)))
            else:
                signal.append(calculate_ema(dates[len(macd) - 9:len(macd)], pd.Series(macd[len(macd) - 9:len(macd)])))
        return pd.Series(macd), pd.Series(signal)


def calculate_fast_ema(date: str, dates: pd.Series, prices: pd.Series) -> float:
    """
    This function calculates the fast exponential moving average for the given date.
    :param date: the date for which the EMA is to be calculated
    :param dates: dates of the data
    :param prices: closing prices of the data
    :return: slow EMA for the given date
    """
    end_index = dates[dates == date].index[0]
    start_index = 0
    if end_index < 12:
        start_index = 0
    else:
        start_index = int(end_index - 12)
    return calculate_ema(dates[start_index:end_index + 1], prices[start_index:end_index + 1])


def calculate_slow_ema(date: str, dates: pd.Series, prices: pd.Series) -> float:
    """
    This function calculates the slow exponential moving average for the given date.
    :param date: the date for which the EMA is to be calculated
    :param dates: dates of the data
    :param prices: closing prices of the data
    :return: fast EMA for the given date
    """
    end_index = dates[dates == date].index[0]
    start_index = 0
    if end_index > 26:
        start_index = end_index - 26
    return calculate_ema(dates[start_index:end_index + 1], prices[start_index:end_index + 1])


def calculate_ema(dates: pd.Series, prices: pd.Series) -> float:
    """
    This function calculates the exponential moving average for the given date
    :param dates: range of dates for which the EMA is to be calculated
    :param prices: closing prices at dates in the range
    :return: EMA for the given range of dates
    """
    N = len(dates) - 1
    alfa = 2 / (N + 1)
    numerator = 0
    denominator = 0
    for i in range(N):
        price = prices.iat[i]
        numerator += price * (1 - alfa) ** (N - i)
        denominator += (1 - alfa) ** (N - i)
    numerator += prices.iat[N]
    denominator += 1
    ema = numerator / denominator
    return ema


if __name__ == '__main__':
    main()

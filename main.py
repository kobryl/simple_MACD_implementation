import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    path = "data.csv"
    macd(path)


def macd(path: str) -> None:
    """
    This function calculates the MACD for the given data and plots the graph
    :param path: path to the data file
    :type path: str
    :return: None
    """
    with open(path, 'r') as f:
        data = pd.read_csv(f)
        dates = data.loc[:, 'Date']
        closing_prices = data.loc[:, 'Close']
        MACS = []
        for date, price in zip(dates, closing_prices):
            slow = calculate_slow_ema(date, dates, closing_prices)
            fast = calculate_fast_ema(date, dates, closing_prices)
            MACS.append(fast - slow)


def calculate_slow_ema(date: str, dates: pd.Series, prices: pd.Series) -> float:
    """
    This function calculates the slow exponential moving average for the given date.
    :param date: the date for which the EMA is to be calculated
    :param dates: dates of the data
    :param prices: closing prices of the data
    :return: slow EMA for the given date
    """
    end_index = dates[dates == date].index[0]
    start_index = 0
    if end_index < 25:                                          # when the current date is earlier than 26th
        start_index = end_index / 2
    else:
        start_index = end_index - 11
    pass


def calculate_fast_ema(date: str, dates: pd.Series, prices: pd.Series) -> float:
    """
    This function calculates the fast exponential moving average for the given date.
    :param date: the date for which the EMA is to be calculated
    :param dates: dates of the data
    :param prices: closing prices of the data
    :return: fast EMA for the given date
    """
    end_index = dates[dates == date].index[0]
    start_index = 0
    if end_index >= 25:                  # when the current date is 26th or later, then we can perform full calculations
        start_index = end_index - 25
    pass


def calculate_ema(dates: pd.Series, prices: pd.Series) -> float:
    """
    This function calculates the exponential moving average for the given date
    :param dates: range of dates for which the EMA is to be calculated
    :param prices: closing prices at dates in the range
    :return: EMA for the given range of dates
    """


if __name__ == '__main__':
    main()

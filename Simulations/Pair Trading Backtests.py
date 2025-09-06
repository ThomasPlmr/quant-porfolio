# simulating pair trading using python, did adf and cointegration tests to measure risk, found sharpe ratio and finally returned PnL over given time frame

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
import os
os.system('cls')

fileName = 'historicalPrices.xlsx'

tickers = ["GLD", "SLV"]                                         # what we are tracking -> gold vs silver

startdate = "2022-01-01"
enddate = "2025-01-01"

data = yf.download(tickers, start=startdate, end=enddate)
closePrices = data['Close']

closePrices.to_excel(fileName)

df = pd.read_excel("historicalPrices.xlsx", index_col=0)        # indexcol means when you print data, it doesn't have 1, 2, 3, 4, ..., 1000, down the side of it

df['Ratio'] = df['GLD']/df['SLV']
df['MovingAv'] = df['Ratio'].rolling(window=30).mean()
df['MovingStd'] = df['Ratio'].rolling(window=30).std()
df['zScore'] = (df['Ratio'] - df['MovingAv']) / df['MovingStd']
df.to_excel(fileName)

print("------------------------------")
#print(df)                                                       # df.head() would print just the starts of the data, e.g. the first 5 rows


def DisplayData(type):                                          # function for displaying data

    if (type == "comparison"):
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['GLD'], label='Gold (GLD)')
        plt.plot(df.index, df['SLV'], label='Silver (SLV)')

        plt.title('Gold vs Silver prices')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')

    elif (type == "ratio"):
        plt.figure(figsize=(12, 6))

        plt.plot(df.index, df['Ratio'], label='GLD/SLV Ratio', color='purple')

        avRatio = df['Ratio'].mean()
        plt.axhline(y=avRatio, color='r', linestyle='--', label=f'Average: {avRatio:.2f}')

        plt.title('Gold to Silver Ratio (GLD/SLV)')
        plt.xlabel('Date')
        plt.ylabel('Ratio')
    
    plt.legend()
    plt.grid(True)

    plt.show()

def displayzScore():                                           # visual example of when we should long and short ETFs
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['zScore'], label='Z-Score')
    
    plt.axhline(y=1.5, color='r', linestyle='--', label='Short (Z > 1.5)')
    plt.axhline(y=-1.5, color='g', linestyle='--', label='Long (Z < -1.5)')
    plt.axhline(y=0.0, color='b', linestyle='-', label='Exit (Z = 0)')
    
    plt.title('Spread Z-Score')
    plt.xlabel('Date')
    plt.ylabel('Standard Deviations')
    plt.legend()
    plt.grid(True)
    plt.show()


#displayzScore()
#DisplayData("ratio")

adfTest = adfuller(df['Ratio'].dropna())

print(f"\nADF Statistic: {adfTest[0]:.4f}")                     # adf tests if mean is reverting (if p < 0.05), it is.
print(f"P-value: {adfTest[1]:.4f}")

CoinResult = coint(df['GLD'], df['SLV'])

print(f"\nCoint Statistic: {CoinResult[0]:.4f}")                # failed implies the long term average of the two is weak and "the leash is too elastic"
print(f"P-value: {CoinResult[1]:.4f}")                          # p-val was greater than 0.05, would be risky to trade

# --------------------------------------------------------------------- Implementing Trades --------------------------------------------------------------------------- #

entry = 1.5
exit = 0.0

df['Position'] = 0                                              # 1 long, -1 short, and 0 nothing
df['PnL'] = 0.0

for i in range(30, len(df)):                                    # loop from first valid z score
    zScore = df['zScore'].iloc[i]
    prevPosition = df['Position'].iloc[i-1]

    if (prevPosition == 0): # if flat
        if (zScore < -entry):
            df.loc[df.index[i], 'Position'] = 1
        elif (zScore > entry):
            df.loc[df.index[i], 'Position'] = -1
        else:
            df.loc[df.index[i], 'Position'] = 0

    elif (prevPosition == 1): # if long
        if zScore >= exit:
            df.loc[df.index[i], 'Position'] = 0 # exit
        else:
            df.loc[df.index[i], 'Position'] = 1 # hold
    
    elif (prevPosition == -1): # if short
        if zScore <= exit:
            df.loc[df.index[i], 'Position'] = 0 # exit
        else:
            df.loc[df.index[i], 'Position'] = -1 # hold

df['GLDreturns'] = df['GLD'].pct_change()                       # calculate daily profit/loss
df['SLVreturns'] = df['SLV'].pct_change()

df['PnL'] = df['Position'].shift(1) * (df['GLDreturns'] - df['SLVreturns'])
df['CumulativePnL'] = df['PnL'].cumsum()

def DisplayPnL():
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['CumulativePnL'], label='Equity Curve')
    plt.title('Backtest Performance: Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profit/Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# DisplayPnL()

print(df.tail(10)) # shows pnl results ect

totalPnL = df['CumulativePnL'].iloc[-1]
DailyPnL = df['PnL'].dropna()

sharpeRatio = (DailyPnL.mean() / DailyPnL.std()) * np.sqrt(252)

print(f"\nTotal Profit/Loss: {totalPnL:.4f}")
print(f"Sharpe Ratio: {sharpeRatio:.2f} \n")

def SummaryPlot():
    
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Ratio'], label='Ratio (Spread)', color='purple', alpha=0.6)
    plt.plot(df.index, df['MovingAv'], label='Moving Average', color='black', linestyle='--')

    long_entries = df[(df['Position'] == 1) & (df['Position'].shift(1) == 0)]
    short_entries = df[(df['Position'] == -1) & (df['Position'].shift(1) == 0)]
    exits = df[(df['Position'] == 0) & (df['Position'].shift(1) != 0)]
    
    plt.scatter(long_entries.index, long_entries['Ratio'], marker='^', color='g', s=100, label='Long Entry') # plot markers
    plt.scatter(short_entries.index, short_entries['Ratio'], marker='v', color='r', s=100, label='Short Entry')
    plt.scatter(exits.index, exits['Ratio'], marker='x', color='b', s=100, label='Exit')

    plt.title('Plot: Spread, Moving Averages and Trades')
    plt.xlabel('Date')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True)
    plt.show()


SummaryPlot()

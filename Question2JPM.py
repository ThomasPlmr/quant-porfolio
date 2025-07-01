import pandas as pd
from sklearn.linear_model import LinearRegression
import os
os.system('cls')

df = pd.read_csv(r'C:\Users\imnaz\OneDrive\Documents\Coding\Work Experience\Nat_Gas.csv')

df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
df['Month'] = df['Dates'].dt.month
df['Year'] = df['Dates'].dt.year

Dates = df["Dates"].values
Prices = df["Prices"].values

def PredictPrice2025(Month):
    MonthData = df[df['Month'] == Month]

    x = MonthData['Year'].values.reshape(-1, 1)
    y = MonthData['Prices'].values

    Model = LinearRegression()
    Model.fit(x, y)

    PredictionPrice = Model.predict([[2025]])[0]

    return PredictionPrice

# Code from above is from task 1, we are told to use it again so that we can reuse our prediction model.
# If you'd like help to understand task 1 above, check my Question1JPM.py file in this repository

def ContractProfit(InjectionDate, WithdrawalDate, DailyInjectionRate, MaxStorageVolume):  # Daily injection rate = daily withdraw rate
    
    # I didn't include any limits into what dates were inputted since I am the one inputting them, however that could easily be added by settings limits, e.g. 1<=Dates<=12

    Returns = 0

    Parts1 = InjectionDate.split('-')
    Parts2 = WithdrawalDate.split('-')
    
    # The following section is for converting dates in format 2025-5-13 into variables Date1 = 13, Month1 = 5.
    
    Month1 = int(Parts1[1])
    Date1 = int(Parts1[2])
    Month2 = int(Parts2[1])
    Date2 = int(Parts2[2])
    
    ProfitLossPerMMBu = (PredictPrice2025(Month2) + (PredictPrice2025(Month2+1)-PredictPrice2025(Month2))*(Date2/30)) - (PredictPrice2025(Month1) + (PredictPrice2025(Month1+1)-PredictPrice2025(Month1))*(Date1/30))
    
    Returns = MaxStorageVolume * ProfitLossPerMMBu

    StorageMonthlyCost = 80000
    InjectionCostPerDay = 10000
    BaseTransportCost = 150000
    
    # In this line I assume storage payment is on first of every month. Rather than actual days spent in storage
    Returns = Returns - StorageMonthlyCost*(Month2-Month1) - (MaxStorageVolume/DailyInjectionRate)*2*InjectionCostPerDay - (BaseTransportCost)*2

    # Didn't use variable 'profit' becuase we may lose money.
    return Returns

InputDate1 = input("Enter injection date in format year-month-date:\n")
InputDate2 = input("Enter withdrawal date in format year-month-date:\n")

Contract = ContractProfit(InputDate1, InputDate2, 100000, 2500000) # 100k daily injection rate and 2.5mil max storage
print(f"Contract Price = ${Contract}")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
os.system('cls')

df = pd.read_csv(r'C:\Users\imnaz\OneDrive\Documents\Coding\Work Experience\Nat_Gas.csv') #df = dataframe

df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y') # Puts 10/31/20 into 2020-10-31 format
df['Month'] = df['Dates'].dt.month
df['Year'] = df['Dates'].dt.year

Dates = df["Dates"].values
Prices = df["Prices"].values

def PredictPrice2025(Month):
    MonthData = df[df['Month'] == Month]

    x = MonthData['Year'].values.reshape(-1, 1)     # MonthData['Year/Prices'].values takes all historical prices from input month, puts in array
    y = MonthData['Prices'].values                  # Reshape(-1, 1) converts [2020, 2021, 2022] into [[2020], [2021], [2022]], because sklearns linear regression requires 2D input array to work
    print("x:", x)
    print("y:", y)

    Model = LinearRegression()
    Model.fit(x, y)

    PredictionPrice = Model.predict([[2025]])[0]    # The [0] removes the brackets, e.g. [13.45] into 13.45

    return PredictionPrice

def PrintScreen():
    
    prediction_dates = []
    prediction_prices = []
    
    for i in range(1, 13):                          # Loop to add 12 months of data (2025's data) to visualise
        result = PredictPrice2025(i)
        pred_date = pd.to_datetime(f'2025-{i:02d}-01')
        prediction_dates.append(pred_date)
        prediction_prices.append(result)

    plt.figure(figsize=(11, 6))
    plt.plot(Dates, Prices, label='Historical Data', color='blue')
    plt.plot(prediction_dates, prediction_prices, label='2025 Predictions', 
             color='red')
    plt.xlabel('Dates')
    plt.ylabel('Prices')
    plt.title('Natural Gas Prices Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()


while True:
    choice = int(input("Enter 1 for prediction calculator, and 2 for prediction visualisation.\n"))
    if (choice == 1):
        MonthPredict = int(input("From 1 to 12 where 1 is Jan and 12 is Dec. Enter number of month you would like 2025 Prediction for:\n--> "))
        result = PredictPrice2025(MonthPredict)
        print(f"Predicted price for month {MonthPredict}: ${result:.2f}")
        break
    elif (choice == 2):
        PrintScreen()
        break
    else:
        print("Invalid input.")
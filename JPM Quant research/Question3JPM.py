import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
import os
warnings.filterwarnings('ignore')
os.system('cls')

df = pd.read_csv(r'C:\Users\imnaz\OneDrive\Documents\Coding\Work Experience\Task 3 and 4_Loan_Data.csv')

print(df.head())

df['debt_to_income'] = df['total_debt_outstanding'] / df['income']
df['credit_utilization'] = df['loan_amt_outstanding'] * df['credit_lines_outstanding']

features = ['income', 'fico_score', 'debt_to_income', 'credit_utilization']
x = df[features]
y = df['default']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)   # Test size = 0.2 means 20% is for testing 80% for trainings

model = LogisticRegression(max_iter=1000)                                                   # Large data range so had to use max iteration of 1000
model.fit(x, y)

probabilities = model.predict_proba(x_test)[:, 1]                                           # Probabilites of defaulting (PD)

LossFromDefault = 0.9                                                                       # only 0.9 because 10% recovery rate

TestLoanAmount = df.loc[x_test.index, 'loan_amt_outstanding']

ExpectedLoss = probabilities * TestLoanAmount * LossFromDefault
    
def CustomerLossCalculator(TotalDebtOutstanding, YearlyIncome, LoansOutstanding, CreditLinesOutstanding, FicoScore):
    
    # Using our LogisticRegression model above, we created a function to return Predicted Loss and Probability of default
    DebtIncomeRatio = TotalDebtOutstanding/YearlyIncome
    CreditUtilisation = LoansOutstanding * CreditLinesOutstanding

    LoanAttributes = [[YearlyIncome, FicoScore, DebtIncomeRatio, CreditUtilisation]]
    
    ProbabilityDefault = model.predict_proba(LoanAttributes)[0, 1]
    
    LossFromDefault = 0.9
    PredictedLoss = ProbabilityDefault * LoansOutstanding * LossFromDefault
    
    return PredictedLoss, ProbabilityDefault

print()
a = int(input("Customers total outstanding debt:\n"))
b = int(input("Customers yearly income:\n"))
c = int(input("Customers total outstanding loans:\n"))
d = int(input("Customers credit lines outstanding:\n"))
e = int(input("Customers fico score:\n"))

Loss, ProbDefault = CustomerLossCalculator(a, b, c, d, e)

print(f"customer's PD: {ProbDefault}, and expected loss: {Loss}")
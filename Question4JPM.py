import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os

os.system('cls')

df = pd.read_csv('C:/Users/imnaz/OneDrive/Documents/Coding/Work Experience/Task 3 and 4_Loan_Data.csv')

x = df[['fico_score']]
y = df['default']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(x, y)

def PredictPD(FicoScore):
    Fico = pd.DataFrame({'fico_score': [FicoScore]})
    return model.predict_proba(Fico)[0, 1]

df = df.sample(n=2000, random_state=42).sort_values(by='fico_score').reset_index(drop=True)
fico_scores = df['fico_score'].to_numpy()
defaults = df['default'].to_numpy()
n = len(fico_scores)

cumulative_defaults = np.cumsum(defaults)

def bucket_log_likelihood(start, end):
    k = cumulative_defaults[end - 1] - (cumulative_defaults[start - 1] if start > 0 else 0)
    total = end - start
    if k == 0 or k == total:
        return -1e9
    p = k / total
    return k * math.log(p) + (total - k) * math.log(1 - p)

z = int(input("\nEnter how many buckets you'd like to split the FICO scores into:\n"))

dp = np.full((z + 1, n + 1), -np.inf)
dp[0][0] = 0
prev = np.zeros((z + 1, n + 1), dtype=int)

for k in range(1, z + 1):
    for j in range(k, n + 1):
        for i in range(k - 1, j):
            ll = dp[k - 1][i] + bucket_log_likelihood(i, j)
            if ll > dp[k][j]:
                dp[k][j] = ll
                prev[k][j] = i

boundaries = []
k = z
j = n
while k > 0:
    i = prev[k][j]
    boundaries.append((fico_scores[i], fico_scores[j - 1]))
    j = i
    k -= 1
boundaries = boundaries[::-1]

print("\nFico bucket ranges:")
for idx, (start, end) in enumerate(boundaries):
    print(f"Bucket {idx + 1}: {int(start)} to {int(end)}")

def get_bucket(fico):
    for idx, (start, end) in enumerate(boundaries):
        if start <= fico <= end:
            return idx
    return -1

df['bucket'] = df['fico_score'].apply(get_bucket)

bucket_pds = df.groupby('bucket')['default'].mean()

print("\nPD for each bucket:")
for idx, pd_val in bucket_pds.items():
    b_start, b_end = boundaries[idx]
    print(f"Bucket {idx + 1} ({int(b_start)}–{int(b_end)}): PD = {pd_val:.3f}")

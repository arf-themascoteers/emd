import pandas as pd
from sklearn.model_selection import train_test_split
from PyEMD import EMD
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics


df = pd.read_csv("sm.csv")
df.head()

datasets = df
datasets.keys()
X = df
y = datasets.soil_moisture

# get hyperspectral bands:
hypbands = []
for col in df.columns:
    try:
        int(col)
    except Exception:
        continue
    hypbands.append(col)

array = np.zeros((len(df), len(hypbands)))
for i in range(len(df)):
    row = df.loc[i]
    array[i] = np.array(row[hypbands].values)


emd = EMD()
IMFs = np.zeros((array.shape[0], array.shape[1]))
for index, row in enumerate(array):
    x = emd(row)
    variations = np.var(x, axis=1)
    index_with_most_variation = np.argmax(variations)
    IMFs[index] = x[index_with_most_variation]

IMFs = IMFs.reshape(IMFs.shape[0],-1)

X_train, X_test, y_train, y_test = train_test_split(
    IMFs, df["soil_moisture"],
    test_size=0.3, random_state=0, shuffle=True)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
score = model_lr.score(X_test, y_test)
print(score)
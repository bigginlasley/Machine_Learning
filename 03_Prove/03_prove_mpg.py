from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

names = ["mpg","cyl","disp","hp","weight","accel","model","origin","car_name"]
data = pd.read_csv("03_Prove/auto-mpg.data", header=None, skipinitialspace=True,
                   names=names, delim_whitespace=True, na_values=["?"])

data = data.dropna()
print(data)
print(data.columns)
print(data.dtypes)

data_d = data.drop(columns=["mpg", "car_name"]).values

data_t = data.drop(columns=["cyl","disp","hp","weight","accel","model","origin","car_name"]).values.flatten()

train_data, test_data, train_target, test_target = train_test_split(data_d, data_t, test_size = .30)

regr = KNeighborsRegressor(n_neighbors=3)
regr.fit(train_data, train_target)
predictions = regr.predict(test_data)

score = r2_score(test_target, predictions)

print("{:.2%}".format(score))

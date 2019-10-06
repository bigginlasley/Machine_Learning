from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# ...
# ... code here to load a training and testing set
# ...

names = ["buying","maint","doors","persons","lug","safety","classif"]

car_data = pd.read_csv("03_Prove/car.data", header=None, skipinitialspace=True,
                   names=names, na_values=["?"])


print(car_data)
print(car_data.columns)
print(car_data.dtypes)

#buying vhigh high med low

car_data["buying"] = car_data.buying.map({"vhigh": 3.0, "high": 2.0, "med": 1.0, "low": 0.0})

#maint vhigh high med low

car_data["maint"] = car_data.maint.map({"vhigh": 3.0, "high": 2.0, "med": 1.0, "low": 0.0})


#doors 2 3 4 5more

car_data["doors"] = car_data.doors.map({"2": 2.0, "3": 3.0, "4": 4.0, "5more": 5.0})


#persons 2 4 more

car_data["persons"] = car_data.persons.map({"2": 2.0, "4": 4.0, "more": 6.0})

#lug small med big

car_data["lug"] = car_data.lug.map({"small": 1.0, "med": 2.0, "big": 3.0})

#safety low med high

car_data["safety"] = car_data.safety.map({"low": 1.0, "med": 2.0, "high": 3.0})

#classification unacc acc good vgood

car_data["classif"] = car_data.classif.map({"unacc": 1.0, "acc": 2.0, "good": 3.0, "vgood": 4.0})


print(car_data)

car_target = car_data["classif"].values.flatten()

car_data = car_data.drop(columns=["classif"]).values

train_data, test_data, train_target, test_target = train_test_split(car_data, car_target, test_size = .30)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(train_data, train_target)
predictions = classifier.predict(test_data)

score = accuracy_score(predictions, test_target)

print("{:.2%}".format(score))

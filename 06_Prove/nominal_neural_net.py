import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

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

model = models.Sequential()

model.add(layers.Dense(50, activation="relu", input_shape=(6, )))

model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))

model.add(layers.Dense(1, activation = "sigmoid"))


model.compile(
 optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"]
)

results = model.fit(
 train_data, train_target,
 epochs= 20,
 batch_size = 10,
 validation_data = (test_data, test_target)
)
_, accuracy = model.evaluate(car_data, car_target)
print('Accuracy: %.2f' % (accuracy*100))
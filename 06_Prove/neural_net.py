import numpy as np
import keras as ks
from keras import layers
from keras.layers import Dense
from sklearn.model_selection import train_test_split

dataset = np.loadtxt('06_Prove/pima-indians-diabetes.data.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
data_train, data_test, target_train, target_test = train_test_split(X, y, test_size = .3)
model = ks.Sequential()
model.add(Dense(60, input_shape=(8,), kernel_initializer='RandomNormal', activation='relu'))
model.add(Dense(30, activation='relu', kernel_initializer='RandomNormal'))
model.add(Dense(15, activation='relu', kernel_initializer='RandomNormal'))
model.add(Dense(1, activation='sigmoid', kernel_initializer='RandomNormal'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
results = model.fit(data_train, target_train, epochs=400, batch_size=25, validation_data=(data_test, target_test))
print("Test-Accuracy:", np.mean(results.history["val_accuracy"]))
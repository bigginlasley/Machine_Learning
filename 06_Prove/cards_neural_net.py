import numpy as np
import keras as ks
from keras import layers
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
import matplotlib.pyplot as plt


dataset = np.loadtxt('06_Prove/poker-hand-testing.data', delimiter=',')
X = dataset[:,0:10]
y = dataset[:,10]
data_train, data_test, target_train, target_test = train_test_split(X, y, test_size = .05)
model = ks.Sequential()
model.add(Dense(128, input_shape=(10,), kernel_initializer='RandomNormal', activation='relu'))
# model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(Dense(64, activation='relu', kernel_initializer='RandomNormal'))
# model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(Dense(32, activation='relu', kernel_initializer='RandomNormal'))
# model.add(layers.Dropout(0.1, noise_shape=None, seed=None))
model.add(Dense(1, activation='sigmoid', kernel_initializer='RandomNormal'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
results = model.fit(data_train, target_train, epochs=10, batch_size=300, validation_data=(data_test, target_test))
print("Test-Accuracy:", np.mean(results.history["val_accuracy"]))
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
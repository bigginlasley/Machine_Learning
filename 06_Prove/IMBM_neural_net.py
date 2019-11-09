import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
def vectorize(sequences, dimension = 10000):
 results = np.zeros((len(sequences), dimension))
 for i, sequence in enumerate(sequences):
  results[i, sequence] = 1
 return results
 
data = vectorize(data)
targets = np.array(targets).astype("float32")
test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

model = models.Sequential()
model.add(layers.Dense(128, activation = "relu", input_shape=(10000, )))
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(32, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
results = model.fit(train_x, train_y, epochs= 4, batch_size = 500, validation_data = (test_x, test_y))
print("Test-Accuracy:", np.mean(results.history["val_accuracy"]))
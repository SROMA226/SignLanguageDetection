from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam

from config import Config

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

X_train_CNN = np.load(str(Config.FEATURES_PATH / "x_train_features_cnn.npy"), allow_pickle=False)
Y_train_CNN = np.load(str(Config.FEATURES_PATH / "y_train_labels_cnn.npy"), allow_pickle=False)


modelCNN = Sequential()

modelCNN.add(Conv2D(128, (3, 3), activation="relu", input_shape=X.shape[1:]))
modelCNN.add(MaxPooling2D(pool_size=(2, 2)))

modelCNN.add(Conv2D(64, (3, 3), activation="relu"))
modelCNN.add(MaxPooling2D(pool_size=(2, 2)))

modelCNN.add(Conv2D(32, (3, 3), activation="relu"))
modelCNN.add(MaxPooling2D(pool_size=(2, 2)))

modelCNN.add(Conv2D(32, (3, 3), activation="relu"))
modelCNN.add(MaxPooling2D(pool_size=(2, 2)))


modelCNN.add(Flatten())
modelCNN.add(Dense(128, activation = "relu"))
modelCNN.add(Dense(128, activation = "relu"))

modelCNN.add(Dense(10, activation="softmax"))

modelCNN.compile(loss="categorical_crossentropy",
              optimizer=Adam(learning_rate=0.0001),
              metrics=["accuracy"])

#history = modelCNN.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_data=(X_test, y_test))
history = modelCNN.fit(X_train_CNN, y_train_CNN, epochs=100, batch_size=32, verbose=0)


# Enregisrement du model
pickle.dump(history, open(str(Config.MODELS_PATH / "model.pk"), mode='wb'))
import pickle # Serialiser des objets (y comporis des modeles) joblib

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, datasets, Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import  Input
from tensorflow.keras.utils import to_categorical
from config import Config


input_shape= (64,64,1)

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

x_train = np.load(str(Config.FEATURES_PATH / "x_train_features_cnn.npy"), allow_pickle=False)
y_train = np.load(str(Config.FEATURES_PATH / "y_train_labels_cnn.npy"), allow_pickle=False)



model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(100, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(200, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(120, activation="relu"),
        layers.Dense(84, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)



batch_size = 256
epochs = 50
model_save_filename=Config.MODELS_PATH / "model_cnn_1.pk"

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
    model_save_filename, monitor="val_accuracy", save_best_only=True
)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[earlystopping_cb, mdlcheckpoint_cb],)



#Enregisrement du model
pickle.dump(model, open(str(Config.MODELS_PATH / "model_cnn.pk"), mode='wb'))

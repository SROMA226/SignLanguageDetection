# Considerer x_train.npy, y_train.npy,x_test.npy, y_test.npy
# Appliquer les pre-traitements néssaires à l'entrainement des modèles
# Enregistrer

import pandas as pd
import numpy as np

from config import Config

Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)

X_train = np.load(str(Config.DATASET_PATH / "x_train.npy"), allow_pickle=False)
X_test = np.load(str(Config.DATASET_PATH / "x_test.npy"), allow_pickle=False)
Y_train = np.load(str(Config.DATASET_PATH / "y_train.npy"), allow_pickle=False)
Y_test = np.load(str(Config.DATASET_PATH / "y_test.npy"), allow_pickle=False)

# Pré-traitement pour appliquer un modèle de DL
# reshape X
X_train_CNN = X_train.reshape(-1,64,64,1)
X_test_CNN = X_test.reshape(-1,64,64,1)
Y_train_CNN = Y_train
Y_test_CNN = Y_test

# print("X Shape:",X.shape)
# print("Y Shape:",y.shape)

# Pré-traitement pour appliquer les algorithmes de ML classiques
X_train_ML = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test_ML = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

X_train_ML = pd.DataFrame(X_train_ML)
X_test_ML = pd.DataFrame(X_test_ML)


Y_train_ML = np.array(pd.Series(Y_train.nonzero()[1]))
Y_test_ML = np.array(pd.Series(Y_test.nonzero()[1]))


# Enregistrement des features pour train et test
np.save(str(Config.FEATURES_PATH / "x_train_features_cnn.npy"), X_train_CNN)
np.save(str(Config.FEATURES_PATH / "x_test_features_cnn.npy"), X_test_CNN)

X_train_ML.to_csv(str(Config.FEATURES_PATH / "x_train_features_ml.csv"), index=None)
X_test_ML.to_csv(str(Config.FEATURES_PATH / "x_test_features_ml.csv"), index=None)

# Enregistrement des labels pour train et test
np.save(str(Config.FEATURES_PATH / "y_train_labels_cnn.npy"), Y_train_CNN)
np.save(str(Config.FEATURES_PATH / "y_test_labels_cnn.npy"), Y_test_CNN)

np.save(str(Config.FEATURES_PATH / "y_train_labels_ml.npy"), Y_train_ML)
np.save(str(Config.FEATURES_PATH / "y_test_labels_ml.npy"), Y_test_ML)

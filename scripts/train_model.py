import pickle # Serialiser des objets (y comporis des modeles)
import scipy

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 

from config import Config

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

X_train_ML = pd.read_csv(str(Config.FEATURES_PATH / "x_train_features_ml.csv"))
Y_train_ML = np.load(str(Config.FEATURES_PATH / "y_train_labels_ml.npy"), allow_pickle=False)

# Entrainement du model
modelRF = RandomForestClassifier(n_estimators=150, random_state=63).fit(X_train_ML, Y_train_ML)

# Enregisrement du model
pickle.dump(modelRF, open(str(Config.MODELS_PATH / "model.pk"), mode='wb'))

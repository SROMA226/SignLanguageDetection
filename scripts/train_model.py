import pickle # Serialiser des objets (y comporis des modeles)
import scipy

import pandas as pd
import numpy as np
#from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

from config import Config

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

X_train_ML = pd.read_csv(str(Config.FEATURES_PATH / "x_train_features_ml.csv"))
Y_train_ML = np.load(str(Config.FEATURES_PATH / "y_train_labels_ml.npy"), allow_pickle=False)

# Entrainement du model
base_model_svc = SVC(kernel="rbf")
legacy_random_state = np.random.RandomState()
modelSVC = HalvingRandomSearchCV(
    base_model_svc,
    param_distributions={
        "C": scipy.stats.expon(scale=100),
        "gamma": scipy.stats.expon(scale=0.1),
        "class_weight": ["balanced", None],
    },
    cv=3,
    n_jobs=3,
    random_state=legacy_random_state,
    verbose=0,
).fit(X_train_ML, Y_train_ML)

# Enregisrement du model
pickle.dump(modelSVC, open(str(Config.MODELS_PATH / "model.pk"), mode='wb'))
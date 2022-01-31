import pickle # Serialiser des objets (y comporis des modeles)
import json 

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, classification_report

from config import Config

X_test_ML = pd.read_csv(str(Config.FEATURES_PATH / "x_test_features_ml.csv"))
Y_test_ML = np.load(str(Config.FEATURES_PATH / "y_test_labels_ml.npy"), allow_pickle=False)

# Restaurer le modèle
modelSVC = pickle.load(open(str(Config.MODELS_PATH / "model.pk"), mode='rb'))

y_pred = modelSVC.predict(X_test_ML)
test_fscore = f1_score(y_true=Y_test_ML, y_pred=y_pred, average='weighted')
test_precision = precision_score(y_true=Y_test_ML, y_pred=y_pred, average='weighted')
test_accuracy = accuracy_score(y_true=Y_test_ML, y_pred=y_pred)
test_recall = recall_score(y_true=Y_test_ML, y_pred=y_pred, average='weighted')


with open(str(Config.METRICS_FILE_PATH), mode='w') as f:
    json.dump(dict(f1_score=test_fscore, precision=test_precision, recall_score=test_recall,accuracy_score=test_accuracy), f)

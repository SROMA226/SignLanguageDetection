import pickle # Serialiser des objets (y comporis des modeles)
import json 

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, classification_report


from config import Config

X_test_CNN = np.load(str(Config.FEATURES_PATH / "x_test_features_cnn.npy"))
Y_test_CNN = np.load(str(Config.FEATURES_PATH / "y_test_labels_cnn.npy"), allow_pickle=False)

# Restaurer le modèle
modelCNN = pickle.load(open(str(Config.MODELS_PATH / "model.pk"), mode='rb'))


# Predict the values from the validation dataset
Y_predictions = model.predict(X_test_CNN)
# Convert predictions classes to one hot vectors 
Y_pred_class = np.argmax(Y_predictions,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_test_CNN,axis = 1)

#print(classification_report(Y_true, Y_pred_class))

with open(str(Config.ASSETS_PATH)'/ metrique_cnn.txt', 'w') as f:
    f.write(classification_report(Y_true, Y_pred_class))

y_pred = modelCNN.predict(X_test_CNN)
test_fscore = f1_score(y_true=Y_test_CNN, y_pred=y_pred, average='weighted')
test_precision = precision_score(y_true=Y_test_CNN, y_pred=y_pred, average='weighted')
test_accuracy = accuracy_score(y_true=Y_test_CNN, y_pred=y_pred)
test_recall = recall_score(y_true=Y_test_CNN, y_pred=y_pred, average='weighted')

evaluation = modelCNN.evaluate(X_test_CNN, Y_test_CNN)


with open(str(Config.METRICS_FILE_PATH), mode='w') as f:
    json.dump(dict(f1_score=test_fscore, precision=test_precision, recall_score=test_recall,accuracy_score=test_accuracy), f)

with open(str(Config.METRICS_FILE_PATH), mode='w') as f:
    json.dump(dict(validation_accuracy=evaluation[1]), f)
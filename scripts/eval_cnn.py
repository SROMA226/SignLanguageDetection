
import pickle # Serialiser des objets (y comporis des modeles)
import json 
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from config import Config



Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

x_test = np.load(str(Config.FEATURES_PATH / "x_test_features_cnn.npy"), allow_pickle=False)
y_test = np.load(str(Config.FEATURES_PATH / "y_test_labels_cnn.npy"), allow_pickle=False)

model= pickle.load(open(str(Config.MODELS_PATH / "model_cnn.pk"), mode='rb'))


score = model.evaluate(x_test, y_test, verbose=0)

with open(str(Config.METRICS_FILE_PATH_CNN), mode='w') as f:
    json.dump(dict(accuracy=score[0], loss=score[1]), f)
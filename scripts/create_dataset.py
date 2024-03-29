# Telecharger le dataset depuis un GDrive
# Split en train et test
# Enregistrer dans "assets/data"

from numpy.core.defchararray import index
from scipy.sparse.construct import rand
import gdown
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import Config

# Set seed
np.random.seed(Config.RANDON_SEED)

# Creer les dossier dont on a besoin dans ce script
# ./assets/original_datasets & ./assets/data
Config.X_ORIGINAL_DATASET_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
Config.Y_ORIGINAL_DATASET_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)

# Telecharge notre fichier
gdown.download(
    "https://drive.google.com/uc?id=1JrAoKnNKBT-v_yawceP26w1mcolWaWYG",
    str(Config.X_ORIGINAL_DATASET_FILE_PATH)
)

gdown.download(
    "https://drive.google.com/uc?id=1kpTR7WgS3venCUUhYU3wHpec01P1INRn",
    str(Config.Y_ORIGINAL_DATASET_FILE_PATH)
)

# Chargement des données
X=np.load(str(Config.X_ORIGINAL_DATASET_FILE_PATH), allow_pickle=False)
Y=np.load(str(Config.Y_ORIGINAL_DATASET_FILE_PATH), allow_pickle=False)

X_train, X_test, Y_train, Y_test= train_test_split(
    X,Y, test_size=Config.TEST_SIZE, 
    random_state=Config.RANDON_SEED,
    stratify=Y
)

np.save(str(Config.DATASET_PATH / "x_train.npy"), X_train)
np.save(str(Config.DATASET_PATH / "x_test.npy"), X_test)
np.save(str(Config.DATASET_PATH / "y_train.npy"), Y_train)
np.save(str(Config.DATASET_PATH / "y_test.npy"), Y_test)
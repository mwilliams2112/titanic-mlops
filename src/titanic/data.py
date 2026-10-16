import os

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

from titanic.config import FEATURES


def download_data():
    path = kagglehub.dataset_download("yasserh/titanic-dataset")
    return pd.read_csv(os.path.join(path, "Titanic-Dataset.csv"))  

def split_data(data):
    X = data[FEATURES]
    y = data["Survived"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
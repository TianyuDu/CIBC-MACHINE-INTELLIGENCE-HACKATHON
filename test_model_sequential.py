import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime
import os

import data_proc
from data_proc import *
import model_methods
from model_methods import *
from training_proc import *

lr_start = float(
    input("Learning rate search start >>> ")
)

lr_end = float(
    input("Learning rate search end >>> ")
)

rates = np.linspace(lr_start, lr_end, 5)

for lr in rates:
    pass

if __name__ == "__main__":
    parameters = {
        "s3_url": "/Users/tianyudu/Downloads/cibc_data_cleaned_to_train.csv",
        "learning_rate": 0.005,
        "epochs": 15,
        "batch_size": 512,
        "shuffle": True,
        "validation_split": 0.1,
        "activation": "relu"
    }

    train_proc(para=parameters)

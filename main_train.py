import datetime
import os

import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import data_proc
import model_methods
from data_proc import *
from model_methods import *
from training_proc import *

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

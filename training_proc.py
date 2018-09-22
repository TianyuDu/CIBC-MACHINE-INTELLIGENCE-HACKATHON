import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime
import os
from pprint import pprint

import data_proc
from data_proc import *
import model_methods
from model_methods import *


parameters = {
    "s3_url": "https://s3.amazonaws.com/cibchack/data/cibc_data_cleaned_to_train.csv",
    "learning_rate": 0.005,
    "epochs": 10,
    "batch_size": 1024,
    "shuffle": True,
    "validation_split": 0.1
}


def train_proc(para: dict=parameters) -> None:
    df = read_cleaned_data(d=para["s3_url"])
    scaled_train, scaler = normalize_data(df)

    print(f"Scaled_train data shape = {scaled_train.shape}")

    model = construct_model(learning_rate=para["learning_rate"])

    start_time = datetime.datetime.now()
    hist = model.fit(
        x=scaled_train,
        y=scaled_train,
        epochs=para["epochs"],
        batch_size=para["batch_size"],
        shuffle=para["shuffle"],
        validation_split=para["validation_split"],
        verbose=1
    )
    end_time = datetime.datetime.now()
    print(f"Time taken {start_time - end_time} seconds.")

    # The model output corresponding to model (standardized)
    df = read_cleaned_data(d=para["s3_url"])
    X, scaler = normalize_data(df)
    model_output = model.predict(
        X, 
        verbose=1
    )
    scores = get_abnormal_score(
        pred=model_output, 
        actual=scaled_train
    )

    # Save result
    print("Saving basic informations...")
    model_methods.save_model(model, history, file_dir=train_name)

    print("Saving scores...")
    np.savetxt(f"./{train_name}/scores.csv", scores)

    print("Save parameters...")
    with open(f"./{train_name}/parameters.txt", "wr") as file:
        file.write(pprint.pformat(para))

    print("Done.")

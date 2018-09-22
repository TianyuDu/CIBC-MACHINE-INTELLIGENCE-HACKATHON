import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime
import os
import pprint

import data_proc
from data_proc import *
import model_methods
from model_methods import *


def train_proc(para: dict) -> None:
    df = read_cleaned_data(d=para["s3_url"])
    scaled_train, scaler = normalize_data(df)

    print(f"Scaled_train data shape = {scaled_train.shape}")

    model = construct_model(
        learning_rate=para["learning_rate"],
        activation=para["activation"])
    print("Model generated.")
    keras.utils.print_summary(model)

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
    print(f"Time taken {end_time - start_time} seconds.")

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
    time_stamp = datetime.datetime.now().timestamp()
    lr = para["learning_rate"]
    train_name = f"T_lr_{lr}_{time_stamp}"

    print("Saving basic informations...")
    model_methods.save_model(model, hist, file_dir=train_name)

    print("Saving scores...")
    # Add header to scores array
    df_score = pd.DataFrame(scores)
    df_score.columns = ["score"]

    df_score.to_csv(f"./saved_models/{train_name}/scores.csv")
    
    # np.savetxt(f"./saved_models/{train_name}/scores.csv", scores)

    print("Save parameters...")
    with open(f"./saved_models/{train_name}/parameters.txt", "w") as file:
        file.write(pprint.pformat(para))

    print("Done.")

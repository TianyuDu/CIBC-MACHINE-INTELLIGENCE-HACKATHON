import keras
import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import datetime
import os

import data_proc
from data_proc import *
import model_methods
from model_methods import *

lr_start = float(input("Learning rate search start >>> "))
lr_end = float(input("Learning rate search end >>> "))

rates = np.linspace(lr_start, lr_end, 5)

for lr in rates:

    df = read_cleaned_data(d="https://s3.amazonaws.com/cibchack/data/cibc_data_cleaned_to_train.csv")
    scaled_train, scaler = normalize_data(df)
    
    model = construct_model(learning_rate=lr)
    
    t_ini = datetime.datetime.now()
    
    history = model.fit(
        scaled_train,
        scaled_train,
        epochs=1,
        batch_size=1024,
        shuffle=True,
        validation_split=0.1,
        verbose=1
    )
    
    t_fin = datetime.datetime.now()
    print(
        f"Time to run the model: {(t_fin - t_ini).total_seconds()} Sec.")
    
    df_history = pd.DataFrame(history.history)
    
    pred = model.predict(scaled_train)
    
    time_stamp = datetime.datetime.now().timestamp()
    train_name = f"training_lr_{lr}_{time_stamp}"
    
    model_methods.save_model(model, file_dir=train_name)

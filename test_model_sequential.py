import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import datetime
import os

import model_methods
from model_methods import *

df = read_cleaned_data()
scaled_train, scaler = normalize_data(df)

model = construct_model()

t_ini = datetime.datetime.now()

history = model.fit(
    scaled_train,
    scaled_train,
    epochs=10,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

t_fin = datetime.datetime.now()
print(
    f"Time to run the model: {(t_fin - t_ini).total_seconds()} Sec.")

df_history = pd.DataFrame(history.history)

pred = autoencoder.predict(scaled_train)

time_stamp = datetime.datetime.now().timestamp()
train_name = f"training_{time_stamp}"

model_methods.save_model(model, file_dir=train_name)

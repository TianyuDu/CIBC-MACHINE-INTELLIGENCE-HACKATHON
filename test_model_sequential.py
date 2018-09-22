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


lr_start = float(
    input("Learning rate search start >>> ")
)

lr_end = float(
    input("Learning rate search end >>> ")
)

rates = np.linspace(lr_start, lr_end, 5)

for lr in rates:
    pass

lr = 0.005
epochs = 10 
batch_size = 1024

df = read_cleaned_data(d=s3_url)
scaled_train, scaler = normalize_data(df)

model = construct_model(learning_rate=lr)

t_ini = datetime.datetime.now()

history = model.fit(
    scaled_train,
    scaled_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

t_fin = datetime.datetime.now()
print(
    f"Time to run the model: {(t_fin - t_ini).total_seconds()} Sec.")


pred = model.predict(scaled_train, verbose=1)

scores = get_abnormal_score(pred=pred, actual=scaled_train)

# Save training result

time_stamp = datetime.datetime.now().timestamp()
train_name = f"T_lr_{lr}_{time_stamp}"

model_methods.save_model(model, history, file_dir=train_name)

print("Saving scores...")
np.savetxt(f"./{train_name}/scores.csv", scores)

print("Saving training parameters")
with open(f"./{train_name}/parameters.txt", "wr") as file:
    file.write(f"Learning rate: {lr}\n")
    file.write(f"Training epochs: {epochs}\n")
    file.write(f"Batch size: {batch_size}")

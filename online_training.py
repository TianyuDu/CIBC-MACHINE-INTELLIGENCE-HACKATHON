import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import os


def read_file(
        d: str="https://s3.amazonaws.com/cibchack/data/claims_final.csv") -> pd.DataFrame:
    df = pd.read_csv(d, header=None)
    col_names = ["fam_id", "fam_mem_id", "pro_id", "pro_type", "state", "date", "med_proc_code", "amt"]
    df.columns = col_names
    return df


def save_model(model, file_dir: str=None) -> None:
    # Try to create record folder.
    try:
        folder = f"./saved_models/{file_dir}/"
        os.system(f"mkdir {folder}")
        print(f"Experiment record directory created: {folder}")
    except:
        print("Current directory: ")
        _ = os.system("pwd")
        raise FileNotFoundError(
            "Failed to create directory, please create directory ./saved_models/")
    
    # Save model structure to JSON
    print("Saving model structure...")
    model_json = model.to_json()
    with open(f"{folder}model_structure.json", "w") as json_file:
        json_file.write(model_json)
    print("Done.")

    # Save model weight to h5
    print("Saving model weights...")
    model.save_weights(f"{folder}model_weights.h5")
    print("Done")

    # Save model illustration to png file.
    print("Saving model visualization...")
    keras.utils.plot_model(
        model,
        to_file=f"{folder}model.png",
        show_shapes=True, 
        show_layer_names=True)
    print("Done.")
    
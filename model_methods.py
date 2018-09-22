import keras
import os
import numpy as np
import pandas as pd

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
    # print("Saving model visualization...")
    # keras.utils.plot_model(
    #     model,
    #     to_file=f"{folder}model.png",
    #     show_shapes=True, 
    #     show_layer_names=True)
    # print("Done.")


def construct_model(input_dim: int=5,
                    nb_epoch: int=100,
                    encoding_dim: int=16,
                    batch_size: int=32,
                    learning_rate: float=0.05) -> keras.Sequential:
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(
            encoding_dim,
            input_dim=input_dim,
            activation="tanh",
            activity_regularizer=keras.regularizers.l1(10e-5),
            name="encoder0"
        )
    )
    model.add(
        keras.layers.Dense(
            int(encoding_dim / 2),
            activation="tanh",
            name="encoder1")
    )
    model.add(
        keras.layers.Dense(
            int(2),
            activation="tanh",
            name="encoder2")
    )
    model.add(
        keras.layers.Dense(
            int(encoding_dim / 2),
            activation="tanh",
            name="decoder0")
    )
    model.add(
        keras.layers.Dense(
            int(encoding_dim),
            activation="tanh",
            name="decoder1")
    )
    model.add(
        keras.layers.Dense(
            input_dim,
            activation="tanh",
            name="decoder2_output")
    )
    optimizer = keras.optimizers.adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    return model

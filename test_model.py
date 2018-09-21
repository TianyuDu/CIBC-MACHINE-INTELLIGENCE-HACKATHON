import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import datetime

toy_X = np.random.rand(3000,10)
norms = np.linalg.norm(toy_X, axis=1)
(min_norm, max_norm) = min(norms), max(norms)

scaler = StandardScaler().fit(toy_X)
X_train_scaled = scaler.transform(toy_X)

input_dim = 10
encoding_dim = 6

input_layer = keras.layers.Input(shape=(input_dim, ))
encoder = keras.layers.Dense(
                            encoding_dim, 
                            activation="tanh",
                            activity_regularizer=keras.regularizers.l1(10e-5)
                            )(input_layer)

encoder = keras.layers.Dense(int(encoding_dim / 2), activation="tanh")(encoder)

encoder = keras.layers.Dense(int(2), activation="tanh")(encoder)

decoder = keras.layers.Dense(int(encoding_dim / 2), activation='tanh')(encoder)

decoder = keras.layers.Dense(int(encoding_dim), activation='tanh')(decoder)

decoder = keras.layers.Dense(input_dim, activation='tanh')(decoder)

autoencoder = keras.Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()

keras.utils.plot_model(autoencoder)

nb_epoch = 100
batch_size = 32
autoencoder.compile(optimizer='adam', loss='mse')

t_ini = datetime.datetime.now()
history = autoencoder.fit(X_train_scaled, 
                            X_train_scaled,
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_split=0.1,
                            verbose=2
                            )
t_fin = datetime.datetime.now()
print(
    f"Time to run the model: {(t_fin - t_ini).total_seconds()} Sec.")

df_history = pd.DataFrame(history.history)

pred = autoencoder.predict(X_train_scaled)
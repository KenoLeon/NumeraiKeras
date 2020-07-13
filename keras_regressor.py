"""Uncle Keno's Keras starter/boilerplate for the Numerai tournament"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import pandas as pd
import random as python_random

print("USING TENSORFLOW:", tf.__version__)

# For mostly reproducible results,
# comment for slighlty better/worse/unique results:
# ----
np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(1234)
# ----

# CONSTANTS:
ROUND = '220'
TOURNAMENT_NAME = "kazutsugi"
TARGET_NAME = f"target_{TOURNAMENT_NAME}"
PREDICTION_NAME = f"prediction_{TOURNAMENT_NAME}"

# FUNCTION  DEFINITIONS:


def plot_the_loss_curve(epochs, mse):
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.plot(epochs, mse, label="Loss")
    plt.legend()
    plt.ylim([mse.min() * 0.95, mse.max() * 1.03])
    plt.draw()


def build_model(learning_rate, layer_size):
    """Build Keras model"""
    model = keras.Sequential([
      layers.Dense(layer_size, activation='relu',
                   input_shape=[len(feature_names)]),
      layers.Dense(layer_size, activation='relu', kernel_regularizer='l2'),
      layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


def train_model(model, feature, label, epochs, batch_size):
    """Train Keras Model"""
    history = model.fit(x=feature,
                        y=label,
                        batch_size=batch_size,
                        epochs=epochs)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    mse = hist["mse"]
    return epochs, mse


# LOAD DATASETS:

training_data = pd.read_csv("Numerai/Datasets/numerai_dataset_"+ROUND+"/numerai_training_data.csv").set_index("id")
tournament_data = pd.read_csv("Numerai/Datasets/numerai_dataset_"+ROUND+"/numerai_tournament_data.csv").set_index("id")


# FEATURES
feature_names = [
    f for f in training_data.columns if f.startswith("feature")
]
print(f"Loaded {len(feature_names)} features columns")

train_Features = training_data[feature_names]
train_Target = training_data[TARGET_NAME]

print(':::::: train_Features ::::::', '\n', train_Features)
print(':::::: train_Target ::::::', '\n', train_Target)

# Parameters (not optimized):
lr = 0.001
epochs = 100
batch_size = 1000
layer_Size = 32

# Run everything...
regressor_model = build_model(lr, layer_Size)
# regressor_model.summary()
epochs, rmse = train_model(regressor_model, train_Features, train_Target, epochs, batch_size)
plot_the_loss_curve(epochs, rmse)
#


# --------------------------- Predictions ---------------------------

tournament_data[PREDICTION_NAME] = regressor_model.predict(tournament_data[feature_names])
df = tournament_data[PREDICTION_NAME]
df.columns = ["id", "prediction_kazutsugi"]
df.to_csv("Numerai/" + TOURNAMENT_NAME + "_submission_YourSubmissionName.csv", header=True)
print(df)

# --------------------------- Predictions ---------------------------

print("Finished")
plt.show()


# REFERENCE:
# https://www.tensorflow.org/tutorials/keras/regression
# https://keras.io/getting_started/intro_to_keras_for_engineers/


# NOTES:
# Tune your parameters, tweak the model,  add your prefered validation and profit ?
# provided as is, there are porbably better NN architectures and a ton of upgrades
# Submit your improvements/results, see the readme ?

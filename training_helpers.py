"""Helper functions for the training set."""

import os

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D, Activation
from tensorflow.keras import Model
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam
from tensorflow import keras


def create_autoencoder_model(img_shape, encoding_dim=6):
    # architecture code taken from arxiv:1808.08992
    input_img = Input(shape=(img_shape[0], img_shape[1], 1))

    layer = input_img
    layer = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(layer)
    layer = MaxPooling2D(pool_size=(2, 2),padding="same")(layer)
    layer = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(layer)
    layer = MaxPooling2D(pool_size=(2, 2),padding="same")(layer)
    layer = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(layer)
    layer = Flatten()(layer)
    layer = Dense(32, activation="relu")(layer)
    layer = Dense(encoding_dim)(layer)
    encoded = layer
    layer = Dense(32, activation="relu")(encoded)
    layer = Dense(12800, activation="relu")(layer)
    layer = Reshape((10,10,128))(layer)
    layer = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(layer)
    layer = UpSampling2D((2,2))(layer)
    layer = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(layer)
    layer = UpSampling2D((2,2))(layer)
    layer = Conv2D(1, kernel_size=(3, 3), padding="same")(layer)
    layer = Reshape((1, img_shape[0] * img_shape[1]))(layer)
    layer = Activation("softmax")(layer)
    decoded = Reshape((img_shape[0], img_shape[1], 1))(layer)
    autoencoder = Model(input_img ,decoded)

    autoencoder.compile(loss=mean_squared_error , optimizer=Adam())
    
    return autoencoder

def train_encoding_dim_search(train_data, nPix, encoding_dim, nEpochs, batch_size, trial, logDir, patience=3):
    print(f"\nRunning training for encoding dim {encoding_dim} -- iteration {trial + 1}")
    # create the autoencoder model
    autoencoder_model = create_autoencoder_model((nPix, nPix), encoding_dim=encoding_dim)

    # callbacks
    early_stopping_callback = EarlyStopping(monitor="loss", patience=patience, verbose=1)
    chkPath = os.path.join(logDir, f"checkpoint_file_iter{trial + 1}.h5")
    model_checkpt_callback = ModelCheckpoint(chkPath, monitor="loss", mode="min",
            save_best_only=True)
    logPath = os.path.join(logDir, f"training_iter{trial + 1}.log")
    csv_logger = CSVLogger(logPath)

    # train the autoencoder
    history = autoencoder_model.fit(x=train_data, y=train_data, epochs=nEpochs, batch_size=batch_size,
            callbacks=[early_stopping_callback, model_checkpt_callback, csv_logger])

    return history

def train_autoencoder_model(train_data, valid_data, nPix, encoding_dim, nEpochs, batch_size, logDir, trial=0, patience=3):
    print(f"\nRunning training for encoding dim {encoding_dim} -- iteration {trial + 1}")
    # create the autoencoder model
    autoencoder_model = create_autoencoder_model((nPix, nPix), encoding_dim=encoding_dim)

    # callbacks
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=patience, verbose=1)
    chkPath = os.path.join(logDir, f"checkpoint_file_iter{trial + 1}.h5")
    model_checkpt_callback = ModelCheckpoint(chkPath, monitor="val_loss", mode="min",
            save_best_only=True)
    logPath = os.path.join(logDir, f"training_iter{trial + 1}.log")
    csv_logger = CSVLogger(logPath)

    # train the autoencoder
    history = autoencoder_model.fit(x=train_data, y=train_data, validation_data=(valid_data, valid_data),
            epochs=nEpochs, batch_size=batch_size,
            callbacks=[early_stopping_callback, model_checkpt_callback, csv_logger])

    return history

def load_autoencoder_model(modelPath):
    autoencoder = keras.models.load_model(modelPath)

    return autoencoder

def evaluate_model_on_set(autoencoder, input_imgs, batch_size=1024):
    output_imgs = autoencoder.predict(input_imgs, batch_size=batch_size, verbose=1)
    # reshape output from model to have the same shape as the input
    output_imgs = output_imgs[:, :, :, 0]
     
    return output_imgs
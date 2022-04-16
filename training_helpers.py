"""Helper functions for the training set."""

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D, Activation
from tensorflow.keras import Model
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam


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

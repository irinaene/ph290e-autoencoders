import h5py
import os
import tensorflow as tf
import training_helpers as trh
import plotting_helpers as ploth

nPix = 40
sampleDir = "training_samples"
batch_size = 1024
validation_split = 0.2
nEpochs = 100
patience = 3
encoding_dim = 6
outDir = "trained_models"

# load the training samples
f_train = h5py.File(os.path.join(sampleDir, "background_train.hdf5"), "r")
bkg_train = f_train["jet_images_proc_normalized"][:]
assert bkg_train.shape[1] == nPix
assert bkg_train.shape[2] == nPix
f_train.close()

# create the autoencoder model
autoencoder_model = trh.create_autoencoder_model((nPix, nPix), encoding_dim=encoding_dim)
autoencoder_model.summary()

# train the autoencoder
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)
history = autoencoder_model.fit(x=bkg_train, y=bkg_train, validation_split=validation_split,
        epochs=nEpochs, batch_size=batch_size, callbacks=[callback])

# save model to hdf5 file
if not os.path.exists(outDir):
    os.mkdir(outDir)
autoencoder_model.save(os.path.join(outDir, "trained_autoencoder.h5"))

# training validation plots
ploth.plot_training_loss(history)

import os

import h5py
import numpy as np
import tensorflow as tf

import helpers.plotting_helpers as ploth
import helpers.training_helpers as trh

nPix = 40
sampleDir = "training_samples"
batch_size = 1024
nEpochs = 2000
patience = 3
encoding_dim = 12
outDir = f"trained_models_dim{encoding_dim}"
rng_seed = 12345

# create output folders
if not os.path.exists(outDir):
    os.mkdir(outDir)

# load the training samples
f_train = h5py.File(os.path.join(sampleDir, "background_train.hdf5"), "r")
bkg_sample = f_train["jet_images_proc_normalized"][:]
assert bkg_sample.shape[1] == nPix
assert bkg_sample.shape[2] == nPix
f_train.close()

# shuffle the jets (again)
rng = np.random.default_rng(rng_seed)
inds = rng.choice(np.arange(bkg_sample.shape[0]), size=125000, replace=False)

# 100k jets for training, 25k jets for validation
bkg_train = bkg_sample[inds[:100000]]
bkg_valid = bkg_sample[inds[100000:]]

# create and train autoencoder model
tf.keras.backend.clear_session()
history = trh.train_autoencoder_model(bkg_train, bkg_valid, nPix, encoding_dim, nEpochs, batch_size, outDir, patience=patience)

# training validation plots
ploth.plot_training_loss(history, outDir=outDir)

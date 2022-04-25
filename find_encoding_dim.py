import h5py
import os
import tensorflow as tf
import time
import numpy as np
import sys
import training_helpers as trh
import plotting_helpers as ploth

nPix = 40
sampleDir = "training_samples"
batch_size = 1024
nEpochs = 2000
patience = 3
encoding_dims = [2, 4, 6, 8, 12, 16, 24, 32]
outDir = "encoding_dim_search"
overWrite = False
doTraining = False
doPlotting = True

# create output folders
if not os.path.exists(outDir):
    os.mkdir(outDir)

if doTraining:
    # load the training samples
    f_train = h5py.File(os.path.join(sampleDir, "background_train.hdf5"), "r")
    bkg_train = f_train["jet_images_proc_normalized"][:]
    assert bkg_train.shape[1] == nPix
    assert bkg_train.shape[2] == nPix
    f_train.close()

    bkg_train = bkg_train[:100000]

    for encoding_dim in encoding_dims:
        subDir = os.path.join(outDir, f"encoding_dim_{encoding_dim}")
        if not os.path.exists(subDir):
            os.mkdir(subDir)

        for trial in range(5):
            tf.keras.backend.clear_session()

            # don't accidentally overwrite things
            if (not overWrite) and (os.path.exists(os.path.join(subDir, f"checkpoint_file_iter{trial + 1}.h5"))):
                print("Previous results already exist and you have chosen not to overwrite. Exiting now!")
                sys.exit(1)

            history = trh.train_encoding_dim_search(bkg_train, nPix, encoding_dim, nEpochs, batch_size, trial,
                    subDir, patience=patience)

            # training validation plots
            ploth.plot_training_loss(history, valid=False, plotName=f"training_loss_iter{trial + 1}.pdf", outDir=subDir)

            # cooldown time
            print("Time for a cooldown!")
            time.sleep(5 * 60)

if doPlotting:
    # assumes the above training loops have been already been done
    loss_vals = np.zeros((5, len(encoding_dims)))
    
    for i, encoding_dim in enumerate(encoding_dims):
        subDir = os.path.join(outDir, f"encoding_dim_{encoding_dim}")
        for trial in range(5):
            logPath = os.path.join(subDir, f"training_iter{trial + 1}.log")
            log_data = np.genfromtxt(logPath, delimiter=",", names=True)
            min_loss = np.min(log_data["loss"])
            loss_vals[trial, i] = min_loss
    avg_loss = np.average(loss_vals, axis=0) * 1e6
    
    ploth.plot_loss_vs_encoding_dim(encoding_dims, avg_loss, outDir=outDir)

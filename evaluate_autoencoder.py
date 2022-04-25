import h5py
import numpy as np
import os
import training_helpers as trh
import evaluation_helpers as evalh
import plotting_helpers as ploth

nPix = 40
sampleDir = "training_samples"
batch_size = 1024
encoding_dim = 12
modelDir = f"trained_models_dim{encoding_dim}"
plotDir = f"evaluation_plots_dim{encoding_dim}"

# load the testing samples
f_test = h5py.File(os.path.join(sampleDir, "background_test.hdf5"), "r")
bkg_test = f_test["jet_images_proc_normalized"][:]
assert bkg_test.shape[1] == nPix
assert bkg_test.shape[2] == nPix
bkg_test_mass = f_test["jet_mass"][:]
f_test.close()

f_test = h5py.File(os.path.join("preprocessed_samples", "signal.hdf5"), "r")
sig_test = f_test["jet_images_proc_normalized"][:]
assert sig_test.shape[1] == nPix
assert sig_test.shape[2] == nPix
f_test.close()

# evaluate the trained autoencoder model on the test set
autoencoder = trh.load_autoencoder_model(os.path.join(modelDir, f"checkpoint_file_iter1.h5"))
print(f"Evaluating model on background test sample")
output_bkg = trh.evaluate_model_on_set(autoencoder, bkg_test)
print(f"Evaluating model on signal test sample")
output_sig = trh.evaluate_model_on_set(autoencoder, sig_test)

# evaluate reconstruction error
reco_err_bkg = evalh.compute_reconstruction_error(bkg_test, output_bkg)
reco_err_sig = evalh.compute_reconstruction_error(sig_test, output_sig)

## plot reconstruction error
print(f"Generation plot of mean reconstruction error")
ploth.plot_reconstruction_error([reco_err_bkg, reco_err_sig], ["QCD jets", "Top jets"], outDir=plotDir)

## plot of average jet mass versus reconstruction error
print(f"Generating plot of jet mass versus reconstruction error")
bins = np.logspace(np.log10(np.min(reco_err_bkg)), np.log10(np.max(reco_err_bkg)), 51)
avg_jet_mass, reco_err = evalh.jet_mass_vs_reco_error(reco_err_bkg, bkg_test_mass, bins=bins)
ploth.plot_mass_vs_reco_err(reco_err, avg_jet_mass, outDir=plotDir)

## ROC curve for CNN
print(f"Generating plot of ROC curve for CNN")
err_cuts = np.logspace(np.log10(np.min(reco_err_bkg)), np.log10(np.max(reco_err_sig)), 100)
sig_eff, bkg_eff = evalh.roc_curve(reco_err_sig, reco_err_bkg, err_cuts, cut_type="greater")
ploth.plot_roc_curve([(sig_eff, bkg_eff)], ["CNN"], "roc_cnn_dim12.pdf", outDir=plotDir, grid=True)
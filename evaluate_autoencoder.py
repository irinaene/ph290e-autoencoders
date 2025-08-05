import os

import h5py
import numpy as np

import helpers.evaluation_helpers as evalh
import helpers.plotting_helpers as ploth
import helpers.training_helpers as trh

nPix = 40
sampleDir = "training_samples"
batch_size = 1024
encoding_dim = 12
modelDir = f"trained_models_dim{encoding_dim}"
plotDir = f"evaluation_plots_dim{encoding_dim}"

# load the testing samples
f_test = h5py.File(os.path.join(sampleDir, "background_test.hdf5"), "r")
bkg_norm = f_test["jet_images_preprocessed"][:]
bkg_test = f_test["jet_images_proc_normalized"][:]
assert bkg_test.shape[1] == nPix
assert bkg_test.shape[2] == nPix
bkg_test_mass = f_test["jet_mass"][:]
is_good = np.where(f_test["tau_2"][:] != 0)
bkg_tau_32 = f_test["tau_3"][:][is_good] / f_test["tau_2"][is_good]
f_test.close()

# compute total pT to "renormalize" output jet images
total_pt_bkg = np.empty(len(bkg_norm))
for i in range(len(bkg_norm)):
    total_pt_bkg[i] = np.sum(bkg_norm[i])

f_test = h5py.File(os.path.join("preprocessed_samples", "signal.hdf5"), "r")
sig_norm = f_test["jet_images_preprocessed"][:]
sig_test = f_test["jet_images_proc_normalized"][:]
assert sig_test.shape[1] == nPix
assert sig_test.shape[2] == nPix
is_good = np.where(f_test["tau_2"][:] != 0)
sig_tau_32 = f_test["tau_3"][is_good] / f_test["tau_2"][:][is_good]
f_test.close()

# compute total pT to "renormalize" output jet images
total_pt_sig = np.empty(len(sig_norm))
for i in range(len(sig_norm)):
    total_pt_sig[i] = np.sum(sig_norm[i])

# evaluate the trained autoencoder model on the test set
autoencoder = trh.load_autoencoder_model(os.path.join(modelDir, f"checkpoint_file_iter1.h5"))
print(f"Evaluating model on background test sample")
output_bkg = trh.evaluate_model_on_set(autoencoder, bkg_test)
bkg_new = np.swapaxes(np.swapaxes(output_bkg, 0, -1) * total_pt_bkg, -1, 0)
bkg_error = np.square(bkg_new - bkg_norm)

print(f"Evaluating model on signal test sample")
output_sig = trh.evaluate_model_on_set(autoencoder, sig_test)
sig_new = np.swapaxes(np.swapaxes(output_sig, 0, -1) * total_pt_sig, -1, 0)
sig_error = np.square(sig_new - sig_norm)

# output jet images
ploth.plot_avg_jet_image(bkg_new, "background", nJets=2000, suffix="output", outDir=plotDir, vmin=1e-1, cmap="Greens")
ploth.plot_avg_jet_image(sig_new, "signal", nJets=2000, suffix="output", outDir=plotDir, vmin=1e-1, cmap="Greens")

ploth.plot_avg_jet_image(bkg_error, "background", nJets=2000, suffix="error", outDir=plotDir, vmin=1e0, vmax=1e2, cmap="Reds")
ploth.plot_avg_jet_image(sig_error, "signal", nJets=2000, suffix="error", outDir=plotDir, vmin=1e0, vmax=1e2, cmap="Reds")

# evaluate reconstruction error
reco_err_bkg = evalh.compute_reconstruction_error(bkg_test, output_bkg)
reco_err_sig = evalh.compute_reconstruction_error(sig_test, output_sig)

## plot reconstruction error
print(f"Generating plot of mean reconstruction error")
ploth.plot_reconstruction_error([reco_err_bkg, reco_err_sig], ["QCD jets", "Top jets"], outDir=plotDir)

## plot of average jet mass versus reconstruction error
print(f"Generating plot of jet mass versus reconstruction error")
bins = np.array([1.0000e-07, 2.0980e-06, 4.0960e-06, 6.0940e-06, 8.0920e-06, 1.0090e-05, 1.2088e-05, 1.4086e-05, 1.6084e-05, 1.8082e-05, 2.0080e-05, 3.060e-05, 5.040e-05, 7.5e-5, 1.0000e-04])
avg_jet_mass, std_jet_mass, reco_err = evalh.jet_mass_vs_reco_error(reco_err_bkg, bkg_test_mass, bins=bins)
ploth.plot_mass_vs_reco_err(reco_err, avg_jet_mass, std_jet_mass, outDir=plotDir)

## plot jet mass distribution for different WPs
print(f"Generating plot of jet mass distribution for different background rejection WPs")
wp_list = [10, 50, 100] # bkg rejection values to consider
cut_list = []
for wp in wp_list:
    cut_list.append(evalh.reco_err_for_wp(reco_err_bkg, 1. / wp))
jet_mass_list = evalh.jet_mass_at_wp(reco_err_bkg, bkg_test_mass, cut_list)
ploth.plot_mass_at_wp(jet_mass_list, wp_list, outDir=plotDir)

## ROC curve for CNN
print(f"Generating plot of ROC curve(s) for trained autoencoder")
err_cuts = np.logspace(np.log10(np.min(reco_err_bkg)), np.log10(np.max(reco_err_sig)), 100)
sig_eff, bkg_eff = evalh.roc_curve(reco_err_sig, reco_err_bkg, err_cuts, cut_type="greater")
ploth.plot_roc_curve([(sig_eff, bkg_eff)], ["CNN"], "roc_cnn_dim12.pdf", outDir=plotDir, grid=True)

tau_cuts = np.linspace(0, 1, 101)
sig_eff_tau, bkg_eff_tau = evalh.roc_curve(sig_tau_32, bkg_tau_32, thresholds=tau_cuts, cut_type="lower")
ploth.plot_roc_curve([(sig_eff, bkg_eff), (sig_eff_tau, bkg_eff_tau)], ["CNN", r"$\tau_{32}$"], "roc_comparison_dim12.pdf", outDir=plotDir, grid=True)

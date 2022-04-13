import h5py
import plotting_helpers as ploth
import evaluation_helpers as evalh
import numpy as np

# where to save the input plots
plotDir = "input_plots"

## jet images
for sample in ["background", "signal"]:
    f = h5py.File(f"preprocessed_samples/{sample}.hdf5", "r")
    
    print(f"Generating jet image plots for sample: {sample}")
    
    # original jet images -- only center + pixelize
    jet_imgs = f["jet_images_original"][:]
    ploth.plot_avg_jet_image(jet_imgs, sample, nJets=2000, suffix="orig", outDir=plotDir)
    
    # fully preprocessed jet images
    jet_imgs_proc = f["jet_images_preprocessed"][:]
    ploth.plot_avg_jet_image(jet_imgs_proc, sample, nJets=2000, suffix="proc", outDir=plotDir)
    
    f.close()

## tau_32 variable
print("Generating plot of tau_32 variables")
f_sig = h5py.File(f"preprocessed_samples/signal.hdf5", "r")
f_bkg = h5py.File(f"preprocessed_samples/background.hdf5", "r")

is_good = np.where(f_sig["tau_2"][:] != 0)
sig_tau_32 = f_sig["tau_3"][is_good] / f_sig["tau_2"][:][is_good]
is_good = np.where(f_bkg["tau_2"][:] != 0)
bkg_tau_32 = f_bkg["tau_3"][:][is_good] / f_bkg["tau_2"][is_good]

ploth.plot_tau_32(sig_tau_32, bkg_tau_32, sig_label="Top jets", bkg_label="QCD jets", outDir=plotDir)

f_sig.close()
f_bkg.close()

## ROC curve for cut on tau_32 variables
print("Generation plot of ROC curve")
tau_cuts = np.linspace(0, 1, 101)
sig_eff, bkg_eff = evalh.roc_curve(sig_tau_32, bkg_tau_32, thresholds=tau_cuts, cut_type="lower")
ploth.plot_roc_curve([(sig_eff, bkg_eff)], [r"$\tau_{32}$"], "roc_tau_32.pdf", outDir=plotDir)

"""Helper functions for evaluating the model performance."""

import numpy as np

def roc_curve(sig_arr, bkg_arr, thresholds, cut_type="greater"):
    n_sig = len(sig_arr)
    n_bkg = len(bkg_arr)
    
    sig_hist, _ = np.histogram(sig_arr, bins=thresholds)
    bkg_hist, _ = np.histogram(bkg_arr, bins=thresholds)
    
    if cut_type == "greater":
        sig_eff = np.cumsum(sig_hist[::-1]) / n_sig
        bkg_eff = np.cumsum(bkg_hist[::-1]) / n_bkg
    elif cut_type == "lower":
        sig_eff = np.cumsum(sig_hist) / n_sig
        bkg_eff = np.cumsum(bkg_hist) / n_bkg
    
    return sig_eff, bkg_eff

def compute_reconstruction_error(input_img, output_img):
    diff_img = input_img - output_img
    sqr_diff = np.square(diff_img)
    mean_reco_err = np.average(sqr_diff, axis=(1, 2))
    return mean_reco_err

def equalObs(x, nbin):
    # define function to calculate equal-frequency bins
    # from https://www.statology.org/equal-frequency-binning-python/
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))

def jet_mass_vs_reco_error(reco_err, jet_mass, bins=25):
    counts, bin_edges = np.histogram(reco_err, bins=bins)
    mass, bin_edges = np.histogram(reco_err, weights=jet_mass, bins=bin_edges)

    print(counts)

    avg_jet_mass = mass / counts
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    
    return avg_jet_mass, bin_centers
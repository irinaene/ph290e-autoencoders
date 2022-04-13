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
        bkg_eff = np.cumsum(bkg_hist) / n_sig
    
    return sig_eff, bkg_eff

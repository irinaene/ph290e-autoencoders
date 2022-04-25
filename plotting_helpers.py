"""Plotting helper functions."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

mpl.rcParams["xtick.top"] = True
mpl.rcParams["ytick.right"] = True
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["xaxis.labellocation"] = "right"
mpl.rcParams["yaxis.labellocation"] = "top"
mpl.rcParams["font.size"] = 12
mpl.rcParams["legend.fontsize"] = 15
mpl.rcParams["savefig.dpi"] = 300

def plot_avg_jet_image(jet_imgs_array, sampleName, nJets=1, suffix=None, outDir=None):
    jet_img = np.average(jet_imgs_array[:nJets, :, :], axis=0)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    pc = ax.imshow(jet_img.T, extent=(-1, 1, -1, 1), origin="lower", norm=mpl.colors.LogNorm())
    ax.set_ylabel("$\phi$'")
    ax.set_xlabel("$\eta$'")
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(pc, label="Calorimeter $E_{T}$ [GeV]", cax=cax)
    plt.tight_layout()
    
    plotName = f"jet_img_{sampleName}"
    if suffix:
        plotName = f"{plotName}_{suffix}"
    if outDir:
        if not os.path.isdir(outDir):
            os.mkdir(outDir)
    outPath = os.path.join(outDir, f"{plotName}.pdf")
    fig.savefig(outPath)

def plot_tau_32(sig_tau, bkg_tau, sig_label=None, bkg_label=None, outDir=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    w_bkg = np.ones(len(bkg_tau)) * 1. / len(bkg_tau)
    w_sig = np.ones(len(sig_tau)) * 1. / len(sig_tau)
    ax.hist(bkg_tau, bins=20, range=[0, 1], weights=w_bkg, histtype="step", color="C0", label=bkg_label)
    ax.hist(sig_tau, bins=20, range=[0, 1], weights=w_sig, histtype="step", color="C1", label=sig_label)
    
    ax.legend(loc="upper left")
    ax.set_xlabel(r"$\tau_3/\tau_2$ of jet", fontsize=15)
    ax.set_xlim([0, 1])
    ax.set_ylabel("Fraction of jets", fontsize=15)
    
    plt.tight_layout()
    if outDir:
        if not os.path.isdir(outDir):
            os.mkdir(outDir)
    outPath = os.path.join(outDir, "tau_32_comparison.pdf")
    fig.savefig(outPath)

def plot_roc_curve(eff_pairs, labels, plotName, outDir=None, grid=False):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    for i, eff_pair in enumerate(eff_pairs):
        sig_eff, bkg_eff = eff_pair
        ax.plot(sig_eff, 1. / bkg_eff, label=labels[i])
    
    ax.legend(loc="upper right")
    ax.set_xlabel(r"Signal efficiency ($\epsilon_s$)", fontsize=15)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylabel(r"Background rejection ($1/\epsilon_b$)", fontsize=15)
    ax.set_yscale("log")
    ax.grid(grid, which="both")
    
    plt.tight_layout()
    if outDir:
        if not os.path.isdir(outDir):
            os.mkdir(outDir)
    outPath = os.path.join(outDir, plotName)
    fig.savefig(outPath)

def plot_training_loss(history, plotName="training_loss_plot.pdf", outDir=None, valid=True):
    fig = plt.figure(figsize=(8, 6))
    epochs = np.arange(1, len(history.history["loss"]) + 1)
    plt.plot(history.history['loss'], label='loss')
    if valid:
        plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    
    plt.tight_layout()
    if outDir:
        if not os.path.isdir(outDir):
            os.mkdir(outDir)
    outPath = os.path.join(outDir, plotName)
    fig.savefig(outPath)

def plot_loss_vs_encoding_dim(encoding_dims, avg_loss, plotName="avg_training_loss_vs_encoding_dim.pdf", outDir=None):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(encoding_dims, avg_loss, label="CNN")
    plt.scatter(encoding_dims, avg_loss)

    plt.xlim([0, 33])
    plt.xlabel('Encoding Dimensions')
    plt.ylabel(r'Loss x $10^6$')
    plt.legend(loc='best')
    
    plt.tight_layout()
    if outDir:
        if not os.path.isdir(outDir):
            os.mkdir(outDir)
    outPath = os.path.join(outDir, plotName)
    fig.savefig(outPath)

def plot_reconstruction_error(error_list, label_list, plotName="mean_reconstruction_error.pdf", outDir=None):
    assert len(error_list) == len(label_list)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    bin_min = 1e-7
    bin_max = 1e-3
    for arr in error_list:
        if np.min(arr) < bin_min: bin_min = np.min(arr)
        if np.max(arr) > bin_max: bin_max = np.max(arr)

    logbins = np.logspace(np.log10(bin_min), np.log10(bin_max), 25)
    for i in range(len(error_list)):
        weights = np.ones(len(error_list[i])) * 1. / len(error_list[i])
        ax.hist(error_list[i], bins=logbins, weights=weights, label=label_list[i], histtype="stepfilled", alpha=0.35)

    ax.set_xlabel("Reconstruction error")
    ax.set_xscale("log")
    ax.set_ylabel("A.U.")
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    if outDir:
        if not os.path.isdir(outDir):
            os.mkdir(outDir)
    outPath = os.path.join(outDir, plotName)
    fig.savefig(outPath)

def plot_mass_vs_reco_err(reco_err, avg_jet_mass, plotName="jet_mass_vs_reco_err.pdf", outDir=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(reco_err * 1e6, avg_jet_mass)
    # plt.hist(avg_jet_mass, bins="auto")

    ax.set_xlabel(r"Reconstruction error x $10^6$")
    ax.set_ylabel("Average jet mass [GeV]")
    
    plt.tight_layout()
    if outDir:
        if not os.path.isdir(outDir):
            os.mkdir(outDir)
    outPath = os.path.join(outDir, plotName)
    fig.savefig(outPath)

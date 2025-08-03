import os
import sys

import h5py
import numpy as np
import ROOT

import helpers.preprocessing_helpers as pph

ROOT.gROOT.SetBatch(True)
# Load Delphes
ROOT.gSystem.Load('libDelphes.so')

# Check for input arguments
if len(sys.argv)==1:
    print('usage: {} input0.root [input1.root ...]')
    sys.exit(1)
inputs=sys.argv[1:]

select_ttbar = False
if "ttbar" in inputs[0]:
    select_ttbar = True

# Load input files
t=ROOT.TChain('Delphes')
for inpath in inputs:
    t.AddFile(inpath)

if select_ttbar:
    nJets = 2104
else:
    nJets = 233178

tau_2_arr = np.empty(nJets)
tau_3_arr = np.empty(nJets)
jet_mass = np.empty(nJets)
nPix = 40
jet_images_orig = np.empty((nJets, nPix, nPix))
jet_images_preproc = np.empty((nJets, nPix, nPix))
jet_images_proc_norm = np.empty((nJets, nPix, nPix))

# Loop the loop
nJets = 0
nEvents = 0
for e in t:
    nEvents += 1
    if nEvents % 1000 == 0:
        print(f"Processing events: {nEvents}")
    
    for fj in e.FatJet:
        # Simple selection
        if fj.PT < 500: continue
        if select_ttbar:
            if abs(fj.Mass - 173) > 20: continue
        
        # tau_2, tau_3 variables for baseline performance
        tau_2_arr[nJets] = fj.Tau[1]
        tau_3_arr[nJets] = fj.Tau[2]
        
        # jet mass
        jet_mass[nJets] = fj.Mass
        
        # for storing the jet images
        eta_arr = []
        phi_arr = []
        et_arr = []

        # Add constituents to the image
        nConstituents = 0
        for c in fj.Constituents:
            if type(c) is not ROOT.Tower:
                continue # not a calorimeter thing
            nConstituents += 1
            eta_arr.append(c.Eta)
            phi_arr.append(c.Phi)
            et_arr.append(c.ET)
        
        if nConstituents == 0: break
        
        eta_arr = np.array(eta_arr)
        phi_arr = np.array(phi_arr)
        et_arr = np.array(et_arr)
        
        # original jet images -- only centered and pixelated
        eta_cent, phi_cent = pph.center(eta_arr, phi_arr, et_arr)
        jet_img = pph.pixelize(eta_cent, phi_cent, et_arr, nPix=nPix)
        jet_images_orig[nJets] = jet_img
        
        # fully pre-processed jet images
        eta_rot, phi_rot = pph.rotate_princ_vertical(eta_cent, phi_cent, et_arr)
        flip = pph.determine_flip(eta_rot, phi_rot, et_arr)
        eta_flip, phi_flip = pph.flip(eta_rot, phi_rot, flip)
        jet_img_proc = pph.pixelize(eta_flip, phi_flip, et_arr, nPix=nPix)
        jet_images_preproc[nJets] = jet_img_proc
        jet_images_proc_norm[nJets] = pph.normalize(jet_img_proc)
        
        nJets += 1
        break # Leading jet only

# now save the data to hdf5 format
outDir = "preprocessed_samples"
if not os.path.isdir(outDir):
    os.mkdir(outDir)
outFile = "signal.hdf5" if select_ttbar else "background.hdf5"
outPath = os.path.join(outDir, outFile)

f1 = h5py.File(outPath, "w")
f1.create_dataset("tau_2", tau_2_arr.shape, dtype="f", data=tau_2_arr)
f1.create_dataset("tau_3", tau_3_arr.shape, dtype="f", data=tau_3_arr)
f1.create_dataset("jet_images_original", jet_images_orig.shape, dtype="f", data=jet_images_orig)
f1.create_dataset("jet_images_preprocessed", jet_images_preproc.shape, dtype="f", data=jet_images_preproc)
f1.create_dataset("jet_images_proc_normalized", jet_images_proc_norm.shape, dtype="f", data=jet_images_proc_norm)
f1.create_dataset("jet_mass", jet_mass.shape, dtype="f", data=jet_mass)
f1.close()

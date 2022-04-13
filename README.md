# Setup

## Preprocessing setup

To setup a conda environment with the packages needed to open the sample files and generate the jet images inputs for training:
```bash
conda create -n ph290e_preproc -c conda-forge root=6.26.0 python=3.10.4 scipy=1.8.0 h5py=3.6.0 matplotlib=3.5.1
conda activate ph290e_preproc
conda install ${CFS}/atlas/kkrizka/E290/delphes-3.5.0-0.tar.bz2
```

## Training setup

# Running the code

Get a copy of the code:
```bash
mkdir myDir && cd myDir
git clone https://github.com/irinaene/ph290e-autoencoders.git
```

## Preprocessing step
First you need to activate the preprocessing environment:
```bash
conda activate ph290e_preproc
```

Then run the preprocessing script to generate the jet images and save them in `hdf5` format for training:
```bash
# for background sample
python ph290e-autoencoders/preprocessing.py ${CFS}/atlas/kkrizka/E290/samples/v0.0.1/dijet/run_01_*/tag_1_delphes_events.root
# for signal sample
python ph290e-autoencoders/preprocessing.py ${CFS}/atlas/kkrizka/E290/samples/v0.0.1/ttbar/run_02_*/tag_1_delphes_events.root
```

This will generate the files `background.hdf5` (containing info about the background jets) and `signal.hdf5` (containing info about the signal jets) inside the folder `preprocessed_samples`.

To generate plots of the input features for background and signal (e.g. jet images, $\tau_32$ variable), run the plotting script:
```bash
python ph290e-autoencoders/plot_inputs.py
```
This will generate and store the plots in the folder `input_plots`.

## Training step

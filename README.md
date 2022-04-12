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

## Training step

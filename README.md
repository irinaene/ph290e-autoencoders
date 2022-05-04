# Setup

## Preprocessing setup

To setup a conda environment with the packages needed to open the sample files and generate the jet images inputs for training:
```bash
conda create -n ph290e_preproc -c conda-forge root=6.26.0 python=3.10.4 scipy=1.8.0 h5py=3.6.0 matplotlib=3.5.1 scikit-learn=1.0.2
conda activate ph290e_preproc
conda install ${CFS}/atlas/kkrizka/E290/delphes-3.5.0-0.tar.bz2
```

## Training setup

To setup a conda environment for training the autoencoder model:
```bash
conda create -n ph290e_train -c conda-forge tensorflow=2.7.0 python=3.10.4 h5py=3.6.0 matplotlib=3.5.1
conda activate ph290e_train
```

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
First you need to activate the training environment:
```bash
conda activate ph290e_train
```

To split the samples into training and testing sets, run the following script:
```bash
python ph290e-autoencoders/prepare_training.py
```
This will split the background sample into a training set and a validation set and save the files in the folder `training_samples`. The default is to have 100k jets in the test set -- if you want to change this, modify the variable `nTest` inside `prepare_training.py`.

### Optional:
If you want to run a search for the optimal encoding dimension of the latent representation, take a look at the script `ph290e-autoencoders/find_encoding_dim.py`. This will loop over various options for the encoding dimension and run 5 training runs over a training set of 100k background jets. In the end, it will generate a plot of the loss (averaged over the 5 training runs) versus encoding dimension, from which one can choose the optimal dimension.
Be warned that this step is very time consuming, so it would ideally be run using a GPU (this means that in the conda enviroment for training one should install the `tensorflow-gpu` package).

To train the autoencoder on the background sample, run the script:
```bash
python ph290e-autoencoders/train_autoencoder.py
```
Note that by default the script uses `encoding_dim=12` and 100k jets for training + 25k jets for validation. If you want to change this, take a look at the variables defined at the top of the script.
The trained model and some diagnostic plots will be saved in the folder `trained_models_dim12`.

Finally, to generate some plots that evaluate the performance of the autoencoder model, run:
```bash
python ph290e-autoencoders/evaluate_autoencoder.py
```
The plots will be stored in the folder `evaluation_plots_dim12`.

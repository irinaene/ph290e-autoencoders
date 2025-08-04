# Detecting anomalies in hadronic resonances using autoencoders

# Running the code

Get a copy of the code:
```bash
mkdir anomaly_detection && cd anomaly_detection
git clone https://github.com/irinaene/ph290e-autoencoders.git
```

The workflow consists of a preprocessing step, a training step, and an evaluation step.

## Preprocessing step

First, create a `conda` environment with the packages needed to open the sample files and generate the jet images inputs for training:
```bash
conda env create -f ph290e-autoencoders/envs/ph290e_preproc.yml
conda activate ph290e_preproc
conda install ${CFS}/atlas/kkrizka/E290/delphes-3.5.0-0.tar.bz2
```

Then run the preprocessing script to generate the jet images and save them in `hdf5` format for training:
```bash
SAMPLE_DIR=${CFS}/atlas/kkrizka/E290/samples/v0.0.1
# for background sample
python ph290e-autoencoders/preprocessing.py ${SAMPLE_DIR}/dijet/run_01_*/tag_1_delphes_events.root
# for signal sample
python ph290e-autoencoders/preprocessing.py ${SAMPLE_DIR}/ttbar/run_02_*/tag_1_delphes_events.root
```

This will generate the files `background.hdf5` (containing info about the background jets) and `signal.hdf5` (containing info about the signal jets) inside the folder `preprocessed_samples`.

To generate plots of the input features for background and signal (e.g. jet images, $\tau_{32}$ variable), run the plotting script:
```bash
python ph290e-autoencoders/plot_inputs.py
```
This will generate and store the plots in the folder `input_plots`.

## Training step

Create the `conda` environment needed for training the autoencoder model:
```bash
conda env create -f ph290e-autoencoders/envs/ph290e_train.yml
conda activate ph290e_train
```

To split the samples into training and testing sets, run the following script:
```bash
python ph290e-autoencoders/prepare_training.py
```
This will split the background sample into a training set and a validation set and save the files in the folder `training_samples`.
The default is to have `100k` jets in the test set (leaving `125k` jets for training) -- if you want to change this, modify the variable `nTest` inside `prepare_training.py`.

**Optional**: If you want to run a search for the optimal encoding dimension of the latent representation, take a look at the script `ph290e-autoencoders/find_encoding_dim.py`.
This will loop over various options for the encoding dimension and run 5 training runs over a training set of `100k` background jets. In the end, it will generate a plot of the loss (averaged over the 5 training runs) versus encoding dimension, from which one can choose the optimal dimension (e.g. at the elbow in the loss curve).
Be aware that this step is very time-consuming, even when running on GPU.
<!-- Be warned that this step is very time consuming, so it would ideally be run using a GPU (this means that in the conda enviroment for training one should install the `tensorflow-gpu` package). -->

To train the autoencoder on the background sample, run the script:
```bash
python ph290e-autoencoders/train_autoencoder.py
```
This step uses `100k` jets for training + `25k` jets for validation.
Note that by default the script uses `encoding_dim=12`. If you want to change this or other training settings, such as the batch size or number of training epochs, take a look at the variables defined at the top of the script.
The trained model and some diagnostic plots will be saved in the folder `trained_models_dim12`.

## Evaluation step

Finally, to generate some plots that evaluate the performance of the autoencoder model, run:
```bash
python ph290e-autoencoders/evaluate_autoencoder.py
```
The plots will be stored in the folder `evaluation_plots_dim12`.
Some of the plots that are created in this step include:
- autoencoder reconstructed jet images
- histograms of the autoencoder mean reconstruction error
- ROC curves (signal efficiency versus background rejection)
- average jet mass versus reconstruction error
- jet mass distribution at different autoencoder working points.

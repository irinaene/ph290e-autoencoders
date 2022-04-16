import h5py
import os
from sklearn.model_selection import train_test_split

# set random state to get same shuffling for all variables
randomState = 42
nTest = int(1e5)

# where to save data
outDir = "training_samples"
if not os.path.isdir(outDir):
    os.mkdir(outDir)
f_train = h5py.File(os.path.join(outDir, "background_train.hdf5"), "w")
f_test = h5py.File(os.path.join(outDir, "background_test.hdf5"), "w")

# split background into training and testing sets
f_bkg = h5py.File(f"preprocessed_samples/background.hdf5", "r")
for key in f_bkg.keys():
    data = f_bkg[key][:]
    print(f"Generating train - test sets for {key}")
    X_train, X_test = train_test_split(data, test_size=nTest, random_state=randomState)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    f_train.create_dataset(key, X_train.shape, dtype="f", data=X_train)
    f_test.create_dataset(key, X_test.shape, dtype="f", data=X_test)

f_bkg.close()
f_train.close()
f_test.close()

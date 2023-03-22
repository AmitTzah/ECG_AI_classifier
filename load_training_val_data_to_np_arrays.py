# loads training and test data to numpy arrays and saves them to disk
import numpy as np
import scipy.io as sio
import os
from imblearn.under_sampling import RandomUnderSampler

# Load the training data
X_train = []
train_diagnoses = []
training_folder = "training_folder"
validation_folder = "validation_folder"

# if it doesn't exist, create a folder to save the training and test numpy arrays
if not os.path.exists("train_val_numpy_arrays"):
    os.mkdir("train_val_numpy_arrays")


for file in os.listdir(f"{training_folder}"):
    if file.endswith(".mat"):
        mat_file = os.path.join(f"{training_folder}", file)
        header_file = os.path.join(f"{training_folder}", file[:-4] + ".hea")

        # Load the .mat file as a numpy array
        mat_data = sio.loadmat(mat_file)['val']

        with open(header_file, 'r') as f:
            header = f.read()
        X_train.append(mat_data)
        lines = header.split('\n')
        for line in lines:
            if line.startswith('#Dx:'):
                train_diagnoses.append(line[5:].strip())

# Load the valiation data
X_val = []
test_diagnoses = []
for file in os.listdir(f"{validation_folder}"):
    if file.endswith(".mat"):
        mat_file = os.path.join(f"{validation_folder}", file)
        header_file = os.path.join(f"{validation_folder}", file[:-4] + ".hea")
        mat_data = sio.loadmat(mat_file)['val']
        with open(header_file, 'r') as f:
            header = f.read()
        X_val.append(mat_data)
        lines = header.split('\n')
        for line in lines:
            if line.startswith('#Dx:'):
                test_diagnoses.append(line[5:].strip())


# define the most common diagnoses codes according to the distribution of the data:
most_common_diagnoses_array = [426783006, 164865005,
                               39732003, 164951009, 164873001, 164934002, 164861001]

classes_names = ["sinus rhythm", "myocardial infarction", "left axis deviation",
                 "abnormal QRS", "left ventricular hypertrophy", "t wave abnormal", "myocardial ischemia", "other"]


# write the Y_train and Y_val
Y_train = []
for diagnosis in train_diagnoses:
    diagnosis_array = np.zeros(8)
    diagnosis = diagnosis.split(',')
    for d in diagnosis:
        if int(d) in most_common_diagnoses_array:
            diagnosis_array[most_common_diagnoses_array.index(int(d))] = 1
        else:
            diagnosis_array[7] = 1

    Y_train.append(diagnosis_array)

# save the unbalanced data per class
for i in range(7):

    y_balanced = [row[i] for row in Y_train]
    # copy X_train to X_train_biased
    X_train_balanced = X_train.copy()

    X_train_balanced = np.array(X_train_balanced).reshape(
        len(X_train_balanced), -1)

    # use RandomUnderSampler to balance the data by downsampling the positive examples
    rus = RandomUnderSampler(random_state=0)
    X_train_balanced, y_balanced = rus.fit_resample(
        X_train_balanced, y_balanced)

    np.save(os.path.join("train_val_numpy_arrays",
                         f'X_train_balanced_{classes_names[i]}.npy'), X_train_balanced)

    np.save(os.path.join("train_val_numpy_arrays",
                         f'y_train_balanced_{classes_names[i]}.npy'), np.array(y_balanced))


# Convert the data to numpy arrays and reshape them
X_train = np.array(X_train).reshape(len(X_train), -1)
Y_train = np.array(Y_train)

Y_val = []
for diagnosis in test_diagnoses:
    diagnosis_array = np.zeros(8)
    diagnosis = diagnosis.split(',')
    for d in diagnosis:
        if int(d) in most_common_diagnoses_array:
            diagnosis_array[most_common_diagnoses_array.index(int(d))] = 1
        else:
            diagnosis_array[7] = 1
    Y_val.append(diagnosis_array)


X_val = np.array(X_val).reshape(len(X_val), -1)
Y_val = np.array(Y_val)


np.save(os.path.join("train_val_numpy_arrays", 'Y_val.npy'), Y_val)
np.save(os.path.join("train_val_numpy_arrays", 'X_val.npy'), X_val)

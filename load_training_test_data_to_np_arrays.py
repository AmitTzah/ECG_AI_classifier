# loads training and test data to numpy arrays and saves them to disk
import numpy as np
import scipy.io as sio
import os

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
        # the [‘val’] is used to access the value of the key ‘val’ in the dictionary returned by loadmat() function.
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
# from most common to least common
# these are the classes we are going to predict, the rest of the classes will be grouped into the 'other' class
most_common_diagnoses_array = [426783006, 164865005,
                               39732003, 164951009, 164873001, 164934002, 164861001]

classes_names = ["sinus rhythm", "myocardial infarction", "left axis deviation",
                 "abnormal QRS", "left ventricular hypertrophy", "t wave abnormal", "myocardial ischemia", "other"]


# write the Y_train and Y_val
# for each elemnt in X_train, the corresponding diagnosis is an array of 1 and 0
# the array is of size 8, a 1 in the i-th position means that the diagnosis is the i-th most common diagnosis
# if the diagnosis is not in the most common diagnoses, then the diagnosis is 'other', which is the 8-th diagnosis
Y_train = []
for diagnosis in train_diagnoses:
    # diagnosis can also be of the format '426783006,164865005'
    # in this case, we put a ones in the corresponding positions according to the most common diagnoses
    diagnosis_array = np.zeros(8)
    diagnosis = diagnosis.split(',')
    for d in diagnosis:
        if int(d) in most_common_diagnoses_array:
            diagnosis_array[most_common_diagnoses_array.index(int(d))] = 1
        else:
            diagnosis_array[7] = 1

    Y_train.append(diagnosis_array)

# augment the data so it is biased towards positive examples, augment by 3 times
for i in range(7):

    y_biased = [row[i] for row in Y_train]
    # copy X_train to X_train_biased
    X_train_biased = X_train.copy()

    # augment the data so it is biased towards positive examples, augment by 4 times
    if not i == 0:
        for j in range(len(y_biased)):
            print("current J is: ", j)
            if y_biased[j] == 1:
                for k in range(4):
                    X_train_biased.append(X_train[j])
                    y_biased.append(1)

        np.save(os.path.join("train_val_numpy_arrays",
                f'X_train_biased_{classes_names[i]}.npy'), np.array(X_train_biased).reshape(len(X_train_biased), -1))

        np.save(os.path.join("train_val_numpy_arrays",
                f'y_train_biased_{classes_names[i]}.npy'), np.array(y_biased))

    else:
        # don't augment the data for the first class

        np.save(os.path.join("train_val_numpy_arrays",
                f'X_train_{classes_names[i]}.npy'), np.array(X_train).reshape(len(X_train), -1))

        np.save(os.path.join("train_val_numpy_arrays",
                f'y_train_{classes_names[i]}.npy'), np.array(y_biased))

        # Convert the data to numpy arrays and reshape them
X_train = np.array(X_train).reshape(len(X_train), -1)
# we convert the list to a numpy array because it is easier to work with
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
np.save(os.path.join("train_val_numpy_arrays", 'Y_train.npy'), Y_train)
np.save(os.path.join("train_val_numpy_arrays", 'X_val.npy'), X_val)
np.save(os.path.join("train_val_numpy_arrays", 'X_train.npy'), X_train)

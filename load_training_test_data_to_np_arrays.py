# loads training and test data to numpy arrays and saves them to disk
import numpy as np
import scipy.io as sio
import os

# Load the training data
X_train = []
train_diagnoses = []
training_folder = "training_folder"
validation_folder = "validation_folder"


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

# Load the testing data
X_test = []
test_diagnoses = []
for file in os.listdir(f"{validation_folder}"):
    if file.endswith(".mat"):
        mat_file = os.path.join(f"{validation_folder}", file)
        header_file = os.path.join(f"{validation_folder}", file[:-4] + ".hea")
        mat_data = sio.loadmat(mat_file)['val']
        with open(header_file, 'r') as f:
            header = f.read()
        X_test.append(mat_data)
        lines = header.split('\n')
        for line in lines:
            if line.startswith('#Dx:'):
                test_diagnoses.append(line[5:].strip())


# define the most common diagnoses codes according to the distribution of the data:
# from most common to least common
# these are the classes we are going to predict, the rest of the classes will be grouped into the 'other' class
most_common_diagnoses_array = [426783006, 164865005,
                               39732003, 164951009, 164873001, 164934002, 164861001]


# Convert the data to numpy arrays and reshape them
X_train = np.array(X_train).reshape(len(X_train), -1)
X_test = np.array(X_test).reshape(len(X_test), -1)

# write the Y_train and Y_test
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

# we convert the list to a numpy array because it is easier to work with
Y_train = np.array(Y_train)

Y_test = []
for diagnosis in test_diagnoses:
    diagnosis_array = np.zeros(8)
    diagnosis = diagnosis.split(',')
    for d in diagnosis:
        if int(d) in most_common_diagnoses_array:
            diagnosis_array[most_common_diagnoses_array.index(int(d))] = 1
        else:
            diagnosis_array[7] = 1
    Y_test.append(diagnosis_array)
Y_test = np.array(Y_test)

# save the Y_test, Y_train, X_test, X_train to directory so that we can use them in the other models
# check if those files already exist, if they do, don't save them again
if not os.path.exists('Y_test.npy') and not os.path.exists('Y_train.npy') and not os.path.exists('X_test.npy') and not os.path.exists('X_train.npy'):

    np.save('Y_test.npy', Y_test)
    np.save('Y_train.npy', Y_train)
    np.save('X_test.npy', X_test)
    np.save('X_train.npy', X_train)

# It looks like, since the data is biased towards not having a heart condition (except sinus rhythm which is the standard), the model chose to simply label everything as "false", take the  small False Negatives  loss (since there aren't many negatives in the data anyway) and thus minimize the loss, at the cost of being completely useless at actually predicting a diagnoses.
# The solution is to artificially increase the number of positive examples in the training data, by duplicating the positive examples in the training data.
# This is called "data augmentation" and is a common technique to deal with unbalanced data.
# as such we will duplicate all the patients that have more than three diagnoses, and we will add them to the training data
# We also remove the patients that have only one diagnosis in the "other" class, since they are not useful for training

X_train_biased = []
Y_train_biased = []

for i in range(len(Y_train)):
    if np.sum(Y_train[i]) > 3:
        X_train_biased.append(X_train[i])
        Y_train_biased.append(Y_train[i]) 
        X_train_biased.append(X_train[i])
        Y_train_biased.append(Y_train[i])
        X_train_biased.append(X_train[i])
        Y_train_biased.append(Y_train[i])
        X_train_biased.append(X_train[i])
        Y_train_biased.append(Y_train[i])
    elif np.sum(Y_train[i]) == 1 and Y_train[i][7] == 1:
        continue
    else:
        X_train_biased.append(X_train[i])
        Y_train_biased.append(Y_train[i])

X_train_biased = np.array(X_train_biased)
Y_train_biased = np.array(Y_train_biased)

# save the biased data
if not os.path.exists('Y_train_biased.npy') and not os.path.exists('X_train_biased.npy'):
    np.save('Y_train_biased.npy', Y_train_biased)
    np.save('X_train_biased.npy', X_train_biased)


# using KNN to classify the ECG data

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
from joblib import dump, load
from tqdm import tqdm

# Load the training data
train_data = []
train_headers = []
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
        train_data.append(mat_data)
        train_headers.append(header)
        lines = header.split('\n')
        for line in lines:
            if line.startswith('#Dx:'):
                train_diagnoses.append(line[5:].strip())

# Load the testing data
test_data = []
test_headers = []
test_diagnoses = []
for file in os.listdir(f"{validation_folder}"):
    if file.endswith(".mat"):
        mat_file = os.path.join(f"{validation_folder}", file)
        header_file = os.path.join(f"{validation_folder}", file[:-4] + ".hea")
        mat_data = sio.loadmat(mat_file)['val']
        with open(header_file, 'r') as f:
            header = f.read()
        test_data.append(mat_data)
        test_headers.append(header)
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
train_data = np.array(train_data).reshape(len(train_data), -1)
test_data = np.array(test_data).reshape(len(test_data), -1)

# write the Y_train and Y_test
# for each elemnt in train_data, the corresponding diagnosis is an array of 1 and 0
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


# find the best K and grpah the K vs accuracy
# keep all the fitted models in a list
Ks = range(1, 20)
accuracies = []
fitted_models = []

# add a progress bar for the training
for K in tqdm(Ks):
    classifier = KNeighborsClassifier(n_neighbors=K)

    # for K in Ks:
    classifier.n_neighbors = K
    classifier.fit(train_data, Y_train)
    Y_pred = classifier.predict(test_data)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"K: {K}, Accuracy: {accuracy}")
    accuracies.append(accuracy)
    fitted_models.append(classifier)

# plot the K vs accuracy
plt.plot(Ks, accuracies)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()

# save the model of the best K
best_K = Ks[np.argmax(accuracies)]
classifier = fitted_models[np.argmax(accuracies)]
dump(classifier, 'knn_best_classifier.joblib')

# print the accuracy of the best K
print(f"Best K: {best_K}")
print(f"Accuracy: {accuracies[np.argmax(accuracies)]}")

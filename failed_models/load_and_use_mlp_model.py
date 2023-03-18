from joblib import dump, load
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss
import numpy as np

import os
import matplotlib.pyplot as plt
import itertools

# load the mlp_biased.joblib model
mlp = load('mlp_biased.joblib')

# Verify the type of model

print("Type of model: ", type(mlp))

# load X_test.npy and Y_test.npy
X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')


if 'Y_pred_mlp_biased.npy' in os.listdir():
    Y_pred = np.load('Y_pred_mlp_biased.npy')

else:
    # predict the labels of the test data
    Y_pred = mlp.predict(X_test)
    # save the predicted labels to a .npy file
    np.save('Y_pred_mlp_biased.npy', Y_pred)

# calculate the accuracy of the model on the test data
accuracy = accuracy_score(Y_test, Y_pred)

# calculate the hammig loss of the model on the test data
hamming_loss = hamming_loss(Y_test, Y_pred)

confusion_matrix = multilabel_confusion_matrix(Y_test, Y_pred)


# print the accuracy and hamming loss
print("Accuracy: ", accuracy)
print("Hamming Loss: ", hamming_loss)

# print the confusion matrix
print("Confusion Matrix: ", confusion_matrix)

# visualize the confusion matrix
# Note that the confusion matrix is a 3D array, multilabel of 8x2x2

classes_names = ["sinus rhythm", "myocardial infarction", "left axis deviation",
                 "abnormal QRS", "left ventricular hypertrophy", "t wave abnormal", "myocardial ischemia", "other"]


# plot confusion matrix per class all the subplots in one figure

fig, axes = plt.subplots(4, 2, figsize=(10, 10))
axes = axes.ravel()
for i, ax in enumerate(axes):
    cm = confusion_matrix[i]
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(classes_names[i])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.xaxis.set_ticklabels(['', '0', '1'])
    ax.yaxis.set_ticklabels(['', '0', '', '1'])

    # set the color of the text in the confusion matrix
    cthresh = cm.max() / 1.1
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > cthresh else "black")

plt.tight_layout()
plt.show()

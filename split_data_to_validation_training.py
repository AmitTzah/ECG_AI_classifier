#split input data folder into test and training data and generate folders for each.

import os
from shutil import copyfile
from sklearn.model_selection import train_test_split

data_folder = "data"
train_folder = "training_folder"
val_folder = "validation_folder"

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

#list of all .mat file names in the data folder
mat_files = [f for f in os.listdir(data_folder) if f.endswith('.mat')]

# split the list of .mat files into training and validation sets with a 80:20 ratio
#Setting random_state a fixed value will guarantee that same sequence of random numbers are generated each time the code is run.
# And unless there is some other randomness present in the process, the results produced will be same as always. This helps in verifying the output.
train_files, val_files = train_test_split(mat_files, test_size=0.2, random_state=42)

# iterate through the training files and copy them to the training folder
for file_name in train_files:
    # copy the .mat file
    mat_file = os.path.join(data_folder, file_name)
    new_mat_file = os.path.join(train_folder, file_name)
    copyfile(mat_file, new_mat_file)
    
    # copy the corresponding .hea file
    header_file = os.path.join(data_folder, file_name[:-4] + ".hea")
    new_header_file = os.path.join(train_folder, file_name[:-4] + ".hea")
    copyfile(header_file, new_header_file)

# iterate through the validation files and copy them to the validation folder
for file_name in val_files:
    # copy the .mat file
    mat_file = os.path.join(data_folder, file_name)
    new_mat_file = os.path.join(val_folder, file_name)
    copyfile(mat_file, new_mat_file)
    
    # copy the corresponding .hea file
    header_file = os.path.join(data_folder, file_name[:-4] + ".hea")
    new_header_file = os.path.join(val_folder, file_name[:-4] + ".hea")
    copyfile(header_file, new_header_file)

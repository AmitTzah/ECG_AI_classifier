import os
import matplotlib.pyplot as plt


def get_diagnoses(folder_path):
    diagnoses = {}
    for header_file in os.listdir(folder_path):
        if header_file.endswith(".hea"):
            with open(os.path.join(folder_path, header_file), 'r') as f:
                for line in f.readlines():
                    if line.startswith("#Dx:"):
                        dxs = line.strip().split(":")[1].strip().split(",")
                        for dx in dxs:
                            if dx in diagnoses:
                                diagnoses[dx] += 1
                            else:
                                diagnoses[dx] = 1
    return diagnoses


folder1 = "./test_code/data"
folder2 = "./test_code/training_folder"
folder3 = "./test_code/validation_folder"

diagnoses1 = get_diagnoses(folder1)
diagnoses2 = get_diagnoses(folder2)
diagnoses3 = get_diagnoses(folder3)

# Get the top 7 diagnoses for each folder
top_diagnoses1 = sorted(
    diagnoses1.items(), key=lambda x: x[1], reverse=True)[:7]
top_diagnoses2 = sorted(
    diagnoses2.items(), key=lambda x: x[1], reverse=True)[:7]
top_diagnoses3 = sorted(
    diagnoses3.items(), key=lambda x: x[1], reverse=True)[:7]

# Extract the diagnosis codes and counts for each folder
dx_codes1 = list(diagnoses1.keys())
dx_counts1 = list(diagnoses1.values())

dx_codes2 = list(diagnoses2.keys())
dx_counts2 = list(diagnoses2.values())

dx_codes3 = list(diagnoses3.keys())
dx_counts3 = list(diagnoses3.values())

# Set the order of the diagnosis codes based on the first folder
dx_order = dx_codes1

# Make sure that all three sets of codes are in the same order
for dx in dx_codes2 + dx_codes3:
    if dx not in dx_order:
        dx_order.append(dx)

# Sort the diagnosis codes based on the first folder
dx_order.sort()

# Create a bar chart for each folder
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
axs[0].bar(dx_order, [diagnoses1.get(dx, 0) for dx in dx_order])
axs[1].bar(dx_order, [diagnoses2.get(dx, 0) for dx in dx_order])
axs[2].bar(dx_order, [diagnoses3.get(dx, 0) for dx in dx_order])

# Set the titles and axis labels
axs[0].set_title(f"{folder1}")
axs[1].set_title(f"{folder2}")
axs[2].set_title(f"{folder3}")

fig.suptitle("Diagnostic Code Histograms")
plt.xlabel("Diagnostic Code")
plt.ylabel("Number of Patients")


# Add the top 7 diagnoses as text annotations to each chart
for i, ax in enumerate(axs):
    text = "\n".join([f"{dx}: {count}" for dx, count in [
                     top_diagnoses1, top_diagnoses2, top_diagnoses3][i]])
    ax.text(0.5, 0.95, text, transform=ax.transAxes, va='top', ha='center')


# use the Dx_map.csv file to get the full name of the diagnosis
# and add it to the chart
with open("Dx_map.csv", 'r') as f:
    dx_map = {line.split(",")[1].strip(): line.split(",")[0]
              for line in f.readlines()}


for i, ax in enumerate(axs):
    for dx, count in [top_diagnoses1, top_diagnoses2, top_diagnoses3][i]:
        ax.text(dx, count, dx_map[dx], ha='center', va='bottom')


plt.show()

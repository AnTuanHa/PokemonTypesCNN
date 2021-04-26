import shutil

from sklearn.datasets import load_files
from sklearn.model_selection import ShuffleSplit
from pathlib import Path

DATASET_PATH = "dataset"
TRAIN_DATASET_PATH = "train_dataset"
TEST_DATASET_PATH = "test_dataset"

TRAIN_SIZE = 0.85  # Percentage out of 100
TEST_SIZE = 0.15  # Percentage out of 100

data = load_files(DATASET_PATH, shuffle=False)
images = data.data
labels = data.target
class_names = data.target_names

# Controls the randomness of the training and testing indices produced.
# We fix an integer here instead of using a randomized integer so that we get reproducible training
# and testing datasets
RANDOM_STATE = 42

# Creates a Train-Test split where Train is TRAIN_SIZE and Test is TEST_SIZE
train_test_split = ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)

train_dataset_folder = Path(TRAIN_DATASET_PATH)
train_dataset_folder.mkdir(parents=True, exist_ok=True)
test_dataset_folder = Path(TEST_DATASET_PATH)
test_dataset_folder.mkdir(parents=True, exist_ok=True)

for train_indices, test_indices in train_test_split.split(images, labels):
    for index in train_indices:
        class_name = class_names[labels[index]]
        type_folder = Path(train_dataset_folder / class_name)
        type_folder.mkdir(parents=True, exist_ok=True)

        src_file = Path(data.filenames[index])
        dst_file = Path(train_dataset_folder / class_name / src_file.name)
        shutil.copy(src_file, dst_file)

        print(f"Copying {src_file.resolve()} to {dst_file.resolve()}")

    for index in test_indices:
        class_name = class_names[labels[index]]
        type_folder = Path(test_dataset_folder / class_name)
        type_folder.mkdir(parents=True, exist_ok=True)

        src_file = Path(data.filenames[index])
        dst_file = Path(test_dataset_folder / class_name / src_file.name)
        shutil.copy(src_file, dst_file)

        print(f"Copying {src_file.resolve()} to {dst_file.resolve()}")

print(f"{train_dataset_folder.name} folder:")
train_total_images_count = 0
for type_folder in train_dataset_folder.iterdir():
    count = len([entry for entry in type_folder.iterdir() if entry.is_file()])
    train_total_images_count += count
    print(f"{type_folder.name} has {count} images")
print(f"{train_dataset_folder.name} has a total of {train_total_images_count} images\n")

print(f"{test_dataset_folder.name} folder:")
test_total_images_count = 0
for type_folder in test_dataset_folder.iterdir():
    count = len([entry for entry in type_folder.iterdir() if entry.is_file()])
    test_total_images_count += count
    print(f"{type_folder.name} has {count} images")
print(f"{test_dataset_folder.name} has a total of {test_total_images_count} images\n")

print(f"# train images + # test images = {train_total_images_count + test_total_images_count}\n")
print(f"Full dataset has a total of {len(images)} images")

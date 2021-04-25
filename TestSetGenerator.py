import shutil

from sklearn.datasets import load_files
from sklearn.model_selection import ShuffleSplit
from pathlib import Path

DATASET_PATH = "dataset"
TEST_DATASET_PATH = "test_dataset"
TEST_SIZE = 0.15  # Percentage out of 100

data = load_files(DATASET_PATH, shuffle=False)
images = data.data
labels = data.target
class_names = data.target_names
rs = ShuffleSplit(n_splits=1, test_size=TEST_SIZE)

test_dataset_folder = Path(TEST_DATASET_PATH)
test_dataset_folder.mkdir(parents=True, exist_ok=True)
for train_indices, test_indices in rs.split(images, labels):
    for index in test_indices:
        class_name = class_names[labels[index]]
        type_folder = Path(test_dataset_folder / class_name)
        type_folder.mkdir(parents=True, exist_ok=True)

        src_file = Path(data.filenames[index])
        dst_file = Path(test_dataset_folder / class_name / src_file.name)
        shutil.copy(src_file, dst_file)

        print(f"Copying {src_file.resolve()} to {dst_file.resolve()}")

total_images_count = 0
for type_folder in test_dataset_folder.iterdir():
    count = len([entry for entry in type_folder.iterdir() if entry.is_file()])
    total_images_count += count
    print(f"{type_folder.name} has {count} images")
print(f"{test_dataset_folder.name} has a total of {total_images_count} images")
print(f"Full dataset has a total of {len(images)} images")

import numpy as np
import tensorflow as tf
import os

from tensorflow.keras import layers, models
from pathlib import Path

BATCH_SIZE = 256
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3

DATASET_PATH = "test_dataset"

data_dir = Path(DATASET_PATH)

test_ds_paths = tf.data.Dataset.list_files(str(data_dir / "*/*"), shuffle=False)
DATASET_SIZE = test_ds_paths.cardinality().numpy() # Returns the total number of images found in the dataset folder

class_names = np.array(
    sorted([item.name for item in data_dir.glob("*") if item.name != "LICENSE.txt"])
)

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

test_ds = test_ds_paths.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_ds = test_ds.batch(BATCH_SIZE)

model = models.load_model("TrainModel_trained")

test_loss, test_acc = model.evaluate(test_ds)
test_acc_loss = f"Accuracy: {test_acc}\tLoss: {test_loss}\n"
print(test_acc_loss)

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sn

from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models

BATCH_SIZE = 256
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3

DATASET_PATH = "dataset"

data_dir = Path(DATASET_PATH)

test_ds_paths = tf.data.Dataset.list_files(str(data_dir / "*/*"), shuffle=False)
DATASET_SIZE = (test_ds_paths.cardinality().numpy())  # Returns the total number of images found in the dataset folder

class_names = np.array(
    sorted([item.name for item in data_dir.glob("*") if item.name != "LICENSE.txt"])
)
NUM_CLASSES = len(class_names)


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


def process_path_img(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


test_ds = test_ds_paths.map(
    process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE
)

test_ds = test_ds.batch(BATCH_SIZE)

test_ds_imgs = test_ds_paths.map(
    process_path_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
)

test_ds_imgs = test_ds_imgs.batch(BATCH_SIZE)

test_ds_labels = test_ds_paths.map(
    get_label, num_parallel_calls=tf.data.experimental.AUTOTUNE
)

model = models.load_model("TrainModel_trained")

test_loss, test_acc = model.evaluate(test_ds)
test_acc_loss = f"Accuracy: {test_acc}\tLoss: {test_loss}\n"
print(test_acc_loss)

predicted_labels = model.predict(test_ds_imgs)
conf_mx = confusion_matrix(list(test_ds_labels), np.argmax(predicted_labels, axis=1))
print("Confusion matrix: \n{0}".format(conf_mx))

plt.figure(figsize=(10, 7))
sn.heatmap(
    conf_mx, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.savefig("confusion_matrix_heatmap")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from tensorflow.keras import layers, models
from pathlib import Path

TRAIN_DATASET_PATH = "train_dataset"
TEST_DATASET_PATH = "test_dataset"

LEARNING_RATE = 0.001
EARLY_STOPPING_THRESHOLD = 200
EPOCHS = 10000
BATCH_SIZE = 256
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3

train_data_dir = Path(TRAIN_DATASET_PATH)

list_ds = tf.data.Dataset.list_files(str(train_data_dir / "*/*"), shuffle=False)
DATASET_SIZE = (list_ds.cardinality().numpy())  # Returns the total number of images found in the train_dataset folder

class_names = np.array(
    sorted(
        [item.name for item in train_data_dir.glob("*") if item.name != "LICENSE.txt"]
    )
)
NUM_CLASSES = len(class_names)

TRAIN_SIZE = int(0.85 * DATASET_SIZE)

list_ds = list_ds.shuffle(DATASET_SIZE, reshuffle_each_iteration=False)
train_ds = list_ds.take(TRAIN_SIZE)
val_ds = list_ds.skip(TRAIN_SIZE)

test_data_dir = Path(TEST_DATASET_PATH)

test_ds = tf.data.Dataset.list_files(str(test_data_dir / "*/*"), shuffle=False)
TEST_SIZE = (test_ds.cardinality().numpy())  # Returns the total number of images found in the test_dataset folder


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


# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def configure_for_performance(ds, is_test_dataset=False):
    # Testing pipeline is similar to training pipeline, except that
    # 1. There is no ds.shuffle() call
    # 2. Caching is done after batching (as batches can be the same between epoch)
    # Reference: https://www.tensorflow.org/datasets/keras_example
    if is_test_dataset:
        ds = ds.batch(BATCH_SIZE)
        ds = ds.cache()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    else:
        ds_size = ds.cardinality().numpy()
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=ds_size)
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds, is_test_dataset=True)

weight_decay = 0.0001

model = models.Sequential()
model.add(
    layers.experimental.preprocessing.Rescaling(
        1.0 / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH)
    )
)
model.add(layers.experimental.preprocessing.RandomFlip("horizontal"))
model.add(layers.experimental.preprocessing.RandomTranslation(0.1, 0.1))
model.add(layers.experimental.preprocessing.RandomZoom(0.1))

model.add(layers.BatchNormalization())
model.add(
    layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=tf.keras.regularizers.L2(l2=weight_decay),
        activation="relu",
    )
)
model.add(layers.BatchNormalization())
model.add(
    layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=tf.keras.regularizers.L2(l2=weight_decay),
        activation="relu",
    )
)
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2), strides=2, padding="valid"))
model.add(layers.Dropout(0.20))

model.add(
    layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=tf.keras.regularizers.L2(l2=weight_decay),
        activation="relu",
    )
)
model.add(layers.BatchNormalization())
model.add(
    layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=tf.keras.regularizers.L2(l2=weight_decay),
        activation="relu",
    )
)
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2), strides=2, padding="valid"))
model.add(layers.Dropout(0.30))

model.add(
    layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=tf.keras.regularizers.L2(l2=weight_decay),
        activation="relu",
    )
)
model.add(layers.BatchNormalization())
model.add(
    layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=tf.keras.regularizers.L2(l2=weight_decay),
        activation="relu",
    )
)
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.40))

model.add(layers.Flatten())
model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["sparse_categorical_accuracy"],
)


def scheduler(epoch, lr):
    lrate = LEARNING_RATE
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate


callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=EARLY_STOPPING_THRESHOLD, restore_best_weights=True
)
callback2 = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[callback, callback2]
)

test_loss, test_acc = model.evaluate(test_ds)
test_acc_loss = f"Accuracy: {test_acc}\tLoss: {test_loss}\n"
print(test_acc_loss)

filename = os.path.basename(__file__).split(".")[0]

model.save(filename + "_trained")

results_dir = filename + "_results/"

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

f = open(results_dir + "test_results.txt", "w")
f.write(test_acc_loss)
f.close()

plt.plot(history.history["sparse_categorical_accuracy"], label="training accuracy")
plt.plot(
    history.history["val_sparse_categorical_accuracy"], label="validation accuracy"
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig(results_dir + "accuracy")

plt.figure()

plt.plot(history.history["loss"], label="training loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig(results_dir + "loss")

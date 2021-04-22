import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from pathlib import Path

DATASET_PATH = "dataset"

EPOCHS = 5
BATCH_SIZE = 32
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3

data_dir = Path(DATASET_PATH)

full_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE
)
DATASET_SIZE = full_ds.cardinality().numpy() # Note this returns TOTAL_NUM_IMAGES / BATCH_SIZE
NUM_CLASSES = len(full_ds.class_names)

TRAIN_SIZE = int(0.70 * DATASET_SIZE)
VAL_SIZE = int(0.15 * DATASET_SIZE)
TEST_SIZE = int(0.15 * DATASET_SIZE)

full_ds = full_ds.shuffle(DATASET_SIZE)
train_ds = full_ds.take(TRAIN_SIZE)
test_ds = full_ds.skip(TRAIN_SIZE)
val_ds = test_ds.take(VAL_SIZE)
test_ds = test_ds.skip(VAL_SIZE)

train_ds = train_ds.cache().shuffle(TRAIN_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
val_ds = val_ds.cache().shuffle(VAL_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
test_ds = test_ds.cache().prefetch(tf.data.experimental.AUTOTUNE)

model = models.Sequential()
model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(layers.MaxPooling2D((2, 2), strides=2, padding="valid"))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(layers.MaxPooling2D((2, 2), strides=2, padding="valid"))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(layers.MaxPooling2D((2, 2), strides=2, padding="valid"))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.50))
model.add(layers.Dense(NUM_CLASSES))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

test_loss, test_acc = model.evaluate(test_ds)
print(f"Accuracy: {test_acc}\tLoss: {test_loss}")

model.save("trained_model")

plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label = 'validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig("ModelAccuracy.png")

plt.figure()

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label = 'validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig("ModelLoss.png")

plt.show()

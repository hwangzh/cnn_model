import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.applications import MobileNetV2, InceptionResNetV2, VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img

## Data Augmentation
train_datagen = ImageDataGenerator(rescale=1 / 255.0, zoom_range=0.2, shear_range=0.3, horizontal_flip=True,
                                   brightness_range=[0.5, 1.5])
test_datagen = ImageDataGenerator(rescale=1 / 255.0)
val_datagen = ImageDataGenerator(rescale=1 / 255.0)

# Creating Batch size and Image shape
BATCH_SIZE = 32
IMG_SHAPE = (224, 224)

# Defining Train, Test and Validation data
train_data = train_datagen.flow_from_directory("/home/howard/Documents/DataSet/archive (1)/dataset/test",
                                               target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode="binary")

test_data = train_datagen.flow_from_directory("/home/howard/Documents/DataSet/archive (1)/dataset/train",
                                              target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode="binary")
val_data = train_datagen.flow_from_directory("/home/howard/Documents/DataSet/archive (1)/dataset/test",
                                             target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode="binary")

print(train_data.class_indices)

# Dictionary with key and correct values as labels
image_class_dict = {0: 'freshapples',
                    1: 'freshbanana',
                    2: 'freshoranges',
                    3: 'rottenapples',
                    4: 'rottenbanana',
                    5: 'rottenoranges'}


# Plotting Augmented Images
def plot_random_images():
    images, labels = train_data.next()
    plt.figure(figsize=(20, 10))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(image_class_dict[labels[i]])


plot_random_images()

# Creating CNN model
cnn_model = tf.keras.models.Sequential([
    Conv2D(16, 3, activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, 3, activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(64, 3, activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(100, activation="relu"),
    Dense(1, activation="sigmoid")
])

cnn_model.summary()

cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics="accuracy")

# Training for 100 EPOCHS

history = cnn_model.fit_generator(train_data, validation_data=(val_data), epochs=100, steps_per_epoch=18,
                                  validation_steps=9)


cnn_model.evaluate(test_data)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.title("Training And Validation Loss")
plt.xlabel("Epochs")
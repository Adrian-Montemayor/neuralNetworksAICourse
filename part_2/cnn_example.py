import tensorflow as tf
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = "cat_or_dog.h5"
IMAGE_PATH = "single_prediction/cat_or_dog_1.jpg"

def preprocess_data():
    print(f"Preprocessing the Training Set")
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
    training_set = train_datagen.flow_from_directory(
        "training_set",
        target_size=(64,64),
        batch_size=32,
        class_mode="binary"
    )
    print(f"Data labels: {training_set.class_indices}..\n")

    print(f"Preprocessing the Test set")
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = test_datagen.flow_from_directory(
        "test_set",
        target_size=(64,64),
        batch_size=32,
        class_mode="binary"
    )
    print(f"Data labels: {test_set.class_indices}...\n")

    return training_set, test_set

def build_cnn():
    print(f"Building the CNN\nIntitialising the CNN")
    cnn = tf.keras.models.Sequential()

    print(f"Convolution")
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=[64, 64, 3]))

    print(f"Pooling")
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) #pool_size 2*2  strides each 2 pixels

    print(f"adding a second convolutional layer")
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    print(f"Flattening")
    cnn.add(tf.keras.layers.Flatten()) #Previous results point to this one dimensional vector

    print(f"Full Connection Layer")
    cnn.add(tf.keras.layers.Dense(units=128, activation="relu"))

    print(f"Output Layer")
    cnn.add(tf.keras.layers.Dense(units=1, activation="sigmoid")) #One neuron due to final result is dog/cat  0/1

    return cnn

def train_cnn(cnn, training_set, test_set):
    print(f"TRAINING THE CNN\nCompiling the CNN")
    cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    cnn.fit(x=training_set, validation_data=test_set, epochs=25)
    cnn.save("cat_or_dog.h5")
    return cnn

def make_prediction(cnn, image_path):
    print(F"Single prediction")
    test_image = image.load_img(image_path, target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)
    if result[0][0] == 1:
        prediction = "dog"
    else:
        prediction = "cat"

    print(prediction)


if __name__ == "__main__":
    print(tf.__version__)
    if os.path.exists(MODEL_PATH):
        cnn = load_model(MODEL_PATH)
        make_prediction(cnn, IMAGE_PATH)
    else:
        training_set, test_set = preprocess_data()
        cnn = build_cnn()
        cnn = train_cnn(cnn, training_set, test_set)
        make_prediction(cnn, IMAGE_PATH)

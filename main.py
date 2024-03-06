import os
import cv2
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
"The dataset consists of 70,000 grayscale images of handwritten digits, "
"evenly divided into a training set (60,000 images) and a testing set (10,000 images)"
(x_train, y_train), (x_test, y_test)= mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) 
"Flatten turns out a grid to a single line (instead of 28*28 matrix yu have a line of 7800 line)"
model.add(tf.keras.layers.Dense(128, activation='relu'))
"Basic Neural networks that connect each neural to the other//relu:rectify linear unit"
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
"Softmax make sure all the outputs (10 neurons) end up to 1"


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
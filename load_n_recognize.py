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


model = tf.keras.models.load_model('handwritten model')

image_number = 1

while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}") #argmax gives the neuron that have the higher prediction
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except: 
        print('Error')
    finally: 
        image_number += 1
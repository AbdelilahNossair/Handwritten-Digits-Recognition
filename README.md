### Handwritten Digit Recognition

## Overview
This project uses TensorFlow and the MNIST dataset to recognize handwritten digits. It involves a simple neural network model that is trained to classify images of digits from 0 to 9.

## Dependencies
- Python 3
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
## Dataset
The MNIST dataset is utilized in this project. It consists of 70,000 grayscale images of handwritten digits, evenly divided into a training set (60,000 images) and a testing set (10,000 images).

## Model
The model is a sequential neural network built with TensorFlow. It has the following layers:

- Flatten Layer: Converts the 28x28 grid of pixels into a single 784 element vector.
- Dense Layer: A fully connected layer with 128 neurons and ReLU activation.
- Dense Layer: Another 128 neuron layer with ReLU activation.
- Dense Layer: A final layer with 10 neurons (one for each digit) and softmax activation, ensuring the output sums up to 1.
  
The model is compiled with the Adam optimizer and uses sparse categorical crossentropy as the loss function. It measures performance with accuracy.
## Training
The model is trained on the normalized training dataset for 3 epochs. This process involves adjusting the model weights to minimize the loss function, improving its accuracy with each epoch.

## Usage

The model is loaded from the saved file 'handwritten model'.
It iterates over images named digit[number].png in the digits directory.
Each image is preprocessed and fed into the model for prediction.
The model outputs the digit it believes is most likely represented by the image.
## Preprocessing Steps
- Image Reading: Images are read using OpenCV.
- Normalization: Images are converted to a grayscale, inverted, and normalized format suitable for the model.
- Prediction: The model predicts the digit and outputs the result.
## Error Handling
The script includes basic error handling to manage issues like file reading errors or prediction failures, ensuring the loop continues with the next image.

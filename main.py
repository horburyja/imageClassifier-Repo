# Basic classification: Classify images of clothing

import os
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # this line deletes annoying red text

# load training set and test set
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# scale down grayscale values before feeding them to network
train_images = train_images / 255.0
test_images = test_images / 255.0

# set up layers to network
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # transforms format of images from 2D to 1D array
    tf.keras.layers.Dense(128, activation='relu'),  # create layer with 128 nodes
    tf.keras.layers.Dense(128, activation='relu'),  # test
    tf.keras.layers.Dense(10)  # create layer that will return logits array of length 10
])

# compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# feed training data to model
model.fit(train_images, train_labels, epochs=10)

# evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTEST ACCURACY:', test_acc, '\n')

# make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# grab image from test dataset (in this case it is test_images[1] which is a pullover)
img = test_images[1]
img = (np.expand_dims(img, 0))  # add image to batch where it is only member
print(img)

# here we can see which of the categories that the neural net thinks the image is and compare it to true value
predictions_single = probability_model.predict(img)
print(predictions_single)
print(np.argmax(predictions_single[0]))

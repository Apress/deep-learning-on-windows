# How to use
#
# Train the model and save the model weights
# python lenet_fashion_mnist_tf_keras.py --train-model 1 --save-trained 1
#
# Train the model and save the model weights to a give directory
# python lenet_fashion_mnist_tf_keras.py --train-model 1 --save-trained 1 --weights data/lenet_fashion_weights.hdf5
#
# Evaluate the model from pre-trained model weights
# python lenet_fashion_mnist_tf_keras.py
#
# Evaluate the model from pre-trained model weights from a give directory
# python lenet_fashion_mnist_tf_keras.py --weights data/lenet_fashion_weights.hdf5

# first, let's import tensorFlow
import tensorflow as tf
import numpy as np

# import the FASHION_MNIST dataset
from tensorflow.keras.datasets import fashion_mnist

# imports used to build the deep learning model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# import the keras util functions
import tensorflow.keras.utils as np_utils

import argparse
import cv2
import matplotlib.pyplot as plt

# Setup the argument parser to parse out command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train-model", type=int, default=-1,
                help="(optional) Whether the model should be trained on the MNIST dataset. Defaults to no")
ap.add_argument("-s", "--save-trained", type=int, default=-1,
                help="(optional) Whether the trained models weights should be saved." +
                "Overwrites existing weights file with the same name. Use with caution. Defaults to no")
ap.add_argument("-w", "--weights", type=str, default="data/lenet_fashion_weights.hdf5",
                help="(optional) Path to the weights file. Defaults to 'data/lenet_fashion_weights.hdf5'")
args = vars(ap.parse_args())


# Getting the FASHION_MNIST dataset from Keras datasets
print("[INFO] Loading the FASHION_MNIST dataset...")
(trainData, trainLabels), (testData, testLabels) = fashion_mnist.load_data()
# The data is already in the form of numpy arrays,
# and already split to training and testing datasets

# Rescale the data from values between [0 - 255] to [0 - 1.0]
trainData = trainData / 255.0
testData = testData / 255.0

# Defining the string labels for the classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Display a sample from the FASHION_MNIST dataset
plt.figure(figsize=(16,16))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainData[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[trainLabels[i]])
plt.show()

# Reshape the data matrix from (samples, height, width) to (samples, height, width, depth)
# Depth (i.e. channels) is 1 since MNIST only has grayscale images
trainData = trainData[:, :, :, np.newaxis]
testData = testData[:, :, :, np.newaxis]

# The labels comes as a single digit, indicating the class.
# But we need a categorical vector as the label. So we transform it.
# So that,
# '0' will become [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# '1' will become [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# '2' will become [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# and so on...
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)


def build_lenet(width, height, depth, classes, weightsPath=None):
    # Initialize the model
    model = Sequential()

    # The first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding="same",
                     input_shape=(height, width, depth)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # The second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # The set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # The softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # If a weights path is supplied, then load the weights
    if weightsPath is not None:
        model.load_weights(weightsPath)

    # Return the constructed network architecture
    return model


def graph_training_history(history):
    plt.rcParams["figure.figsize"] = (12, 9)

    plt.style.use('ggplot')

    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')

    plt.tight_layout()

    plt.show()


# Build and Compile the model
print("[INFO] Building and compiling the LeNet model...")
opt = SGD(lr=0.01)
model = build_lenet(width=28, height=28, depth=1, classes=10,
                    weightsPath=args["weights"] if args["train_model"] <= 0 else None)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

# Check the argument whether to train the model
if args["train_model"] > 0:
    print("[INFO] Training the model...")

    history = model.fit(trainData, trainLabels,
                        batch_size=128,
                        epochs=50,
                        validation_data=(testData, testLabels),
                        verbose=1)

    # Use the test data to evaluate the model
    print("[INFO] Evaluating the model...")

    (loss, accuracy) = model.evaluate(
        testData, testLabels, batch_size=128, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

    # Visualize the training history
    graph_training_history(history)

# Check the argument on whether to save the model weights to file
if args["save_trained"] > 0:
    print("[INFO] Saving the model weights to file...")
    model.save_weights(args["weights"], overwrite=True)

# Training of the model is now complete

# Randomly select a few samples from the test dataset to evaluate
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
    # Use the model to classify the digit
    probs = model.predict(testData[np.newaxis, i])
    prediction = probs.argmax(axis=1)

    # Convert the digit data to a color image
    image = (testData[i] * 255).astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # The images are in 28x28 size. Much too small to see properly
    # So, we resize them to 280x280 for viewing
    image = cv2.resize(image, (280, 280), interpolation=cv2.INTER_LINEAR)

    # Add the predicted value on to the image
    cv2.putText(image, str(class_names[prediction[0]]), (20, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 1)

    # Show the image and prediction
    print("[INFO] Predicted: \"{}\", Actual: \"{}\"".format(
        class_names[prediction[0]], class_names[np.argmax(testLabels[i])]))
    cv2.imshow("Digit", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import cv2

image_path = 'data/validation/ALBATROSS/1.jpg'
img_width, img_height = 224, 224

# load the trained model
model = load_model('bird_classify_finetune.h5')

# load the class label dictionary
class_dictionary = np.load('class_indices.npy', allow_pickle=True).item()

# load the image and resize it to the size required by our model
image_orig = load_img(image_path, target_size=(img_width, img_height), interpolation='lanczos')
image = img_to_array(image_orig)

# important! otherwise the predictions will be '0'
image = image / 255.0

# add a new axis to make the image array confirm with
# the (samples, height, width, depth) structure
image = np.expand_dims(image, axis=0)

# get the probabilities for the prediction
probabilities = model.predict(image)

# decode the prediction
prediction_probability = probabilities[0, probabilities.argmax(axis=1)][0]
class_predicted = np.argmax(probabilities, axis=1)
inID = class_predicted[0]

# invert the class dictionary in order to get the label for the id
inv_map = {v: k for k, v in class_dictionary.items()}
label = inv_map[inID]

print("[Info] Predicted: {}, Confidence: {:.5f}%".format(label, prediction_probability*100))

# display the image and the prediction using OpenCV
image_cv = cv2.imread(image_path)
image_cv = cv2.resize(image_cv, (600, 600), interpolation=cv2.INTER_LINEAR)

cv2.putText(image_cv, 
            "Predicted: {}".format(label), 
            (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(image_cv, 
            "Confidence: {:.5f}%".format(prediction_probability*100), 
            (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow("Prediction", image_cv)
cv2.waitKey(0)

cv2.destroyAllWindows()
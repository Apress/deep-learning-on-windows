from flask import Flask, request, render_template, url_for, make_response, send_from_directory, flash, redirect, jsonify
from werkzeug.utils import secure_filename

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from PIL import Image
from io import BytesIO
import os
import os.path
import sys
import base64
import uuid
import time

# avoiding some compatibility problems in TensorFlow, cuDNN, and Flask
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# dimensions of our images.
img_width, img_height = 224, 224
# limiting the allowed filetypes
ALLOWED_FILETYPES = set(['.jpg', '.jpeg', '.gif', '.png'])

model_path = 'models/bird_classify_finetune_IV3_final.h5'

# loading the class dictionary and the model
class_dictionary = np.load('models/class_indices.npy', allow_pickle=True).item()

model = load_model(model_path)

# function for classifying the image using the model
def classify_image(image):
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255.0

    # add a new axis to make the image array confirm with
    # the (samples, height, width, depth) structure
    image = np.expand_dims(image, axis=0)

    # get the probabilities for the prediction
    # with graph.as_default():
    probabilities = model.predict(image)

    prediction_probability = probabilities[0, probabilities.argmax(axis=1)][0]

    class_predicted = np.argmax(probabilities, axis=1)

    inID = class_predicted[0]

    # invert the class dictionary in order to get the label for the id
    inv_map = {v: k for k, v in class_dictionary.items()}
    label = inv_map[inID]

    print("[Info] Predicted: {}, Confidence: {}".format(label, prediction_probability))

    return label, prediction_probability

# get a thumbnail version of the uploaded image
def get_iamge_thumbnail(image):
    image.thumbnail((400, 400), resample=Image.LANCZOS)
    image = image.convert("RGB")
    with BytesIO() as buffer:
        image.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

# request handler function for the home/index page
def index():
    # handling the POST method of the submit
    if request.method == 'POST':
        # check if the post request has the submitted file
        if 'bird_image' not in request.files:
            print("[Error] No file uploaded.")
            flash('No file uploaded.')
            return redirect(url_for('index'))
        
        f = request.files['bird_image']

        # if user does not select a file, some browsers may
        # submit an empty field without the filename
        if f.filename == '':
            print("[Error] No file selected to upload.")
            flash('No file selected to upload.')
            return redirect(url_for('index'))

        sec_filename = secure_filename(f.filename)
        file_extension = os.path.splitext(sec_filename)[1]

        if f and file_extension.lower() in ALLOWED_FILETYPES:
            file_tempname = uuid.uuid4().hex
            image_path = './uploads/' + file_tempname + file_extension
            f.save(image_path)

            image = load_img(image_path, target_size=(img_width, img_height), interpolation='lanczos')

            label, prediction_probability = classify_image(image=image)
            prediction_probability = np.around(prediction_probability * 100, decimals=4)

            orig_image = Image.open(image_path)
            image_data = get_iamge_thumbnail(image=orig_image)

            with application.app_context():
                return render_template('index.html', 
                                        label=label, 
                                        prob=prediction_probability,
                                        image=image_data
                                        )
        else:
            print("[Error] Unauthorized file extension: {}".format(file_extension))
            flash("The file type you selected: '{}' is not supported. Please select a '.jpg', '.jpeg', '.gif', or a '.png' file.".format(file_extension))
            return redirect(url_for('index'))
    else:
        # handling the GET, HEAD, and any other methods

        with application.app_context():
            return render_template('index.html')

# handle 'filesize too large' errors
def http_413(e):
    print("[Error] Uploaded file too large.")
    flash('Uploaded file too large.')
    return redirect(url_for('index'))

# setting up the application context
application = Flask(__name__)
# set the application secret key. Used with sessions.
application.secret_key = '@#$%^&*@#$%^&*'

# add a rule for the index page.
application.add_url_rule('/', 'index', index, methods=['GET', 'POST'])

# limit the size of the uploads
application.register_error_handler(413, http_413)
application.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()
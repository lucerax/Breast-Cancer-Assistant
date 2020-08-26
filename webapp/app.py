import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil
import cv2


# Declare a flask app
app = Flask(__name__)



# Model saved with Keras model.save()
MODEL_PATH = 'models/my_model2.h5'


# Load your own trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img, model):
    IMG = []
    img = np.asarray(img.convert("RGB"))
    img = cv2.resize(img, (224, 224))
    IMG.append(np.array(img))
    x = np.array(IMG)
    print("input form: ", x)
    # Preprocessing the image
    #x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)
        print("b/m", preds)

        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        print("pred_proba", pred_proba)
        print(type(preds[0]))

        if float(preds[0][1]) >= 0.5:
            result = "malignant"
        else:
            result = "benign"


        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

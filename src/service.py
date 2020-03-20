# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

import os
import json
import time
import sys
import cv2
import numpy as np
import flask
import processing.preprocessing as preprocessor
import processing.face_net as models

prefix = '/opt/program/'
model_path = os.path.join(prefix, 'models')

# The flask app for serving predictions
recognition_app = flask.Flask(__name__)


def get_embeddings(img):

    img = preprocessor.extract_face(img)

    img = cv2.resize(img, (160,160))
        
    img = preprocessor.normalize(img)

    return models.FaceNet.predict(img)[0]


# Recognition endpoint
@recognition_app.route("/recognize", methods=["POST"])
def recognize():

    try:

        t1 = time.time()

        customer_face_img = flask.request.get_data()

        customer_face_img = cv2.imdecode(np.fromstring(customer_face_img, np.uint8), cv2.IMREAD_COLOR)

        customer_face_embeddings = get_embeddings(customer_face_img)

        test_face_img = cv2.imread("jim_carrey_1.jpeg")

        test_face_embeddings = get_embeddings(test_face_img)

        distance = np.linalg.norm(customer_face_embeddings - test_face_embeddings)

        t2 = time.time()

        print(f"Elapsed time :{t2-t1} seconds")

        return flask.jsonify({
            "error": "false",
            "distance": str(distance)
        })

    except Exception as err:
        print(err)
        return flask.jsonify({
            "error": "true",
            "message": "Something bad happens"
        })

# Testing endpoint
@recognition_app.route("/test", methods = ["POST"])
def test():

    try:
        print("Hi")
    except Exception as err:
        print(err)
        return flask.jsonify({
            "error": "true",
            "message": "Something bad happens"
        })
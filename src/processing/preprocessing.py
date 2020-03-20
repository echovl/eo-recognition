import cv2
import os
import math
import numpy as np
import glob
import pathlib

prefix = '/opt/program/'
model_path = os.path.join(prefix, 'models')
    
# Haarcascade face detector
face_detector = cv2.CascadeClassifier(os.path.join(model_path, "haarcascade_frontalface_default.xml"))

# Haarcascade parameters
haar_scale_factor = 1.3
haar_min_neighbors = 5

def extract_face(img):
    # Image original height and width
    img_height = img.shape[0]
    img_width = img.shape[1]

    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    

    # Detect faces with OpenCV
    faces_box = face_detector.detectMultiScale(gray_img, haar_scale_factor , haar_min_neighbors)

    if len(faces_box) == 0:
        raise Exception("No faces detected")

    cropped_img = None

    for (x, y, w, h) in faces_box:

        x1, y1 = x, y
        x2, y2 = x + w, y + h
                
        cropped_img = img[y1:y2,x1:x2]
        
    return cropped_img

def normalize(img):
    img = img.astype("float32")

    mean, std = img.mean(), img.std()
    
    img = (img - mean) / std

    return np.expand_dims(img, axis = 0)


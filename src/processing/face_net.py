import os
from tensorflow.python.keras.models import load_model

prefix = '/opt/program/'
model_path = os.path.join(prefix, 'models')

class FaceNet(object):
    model = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            print("Building FaceNet ...")

            cls.model = load_model(os.path.join(model_path, "facenet_keras.h5"), compile = False)
        return cls.model

    @classmethod
    def predict(cls, input):
        clf = cls.get_model()
        return clf.predict(input)
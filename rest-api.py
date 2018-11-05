from PIL import Image
import numpy as np
import flask
import io
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = load_model('saved_models/Emotion_Voice_Detection_Model.h5')
def prepare_audio(audio, target):

    # resize the input image and preprocess it
    audio = audio.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

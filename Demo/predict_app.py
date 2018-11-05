# import the necessary packages
import numpy as np
import flask
import io
import keras
from keras.models import Sequential
from keras.models import load_model
import base64
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

def get_model():
	global model
	model = load_model('Emotion_Voice_Detection_Model.h5')
	print("Model loaded")

def preprocess_audio(audio):
	audio = np.array(audio)
	audio = np.expand_dims(audio, axis=0)
	return audio

print("Loading keras model")
get_model()

@app.route('/predict', methods=["POST"])
def predict():
	message = request.get_json(force = True)
	encoded = message['audio']
	decoded = base64.b64decode(encoded)
	preprocessed_audio = preprocess_audio(decoded)
	predection = model.predict(preprocessed_audio).tolist()
	response = {
		'predection':{
			 'Angry' : prediction[0][0],
			 'Disgust' : prediction[0][1],
			 'Fear' : prediction[0][2],
			 'Happy' : prediction[0][3],
			 'Neutral' : prediction[0][4],
			 'Sad' : prediction[0][5],
			 'Surprise' : prediction[0][6],
			 }
	}
	return jsonify(response)

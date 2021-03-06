from voice import app
from flask import request, render_template, jsonify

import tempfile
import shutil
import os


from flask import Flask, render_template, request
from werkzeug import secure_filename

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram

import json
from pprint import pprint

import keras
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras import regularizers

import pandas as pd
import os
import glob 
import csv
from keras.models import load_model
import warnings


@app.route('/')
def index():
    return render_template('index.html', key=None)

@app.route('/submit', methods=['POST'])
def submit():
    dirpath = "";
    path = os.path.join(dirpath, 'audio.wav')
    request.files['blob'].save(path)

    # TODO: process the wav file save at path
    size = os.path.getsize(path)



    shutil.rmtree(dirpath)

    return render_template('synth.html', key=None)

    # TODO: return the result

@app.route('/submit2')
def submit2():
    df = pd.DataFrame(columns=['feature'])
                    
    bookmark=0

    X, sample_rate = librosa.load('audio.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)

    plt.figure(figsize=(15, 5))

    librosa.display.waveplot(X, sr=sample_rate)
    plt.savefig('static/img/waveplot.png')


                
    sample_rate = np.array(sample_rate)
    
    mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                            sr=sample_rate, 
                                            n_mfcc=13),
                            axis=0)

    kogo = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)


    plt.figure(figsize=(10, 4))
    librosa.display.specshow(kogo, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()

    plt.savefig("static/img/mfccplot.png")

                
                
    feature = mfccs
    #[float(i) for i in feature]
    #feature1=feature[:135]
    df.loc[bookmark] = [feature]
    bookmark=bookmark+1

    df = pd.DataFrame(df['feature'].values.tolist())
    df=df.fillna(0)
    df.to_csv('features-test.csv',sep='\t', encoding='utf-8')


    Test = np.array(df)

    print(Test.shape)


    temp = np.zeros((195,1))

    Test = Test.reshape(Test.shape[1],1)


    print(Test.shape[0])
    temp[:Test.shape[0],:Test.shape[1]] = Test

    Test = temp

    x_testncnn =np.expand_dims(Test, axis=0)
    # x_testncnn =np.expand_dims(Test, axis=1)

    print(x_testncnn.shape)
    # npdf=np.array(df)
    # print(npdf[0])

    model = load_model('../saved_models/Emotion_Voice_Detection_Model.h5')

    print("Loaded model from disk")
    preds = model.predict(x_testncnn, 
                                batch_size=23, 
                                verbose=1)
    print("preds0")
    print(preds)
    preds1=preds.argmax(axis=1)
    print(preds1)

                        

    emot = {
            0:"angry",
            1:"disgust",
            2:"fear",
            3:"happy",
            4:"neutral",
            5:"sad",
            6:"surprised"
            }

                        
    data = {
            "emotion":emot[preds1[0]],
            "angry":format(preds[0][0]*100, '.2f'),
            "disgust": format(preds[0][1]*100, '.2f'),
            "fear": format(preds[0][2]*100, '.2f'),
            "happy": format(preds[0][3]*100, '.2f'),
            "neutral": format(preds[0][4]*100, '.2f'),
            "sad": format(preds[0][5]*100, '.2f'),
            "surprised": format(preds[0][6]*100, '.2f'),
            "audiofilename":"audio.wav"
            }

    print(data)

    return render_template('synthesis.html',data=data)



    # TODO: return the result

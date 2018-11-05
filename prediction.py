#Importing the required libraries
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
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
from keras import regularizers
import pandas as pd
import os
import glob 
import csv
from keras.models import load_model
import warnings
from keras.callbacks import History 
#List of dataset audio
Testlist= os.listdir('test/')
print(Testlist[0])
df = pd.DataFrame(columns=['feature'])
bookmark=0

print(Testlist)
for index,y in enumerate(Testlist):
        X, sample_rate = librosa.load('test/'+y, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)

        plt.figure()
        plt.subplot(3, 1, 1)
        librosa.display.waveplot(y, sr=sr)
        plt.title('Monophonic')
        
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                            sr=sample_rate, 
                                            n_mfcc=13),
                        axis=0)
        feature = mfccs
        #[float(i) for i in feature]
        #feature1=feature[:135]
        df.loc[bookmark] = [feature]
        bookmark=bookmark+1

        break

df = pd.DataFrame(df['feature'].values.tolist())
df=df.fillna(0)
df.to_csv('features-test.csv',sep='\t', encoding='utf-8')

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
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

model = load_model('saved_models/Emotion_Voice_Detection_Model.h5')

print("Loaded model from disk")
preds = model.predict(x_testncnn, 
                         batch_size=23, 
                         verbose=1)
print(preds)
preds1=preds.argmax(axis=1)
print(preds1)



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
#List of dataset audio
mylist= os.listdir('dataset/')
print(mylist[0])
#Setting the labels
feeling_list=[]
for filename in os.listdir('dataset/'):
 if filename.startswith('1'):
        feeling_list.append('angry') 
 if filename.startswith('2'):
        feeling_list.append('disgust') 
 if filename.startswith('3'):
        feeling_list.append('fear') 
 if filename.startswith('4'):
        feeling_list.append('happy') 
 if filename.startswith('5'):
        feeling_list.append('neutral') 
 if filename.startswith('6'):
        feeling_list.append('sad') 
 if filename.startswith('7'):
        feeling_list.append('surprise')
labels = pd.DataFrame(feeling_list)
#print(labels[:10])
#Getting the features of audio files using librosa
df = pd.DataFrame(columns=['feature'])
bookmark=0
for index,y in enumerate(mylist):
        X, sample_rate = librosa.load('dataset/'+y, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
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
#print(df[:5])

df3 = pd.DataFrame(df['feature'].values.tolist())
#print(df3[:5])

newdf = pd.concat([df3,labels], axis=1)
rnewdf = newdf.rename(index=str, columns={"emotion": "label"})
rnewdf=rnewdf.fillna(0)
#print(rnewdf[:5])

# from sklearn.utils import shuffle
# rnewdf = shuffle(newdf)
# #print(rnewdf[:10])

# rnewdf=rnewdf.fillna(0)
# #print(rnewdf)
rnewdf.to_csv('features.csv',sep='\t', encoding='utf-8')
#Dividing the data into test and train
newdf1 = np.random.rand(len(rnewdf)) < 0.8
train = rnewdf[newdf1]
test = rnewdf[~newdf1]
trainfeatures = train.iloc[:, :-1]
trainlabel = train.iloc[:, -1:]
testfeatures = test.iloc[:, :-1]
testlabel = test.iloc[:, -1:]

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

X_train = np.array(trainfeatures)
y_train = np.array(trainlabel)
X_test = np.array(testfeatures)
y_test = np.array(testlabel)

lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))
print(y_train)
print(X_train.shape)

#Padding sequence for CNN model
print('Pad sequences')
x_traincnn =np.expand_dims(X_train, axis=2)
x_testcnn= np.expand_dims(X_test, axis=2)
print(x_testcnn)

# model = Sequential()

# model.add(Conv1D(128, 5,padding='same',
#                  input_shape=(195,1)))
# model.add(Activation('relu'))
# model.add(Conv1D(128, 5,padding='same'))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))
# model.add(MaxPooling1D(pool_size=(8)))
# model.add(Conv1D(128, 5,padding='same',))
# model.add(Activation('relu'))
# model.add(Conv1D(128, 5,padding='same',))
# model.add(Activation('relu'))
# model.add(Conv1D(128, 5,padding='same',))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Conv1D(128, 5,padding='same',))
# model.add(Activation('relu'))
# model.add(Flatten())
# model.add(Dense(7))
# model.add(Activation('softmax'))
# opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
# model.summary()
# model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
# cnnhistory=model.fit(x_traincnn, y_train, batch_size=32, epochs=200, validation_data=(x_testcnn, y_test))

#sigmoid
# plt.plot(cnnhistory.history['acc'])
# plt.plot(cnnhistory.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# plt.plot(cnnhistory.history['loss'])
# plt.plot(cnnhistory.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# model_name = 'Emotion_Voice_Detection_Model.h5'
# save_dir = os.path.join(os.getcwd(), 'saved_models')
# # Save model and weights
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# model_path = os.path.join(save_dir, model_name)
# model.save(model_path)
# print('Saved trained model at %s ' % model_path)
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

model = load_model('saved_models/Emotion_Voice_Detection_Model.h5')
print("Loaded model from disk")
preds = model.predict(x_testcnn, 
                         batch_size=32, 
                         verbose=1)
print(preds)
preds1=preds.argmax(axis=1)
print(preds1)
abc = preds1.astype(int).flatten()
predictions = (lb.inverse_transform((abc)))
preddf = pd.DataFrame({'predictedvalues': predictions})
preddf.to_csv('Predictions.csv', index=False)
print(preddf[:10])
objective_score = model.evaluate(x_testcnn, y_test,
batch_size=32)
print(objective_score)
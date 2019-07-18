# -*- coding: utf-8 -*-
"""
Created on Fri July 10 07:43:46 2019

@author: venkatesh avula
"""
from PIL import ImageFont, ImageDraw, Image, ImageFilter
import Lib
from Lib import add_noise_img, ImgData, ImgSize, Fonts, MakeDataset, MakeDataset_CNN
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.utils import plot_model

from tqdm import tqdm
#inputs---------------------------------------------------
N=10000

##train---------------------------------------------------
#build dataset
nLabels=len(Fonts)
x,y=MakeDataset_CNN(N)     
y = to_categorical(y)
x, y = shuffle(x, y, random_state=10)
#Split The Data Into Train And Test Sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

#build CNN-----------------------------------------------
model = Sequential()
#2 convolutional layers: 32 and 64 3x3 filters.
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(ImgSize,ImgSize,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Dropout Regularization
model.add(Dropout(0.25))

model.add(Flatten())

#dense hidden layers
#model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))

#Dropout Regularization
model.add(Dropout(0.5))

#output layer, Binary Softmax classifier
model.add(Dense(nLabels, activation='softmax'))

#keras optimizer Adam, a module that contains different types of back propagation algorithm 
#for training our model.
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
history=model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# making predictions
preds = model.predict_classes(X_test)

#for roc
#preds_prob = model.predict_proba(X_test)

y_test_results=np.argmax(y_test, axis=1)

accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

#Check The Accuracy Of The Model
print("Accuracy:", accuracy_score(y_test_results,preds))

#Visualazation--------------------------------------------------------
model.summary()

# Plot training & validation accuracy values
plt.rcParams.update({'font.size': 21})

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.grid(color='grey', linestyle='--', linewidth=1)
plt.savefig('model_Accuracy.png', bbox_inches='tight')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.grid(color='grey', linestyle='--', linewidth=1)
plt.savefig('model_Loss.png', bbox_inches='tight')

plot_model(model, to_file='mode_CNN.png',show_shapes=True, show_layer_names=True)
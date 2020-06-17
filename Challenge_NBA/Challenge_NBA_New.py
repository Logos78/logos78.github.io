# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:07:26 2020

@author: Geoffrey
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot as plt


### Pre-process data
directory = "C:\\Users\\Geoffrey\\Desktop\\Cours\\M2\\Projet statistiques\\Projet"
df = pd.read_csv(directory+"\\train.csv")
df = df.set_index('ID')
#df = df.sort_values(by=['ID'])
df = df.drop(columns = [i for i in df.columns if "defensive foul" in i])


results = pd.read_csv(directory+"\\train_results.csv", sep=";")
results = results.set_index('ID')
#results = results.sort_values(by=['ID'])

size = 12000
dt = 10
limit = 16*60
timesteps = int(min(1440,limit)/dt)
df_part = df[:size]
df_part = df_part[[i for i in df_part.columns if int(i.split("_")[1])%dt == 0]]
df_part = df_part[[i for i in df_part.columns if int(i.split("_")[1]) <= limit]]
X = df_part.to_numpy()
X = np.reshape(X, (size, timesteps, 10))
Y = results[:size].to_numpy()

#from sklearn import preprocessing
#X = preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.25, random_state=4)


### Create model
n_units = 8
n_epochs = 30
batch_size = 50

model = Sequential()
model.add(LSTM(n_units, return_sequences=True))
model.add(LSTM(n_units))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs = n_epochs, batch_size = batch_size,
                    validation_data=(X_test, Y_test), shuffle=False)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model train vs validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

plt.show()
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:07:26 2020

@author: Geoffrey
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from matplotlib import pyplot as plt


### Pre-process data
directory = "C:\\Users\\Geoffrey\\Desktop\\Cours\\M2\\Projet statistiques\\Projet"
df = pd.read_csv(directory+"\\train.csv")
df = df.set_index('ID')
df = df.drop(columns = [i for i in df.columns if "defensive foul" in i])

results = pd.read_csv(directory+"\\train_results.csv", sep=";")
results = results.set_index('ID')


### Create dataset
def create_dataset(df, train_size, test_size=3000, time_limit=24, dt=1) :
    # Select the samples
    samples = train_size+test_size
    df_part = df[:samples]
        
    # Take only the firts "time_limit" minutes
    df_part = df_part[df.columns[:time_limit*60*10]]
        
    # Select the desired timesteps
    timesteps = int(min(24*60,time_limit*60)/dt)
    #df_part = df_part[[i for i in df_part.columns if int(i.split("_")[1])%dt == 0]]
    
    # Prepare datas for the training
    X = df_part.to_numpy()
    X = np.reshape(X, (samples, timesteps, 10))
    Y = results[:samples].to_numpy()
    #for i in range(timesteps) :
        #X[:,i,:] = preprocessing.StandardScaler().fit(X[:,i,:]).transform(X[:,i,:])
    
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=test_size/samples, random_state=4)
    
    return [X_train, X_test, Y_train, Y_test]


### Create model
def train_model(dataset, n_units, n_epochs, batch_size) :
    X_train, X_test, Y_train, Y_test = dataset
    
    # Create model
    model = Sequential()
    for l in range(len(n_units)-1) : 
        model.add(LSTM(n_units[l], return_sequences=True))
        #model.add(Dropout(0.3))
    model.add(LSTM(n_units[-1]))
    #model.add(Dropout(0.3))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train model
    history = model.fit(X_train, Y_train, epochs = n_epochs, batch_size = batch_size,
                        validation_data=(X_test, Y_test), shuffle=False)
    
    return history


### Display results
def display_results(history) :
    # Plot loss    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    
    # Plot accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model train vs validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


### Execute
train_size = 1000      # 9576 maximum
test_size = 3000
dt = 1
time_limit = 24  # 24 minutes max

n_units = [32]
n_epochs = 10
batch_size = 50

dataset = create_dataset(df, train_size, 3000, test_size, dt)
history = train_model(dataset, n_units, n_epochs, batch_size)
display_results(history)


### Conclusion
# A one-layer network with 32 hidden units achieve the best result with around 72-73% accuracy
# Unfortunately, the LSTM network doesn't perform better than the previous models based on linear or logistic regressions
# Since the model is subject to overfitting, the complexity of the network had been reduced from several layers to one..
# Then dropout layers  were implemented to fix this problem.
# These two solutions didn't really work.
# As stated in the report, the main problem is the dataset. Most variables are not related enough to the score.
# Collecting new variables seems to be the only solution to improve the performance of the model.

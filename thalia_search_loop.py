# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
#Usage: copy the df line to read as many of the feature files as you would like to test
#Then add the dataframes to the dfs list, and their names to the dfnames list.
#Make sure to change the shapes list to fit the number of features in each set.

#This program will automatically search for hyperparameters that will fit these sets best.
#This is meant to be used on a computer cluster, since each search takes around 2 hours to run.
df = pd.read_table('')
dfs = []
dfnames = []
shapes =[1306, 1306, 1306, 1306, 1290, 1290, 1290, 1290]

#seperating targets from training data
def shuffle_sep(df):
  df = df.sample(frac=1)
  df_targy = df.filter(regex='Genotype')
  df_targy['Genotype'] = pd.factorize(df['Genotype'])[0]
  df = df.drop('Genotype', 1)
  return df, df_targy

def sets_numpy(df, df_targy):
  train = df.to_numpy()
  targy = df_targy.to_numpy()
  return train, targy

#vectorization
from tensorflow.keras.utils import to_categorical
def vectorize_and_holdout(train, targy):
  v_train = train
  v_targy = to_categorical(targy)
  v_train = v_train[100:]
  v_targy = v_targy[100:]
  return v_train, v_targy

#k-fold validation training                                   
def run_thalia(shape, trainset, targyset, k, num_epochs, numlayers, weights, kernel, dropout):          
    all_scores = []                                           
    all_losses = []                                           
    all_tlosses = []                                          
    num_val_samples = len(trainset) // k
    all_scores = []                                           
    for i in range(k):                                        
            model = tf.keras.models.Sequential()              
            if kernel == True:                                
                model.add(tf.keras.layers.Dense(weights, kernel_regularizer= tf.
keras.regularizers.l2(0.001), activation='relu', input_shape=(shape,)))
            if kernel != True:                                
                model.add(tf.keras.layers.Dense(weights, activation='relu', input_shape=(shape,)))
            if dropout == True:                               
                model.add(tf.keras.layers.Dropout(0.25))
            for j in range(numlayers):                        
                if kernel == True:                            
                    model.add(tf.keras.layers.Dense(weights, kernel_regularizer=
tf.keras.regularizers.l2(0.001), activation='relu'))
                if kernel != True:                            
                    model.add(tf.keras.layers.Dense(weights, activation='relu'))
                if dropout == True:                           
                    model.add(tf.keras.layers.Dropout(0.25))  
            model.add(tf.keras.layers.Dense(6, activation='softmax'))
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            val_data = trainset[i * num_val_samples: (i + 1) * num_val_samples]
            val_targy = targyset[i * num_val_samples: (i + 1) * num_val_samples]
            partial_v_train = np.concatenate([trainset[:i * num_val_samples], trainset[(i + 1) * num_val_samples:]], axis = 0)
            partial_v_targy = np.concatenate([targyset[:i * num_val_samples], targyset[(i + 1) * num_val_samples:]], axis = 0)
            history = model.fit(partial_v_train, partial_v_targy, epochs= num_epochs, batch_size=512, verbose = 0, validation_data = (val_data, val_targy)) 
            val_loss, val_mae = model.evaluate(val_data, val_targy, verbose = 0)
            all_scores.append(val_mae)
    return history, model                                     
#search function, change weights or add more loops as needed!                                                              
def thalia_search(shape, trainset, targyset):
    weightvals = [32, 64, 128, 256, 512, 1024]                
    torf = [True, False]                                      
    torf2 = [True, False]
    accuracies = []                                           
    hparams = []                                              
    for i in range(9):                                        
        for j in weightvals:                                  
            for k in torf:                                    
                for l in torf2:                               
                    paramstring = str(i) + " " + str(j) + " " + str(k) + " " + str(l)                                                         
                    hparams.append(paramstring)               
                    history, model = run_thalia(shape, trainset, targyset, 4, 10, i, j, k, l)
                    val_accuracy_list = history.history["val_accuracy"]
                    val_accuracy = max(val_accuracy_list)     
                    accuracies.append(val_accuracy)
                                                              
    maxindex = accuracies.index(max(accuracies))
    best_hparams = hparams[maxindex]                          
    print(max(accuracies))                                    
    return best_hparams  
counter = 0
for df in dfs:
    df, df_targy = shuffle_sep(df)
    train, targy = sets_numpy(df, df_targy)
    v_train, v_targy = vectorize_and_holdout(train, targy)
    best = thalia_search(shapes[counter], v_train, v_targy)
    print(dfnames[counter])
    print(best)
    counter = counter + 1
    


                                     

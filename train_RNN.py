##import required packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from tensorflow.keras.layers import Dense,LSTM, Dropout
import matplotlib.pyplot as plt

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__": 
    # 1. load your training data

    #original_dataset consists the data from original file 
    original_dataset = pd.read_csv("./data/q2_dataset.csv")
    new_df=np.zeros((1258,13))

    # literating throught the for loop in order to create a a dataset in which
    # the open of the next day is calculated using the past 3 days Open, High, and Low prices and volume. 
    for i in range (len(new_df)-2):
    
        new_df[i][12] = original_dataset.iloc[i+3][3]  #target column
        
        new_df[i][0] = original_dataset.iloc[i+1][3]    #open1
        new_df[i][1] = original_dataset.iloc[i+2][3]    #open2 
        new_df[i][2] = original_dataset.iloc[i][3]      #open3
        
        new_df[i][3] = original_dataset.iloc[i+2][4]    #High1
        new_df[i][4] = original_dataset.iloc[i+1][4]    #High2
        new_df[i][5] = original_dataset.iloc[i][4]      #High3
        
        new_df[i][6] = original_dataset.iloc[i+2][5]    #Low1
        new_df[i][7] = original_dataset.iloc[i+1][5]    #Low2
        new_df[i][8] = original_dataset.iloc[i][5]      #Low3
        
        new_df[i][9] = original_dataset.iloc[i+2][2]    #Volume1
        new_df[i][10] = original_dataset.iloc[i+1][2]   #Volume2
        new_df[i][11] = original_dataset.iloc[i][2]     #Volume3

    column_names = ['Open_day1','Open_day2','Open_day3','High_day1','High_day2','High_day3','Low_day1','Low_day2','Low_day3','Volume_day1','Volume_day2','Volume_day3','Target']
    threeday_df = pd.DataFrame(new_df[:-2,:],columns=column_names)

    #dropping the Target feature
    df_dropped_target = threeday_df.drop(['Target'],axis=1)

    # Splitting the dataset into 30% testing and 70% training 
    x_train, x_test, y_train, y_test = train_test_split(df_dropped_target, threeday_df['Target'], test_size=0.3)
    Train_Dataset = pd.concat([x_train,y_train],axis=1)
    Test_Dataset=pd.concat([x_test,y_test],axis=1)

    # Commenting the exporting of ‘train_data_RNN.csv’ and ‘test_data_RNN.csv’
    # Train_Dataset.to_csv(r'./Train_Dataset_Rnn.csv', index = False, header=True)
    # Test_Dataset.to_csv(r'./Test_Dataset_Rnn.csv', index = False, header=True)
    
    
    #Loading the training data from Train_Dataset_Rnn.csv
    Train_Data = pd.read_csv("./data/Train_Dataset_Rnn.csv")

    #creating x-train
    x_train=Train_Data.drop(['Target'],axis=1)
    #creating y-train
    y_train=Train_Data['Target']

    #Normlization using min-max scaler
    scaler=MinMaxScaler(feature_range=(0,1))
    x_train=scaler.fit_transform(x_train)

    #Converting into numpy-array
    x_train=np.array(x_train)

    #reshaping the 2-D array into 3-D array which is needed for LSTM
    x_train =x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
    
    
    # 2. Train your network

    # Building a Model
    My_LSTM_model = Sequential()
    #adding LSTM layer with 60 LSTM units
    My_LSTM_model.add(LSTM(50,input_shape=(x_train.shape[1],1),return_sequences=True))
    #The function of the dropout layer is just to add noise so the model learns to generalize better
    # My_LSTM_model.add(Dropout(0.2))
    #adding LSTM layer with 180 LSTM units
    My_LSTM_model.add(LSTM(150))
    # My_LSTM_model.add(Dropout(0.2))
    #adding dense layer
    My_LSTM_model.add(Dense(1,activation='linear'))


    #'mean_squared_error' has been used as loss function
    # Optimizer: Here adam optimizer has been used.
    # Adam is an adaptive learning rate optimization algorithm that’s been designed specifically for
    # training deep neural networks.
    My_LSTM_model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])

    History = My_LSTM_model.fit(x_train,y_train,epochs=600,batch_size=64,verbose=1)

    # Make sure to print your training loss within training to show progress
    # Make sure you print the final training loss
    print('The final training loss(total losss) is ',History.history['loss'][-1])
    print('The final training loss(mean squared error) is ',History.history['mae'][-1])
    
    
    # 3. Save your model
    My_LSTM_model.save('./models/Group31_RNN_model.h5')
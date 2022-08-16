# import required packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from tensorflow.keras.layers import Dense,LSTM, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__": 

    # 1. Load your saved model
    my_model = keras.models.load_model('./models/Group31_RNN_model.h5')


    # 2. Load your testing data
    testing_data = pd.read_csv("./data/Test_Dataset_Rnn.csv")
    x_test = testing_data.drop(['Target'],axis=1)
    y_test = testing_data['Target']

    #Normalization
    scaler = MinMaxScaler(feature_range=(0,1))
    x_test = scaler.fit_transform(x_test)

    #Converting into a numpy_array for further reshaping
    x_test=np.array(x_test)
    y_test=np.array(y_test)
    
    #Reshaping the array into 3-Dimensional data
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)
    
    
    # 3. Run prediction on the test data and print the test accuracy
    y_pred = my_model.predict(x_test)


    total_loss_mse = mean_squared_error(y_test,y_pred)
    print(f'Total loss(mean_squared_error) of the Test Data is : {total_loss_mse}')
    total_loss_mae = mean_absolute_error(y_test,y_pred)
    print(f'Total loss(mean absolute error) of the Test Data is : {total_loss_mae}')
  


    plt.figure(figsize=(20,10))
    plt.plot(y_test, color="red", marker='o', linestyle='solid', label="real stock price")
    plt.plot(y_pred, color="blue", marker='o', linestyle='solid', label="predicted stock price")
    plt.title("stock price prediction")
    plt.xlabel("Date(random)")
    plt.ylabel("stock price")
    plt.legend()
    plt.show()
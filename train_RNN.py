# import required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import os
import sys


# YOUR IMPLEMENTATION
def load_data():
	df= pd.read_csv("data/q2_dataset.csv")
	return df

def model(X_train):
	# Initialising the RNN
	regressor = Sequential()

	# Adding the first LSTM layer and some Dropout regularisation
	regressor.add(LSTM(units = 512, input_shape = (X_train.shape[1], 4)))


	# Adding the output layer
	regressor.add(Dense(units = 1))

	# Compiling the RNN
	regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

	return regressor

	
# Thoroughly comment your code to make it easy to follow

def main():

	# 1. load your training data
	# Loading the data from q2_dataset.csv

	df=load_data() 
	
	# Considering the columns volume,open,high,low
	training_set = df.iloc[:, 2:].values 

	# Creating a custom data structure with 3 timesteps and 1 output
	X_train_data = []
	y_train_data = []
	for i in range(3, 1259):
	    X_train_data.append(training_set[i-3:i, :])
	    y_train_data.append(training_set[i, 1])
	X_train_data, y_train_data = np.array(X_train_data), np.array(y_train_data)

	# RANDOMLY SPLITTING THE DATASET
	X_train, X_test, y_train, y_test = train_test_split(X_train_data, y_train_data, test_size = 0.30, random_state = 42)

	''' FOLLOWING IS THE CODE FOR STORING THE DATASET IN TRAIN AND TEST CSV 

	# CODE FOR STORING THE DATASET IN TRAIN AND TEST CSV
	flatten_list=[]
	for i in range (X_train.shape[0]):
	    x_flat=np.ravel(X_train[i])
	    flatten_list.append(x_flat)

	df_1_train= pd.DataFrame(data=flatten_list)
	df_2_train=pd.DataFrame(data=y_train,columns=["Labels"])
	train_frames = [df_1_train,df_2_train]
	tot_df_train= pd.concat(train_frames,axis=1)

	# CODE FOR STORING THE TESTING DATASET 
	flatten_list_test=[]
	for i in range (X_test.shape[0]):
	    x_flat_test=np.ravel(X_test[i])
	    flatten_list_test.append(x_flat_test)

	df_1_test= pd.DataFrame(data=flatten_list_test)
	df_2_test=pd.DataFrame(data=y_test,columns=["Labels"])
	test_frames = [df_1_test,df_2_test]
	tot_df_test= pd.concat(test_frames,axis=1)

	# WRITING TO CSV FILES
	tot_df_train.to_csv("data/train_data_RNN.csv",index=False)
	tot_df_test.to_csv("data/test_data_RNN.csv",index=False)
	'''
	

	# READING THE DATA
	df_train= pd.read_csv("data/train_data_RNN.csv")

	# PREPROCESSING FOR DATA AND LABELS
	sc = MinMaxScaler(feature_range = (0, 1))
	training_set_scaled = sc.fit_transform(df_train.iloc[:,:12])
	training_y_label_scaled=sc.fit_transform(np.array(df_train["Labels"]).reshape(-1,1))

	# Saving the scaler for using the same normalization for testing

	joblib.dump(sc, "models/sc_RNN.save")
	

	# RESHAPING THE DATA 
	X_train_data=training_set_scaled.reshape(-1,3,4)
	y_train_data=training_y_label_scaled
	

	# 2. Train your network

	regressor= model(X_train)
	print(regressor.summary())
	# Fitting the RNN to the Training set
	regressor.fit(X_train_data, y_train_data, epochs = 50, batch_size = 32,validation_split=0.2)


	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss

	# 3. Save your model
	regressor.save("models/20859891_RNN_model.hdf5")

	




if __name__ == "__main__": 
	
	main() 

	

	

	
	

	
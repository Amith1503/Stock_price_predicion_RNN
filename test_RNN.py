# import required packages
	
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import os
import sys
from keras.models import load_model
from sklearn.metrics import mean_squared_error
# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

def plot_graph(y_test_data,predicted_stock_price):
	plt.plot(y_test_data, color = 'red', label = 'Real Stock Price')
	plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
	plt.title('Stock Price Prediction BOTH Y_true and Y_predicted')
	plt.xlabel('Time')
	plt.ylabel('Stock Price')
	plt.legend()
	plt.show()

def plot_graph_real(y_test_data):
	plt.plot(y_test_data, color = 'red', label = 'Real Stock Price')
	plt.title('Stock Price Prediction only Y_true')
	plt.xlabel('Time')
	plt.ylabel('Stock Price')
	plt.legend()
	plt.show()

def plot_graph_predicted(predicted_stock_price):
	plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
	plt.title('Stock Price Prediction only Y_predicted')
	plt.xlabel('Time')
	plt.ylabel('Stock Price')
	plt.legend()
	plt.show()

def main():
	# 1. Load your saved model
	regressor_model=load_model("models/20859891_RNN_model.hdf5")

	# 2. Load your testing data
	df_test= pd.read_csv("data/test_data_RNN.csv")

	# Loading the normalization which was done for training set
	scaler = joblib	.load("models/sc_RNN.save")

	# Normalizing the testing set 
	test_set_scaled = scaler.fit_transform(df_test.iloc[:,:12])
	test_y_label_scaled=scaler.fit_transform(np.array(df_test["Labels"]).reshape(-1,1))

	# RESHAPING THE TEST SET
	X_test_data=test_set_scaled.reshape(-1,3,4)
	y_test_data=test_y_label_scaled

	# 3. Run prediction on the test data and output required plot and loss
	predicted_stock_price = regressor_model.predict(X_test_data)
	loss= mean_squared_error(y_test_data,predicted_stock_price)
	print("The test loss is ",loss)
	

	
	# PLOTTING THE GRAPH
	plot_graph(y_test_data,predicted_stock_price)
	plot_graph_real(y_test_data)
	plot_graph_predicted(predicted_stock_price)


if __name__ == "__main__":
	
	main()
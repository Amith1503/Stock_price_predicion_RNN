# Stock_price_predicion_RNN

Dataset for stock price prediction for 5 years with one sample per day
(q2_dataset.py). Create a Recurrent Neural Network using the machine learning platform of your choice
(PyTorch, Tensorflow, or Keras) to predict the next day opening price using the past 3 days Open, High, and
Low prices and volume. Therefore, each sample will have (4*3 = ) 12 features.

The following steps are performed:-

1. In train_RNN.py file: Before any preprocessing, create the dataset by using the latest 3 days as
the features and the next day’s opening price as the target. Randomize the created data and split it
into 70% training and 30% testing and save it to ‘train_data_RNN.csv’ and ‘test_data_RNN.csv’ in
the data directory respectively. Keep this code in your file but comment it out!

2. Populate the file train_RNN.py so that it reads train_data_RNN.csv, preprocesses the data, and
trains your RNN network. After training, the file should save your model with the name
‘YOUR_ID_RNN_model’ in the models directory. You can use any extension you want as you will
load your own model. 

3. Populate the file test_RNN.py so that it reads test_data_RNN.csv and runs the prediction
model. Print the loss on your test data and show a plot of the true and predicted values (use
appropriate labels for the axes and legend).

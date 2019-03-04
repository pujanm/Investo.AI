import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf

seq_len = 20
n_steps = seq_len-1 
n_inputs = 5 
n_neurons = 200 
n_outputs = 5
n_layers = 2
learning_rate = 0.001
batch_size = 50
n_epochs = 300 

dataset = pd.read_csv('./gold/gold.csv')
x_test = dataset.iloc[:, :-1]
y_test = dataset.iloc[:, -1]
print(x_test.shape)

def normalize_data(dataset):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    dataset['Open'] = min_max_scaler.fit_transform(dataset.Open.values.reshape(-1,1))
    dataset['High'] = min_max_scaler.fit_transform(dataset.High.values.reshape(-1,1))
    dataset['Low'] = min_max_scaler.fit_transform(dataset.Low.values.reshape(-1,1))
    dataset['Close'] = min_max_scaler.fit_transform(dataset['Close'].values.reshape(-1,1))
    return dataset

# function to create train, validation, test data given stock data and sequence length
def load_data(stock, seq_len):
    data_raw = stock.as_matrix() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len): 
        data.append(data_raw[index: index + seq_len])
    
    data = np.array(data);
    
    x_test = data[:,:-1,:]
    y_test = data[:,-1,:]
    
    return [x_test, y_test]

# choose one stock
# dataset.drop(['Volume'],1,inplace=True)

cols = list(dataset.columns.values)
print('df_stock.columns.values = ', cols)

# normalize stock
df_stock_norm = dataset.copy()
df_stock_norm = normalize_data(df_stock_norm)

# create train, test data
seq_len = 20 # choose sequence length
x_test, y_test = load_data(df_stock_norm, seq_len)



tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])
# use Basic RNN Cell
layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
          for layer in range(n_layers)]
                          
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons]) 
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
outputs = outputs[:,n_steps-1,:] # keep only last output of sequence
                                              
# run graph
saver = tf.train.Saver()

with tf.Session() as sess: 
    saver.restore(sess, "./gold/model.ckpt")
    y_test_pred = sess.run(outputs, feed_dict={X: x_test})
    
print(y_test_pred)
print(y_test)
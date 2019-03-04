import bulbea as bb
from bulbea.learn.evaluation import split
# from bulbea.learn.models import RNN
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

import os

# bulbea
from six import with_metaclass

from keras.models import Sequential
from keras.layers import recurrent
from keras.layers import core
from keras.models import load_model
from bulbea.learn.models import Supervised

class ANN(Supervised):
    pass

class RNNCell(object):
    RNN  = recurrent.SimpleRNN
    GRU  = recurrent.GRU
    LSTM = recurrent.LSTM

class RNN(ANN):
    def __init__(self, sizes,
                 cell       = RNNCell.LSTM,
                 dropout    = 0.2,
                 activation = 'linear',
                 loss       = 'mse',
                 optimizer  = 'rmsprop'):
        self.model = Sequential()
        self.model.add(cell(
            input_dim        = sizes[0],
            output_dim       = sizes[1],
            return_sequences = True
        ))

        for i in range(2, len(sizes) - 1):
            self.model.add(cell(sizes[i], return_sequences = False))
            self.model.add(core.Dropout(dropout))

        self.model.add(core.Dense(output_dim = sizes[-1]))
        self.model.add(core.Activation(activation))

        self.model.compile(loss = loss, optimizer = optimizer)

    def fit(self, X, y, *args, **kwargs):
        return self.model.fit(X, y, *args, **kwargs)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, file):
        return self.model.save(file)

    def load(self, file):

        return load_model(file)

# Saving a file in python

# file = "f.txt"
# path = os.path.join("test/", file)
# file = open(path, "w")

stock_list = ['ADM', 'FLS', 'ADI', 'ADBE', 'CB', 'FAST', 'ABBV', 'SJM', 'DHI', 'ACN', 'AAP', 'ZTS', 'SIG', 'CME', 'XOM', 'CMCSA', 'ABC', 'ABT', 'JBHT', 'DHR', 'GOOGL', 'AAL', 'XLNX', 'MMC', 'RRC', 'ROST', 'GPC', 'AAPL', 'DLTR', 'WM']

for i in stock_list:
    stock = bb.Share('Wiki', i)

    Xtrain, Xtest, ytrain, ytest = split(stock, 'Close', normalize = True)
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
    Xtest  = np.reshape( Xtest, ( Xtest.shape[0],  Xtest.shape[1], 1))
    rnn = RNN([1, 100, 100, 1]) # number of neurons in each layer
    rnn.fit(Xtrain, ytrain)

    rnn.save("models/" + i + ".h5")
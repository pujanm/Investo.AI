import os

os.environ["BULBEA_QUANDL_API_KEY"] = 'kyBMzL7tVaYQrYwmwW-m'

import bulbea as bb
share = bb.Share('WIKI', 'GOOGL')
# print(share.data)

from bulbea.learn.evaluation import split
Xtrain, Xtest, ytrain, ytest = split(share, 'Close', normalize = True)

import numpy as np
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
Xtest  = np.reshape( Xtest, ( Xtest.shape[0],  Xtest.shape[1], 1))

from bulbea.learn.models import RNN
rnn = RNN([1, 100, 100, 1]) # number of neurons in each layer
rnn.fit(Xtrain, ytrain)

from sklearn.metrics import mean_squared_error
p = rnn.predict(Xtest)
mean_squared_error(ytest, p)
import matplotlib.pyplot as pplt
# pplt.plot(ytest[:-500])
# pplt.plot(p)
# pplt.show()

share.plot(bollinger_bands = True, period = 100, bandwidth = 2)
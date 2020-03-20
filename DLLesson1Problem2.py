import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#read in new file
dataset = pd.read_csv("breastcancer.csv", header=0)

#replace non-numeric data with numeric data
targetMap = {'B': 0, 'M': 1}
dataset['diagnosis'] = dataset['diagnosis'].map(targetMap)

#drop the ID column and split into x and y datasets
dataset1 = dataset.drop(['id'], axis=1)
xdata = dataset1.drop(['diagnosis'], axis=1).values
ydata = dataset1['diagnosis'].values

#train test split
X_train, X_test, Y_train, Y_test = train_test_split(xdata, ydata,
                                                    test_size=0.25, random_state=87)
#deep learning part
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=30, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, #fit data to model
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test)) #results

#accuracy: [0.3422813026325686, 0.8881118893623352]

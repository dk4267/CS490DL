import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

dataset = pd.read_csv("breastcancer.csv", header=0)

targetMap = {'B': 1, 'M': 2}
dataset['diagnosis'] = dataset['diagnosis'].map(targetMap)

dataset1 = dataset.drop(['id'], axis=1)
xdata = dataset1.drop(['diagnosis'], axis=1).values
ydata = dataset1['diagnosis'].values

#perform standard scaler on x dataset
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
sc.fit(xdata)
X_scaled_array = sc.transform(xdata)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled_array, ydata,
                                                    test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=30, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test))

#new accuracy: [-271.71047461449683, 0.6503496766090393]
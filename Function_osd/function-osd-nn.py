# mlp for multiclass classification
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from keras.utils import np_utils
import json


# load the dataset
path = 'function-osd.csv'
df = read_csv(path, header=None)
# split into input and output columns
X, y = df.values[1:,:2] , df.values[1:, -1]
# ensure all data are floating point values
X = X.astype('float32')
#X[:,[0,1,2,3]]= np.log(X[:,[0,1,2,3]])/6.0
#X[:,[1]] =5*X[:,[1]]

# encode strings to integer
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
y = encoded_Y

# split into train and test datasets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#X_train , X_test , y_train , y_test = X,X,y,y
# X_dum = np.repeat(X,[3 for i in range(14)],axis = 0)
# y_dum = np.repeat(y,[3 for i in range(14)],axis = 0)

X_train, X_test, y_train, y_test = X,X,y,y
# determine the number of input features
n_features = 2
X_test,y_test = X,y
# define model
model = Sequential()
model.add(Dense(512, activation='sigmoid', kernel_initializer='he_normal',input_shape=(n_features,)))
# model.add(Dense(512, activation='sigmoid', kernel_initializer='he_normal'))
# model.add(Dense(512, activation='sigmoid', kernel_initializer='he_normal'))
# model.add(Dense(512, activation='sigmoid', kernel_initializer='he_normal'))

# model.add(Dense(256, activation='sigmoid', kernel_initializer='he_normal'))
# model.add(Dense(128, activation='sigmoid', kernel_initializer='he_normal'))

#model.add(Dense(128, activation='sigmoid', kernel_initializer='he_normal', input_shape=(n_features,)))
#model.add(Dense(64, activation='sigmoid', kernel_initializer='he_normal'))

model.add(Dense(12, activation='softmax'))
# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(X_train, y_train, epochs=400, batch_size=1,verbose=None)
# evaluate the model
loss, acc = model.evaluate(X_test, y_test)
print('Test Accuracy: %.3f' % acc)
# make a prediction
# row = [548,	182,139,29,	0,	0,	0,	0,	0,	0]
# yhat = [argmax(model.predict(row)) for row in X_test]  
# print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
for i in range(12):
    print(argmax(model.predict([X])[i]))
    
model.save("function-osd")
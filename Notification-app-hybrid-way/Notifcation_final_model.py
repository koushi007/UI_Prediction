from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from keras.utils import np_utils
# load the dataset
path = 'Notification_final.csv'
df = read_csv(path, header=None)
# split into input and output columns
X, y = df.values[1:,:-1] , df.values[1:, -1]
# ensure all data are floating point values
X = X.astype('float32')
X[:,[0,1,2,3]]= np.log(X[:,[0,1,2,3]])/6.0

# encode strings to integer
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
y = np_utils.to_categorical(encoded_Y)
# split into train and test datasets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train , X_test , y_train , y_test = X,X,y,y
n_features = X_train.shape[1]
X_test,y_test = X,y
# define model
model = Sequential()
model.add(Dense(1024, activation='relu', kernel_initializer='he_normal'))
#model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(256, activation='sigmoid', kernel_initializer='he_normal'))
# model.add(Dense(128, activation='sigmoid', kernel_initializer='he_normal'))

#model.add(Dense(128, activation='sigmoid', kernel_initializer='he_normal', input_shape=(n_features,)))
#model.add(Dense(64, activation='sigmoid', kernel_initializer='he_normal'))

model.add(Dense(14, activation='softmax'))
# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(X_train, y_train, epochs=2500, batch_size=1,verbose=None)
# evaluate the model
loss, acc = model.evaluate(X_test, y_test)
print('Test Accuracy: %.3f' % acc)
# make a prediction
# row = [548,	182,139,29,	0,	0,	0,	0,	0,	0]
# yhat = [argmax(model.predict(row)) for row in X_test]  
# print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
for i in range(14):
    print(argmax(model.predict([X])[i]))
    
model.save('Notification-app-final')
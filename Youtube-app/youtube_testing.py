from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from keras.utils import np_utils
from tensorflow import keras
# load the dataset
path = 'youtube_test_ro.csv'
df = read_csv(path, header=None)
# # split into input and output columns
X, y = df.values[1:,1:-1] , df.values[1:, -1]
# ensure all data are floating point values
X = X.astype('float32')
X = np.log(X+4)/6.0

# encode strings to integer
# encoder = LabelEncoder()
# encoder.fit(y)
# encoded_Y = encoder.transform(y)
# y = np_utils.to_categorical(encoded_Y)
# split into train and test datasets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
model2 = keras.models.load_model('Youtube-toy.h5')

#loss, acc = model.evaluate(X, y)
#print('Test Accuracy: %.3f' % acc)
for i in range(9):
    print(np.argmax(model2.predict(X)[i]))

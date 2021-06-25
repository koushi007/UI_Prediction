from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import tensorflow as tf
import pandas as pd

class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs["accuracy"]
        if val_acc >= self.threshold:
            self.model.stop_training = True
# load the dataset
path = 'combined_toy_train_ro.csv'
df = read_csv(path, header=None)
# split into input and output columns
X, y = df.values[1:,1:-1] , df.values[1:, -1]
# ensure all data are floating point values
X = X.astype('float32')
X[:,:240] = np.log(X[:,:240]+4)/6.0

# encode strings to integer
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
y = np_utils.to_categorical(encoded_Y)

df_encoder = pd.DataFrame(list(encoder.classes_),columns=['states'])
df_encoder.to_csv('Combined_states_map.csv')
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#X_train , X_test , y_train , y_test = X,X,y,y
n_features = X_train.shape[1]


model = Sequential()
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
#model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(1024, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(1024, activation='relu', kernel_initializer='he_normal'))

model.add(Dense(y.shape[1], activation='softmax'))
# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fit the model

#es = EarlyStopping(monitor='accuracy', mode='max', verbose=1,baseline = 0.98)
my_callback = MyThresholdCallback(threshold=0.98)
model.fit(X_train, y_train, epochs=5000, batch_size=12,validation_split = 0.1,verbose=None,callbacks=[my_callback])
# evaluate the model
loss, acc = model.evaluate(X_test, y_test)
print('Test Accuracy: %.3f' % acc)

# for i in range(72):
#     print(argmax(model.predict([X])[i]))
    
model.save('Combined-toy{}'.format(acc))
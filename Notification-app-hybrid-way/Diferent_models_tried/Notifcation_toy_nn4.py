import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers

dataframe = pd.read_csv('Notification_toy.csv')

dataframe.head()

train, test = dataframe
train, val = train_test_split(train, test_size=0.2)
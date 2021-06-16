# import pandas as pd

# pd_dict = dict()
# pd_dict["x"] = [i[0] for i in focus_box_list]
# pd_dict["y"] = [i[1] for i in focus_box_list]
# pd_dict["w"] = [i[2] for i in focus_box_list]
# pd_dict["h"] = [i[3] for i in focus_box_list]
# pd_dict["name"] = file_ordering
# pd_dict["cam.png"] = [1 if i[:3]=="cam" else 0 for i in file_ordering]
# pd_dict["usb.png"] = [1 if i[:3]=="usb" else 0 for i in file_ordering]
# pd_dict["samsung_health.png"] = [1 if i[:2]=="sh" else 0 for i in file_ordering]
# pd_dict["smart_things.png"] = [1 if i[:2]=="st" else 0 for i in file_ordering]
# pd_dict["network_disconnected.png"] = [1 if i=="network#main#ok-2.png" else 0 for i in file_ordering]
# pd_dict["network_connected.png"] = [1 if i=="network#main#ok-1.png" else 0 for i in file_ordering]

# #pd_dict.to_csv("Notification_toy.csv")

# pd.DataFrame(pd_dict).to_csv("Notification_toy.csv")


# multi-class classification with Keras
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
# load dataset
dataframe = pandas.read_csv("Notification_toy.csv", header=None)
dataset = dataframe.values
X = dataset[1:,0:10].astype(float)
Y = dataset[1:,10]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=10, activation='relu'))
	model.add(Dense(14, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
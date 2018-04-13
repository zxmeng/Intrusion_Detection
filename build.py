from keras.models import Sequential
from keras.layers import Dense

def build_model():
	model = Sequential()
	model.add(Dense(12, activation='sigmoid', input_dim=6))
	model.add(Dense(18, activation='sigmoid'))
	model.add(Dense(12, activation='sigmoid'))
	model.add(Dense(6, activation='sigmoid'))
	model.add(Dense(3, activation='sigmoid'))
	model.add(Dense(2, activation='softmax'))

	return model
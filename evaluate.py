import numpy as np
from numpy import genfromtxt
import keras
from keras.models import load_model
import time
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score


def scale(x_train, x_test):
	x_train = x_train.astype(float)
	x_test = x_test.astype(float)

	for i in range(6):
		train_max = x_train[:,i].max()
		x_train[:,i] /= train_max
		x_test[:,i] /= train_max

		train_mean = np.mean(x_train[:,i])
		x_train[:,i] -= train_mean
		x_test[:,i] -= train_mean

	return x_train, x_test


start_time = time.time()

print "loading data..."
train = genfromtxt("train.csv", delimiter=',')
test = genfromtxt("test.csv", delimiter=',')

x_train = train[:,:-1]
y_train = train[:,-1]

x_test = test[:,:-1]
y_test = test[:,-1]

x_train, x_test = scale(x_train, x_test)

print "loading model..."
MODEL= "../model/second_0.h5"
model = load_model(MODEL)

prob = np.asarray(model.predict(x_test))
row, col = prob.shape
y_pred = np.zeros((row)).astype(int)
for i in range(row):
	if prob[i,0] < prob[i,1]:
		y_pred[i] = 1

y_true = y_test.astype(int)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print('test accuracy: ', accuracy)
print('test precision: ', precision)
print('test recall: ', recall)
print('test fmeasure: ', f1)

print "--- %s seconds ---" % (time.time() - start_time)



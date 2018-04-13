import numpy as np
from numpy import genfromtxt
import keras
from keras.optimizers import Adam
from keras.models import load_model
import build
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

y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

print "training..."
model = build.build_model()

# training and testing
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', metrics.precision, metrics.recall, metrics.fmeasure])
for i in range(3):
	history = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test), shuffle=True)
	# save trained model for future usage
	model.save("../model/model_" + str(i) + ".h5")

# evaluate model
score = model.evaluate(x_test, y_test)
print('test loss: ', score[0])
print('test accuracy: ', score[1])

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



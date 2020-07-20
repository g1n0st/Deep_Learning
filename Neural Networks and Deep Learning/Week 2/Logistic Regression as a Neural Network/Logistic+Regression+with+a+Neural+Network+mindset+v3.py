import numpy as np
import matplotlib.pyplot  as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

def sigmoid(z):
	s = 1 / (1 + np.exp(-z))
	return s

def initialize(dim):
	w = np.zeros((dim, 1))
	b = 0

	return w, b

def propagate(w, b, X, Y):
	m = X.shape[1]
	A = sigmoid(w.T.dot(X) + b)
	cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
	dw = (1 / m)  * X.dot((A - Y).T)
	db = (1 / m)  * np.sum(A - Y)

	assert(dw.shape == w.shape)
	assert(db.dtype == float)
	cost = np.squeeze(cost)
	assert(cost.shape == ())

	grads = {"dw" : dw, "db" : db}

	return grads, cost

def optimize(w, b, X, Y, it, rate, pr = False):
	costs = []

	for i in range(it):
		grads, cost = propagate(w, b, X, Y)

		dw = grads["dw"]
		db = grads["db"]

		w = w - rate * dw
		b = b - rate * db

		if i % 100 == 0:
			costs.append(cost)
		if pr and i % 100 == 0:
			print ("Cost after iteration %i: %f" % (i, cost))

	params = {"w" : w, "b" : b}
	grads = {"dw" : dw, "db" : db}

	return params, grads, costs

def predict(w, b, X):
	m = X.shape[1]
	prey = np.zeros((1, m))
	w = w.reshape(X.shape[0], 1)

	A = sigmoid(w.T.dot(X) + b)

	for i in range(A.shape[1]):
		prey [0, i] = 1 if A[0, i] > 0.5 else 0
	
	assert(prey.shape == (1, m))

	return prey

def model(x_train, y_train, x_test, y_test, it = 2000, rate = 0.5, cost = False):
	w, b = initialize(x_train.shape[0])
	params, grads, costs = optimize(w, b, x_train, y_train, it, rate, cost)
	
	w = params["w"]
	b = params["b"]

	prey_train = predict(w, b, x_train)
	prey_test = predict(w, b, x_test)

	print("test accuracy: {} %".format(100 - np.mean(np.abs(prey_train - y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(prey_test - y_test)) * 100))

	return prey_test

prey_test = model(train_set_x, train_set_y, test_set_x, test_set_y, it = 20000, rate = 0.004, cost = True)

index = 1
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
print("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" + str(prey_test[0, index]) + "\" picture. ")

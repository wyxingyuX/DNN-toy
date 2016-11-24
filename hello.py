# -*- coding:utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers.recurrent import LSTM

iris = datasets.load_iris()
X = iris.data[:, :]  # we only take the first two features.
Y = iris.target
label_set = set(list(Y))
Y = to_categorical(Y, len(label_set))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=128)
print X.shape, Y.shape

print len(X[0]), len(label_set)

model = Sequential()
model.add(LSTM(input_dim=len(X[0]), output_dim=len(X[0]), return_sequences=False))
model.add(Dense(output_dim=len(label_set)))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, Y_train, nb_epoch=30,batch_size=1)
loss_and_metrics = model.evaluate(X_test, Y_test)

# -*- coding:utf-8 -*-
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from  keras.layers.core import Dropout
from keras.layers.core import Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import AveragePooling1D
from keras.layers import Flatten
from keras.utils.visualize_util import plot
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D


def read_X_y(file):
    print "strat read ...."
    f = open(file)
    X = []
    y = []
    for line in f:
        elms = line.split()
        X.append([float(e) for e in elms[1:-1]])
        y.append(float(elms[-1]))
    f.close()
    print "read complete."
    return X, y


if __name__ == "__main__":
    print "hello"
    X_train, y_train = read_X_y(
        "/media/wyjn/Work/ExpData/DataFromOther/security detection/kddcup99/preprocess/kddcup.data_10_percent_corrected.train")
    X_train, y_train = shuffle(X_train, y_train, random_state=128)
    X_train = X_train[0:20000]
    y_train = y_train[0:20000]
    y_train = np_utils.to_categorical(y_train, 24)

    X_test, y_test = read_X_y(
        "/media/wyjn/Work/ExpData/DataFromOther/security detection/kddcup99/preprocess/corrected.test")
    X_test, y_test = shuffle(X_test, y_test, random_state=128)
    X_test = X_test[0:5000]
    y_test = y_test[0:5000]
    y_test = np_utils.to_categorical(y_test, 24)

    model = Sequential()
    # model.add(Dense(output_dim=34, input_dim=len(X_train[0])))
    # model.add(Activation("sigmoid"))
    # model.add(Dense(output_dim=24))
    # model.add(Activation("softmax"))

    model.add(Dense(output_dim=80, input_dim=len(X_train[0]), activation="sigmoid"))
    model.add(Reshape((10, 8)))
    model.add(Convolution1D(nb_filter=20, filter_length=3, border_mode='valid'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(output_dim=24))
    model.add(Activation("softmax"))

    # model.add(Reshape((4, 5, 4)))
    # model.add(Convolution2D(nb_filter=3, nb_row=3, nb_col=3, border_mode="valid"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), border_mode="valid"))
    # model.add(Flatten())
    # model.add(Dense(output_dim=24, activation="softmax"))

    plot(model, show_shapes=True, to_file='./model.png')
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    model.fit(X_train, y_train, nb_epoch=10, batch_size=1, validation_split=0.2, callbacks=[early_stopping])
    score = model.evaluate(X_test, y_test)
    print score

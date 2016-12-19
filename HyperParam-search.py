# -*- coding:utf-8 -*-
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.layers.core import Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from keras.utils.visualize_util import plot
from test import read_train_test
from keras.models import Sequential


def create_model(neurons=80):
    print "nn—cnn-model"
    np.random.seed(64)

    model = Sequential()
    model.add(Dense(output_dim=neurons, input_dim=122, activation="tanh"))
    model.add(Reshape((20, neurons / 20)))
    model.add(Convolution1D(nb_filter=8, filter_length=8, border_mode='valid', activation="tanh"))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Convolution1D(nb_filter=16, filter_length=4, border_mode='valid', activation="tanh"))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(output_dim=2, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model


if __name__ == "__main__":
    print "hello"

    # fix random seed for reproducibility
    seed = 64
    np.random.seed(seed)

    # load data
    train_file = "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/KDDTrain+_20Percent.txt"
    test_file = "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/KDDTest-21.txt"
    X_train, X_test, y_train, y_test, y_train_category, y_test_category = read_train_test(train_file, test_file)

    # create model
    model = KerasClassifier(build_fn=create_model, verbose=0, batch_size=256, nb_epoch=200)
    # define the grid search parameters

    neurons = [20 * 2, 20 * 3, 20 * 4, 20 * 5, 20 * 6, 20 * 7, 20 * 8, 20 * 9, 20 * 10]
    param_grid = dict(neurons=neurons)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=8)
    grid_result = grid.fit(X_train, y_train_category)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.grid_scores_:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

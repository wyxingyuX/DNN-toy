# -*- coding:utf-8 -*-
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import naive_bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
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
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras import regularizers
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.lda import LDA
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def my_report(y_test, y_pred, start_label=1):
    count_mat = np.zeros((2, 2))
    for i in range(len(y_test)):
        y_label_pred = y_pred[i] - start_label
        y_label_test = y_test[i] - start_label
        row = int(y_label_pred)
        col = int(y_label_test)
        count_mat[row][col] += 1
    rate_mat = np.zeros((2, 3))
    for i in range(len(count_mat)):
        precision = (1.0 * count_mat[i][i]) / (count_mat[i][0] + count_mat[i][1])
        recall = (1.0 * count_mat[i][i]) / (count_mat[0][i] + count_mat[1][i])
        rate_mat[i][0] = precision
        rate_mat[i][1] = recall
        rate_mat[i][2] = 2.0 * precision * recall / (precision + recall)

    print "precsion \trecall \tF1"
    for i in range(len(rate_mat)):
        print str(i + 1), "  ",
        for j in range(len(rate_mat[i])):
            print str(rate_mat[i][j]), "  ",
        print ""


def read_X_y(file, feature_end_diem=-1, label_index=-1):
    print "strat read ...."
    f = open(file)
    X = []
    y = []
    for line in f:
        elms = line.split()
        X.append([float(e) for e in elms[0:feature_end_diem]])
        y.append(int(elms[label_index]))
    f.close()
    print "read complete."
    return X, y


def read_train_test():
    # X, y = read_X_y("/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/KDDTrain+_20Percent.txt",
    #                 feature_end_diem=-3, label_index=-1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=128)

    label_idx = -1
    label_num = 2
    if label_idx == -1:
        label_num = 2
    if label_idx == -2:
        label_num = 5
    if label_idx == -3:
        label_num = 23

    X_train, y_train = read_X_y("/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/KDDTrain+_20Percent.txt",
                                feature_end_diem=-3, label_index=label_idx)
    X_train, y_train = shuffle(X_train, y_train, random_state=128)
    scalar = preprocessing.MinMaxScaler()
    X_train = scalar.fit_transform(X_train)

    X_test, y_test = read_X_y("/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/KDDTest+.txt",
                              feature_end_diem=-3, label_index=label_idx)
    X_test, y_test = shuffle(X_test, y_test, random_state=128)
    X_test = scalar.transform(X_test)

    y_train_category = np_utils.to_categorical([(label - 1) for label in y_train], label_num)
    y_test_category = np_utils.to_categorical([(label - 1) for label in y_test], label_num)

    return X_train, X_test, y_train, y_test, y_train_category, y_test_category


def report_clf(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print clf.__class__, str(score)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))
    print "\n"


def base_line_clf(X_train, X_test, y_train, y_test):
    print "base_line_clf"

    report_clf(LogisticRegression(), X_train, y_train, X_test, y_test)
    report_clf(svm.LinearSVC(), X_train, y_train, X_test, y_test)
    report_clf(RandomForestClassifier(n_estimators=50), X_train, y_train, X_test, y_test)
    report_clf(naive_bayes.MultinomialNB(), X_train, y_train, X_test, y_test)


def bp(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(output_dim=200, input_dim=len(X_train[0])))
    model.add(Activation("sigmoid"))
    model.add(Dense(output_dim=2))
    model.add(Activation("softmax"))
    plot(model, show_shapes=True, to_file='./model.png')
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=1)

    model.fit(X_train, y_train, nb_epoch=5, batch_size=1, validation_split=0.2, callbacks=[early_stopping])
    score = model.evaluate(X_test, y_test)
    print score

    y_pred = model.predict_classes(X_test)
    my_report(category_2_label(y_test), y_pred, start_label=0)


def category_2_label(y_category):
    y = []
    for c in y_category:
        y.append(list(c).index(1))
    return y


def smr(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(output_dim=2, input_dim=len(X_train[0])))
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=1)

    model.fit(X_train, y_train, nb_epoch=5, batch_size=1, validation_split=0.2, callbacks=[early_stopping])
    score = model.evaluate(X_test, y_test)
    print score

    y_pred = model.predict_classes(X_test)
    my_report(category_2_label(y_test), y_pred, start_label=0)


def nn_cnn(X_train, X_test, y_train, y_test):
    model = Sequential()

    model.add(Dense(output_dim=80, input_dim=len(X_train[0]), activation="tanh"))
    model.add(Reshape((8, 10)))
    model.add(Convolution1D(nb_filter=5, filter_length=5, border_mode='valid'))
    model.add(MaxPooling1D(pool_length=2))
    # model.add(Convolution1D(nb_filter=3, filter_length=3, border_mode='valid'))
    # model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(output_dim=2))
    model.add(Activation("softmax"))

    plot(model, show_shapes=True, to_file='./model.png')
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    model.fit(X_train, y_train, nb_epoch=30, batch_size=1, validation_split=0.2, callbacks=[early_stopping])

    score = model.evaluate(X_test, y_test)
    print score

    y_pred = model.predict_classes(X_test)
    my_report(category_2_label(y_test), y_pred, start_label=0)


def pure_cnn(X_train, X_test, y_train, y_test):
    input_img = Input(shape=(len(X_train[0]),))
    f_input = Reshape((122, 1))(input_img)
    c_conv1 = Convolution1D(nb_filter=5, filter_length=100, border_mode='valid', activation="sigmoid")(f_input)
    m_pool1 = MaxPooling1D(pool_length=2)(c_conv1)
    c_conv2 = Convolution1D(nb_filter=3, filter_length=3, border_mode='valid', activation="sigmoid")(m_pool1)
    m_pool2 = MaxPooling1D(pool_length=2)(c_conv2)
    flatten = Flatten()(m_pool2)
    # flatten = Flatten()(m_pool1)

    out_dense = Dense(output_dim=2, activation="softmax")(flatten)

    model = Model(input=input_img, output=out_dense)

    plot(model, show_shapes=True, to_file='./model.png')
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # best_model_filepath = "./model.best.hdf5"
    # model_check_point = ModelCheckpoint(filepath=best_model_filepath, save_best_only=True, monitor="val_loss",
    #                                     mode="min")
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.fit(X_train, y_train, nb_epoch=30, batch_size=1, validation_split=0.2, callbacks=[early_stopping])

    # model = load_model(best_model_filepath)
    score = model.evaluate(X_test, y_test)
    print score

    y_pred_prob = model.predict(X_test)
    y_pred = y_prob_2_y_class(y_pred_prob)
    my_report(category_2_label(y_test), y_pred, start_label=0)


def y_prob_2_y_class(y_prob, start_label=0):
    y_class = []
    for probs in y_prob:
        y_class.append(probs.argmax())
    return y_class


def auto_encoder(X_train, X_test):
    # this is the size of our encoded representations
    encoding_dim = 80

    # this is our input placeholder
    input_img = Input(shape=(len(X_train[0]),))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='tanh')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(len(X_train[0]), activation='tanh')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input=input_img, output=decoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(X_train, X_train,
                    nb_epoch=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(X_train, X_train))

    X_test_encoded = autoencoder.predict(X_test)
    X_train_encoded = autoencoder.predict(X_train)
    return X_train_encoded, X_test_encoded


def sparse_auto_encoder(X_train, X_test):
    encoding_dim = 80
    input_img = Input(shape=(len(X_train[0]),))
    # add a Dense layer with a L1 activity regularizer
    encoded = Dense(encoding_dim, activation='relu',
                    activity_regularizer=regularizers.activity_l1(10e-5))(input_img)
    decoded = Dense(len(X_train[0]), activation='sigmoid')(encoded)

    autoencoder = Model(input=input_img, output=decoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(X_train, X_train,
                    nb_epoch=100,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(X_train, X_train))

    X_test_encoded = autoencoder.predict(X_test)
    X_train_encoded = autoencoder.predict(X_train)
    return X_train_encoded, X_test_encoded


def pca_extract(X_train, X_test):
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca


def nmf_extract(X_train, X_test):
    model = NMF(n_components=80, init='random', random_state=0)
    X_train_nmf = model.fit_transform(X_train)
    X_test_nmf = model.transform(X_test)
    return X_train_nmf, X_test_nmf


def lda_extract(X_train, X_test, y_train, y_test):
    clf = LDA(n_components=30)
    clf.fit(X_train, y_train)
    X_train_lda = clf.transform(X_train)
    X_test_lda = clf.transform(X_test)
    return X_train_lda, X_test_lda


if __name__ == "__main__":
    print "base_line_clf"
    X_train, X_test, y_train, y_test, y_train_category, y_test_category = read_train_test()

    # X_train_encoded, X_test_encoded = auto_encoder(X_train, X_test)
    # X_train_encoded, X_test_encoded = sparse_auto_encoder(X_train, X_test)
    # X_train_encoded, X_test_encoded = pca_extract(X_train, X_test)
    # X_train_encoded, X_test_encoded = nmf_extract(X_train, X_test)
    # X_train_encoded, X_test_encoded = lda_extract(X_train, X_test, y_train, y_test)
    # report_clf(svm.LinearSVC(), X_train_encoded, y_train, X_test_encoded, y_test)
    # base_line_clf(X_train, X_test, y_train, y_test)
    # bp(X_train, X_test, y_train_category, y_test_category)
    # nn_cnn(X_train, X_test, y_train_category, y_test_category)
    pure_cnn(X_train, X_test, y_train_category, y_test_category)
    # smr(X_train_encoded, X_test_encoded, y_train_category, y_test_category)
    # smr(X_train, X_test, y_train_category, y_test_category)

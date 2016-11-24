# -*- coding:utf-8 -*-
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from keras.optimizers import SGD

# model = Sequential()
#
# model.add(Dense(output_dim=64, input_dim=100))
# model.add(Activation("relu"))
# model.add(Dense(output_dim=10))
# model.add(Activation("softmax"))
#
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

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

def score_clf(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print clf.__class__,str(score),"\n"


if __name__ == "__main__":
    print "hello"
    clf = RandomForestClassifier()

    X_train, y_train = read_X_y(
        "F:\ExpData\DataFromOther\security detection\kddcup99\\preprocess\\kddcup.data_10_percent_corrected.train")
    X_train, y_train = shuffle(X_train, y_train,random_state=128)
    X_train = X_train[0:20000]
    y_train = y_train[0:20000]

    X_test, y_test = read_X_y("F:\ExpData\DataFromOther\security detection\kddcup99\\preprocess\\corrected.test")
    X_test, y_test = shuffle(X_test, y_test,random_state=128)
    X_test = X_test[0:5000]
    y_test = y_test[0:5000]

    score_clf(LogisticRegression(),X_train,y_train,X_test,y_test)
    score_clf(svm.LinearSVC(), X_train, y_train, X_test, y_test)
    score_clf(RandomForestClassifier(),X_train,y_train,X_test,y_test)
    score_clf(GradientBoostingClassifier(), X_train, y_train, X_test, y_test)

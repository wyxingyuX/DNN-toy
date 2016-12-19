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
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from mlxtend.classifier import StackingClassifier
from sklearn import tree
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l1l2, l1, l2
from keras import initializations
from sklearn.preprocessing import normalize
from keras.layers.advanced_activations import SReLU
from keras.layers.normalization import BatchNormalization
import keras.backend as K


def my_report(y_test, y_pred, start_label=0):
    count_mat = np.zeros((2, 2))
    for i in range(len(y_test)):
        y_label_pred = y_pred[i] - start_label
        y_label_test = y_test[i] - start_label
        row = int(y_label_pred)
        col = int(y_label_test)
        count_mat[row][col] += 1
    print "===confusion Matrix========"
    print "\tnormal(0)\t attack(1)\t total"
    for i in range(len(count_mat)):
        print str(i), "\t", str(count_mat[i][0]), " \t", str(count_mat[i][1]), "\t\t", str(sum(count_mat[i]))
    col_sum = sum(count_mat)
    print "\t", str(col_sum[0]), "\t", str(col_sum[1]), "\t\t", str(sum(col_sum))

    print "Accuary:", (count_mat[0][0] + count_mat[1][1]) / (1.0 * len(y_test))
    rate_mat = np.zeros((2, 5))
    for i in range(len(count_mat)):
        tprate = (1.0 * count_mat[i][i]) / (count_mat[0][i] + count_mat[1][i])
        fprate = (1.0 * count_mat[i][1 - i]) / (count_mat[0][1 - i] + count_mat[1][1 - i])
        precision = (1.0 * count_mat[i][i]) / (count_mat[i][0] + count_mat[i][1])
        recall = (1.0 * count_mat[i][i]) / (count_mat[0][i] + count_mat[1][i])

        rate_mat[i][0] = tprate
        rate_mat[i][1] = fprate
        rate_mat[i][2] = precision
        rate_mat[i][3] = recall
        rate_mat[i][4] = 2.0 * precision * recall / (precision + recall)

    print "\tTP Rate | \t FP Rate |\tPrecsion |\tRecall |\tF1"
    for i in range(len(rate_mat)):
        print str(i), "  ",
        for j in range(len(rate_mat[i])):
            print str(round(rate_mat[i][j], 4)), "  ",
        print ""
    print "DR(Attack Detection Rate):", round((1.0 * count_mat[1][1] / (count_mat[0][1] + count_mat[1][1])), 4)
    print "FPR(False Alarm Rate):", round((1.0 * count_mat[1][0] / (count_mat[0][0] + count_mat[1][0])), 4)


def read_X_y(file, feature_end_diem=-1, label_index=-1, label_scalar_start=1):
    print "strat read ...."
    f = open(file)
    X = []
    y = []
    for line in f:
        elms = line.split()
        X.append([float(e) for e in elms[0:feature_end_diem]])
        y.append(int(elms[label_index]) - label_scalar_start)
    f.close()
    print "read complete."
    return X, y


def read_train_test(train_file, test_file, label_idx=-1, scalar=preprocessing.MinMaxScaler(), isNormalize=False):
    # X, y = read_X_y("/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/KDDTrain+_20Percent.txt",
    #                 feature_end_diem=-3, label_index=-1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=128)

    # label_idx = -1
    label_num = 2
    if label_idx == -1:
        label_num = 2
    if label_idx == -2:
        label_num = 5
    if label_idx == -3:
        label_num = 23

    X_train, y_train = read_X_y(train_file, feature_end_diem=-3, label_index=label_idx, label_scalar_start=1)
    X_train = scalar.fit_transform(X_train)

    X_test, y_test = read_X_y(test_file, feature_end_diem=-3, label_index=label_idx)
    X_test = scalar.transform(X_test)

    y_train_category = np_utils.to_categorical([(label) for label in y_train], label_num)
    y_test_category = np_utils.to_categorical([(label) for label in y_test], label_num)

    if isNormalize:
        X_train = normalize(X_train)
        X_test = normalize(X_test)

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
    report_clf(RandomForestClassifier(), X_train, y_train, X_test, y_test)
    report_clf(naive_bayes.MultinomialNB(), X_train, y_train, X_test, y_test)
    report_clf(tree.DecisionTreeClassifier(), X_train, y_train, X_test, y_test)
    report_clf(GradientBoostingClassifier(), X_train, y_train, X_test, y_test)
    stack_clf = StackingClassifier(
        classifiers=[LogisticRegression(), RandomForestClassifier(), GradientBoostingClassifier()],
        meta_classifier=LogisticRegression())
    report_clf(stack_clf, X_train, y_train, X_test, y_test)


def stack_clf(X_train, X_test, y_train, y_test):
    pass


def bp(X_train, X_test, y_train, y_test):
    np.random.seed(64)
    model = Sequential()
    model.add(Dense(output_dim=110, input_dim=len(X_train[0]), activation="tanh"))
    model.add(Dense(output_dim=80, activation="relu"))
    model.add(Dense(output_dim=40, activation="tanh"))
    model.add(Dense(output_dim=20, activation="relu"))
    model.add(Dense(output_dim=2, activation="softmax"))
    plot(model, show_shapes=True, to_file='./model.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    # model.fit(X_train, y_train, nb_epoch=20, batch_size=1, validation_split=0.2, callbacks=[early_stopping])
    model.fit(X_train, y_train, nb_epoch=300, batch_size=128, validation_split=0.2, validation_data=(X_test, y_test))
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
    model.add(Dense(output_dim=2, input_dim=len(X_train[0]), activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=1)

    # model.fit(X_train, y_train, nb_epoch=8, batch_size=1, validation_split=0.2, callbacks=[early_stopping])
    model.fit(X_train, y_train, nb_epoch=300, batch_size=128, validation_split=0.2, callbacks=[early_stopping])
    score = model.evaluate(X_test, y_test)
    print score

    y_pred = model.predict_classes(X_test)
    my_report(category_2_label(y_test), y_pred, start_label=0)


def nn_cnn(X_train, X_test, y_train, y_test):
    print "nn——cnn"
    np.random.seed(64)
    model = Sequential()

    model.add(Dense(output_dim=80, input_dim=len(X_train[0])))
    model.add(Reshape((20, 4)))
    model.add(Convolution1D(nb_filter=8, filter_length=8, border_mode='valid'))
    model.add(MaxPooling1D(pool_length=2))
    # model.add(Convolution1D(nb_filter=3, filter_length=3, border_mode='valid'))
    # model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(output_dim=2))
    model.add(Activation("softmax"))

    plot(model, show_shapes=True, to_file='./model.png')
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    model.fit(X_train, y_train, nb_epoch=500, batch_size=256, validation_split=0.2, callbacks=[early_stopping])

    score = model.evaluate(X_test, y_test)
    print score

    y_pred = model.predict_classes(X_test)
    my_report(category_2_label(y_test), y_pred, start_label=0)


def y_prob_2_y_class(y_prob):
    y_class = []
    for probs in y_prob:
        y_class.append(probs.argmax())
    return y_class


def auto_encoder(X_train, X_test):
    # this is the size of our encoded representations
    np.random.seed(64)
    encoding_dim = 80

    # this is our input placeholder
    input_img = Input(shape=(len(X_train[0]),))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim)(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(len(X_train[0]))(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input=input_img, output=decoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(X_train, X_train,
                    nb_epoch=300,
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
    encoded = Dense(encoding_dim, activation='tanh',
                    activity_regularizer=regularizers.activity_l1(10e-5))(input_img)
    decoded = Dense(len(X_train[0]), activation='tanh')(encoded)

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
    clf = LDA(n_components=80)
    clf.fit(X_train, y_train)
    X_train_lda = clf.transform(X_train)
    X_test_lda = clf.transform(X_test)
    return X_train_lda, X_test_lda


def xgb(X_train, y_train, X_test, y_test):
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',  # 多分类的问题
        'num_class': 2,  # 类别数，与 multisoftmax 并用
        'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 12,  # 构建树的深度，越大越容易过拟合
        'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.7,  # 随机采样训练样本
        'colsample_bytree': 0.7,  # 生成树时进行的列采样
        'min_child_weight': 3,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.007,  # 如同学习率
        'seed': 1000,
        'nthread': 7,  # cpu 线程数
        # 'eval_metric': 'auc'
    }

    plst = list(params.items())
    num_rounds = 5000  # 迭代次数

    X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(X_train, y_train, test_size=0.2)
    xgb_val = xgb.DMatrix(X_val_xgb, label=y_val_xgb)
    xgb_train = xgb.DMatrix(X_train_xgb, label=y_train_xgb)

    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    xgb_model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
    y_pred_prob = xgb_model.predict(xgb.DMatrix(X_test))
    y_pred = y_prob_2_y_class(y_pred_prob)
    my_report(y_test, y_pred)


def stat(y):
    cate_num_dic = {}
    for yi in y:
        yi = int(yi)
        if cate_num_dic.has_key(yi):
            cate_num_dic[yi] = cate_num_dic[yi] + 1
        else:
            cate_num_dic[yi] = 1
    sort_dic_items = sorted(cate_num_dic.items(), key=lambda x: x[0])
    for item in sort_dic_items:
        print item[0], item[1]
    print "================"


def nn_cnn_model(X_train, X_test, y_train, y_test):
    print "nn—cnn-model"
    np.random.seed(64)
    X_train, y_train = shuffle(X_train, y_train, random_state=100 * 7)

    input_img = Input(shape=(len(X_train[0]),))
    dense_1 = Dense(init=my_init, output_dim=122, input_dim=len(X_train[0]), activation="relu")(input_img)
    reshape = Reshape((122, 1))(dense_1)

    conv_1 = Convolution1D(init=my_init, nb_filter=8, filter_length=4, activation="relu")(reshape)
    pool_1 = MaxPooling1D(pool_length=2)(conv_1)

    conv_2 = Convolution1D(init=my_init, nb_filter=16, filter_length=4, activation="relu")(pool_1)
    pool_2 = MaxPooling1D(pool_length=2)(conv_2)

    conv_3 = Convolution1D(init=my_init, nb_filter=16, filter_length=2, activation="relu")(pool_2)
    pool_3 = MaxPooling1D(pool_length=2)(conv_3)

    conv_4 = Convolution1D(init=my_init, nb_filter=8, filter_length=2, activation="relu")(pool_3)
    pool_4 = MaxPooling1D(pool_length=2)(conv_4)

    conv_5 = Convolution1D(init=my_init, nb_filter=8, filter_length=2, activation="relu")(pool_4)
    pool_5 = MaxPooling1D(pool_length=2)(conv_5)

    flatten = Flatten()(pool_5)
    out_dense = Dense(output_dim=2, activation="softmax")(flatten)

    model = Model(input=input_img, output=out_dense)
    plot(model, show_shapes=True, to_file='./model.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_acc', patience=1)
    # model.fit(X_train, y_train, nb_epoch=50, batch_size=300, validation_split=0.2,
    #           validation_data=(X_test, y_test), shuffle=False)
    # model.fit(X_train, y_train, nb_epoch=50, batch_size=2048, validation_data=(X_test, y_test), shuffle=False)
    X_test_test, X_test_val, Y_test_test, Y_test_val = train_test_split(X_test, y_test, test_size=0.1, random_state=1)
    model.fit(X_train, y_train, nb_epoch=50, batch_size=2048, validation_data=(X_test_val, Y_test_val), shuffle=False)
    # model.fit(X_train, y_train, nb_epoch=50, batch_size=2048, validation_data=(X_test_val, Y_test_val), shuffle=False)
    # model.fit(X_train, y_train, nb_epoch=100, batch_size=2000, validation_split=0.2, callbacks=[early_stopping])

    score = model.evaluate(X_test, y_test)
    print score

    y_pred_prob = model.predict(X_test)
    y_pred = y_prob_2_y_class(y_pred_prob)
    my_report(category_2_label(y_test), y_pred, start_label=0)


def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.05, name=name)


def nn_cnn_adjust(X_train, X_test, y_train, y_test):
    print "nn_cnn_adjust"
    np.random.seed(64)
    X_train, y_train = shuffle(X_train, y_train, random_state=100 * 7)

    input_img = Input(shape=(len(X_train[0]),))
    dense_1 = Dense(output_dim=122, init=my_init, input_dim=len(X_train[0]), activation="relu")(input_img)
    reshape = Reshape((122, 1))(dense_1)

    # conv_1 = Convolution1D(nb_filter=8, init=my_init, filter_length=4, activation="relu")(reshape)
    # pool_1 = MaxPooling1D(pool_length=2)(conv_1)
    # conv_2 = Convolution1D(nb_filter=16, init=my_init, filter_length=4, activation="relu")(pool_1)
    # pool_2 = MaxPooling1D(pool_length=2)(conv_2)
    # conv_3 = Convolution1D(nb_filter=16, init=my_init, filter_length=2, activation="relu")(pool_2)
    # pool_3 = MaxPooling1D(pool_length=2)(conv_3)
    # conv_4 = Convolution1D(nb_filter=8, init=my_init, filter_length=2, activation="relu")(pool_3)
    # pool_4 = MaxPooling1D(pool_length=2)(conv_4)
    # conv_5 = Convolution1D(nb_filter=8, init=my_init, filter_length=2, activation="relu")(pool_4)
    # pool_5 = MaxPooling1D(pool_length=2)(conv_5)

    # flatten = Flatten()(pool_2)
    # flatten = Flatten()(pool_3)
    # flatten = Flatten()(pool_4)
    # flatten = Flatten()(pool_5)

    conv_1 = Convolution1D(nb_filter=4, init=my_init, filter_length=4, activation="relu")(reshape)
    conv_2 = Convolution1D(nb_filter=4, init=my_init, filter_length=60, activation="relu")(conv_1)
    conv_3 = Convolution1D(nb_filter=8, init=my_init, filter_length=4, activation="relu")(conv_2)
    conv_4 = Convolution1D(nb_filter=8, init=my_init, filter_length=28, activation="relu")(conv_3)
    conv_5 = Convolution1D(nb_filter=2, init=my_init, filter_length=2, activation="relu")(conv_4)
    conv_6 = Convolution1D(nb_filter=2, init=my_init, filter_length=13, activation="relu")(conv_4)

    flatten = Flatten()(conv_6)
    out_dense = Dense(output_dim=2, activation="softmax")(flatten)

    model = Model(input=input_img, output=out_dense)

    plot(model, show_shapes=True, to_file='./model.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, nb_epoch=50, batch_size=2000, shuffle=False,
              validation_data=(X_test, y_test))
    # model.fit(X_train, y_train, nb_epoch=50, batch_size=400, shuffle=False, validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test)
    print score

    y_pred_prob = model.predict(X_test)
    y_pred = y_prob_2_y_class(y_pred_prob)
    my_report(category_2_label(y_test), y_pred, start_label=0)


def pure_cnn(X_train, X_test, y_train, y_test):
    print "pure_cnn"
    np.random.seed(64)
    X_train, y_train = shuffle(X_train, y_train, random_state=100 * 7)

    input_img = Input(shape=(len(X_train[0]),))
    reshape = Reshape((len(X_train[0]), 1))(input_img)
    conv_1 = Convolution1D(init=my_init, nb_filter=8, filter_length=4, activation="relu")(reshape)
    pool_1 = MaxPooling1D(pool_length=2)(conv_1)
    conv_2 = Convolution1D(init=my_init, nb_filter=16, filter_length=4, activation="relu")(pool_1)
    pool_2 = MaxPooling1D(pool_length=2)(conv_2)
    conv_3 = Convolution1D(init=my_init, nb_filter=16, filter_length=2, activation="relu")(pool_2)
    pool_3 = MaxPooling1D(pool_length=2)(conv_3)
    conv_4 = Convolution1D(init=my_init, nb_filter=8, filter_length=2, activation="relu")(pool_3)
    pool_4 = MaxPooling1D(pool_length=2)(conv_4)
    conv_5 = Convolution1D(init=my_init, nb_filter=8, filter_length=2, activation="relu")(pool_4)
    pool_5 = MaxPooling1D(pool_length=2)(conv_5)
    # flatten = Flatten()(pool_2)
    # flatten = Flatten()(pool_3)
    # flatten = Flatten()(pool_4)
    flatten = Flatten()(pool_5)
    out_dense = Dense(output_dim=2, activation="softmax")(flatten)

    model = Model(input=input_img, output=out_dense)
    plot(model, show_shapes=True, to_file='./model.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # best_model_filepath = "./model.best.hdf5"
    # model_check_point = ModelCheckpoint(filepath=best_model_filepath, save_best_only=True, monitor="val_loss",
    #                                     mode="min")
    early_stopping = EarlyStopping(monitor='val_acc', patience=4)
    model.fit(X_train, y_train, nb_epoch=50, batch_size=2048, shuffle=False, validation_data=(X_test, y_test))
    # model.fit(X_train, y_train, nb_epoch=50, batch_size=512, shuffle=False, validation_data=(X_test, y_test))

    # model = load_model(best_model_filepath)
    score = model.evaluate(X_test, y_test)
    print score

    y_pred_prob = model.predict(X_test)
    y_pred = y_prob_2_y_class(y_pred_prob)
    my_report(category_2_label(y_test), y_pred, start_label=0)


def nn_cnn_adjust_test(X_train, X_test, y_train, y_test):
    print "nn_cnn_adjust_test"
    np.random.seed(64)
    X_train, y_train = shuffle(X_train, y_train, random_state=100 * 7)

    neunum = 122
    input_img = Input(shape=(len(X_train[0]),))
    dense_1 = Dense(output_dim=neunum, init=my_init, input_dim=len(X_train[0]), activation="relu")(input_img)
    reshape = Reshape((neunum, 1))(dense_1)
    print "dense-1：", str(neunum)

    conv_1 = Convolution1D(nb_filter=8, init=my_init, filter_length=4, activation="relu")(reshape)
    pool_1 = MaxPooling1D(pool_length=2)(conv_1)
    print "pool_1"

    conv_2 = Convolution1D(nb_filter=16, init=my_init, filter_length=4, activation="relu")(pool_1)
    pool_2 = MaxPooling1D(pool_length=2)(conv_2)
    print "pool_2"

    conv_3 = Convolution1D(nb_filter=16, init=my_init, filter_length=2, activation="relu")(pool_2)
    pool_3 = MaxPooling1D(pool_length=2)(conv_3)
    print "pool_3"

    conv_4 = Convolution1D(nb_filter=8, init=my_init, filter_length=2, activation="relu")(pool_3)
    pool_4 = MaxPooling1D(pool_length=2)(conv_4)
    print "pool_4"

    conv_5 = Convolution1D(nb_filter=8, init=my_init, filter_length=2, activation="relu")(pool_4)
    pool_5 = MaxPooling1D(pool_length=2)(conv_5)
    print "pool_5"

    # flatten = Flatten()(pool_1)
    # flatten = Flatten()(pool_2)
    # flatten = Flatten()(pool_3)
    # flatten = Flatten()(pool_4)
    flatten = Flatten()(pool_5)

    out_dense = Dense(output_dim=2, activation="softmax")(flatten)

    model = Model(input=input_img, output=out_dense)

    plot(model, show_shapes=True, to_file='./model.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, nb_epoch=50, batch_size=2000, shuffle=False, validation_data=(X_test, y_test))
    # model.fit(X_train, y_train, nb_epoch=50, batch_size=400, shuffle=False, validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test)
    print score

    y_pred_prob = model.predict(X_test)
    y_pred = y_prob_2_y_class(y_pred_prob)
    my_report(category_2_label(y_test), y_pred, start_label=0)


if __name__ == "__main__":
    print "hello"

    train_file = "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/KDDTrain+_20Percent.txt"
    # train_file = "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/KDDTrain+.txt"
    test_file = "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/KDDTest-21.txt"
    # test_file = "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/KDDTest+.txt"
    X_train, X_test, y_train, y_test, y_train_category, y_test_category = read_train_test(train_file, test_file,label_idx=-2)
    stat(y_train)
    stat(y_test)

    # X_train_encoded, X_test_encoded = auto_encoder(X_train, X_test)
    # X_train_encoded, X_test_encoded = sparse_auto_encoder(X_train, X_test)
    # X_train_encoded, X_test_encoded = pca_extract(X_train, X_test)
    # X_train_encoded, X_test_encoded = nmf_extract(X_train, X_test)
    # X_train_encoded, X_test_encoded = lda_extract(X_train, X_test, y_train, y_test)
    # report_clf(svm.LinearSVC(), X_train_encoded, y_train, X_test_encoded, y_test)
    # smr(X_train_encoded, X_test_encoded, y_train_category, y_test_category)
    # smr(X_train, X_test, y_train_category, y_test_category)
    # base_line_clf(X_train, X_test, y_train, y_test)
    # bp(X_train, X_test, y_train_category, y_test_category)
    # nn_cnn(X_train, X_test, y_train_category, y_test_category)
    # nn_cnn_adjust_test(X_train, X_test, y_train_category, y_test_category)
    # nn_cnn_adjust(X_train, X_test, y_train_category, y_test_category)
    # nn_cnn_model(X_train, X_test, y_train_category, y_test_category)
    # pure_cnn(X_train, X_test, y_train_category, y_test_category)

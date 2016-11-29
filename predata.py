# -*- coding:utf-8 -*-

if __name__ == "__main__":
    fr_train = open("/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/KDDTrain+_20Percent.txt")
    fw_train = open("/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/test_KDDTrain+_20Percent.txt", "w")
    dic1 = {}
    dic2 = {}
    dic3 = {}
    dic_label = {}
    label_idx = -2
    # train
    for line in fr_train:
        elms = line.split(",")
        fw_train.write(elms[0] + '\t')
        if not dic1.has_key(elms[1]):
            dic1[elms[1]] = len(dic1) + 1
        fw_train.write(str(dic1[elms[1]]) + '\t')

        if not dic2.has_key(elms[2]):
            dic2[elms[2]] = len(dic2) + 1
        fw_train.write(str(dic2[elms[2]]) + '\t')

        if not dic3.has_key(elms[3]):
            dic3[elms[3]] = len(dic3) + 1
        fw_train.write(str(dic3[elms[3]]) + '\t')

        for item in elms[4:label_idx]:
            fw_train.write(item + '\t')
        if not dic_label.has_key(elms[label_idx]):
            dic_label[elms[label_idx]] = len(dic_label) + 1
        fw_train.write(str(dic_label[elms[label_idx]]))
        fw_train.write('\n')
    fr_train.close()
    fw_train.close()

    # test
    fr_test = open("/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/KDDTest-21.txt")
    fw_test = open("/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/test_KDDTest-21.txt", "w")
    for line in fr_test:
        elms = line.split(",")
        fw_test.write(elms[0] + '\t')
        fw_test.write((str(dic1[elms[1]]) if dic1.has_key(elms[1]) else '0') + '\t')
        fw_test.write((str(dic2[elms[2]]) if dic2.has_key(elms[2]) else '0') + '\t')
        fw_test.write((str(dic3[elms[3]]) if dic3.has_key(elms[3]) else '0') + '\t')
        for item in elms[4:label_idx]:
            fw_test.write(item + '\t')
        fw_test.write(str(dic_label[elms[label_idx]]) if dic_label.has_key(elms[label_idx]) else '0')
        fw_test.write('\n')
    fr_test.close()
    fw_test.close()

# -*- coding:utf-8 -*-
def gen_origin_fildes(file, dest):
    num_fileds = {}
    fr = open(file)
    for line in fr:
        elms = line.split(",")
        for i in range(len(elms)):
            v = elms[i]
            if not is_num(v):
                num_fileds.setdefault(i, set()).add(v)
            else:
                if not num_fileds.has_key(i):
                    num_fileds[i] = set()
    fr.close()

    sort_num_fileds = sorted(num_fileds.items(), key=lambda t: t[0])
    fw = open(dest, "w")
    for tu in sort_num_fileds:
        diem = tu[0]
        value = tu[1]
        if len(value) == 0:
            fw.write(str(diem) + "\t" + "num\n")
        else:
            fw.write(str(diem) + "\t" + "symbolic")
            print diem, len(value)
            for s in value:
                fw.write("\t" + s)
            fw.write("\n")
    fw.close()
    return num_fileds


def is_num(s):
    try:
        if type(eval(s) == int):
            return True
        if type(eval(s) == float):
            return True
    except:
        return False
    return False


def attac_code(attac_file, attac_code_file):
    dic1 = {}
    dic2 = {}
    fr = open(attac_file)
    fw = open(attac_code_file, "w")
    for content in fr:
        ls = content.split("\r")
        for line in ls:
            print line
            elms = line.split(",")
            dic1.setdefault(elms[0], len(dic1) + 1)
            dic2.setdefault(elms[1], len(dic2) + 1)
            fw.write(elms[0] + ":" + str(dic1[elms[0]]) + "\t" + elms[1] + ":" + str(dic2[elms[1]]) + "\t")
            if elms[0] == "normal":
                fw.write("normal:1\n")
            else:
                fw.write("anomaly:2\n")
    fr.close()
    fw.close()


def read_attacs(file):
    attac_code_dic = {}
    fr = open(file)
    for line in fr:
        elms = line.split()
        e1 = elms[0].split(":")
        e2 = elms[1].split(":")
        e3 = elms[2].split(":")
        attac_code_dic.setdefault(e1[0], [int(e1[1]), int(e2[1]), int(e3[1])])
    fr.close()
    return attac_code_dic


def read_fileds(file):
    fileds_dic = {}
    fr = open(file)
    for line in fr:
        elms = line.split()
        if elms[1] == "num":
            fileds_dic.setdefault(int(elms[0]), [])
        else:
            fileds_dic.setdefault(int(elms[0]), elms[2:])
    return fileds_dic


def preprocess_feature(fileds_dic, attac_dic, raw_feature_file, dest):
    add_attactype_dic = {"saint": 4, "mscan": 4, "apache2": 1, "mailbomb": 1, "udpstorm": 1,
                         "processtable": 1, "httptunnel": 2,
                         "ps": 2, "sqlttack": 2, "xterm": 2, "named": 3, "xsnoop": 3, "xlock": 3,
                         "sendmail": 3, "worm": 3, "snmpgetattack": 3, "snmpguess": 3}
    fr = open(raw_feature_file)
    fw = open(dest, "w")
    print "in preprocess_feature"
    for line in fr:
        print line
        n_feature = 0
        elms = line.split(",")
        for i in range(len(elms)):
            if i < 41:
                if len(fileds_dic[i]) == 0:
                    fw.write(elms[i] + "\t")
                    n_feature += 1
                else:
                    print i, len(fileds_dic[i])
                    for filed_symbolic in fileds_dic[i]:
                        n_feature += 1
                        if filed_symbolic == elms[i]:
                            fw.write("1\t")
                        else:
                            fw.write("0\t")
            if i == 41:
                attac = elms[i]
                attac_codes = [18, 3, 2]
                if attac_dic.has_key(attac):
                    attac_codes = attac_dic[attac]
                else:
                    if add_attactype_dic.has_key(attac):
                        attac_codes[1] = add_attactype_dic[attac]
                for code in attac_codes:
                    fw.write(str(code) + "\t")
                fw.write("\n")
            if i > 41:
                break
        print n_feature
    print "end preprocess_feature"
    fr.close()
    fw.close()


def preprocess_feature2(fileds_dic, attac_dic, raw_feature_file, dest):
    add_attactype_dic = {"saint": 4, "mscan": 4, "apache2": 1, "mailbomb": 1, "udpstorm": 1,
                         "processtable": 1, "httptunnel": 2,
                         "ps": 2, "sqlttack": 2, "xterm": 2, "named": 3, "xsnoop": 3, "xlock": 3,
                         "sendmail": 3, "worm": 3, "snmpgetattack": 3, "snmpguess": 3}
    fr = open(raw_feature_file)
    fw = open(dest, "w")
    print "in preprocess_feature"
    for line in fr:
        print line
        n_feature = 0
        elms = line.split(",")
        for i in range(len(elms)):
            if i < 41:
                if len(fileds_dic[i]) == 0:
                    fw.write(elms[i] + "\t")
                else:
                    fw.write(str(fileds_dic[i].index(elms[i])) + "\t")
                n_feature += 1
            if i == 41:
                attac = elms[i]
                attac_codes = [18, 3, 2]
                if attac_dic.has_key(attac):
                    attac_codes = attac_dic[attac]
                else:
                    if add_attactype_dic.has_key(attac):
                        attac_codes[1] = add_attactype_dic[attac]
                for code in attac_codes:
                    fw.write(str(code) + "\t")
                fw.write("\n")
            if i > 41:
                break
        print n_feature
    print "end preprocess_feature"
    fr.close()
    fw.close()


if __name__ == "__main__":
    print "hello"
    # gen_origin_fildes("/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/KDDTrain+.txt", \
    #                   "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/fileds.txt")
    # attac_code("/home/wyjn/下载/NSL_KDD-master/Attack Types.csv",
    #            "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/attack_types_code.txt")
    fileds_dic = read_fileds("/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/fileds.txt")
    attac_dic = read_attacs("/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/attack_types_code.txt")

    # process1
    # preprocess_feature(fileds_dic, attac_dic,
    #                    "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/KDDTrain+.txt",
    #                    "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/KDDTrain+.txt")
    # preprocess_feature(fileds_dic, attac_dic,
    #                    "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/KDDTrain+_20Percent.txt",
    #                    "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/KDDTrain+_20Percent.txt")
    # preprocess_feature(fileds_dic, attac_dic,
    #                    "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/KDDTest+.txt",
    #                    "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/KDDTest+.txt")
    # preprocess_feature(fileds_dic, attac_dic,
    #                    "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/KDDTest-21.txt",
    #                    "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess/KDDTest-21.txt")

    # precess 2
    # preprocess_feature2(fileds_dic, attac_dic,
    #                     "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/KDDTrain+.txt",
    #                     "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess2/KDDTrain+.txt")
    # preprocess_feature2(fileds_dic, attac_dic,
    #                     "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/KDDTrain+_20Percent.txt",
    #                     "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess2/KDDTrain+_20Percent.txt")
    # preprocess_feature2(fileds_dic, attac_dic,
    #                     "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/KDDTest+.txt",
    #                     "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess2/KDDTest+.txt")
    # preprocess_feature2(fileds_dic, attac_dic,
    #                     "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/KDDTest-21.txt",
    #                     "/home/wyjn/下载/NSL_KDD-master/Original NSL KDD Zip/preprocess2/KDDTest-21.txt")

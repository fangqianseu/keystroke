import os

from sklearn.metrics import classification_report
import pickle

from final.tool import *
from final.endpointDetection import EndPointDetect
from final.preProcess import PreProcess
from final.tdoa import get_tdoa
from final.feature import *
from final.modelsTrainer import ModelsTrainer

from sklearn.model_selection import train_test_split


def test_preProcess(datas, rate):
    preProcess = PreProcess()
    preProcess.doPreProcess(rate, datas)

    # show_figure(datas)

    return preProcess.data


def test_endDetrction(pre_data):
    end_point_detect = EndPointDetect(pre_data)
    final_data = end_point_detect.endPoint_data

    index = 0
    # figure()
    for sclice in final_data:
        plot2(range(index, index + len(sclice)), sclice)
        index = index + len(sclice) + 1000
    # show()

    return final_data


def test_tdoa(datas, rate):
    tau = get_tdoa(datas, rate)

    return tau


def test_feature(datas, rate):
    features = combine_features(datas, rate)

    # matshow_figure(feature)

    return features


def dump_object(path, object):
    with open(path, 'wb') as dump_file:
        pickle.dump(object, dump_file)


def load_object(path):
    return pickle.load(open(path, 'rb'))


def save_data():
    print("collect features ------------")
    # 存储数据
    X = []
    Y = []
    labels = []
    tags = []
    label = 0
    for dirname in os.listdir(base_path):
        label += 1

        base_dir = os.path.join(base_path, dirname)
        print("*" * 20, " tag", dirname, ' label', label)

        tag = dirname

        for file_name in os.listdir(base_dir):
            file_path = os.path.join(base_dir, file_name)

            rate, datas = read_sign(file_path)

            pre_datas = test_preProcess(datas, rate)
            # tdoa = test_tdoa(pre_datas, rate)
            cut_datas = main_signals_cut(pre_datas, rate)
            feature = test_feature(cut_datas, rate)

            print(feature.shape)

            X.append(feature)
            Y.append(label)

            if label not in labels:
                labels.append(label)

            if tag not in tags:
                tags.append(tag)

    dump_object(features_path, X)
    dump_object(Y_path, Y)
    dump_object(labels_path, labels)
    dump_object(tags_path, tags)


if __name__ == '__main__':
    # path = '/Users/qian/taogroup/data/0-10.wav'
    # path = '/Users/qian/taogroup/data/0-10two.wav'
    # path = '/Users/qian/taogroup/data/e-3.wav'
    # rate, datas = tool.Tool.read_sign(path)
    # datas = test_preProcess(datas, rate)
    # final_datas = test_endDetrction(datas)

    # base_path = '/Users/qian/taogroup/data/test/datebace'
    # base_path = '/Users/qian/taogroup/data/test/datebace1'
    base_path = '/Users/qian/taogroup/data/test/v20'
    features_path = 'data/features.dump'
    Y_path = 'data/Y.dump'
    labels_path = 'data/labels.dump'
    tags_path = 'data/tags.dump'

    save_data()

    X = load_object(features_path)
    Y = load_object(Y_path)
    labels = load_object(labels_path)
    tags = load_object(tags_path)

    modelsTrainer = ModelsTrainer()

    # label 和 tag 对应
    label_map = {}
    for i in range(len(tags)):
        label_map[Y[i]] = tags[i]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)
    for i, _ in enumerate(x_train):
        feature = x_train[i]
        label = y_train[i]
        modelsTrainer.collect_features(label, feature)

    print("train ------------")

    modelsTrainer.train()

    print("predict -------------")

    predicts_one = {}
    predicts_three = {}
    for i, _ in enumerate(x_test):
        feature = x_test[i]
        real_label = y_test[i]

        models_results = modelsTrainer.predict(feature)

        for model, predict in models_results.items():
            if model not in predicts_one.keys():
                predicts_one[model] = []

            predicts_one[model].append(predict[0])

            if model in ['hmm', 'gmm']:
                if model not in predicts_three.keys():
                    predicts_three[model] = []
                if real_label in predict:
                    predicts_three[model].append(real_label)
                else:
                    predicts_three[model].append(predict[0])

    print("one key")
    for name, predicts in predicts_one.items():
        print('@@@@@', name)
        print(classification_report(y_test, predicts, labels=labels, target_names=tags))

    print("three keys")
    for name, predicts in predicts_three.items():
        print('@@@@@', name)
        print(classification_report(y_test, predicts, labels=labels, target_names=tags))

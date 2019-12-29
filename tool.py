import pickle

import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
from scipy.spatial.distance import pdist
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns


def read_sign(path):
    rate, datas = wav.read(path)
    return rate, datas


def main_signals_cut(datas, rate):
    data = datas[:, 0]

    # 返回 绝对值 最大的下标
    index = np.argmax(np.abs(data))

    begin = max(0, int(index - rate * 0.003))
    end = min(len(data), int(index + rate * 0.03))

    print(index, begin, end, len(datas[:, 0]))
    return np.array(datas[begin:end, :])


def convent2_line(feature):
    return np.array(feature).reshape(-1, 1)


def convent2_row(feature):
    return np.array(feature).reshape(1, -1)


def dump_object(path, object):
    with open(path, 'wb') as dump_file:
        pickle.dump(object, dump_file)


def load_object(path):
    return pickle.load(open(path, 'rb'))


def cosine_distance(data1, data2):
    return cosine_similarity(data1, data2)


def points_distance(data1, data2, metric='euclidean'):
    d2 = pdist(np.vstack([data1, data2]), metric)
    return d2


def data_stander(data):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    return min_max_scaler.fit_transform(data)


def np_array(data):
    return np.array(data)


def np_newarray():
    return np.asarray()


def figure():
    plt.figure()


def plot(data):
    plt.plot(data)


def plot2(X, Y):
    plt.plot(X, Y)


def show():
    plt.show()


def plot_figure(data, title=''):
    plt.figure()
    plt.title(title)
    plt.plot(data)
    plt.show()


def matshow_figure(data, title=''):
    plt.matshow(data)
    plt.title(title)
    plt.show()


def imshow_figure(data, title=''):
    plt.imshow(data)
    plt.title(title)
    plt.show()


def heat_figure(data):
    sns.heatmap(data, cmap='GnBu', )
    plt.show()

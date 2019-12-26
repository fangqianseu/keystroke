import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np


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
    sign = []
    row, line = feature.shape
    for j in range(line):
        for i in range(row):
            sign.append(feature[i, j])
    return np.array(sign)


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

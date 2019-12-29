import numpy as np
from final.tool import *
import math


def gcc_phat(sig, refsig, fs=48000, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    基于 gcc-path 的 多通道 时延计算方法
    '''

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    # return tau, cc
    return abs(shift), cc


def get_tdoa(datas, rate):
    toda_datas = main_signals_cut(datas, rate)
    # show_figure(toda_datas, "cut data")

    data1 = np_array(toda_datas[:, 0])
    data2 = np_array(toda_datas[:, 1])

    audio_length = len(data1)
    window = np.hanning(audio_length)
    # window = 1

    tau, cc = gcc_phat(data1 * window, data2 * window, fs=rate)

    return tau


# DTW 算法...
def dtw(M1, M2):
    # 初始化数组 大小为 M1 * M2
    M1_len = len(M1)
    M2_len = len(M2)
    cost = [[0 for i in range(M2_len)] for i in range(M1_len)]

    # 初始化 dis 数组
    dis = []
    for i in range(M1_len):
        dis_row = []
        for j in range(M2_len):
            dis_row.append(distance(M1[i], M2[j]))
        dis.append(dis_row)

    # 初始化 cost 的第 0 行和第 0 列
    cost[0][0] = dis[0][0]
    for i in range(1, M1_len):
        cost[i][0] = cost[i - 1][0] + dis[i][0]
    for j in range(1, M2_len):
        cost[0][j] = cost[0][j - 1] + dis[0][j]

    # 开始动态规划
    for i in range(1, M1_len):
        for j in range(1, M2_len):
            cost[i][j] = min(cost[i - 1][j] + dis[i][j] * 1,
                             cost[i - 1][j - 1] + dis[i][j] * 2,
                             cost[i][j - 1] + dis[i][j] * 1)
    return cost[M1_len - 1][M2_len - 1]


# 两个维数相等的向量之间的距离
def distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum = sum + abs(x1[i] - x2[i])
    return sum


if __name__ == '__main__':
    fs, far = read_sign('../audio/alexa-01.wav')
    _, near = read_sign('../audio/alexa-02.wav')

    # max_tau = 0.14 / 340
    # audio_length = len(far)
    # block_length = math.floor(fs / 2)
    # n = math.floor(audio_length / block_length)
    # samples = math.floor(max_tau * fs) * 2 + 1
    # window = np.hanning(block_length)
    #
    # for k in range(1, n + 1):
    #     i = (k - 1) * block_length + 1
    #     sig = near[i:(i + block_length)] * window
    #     refsig = far[i:(i + block_length)] * window
    #     tau, cc = gcc_phat(sig, refsig, fs, max_tau)
    #     print(tau)
    # window = np.hanning(len(far))
    sig = near[:2000]
    refsig = far[:2000]
    tau, cc = gcc_phat(sig, refsig, fs=fs)
    print(tau)

    dtw_ = dtw(sig, refsig)
    print(dtw_)

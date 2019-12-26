import numpy as np
from final.tool import *
import math


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
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


if __name__ == '__main__':
    fs, far = read_sign('../audio/alexa-01.wav')
    fs, near = read_sign('../audio/alexa-02.wav')

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
    window = 1
    sig = near * window
    refsig = far * window
    tau, cc = gcc_phat(sig, refsig, fs=fs)
    print(tau)

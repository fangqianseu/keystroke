import math

import python_speech_features
from python_speech_features import mfcc, delta
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA

from final.tool import *


def get_mfcc(recording, sample_rate, winlen=0.025, numcep=13, nfilt=26, appendEnergy=True, winfunc=np.hanning):
    frame_length = math.ceil(winlen * sample_rate)  # 默认 512
    mfcc_feature = mfcc(recording, sample_rate, numcep=numcep, nfilt=nfilt, nfft=frame_length,
                        appendEnergy=appendEnergy,
                        winfunc=winfunc)
    return mfcc_feature


def get_fft(recording, sample_rate):
    fft = np.fft.fft(recording)
    number_of_samples = len(recording)
    sample_length = 1. / sample_rate
    frequencies = np.fft.fftfreq(number_of_samples, sample_length)
    positive_frequency_indices = np.where(frequencies > 0)
    # frequencies = frequencies[positive_frequency_indices]
    magnitudes = abs(fft[positive_frequency_indices])
    # magnitudes = abs(fft)
    magnitudes = magnitudes / np.linalg.norm(magnitudes)
    return np.array(magnitudes)


def get_delta(mfccfeature, n=1):
    deltafeature = delta(mfccfeature, n)
    return deltafeature


def get_log_fbank(recording, sample_rate, winlen=0.025, numcep=13, nfilt=26):
    frame_length = math.ceil(winlen * sample_rate)
    fback, energy = python_speech_features.fbank(recording, samplerate=sample_rate, nfft=frame_length,
                                                 winfunc=np.hanning)
    return np.log(fback), np.log(energy)


def combine_features(datas, rate):
    mfcc_feature = get_mfcc(datas, rate)
    mfcc_feature = preprocessing.scale(mfcc_feature)

    mfcc_grad = np.gradient(mfcc_feature, axis=0)
    delta = get_delta(mfcc_feature, 2)
    ddelta = get_delta(delta, 2)

    fbank, energy = get_log_fbank(datas, rate)
    fbank = np.log(fbank)[:, 0:mfcc_feature.shape[1]]
    fbank = preprocessing.scale(fbank)

    features = np.hstack((mfcc_feature, delta, ddelta, mfcc_grad, fbank))
    return features


def hz2mel(f):
    return 2595. * np.log10(1. + f / 700.)


def mel2hz(z):
    return 700. * (np.power(10., z / 2595.) - 1.)


def get_dct_coeff(in_channel, out_channel):
    dct_coef = np.zeros((out_channel, in_channel), dtype=np.float32)
    for i in range(out_channel):
        n = np.linspace(0, in_channel - 1, in_channel)
        dct_coef[i, :] = np.cos((2 * n + 1) * i * np.pi / (2 * in_channel))
    return dct_coef


def get_fft_mel_mat(nfft, sr=8000, nfilts=None, width=1.0, minfrq=20, maxfrq=None, constamp=0):
    if nfilts is None:
        nfilts = nfft
    if maxfrq is None:
        maxfrq = sr // 2
    wts = np.zeros((int(nfilts), int(nfft // 2 + 1)))
    fftfrqs = np.arange(0, nfft // 2 + 1) / (1. * nfft) * (sr)
    minmel = hz2mel(minfrq)
    maxmel = hz2mel(maxfrq)
    binfrqs = mel2hz(minmel + np.arange(0, nfilts + 2) / (nfilts + 1.) * (maxmel - minmel))
    # binbin = np.round(binfrqs / maxfrq * nfft)
    for i in range(nfilts):
        fs = binfrqs[[i + 0, i + 1, i + 2]]
        fs = fs[1] + width * (fs - fs[1])
        loslope = (fftfrqs - fs[0]) / (fs[1] - fs[0])
        hislope = (fs[2] - fftfrqs) / (fs[2] - fs[1])
        wts[i, :] = np.maximum(0, np.minimum(loslope, hislope))
    return wts


def mfcc_extractor(xx, sr, win_len=0.025, shift_len=0.01, mel_channel=26, dct_channel=13, win_type='hamming',
                   include_delta=False):
    win_len = int(sr * win_len)
    shift_len = int(sr * shift_len)

    my_melbank = get_fft_mel_mat(win_len, sr, mel_channel)

    pre_emphasis_weight = 0.9375

    # x = xx * (1-pre_emphasis_weight)
    x = np.append(xx[0], xx[1:] - pre_emphasis_weight * xx[:-1])
    dctcoef = np.zeros((dct_channel, mel_channel), dtype=np.float32)
    for i in range(dct_channel):
        n = np.linspace(0, mel_channel - 1, mel_channel)
        dctcoef[i, :] = np.cos((2 * n + 1) * i * np.pi / (2 * mel_channel))

    w = 1 + 6 * np.sin(np.pi * np.linspace(0, dct_channel - 1, dct_channel) / (dct_channel - 1))
    w /= w.max()
    w = np.reshape(w, newshape=(dct_channel, 1))

    samples = x.shape[0]
    frames = int(samples - win_len) // shift_len
    stft = np.zeros((win_len, frames), dtype=np.complex64)
    spectrum = np.zeros((win_len // 2 + 1, frames), dtype=np.float32)

    mfcc = np.zeros((dct_channel, frames), dtype=np.float32)

    if win_type == 'hanning':
        window = np.hanning(win_len)
    elif win_type == 'hamming':
        window = np.hamming(win_len)
    elif win_type == 'triangle':
        window = (1 - (np.abs(win_len - 1 - 2 * np.arange(1, win_len + 1, 1)) / (win_len + 1)))
    else:
        window = np.ones(win_len)

    for i in range(frames):
        one_frame = x[i * shift_len: i * shift_len + win_len]
        windowed_frame = np.multiply(one_frame, window)
        stft[:, i] = np.fft.fft(windowed_frame, win_len)
        spectrum[:, i] = np.power(np.abs(stft[0:win_len // 2 + 1, i]), 2)

    c1 = np.matmul(my_melbank, spectrum)
    c1 = np.where(c1 == 0.0, np.finfo(float).eps, c1)
    mfcc[:dct_channel, :] = np.multiply(np.matmul(dctcoef, np.log(c1)), np.repeat(w, frames, 1))

    if include_delta:
        dtm = np.zeros((dct_channel, frames), dtype=np.float32)
        ddtm = np.zeros((dct_channel, frames), dtype=np.float32)
        for i in range(2, frames - 2):
            dtm[:, i] = 2 * mfcc[:, i + 2] + mfcc[:, i + 1] - mfcc[:, i - 1] - 2 * mfcc[:, i - 2]
        dtm /= 3.0
        for i in range(2, frames - 2):
            ddtm[:, i] = 2 * dtm[:, i + 2] + dtm[:, i + 1] - dtm[:, i - 1] - 2 * dtm[:, i - 2]
        ddtm /= 3.0
        mfcc = np.row_stack((mfcc[:, 4:frames - 4], dtm[:, 4:frames - 4], ddtm[:, 4:frames - 4]))

    return mfcc, spectrum


if __name__ == '__main__':
    path = '/Users/qian/taogroup/data/e-3.wav'
    rate, datas = read_sign(path)

    data = datas[:, 0]
    plot_figure(combine_features(data, rate))

import copy

import math
import numpy as np


class PreProcess:
    '''
    预处理 去噪
    '''

    def doPreProcess(self, rate, data):
        self.rate = rate
        self.data = copy.copy(data)

        # 归一化
        # self.normalizePCM()
        self.denoising()

    def normalizePCM(self):
        max = self.data[0]
        for i in self.data:
            if max < abs(i):
                max = abs(i)
        self.data = self.data / max

    def denoising(self):
        voiced = copy.copy(self.data[:, 0])
        samplePerFrame = (int)(self.rate * 0.01)
        # firstSamples = min((int)(samplePerFrame * 20), len(data))
        firstSamples = samplePerFrame

        sum = 0

        for i in range(firstSamples):
            sum += voiced[i]

        m = sum / firstSamples
        sum = 0

        for i in range(firstSamples):
            sum += math.pow(voiced[i] - m, 2)

        sd = math.sqrt(sum / firstSamples)
        if sd == 0:
            sd = 1

        for i in range(len(voiced)):
            if abs((voiced[i] - m) / sd) > 2:
                voiced[i] = 1
            else:
                voiced[i] = 0

        new = []

        for i in range(len(voiced)):
            if voiced[i] == 1:
                new.append(self.data[i, :])

        self.data = np.array(new)

        # return
        # frameCount = 0
        # voicedFrame = [-1 for _ in range(len(voiced) // samplePerFrame)]
        # loopCount = len(voiced) - len(voiced) % samplePerFrame
        #
        # indes = 0
        # while indes < loopCount:
        #     count_voiced = 0
        #     count_unvoiced = 0
        #
        #     j = indes
        #     while j < indes + samplePerFrame:
        #         if voiced[j] == 1:
        #             count_voiced += 1
        #         else:
        #             count_unvoiced += 1
        #         j += 1
        #
        #     if count_voiced > count_unvoiced:
        #         voicedFrame[frameCount] = 1
        #     else:
        #         voicedFrame[frameCount] = 0
        #
        #     frameCount += 1
        #     indes += samplePerFrame
        #
        # self.voice = voicedFrame
        #
        # for i in range(frameCount):
        #     if voicedFrame[i] == 0:
        #         j = i * samplePerFrame
        #         while j < i * samplePerFrame + samplePerFrame:
        #             self.data[j] = 0
        #             j += 1

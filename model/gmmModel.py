from sklearn.mixture import GaussianMixture
import numpy as np
import math


class GmmModel:
    '''
    hmm 模型的训练接口
    '''

    def __init__(self):
        self.name = 'gmm'
        self.gmm_models = []

    def train(self, label, features):
        gmm_trainer = GMM()
        feature = np.asarray(features[0])

        for iterm in features[1:]:
            feature = np.append(feature, iterm, axis=0)

        gmm_trainer.train(feature)

        self.gmm_models.append((label, gmm_trainer))

    def predict(self, feature):
        predicts = {}

        # 对每个模型 获取分数
        for label, hmm_model in self.gmm_models:
            score = hmm_model.get_score(feature)
            predicts[label] = score

        # 按照 score 排序
        predicts = sorted(predicts.items(), key=lambda x: x[1], reverse=True)[:3]

        result = []
        for label, score in predicts:
            result.append(label)
        return result


class GMM:
    def __init__(self):
        self.model_name = 'GMM'

        # self.model = GaussianMixture(n_components=16, max_iter=1000, covariance_type='diag', n_init=3)
        self.model = GaussianMixture(n_components=6, max_iter=1000, covariance_type='diag', n_init=3)

    def train(self, feature):
        np.seterr(all='ignore')
        self.model.fit(feature)

    # 对输入数据运行模型
    def get_score(self, input_data):
        score = np.sum(self.model.score(input_data))
        return score

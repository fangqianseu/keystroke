from hmmlearn import hmm
import numpy as np


class HmmModel:
    '''
    hmm 模型的训练接口
    '''

    def __init__(self):
        self.name = 'hmm'
        self.hmm_models = []

    def train(self, label, features):
        hmm_trainer = HMM()

        feature = np.asarray(features[0])
        for iterm in features[1:]:
            feature = np.append(feature, iterm, axis=0)
        hmm_trainer.train(feature)

        self.hmm_models.append((label, hmm_trainer))

    def predict(self, feature):
        predicts = {}

        # 对每个模型 获取分数
        for label, hmm_model in self.hmm_models:
            score = hmm_model.get_score(feature)
            predicts[label] = score

        # 按照 score 排序
        predicts = sorted(predicts.items(), key=lambda x: x[1], reverse=True)[:3]

        result = []
        for label, score in predicts:
            result.append(label)
        return result


class HMM(object):
    '''用到高斯隐马尔科夫模型
    n_components：定义了隐藏状态的个数
    cov_type：定义了转移矩阵的协方差类型
    n_iter:定义了训练的迭代次数
    '''

    def __init__(self, model_name='GaussianHMM', n_components=8, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter

        self.model = hmm.GaussianHMM(n_components=self.n_components,
                                     covariance_type=self.cov_type, n_iter=self.n_iter,tol=1e-3)

    def train(self, feature):
        np.seterr(all='ignore')
        self.model.fit(feature)

    # 对输入数据运行模型
    def get_score(self, input_data):
        return self.model.score(input_data)

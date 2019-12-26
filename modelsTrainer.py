import numpy as np

from final.model.hmmModel import HmmModel
from final.model.svmModel import SvmModel
from final.model.gmmModel import GmmModel


class ModelsTrainer:
    '''
    统一的 模型接口
    '''

    def __init__(self):
        # self.models = [GmmModel(), HmmModel(),SvmModel()]
        self.models = [GmmModel(), HmmModel()]
        self.labels = []
        self.features = []

    def collect_features(self, label, feature):
        if label not in self.labels:
            self.labels.append(label)
            self.features.append([feature])

        else:
            for i, l in enumerate(self.labels):
                if label == l:
                    self.features[i].append(feature)
                    return

    def train(self):
        for i, label in enumerate(self.labels):
            features = self.features[i]
            for model in self.models:
                model.train(label, features)

    def predict(self, feature):
        result = {}
        for model in self.models:
            predicts = model.predict(feature)
            result[model.name] = predicts

        return result


if __name__ == '__main__':
    trainer = ModelsTrainer()

    for i in range(10):
        for j in range(3):
            trainer.collect_features(i, np.array([j]))

    print(trainer)

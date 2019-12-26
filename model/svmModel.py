from sklearn.svm import SVC
import numpy as np
from final.tool import convent2_line


class SvmModel:
    '''
    hmm 模型的训练接口
    '''

    def __init__(self):
        self.name = 'svm'
        self.svm_model = SVC(kernel='linear', C=0.8, decision_function_shape='ovr')
        self.flag = False
        self.X = []
        self.Y = []

    def train(self, label, features):
        if len(self.X) == 0:
            self.X = np.asarray(convent2_line(features[0]))
            self.Y = np.asarray((label))

        for feature in features:
            self.X = np.vstack((self.X, convent2_line(feature)))
            self.Y = np.vstack((self.Y, label))

    def predict(self, feature):
        if not self.flag:
            self.svm_model.fit(self.X, self.Y)
            self.flag = True

        predict = self.svm_model.predict(np.asarray([convent2_line(feature)]))
        return [predict]

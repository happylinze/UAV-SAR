import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

# 0: 128frame_1\epoch1_test_score.pkl
# 1: 32frame_1\epoch1_test_score.pkl
# 2: angle_1\epoch1_test_score.pkl
# 3: FocalLoss_1\epoch1_test_score.pkl
# 4: FR_Head_1\epoch1_test_score.pkl
# 5: FR_Head_2\epoch1_test_score.pkl
# 6: FR_Head_6\epoch1_test_score.pkl
# 7: motion_1\epoch1_test_score.pkl
# 8: motion_2\epoch1_test_score.pkl
# 9: motion_6\epoch1_test_score.pkl

# 10: angle_1\epoch1_test_score.pkl
# 11: motion_1\epoch1_test_score.pkl
# 12: motion_2\epoch1_test_score.pkl
# 13: motion_6\epoch1_test_score.pkl
# 14: _1\epoch1_test_score.pkl
# 15: _2\epoch1_test_score.pkl
# 16: _6\epoch1_test_score.pkl

# 17: angle\epoch1_test_score.pkl
# 18: b\epoch1_test_score.pkl
# 19: j\epoch1_test_score.pkl
# 20: m\epoch1_test_score.pkl

# CSv1: best_alphas:[0, 0.5, 2, 0.5, 0.5, 1, 0.5, 0.5, 1, 0.5,
#                     1, 1, 1, 1, 0.5, 0.5, 0.5,
#                     0.5, 1.5, 1.5, 1]
# CSv2: best_alphas:[1, 0, 0, 0, 1, 1.5, 1, 0.5, 0.5, 0.5,
#                     2, 1.5, 1.5, 1.5, 1, 1, 1,
#                     2, 1.5, 1.5, 1]

class EvalModel():
    def __init__(self, dir_path, datacase):
        self.datacase = datacase
        self.dir_path = dir_path
        self.ensemble_alphas = None
        self.infogcn_alphas = None
        self.mixformer_alphas = None
        self.sttformer_alphas = None
        self.scores = None
        self.N = 0
        self.num_class = 155
        self.load_scores()

    def load_scores(self):
        self.scores = []
        for model_name in os.listdir(self.dir_path):
            for dc_name in os.listdir(os.path.join(self.dir_path, model_name)):
                if dc_name[-5:] == self.datacase:
                    for m in os.listdir(os.path.join(self.dir_path, model_name, dc_name)):
                        pkl_path = os.path.join(self.dir_path, model_name, dc_name, m, 'epoch1_test_score.pkl')
                        with open(pkl_path, 'rb') as f:
                            a = list(pickle.load(f).items())
                            b = []
                            for i in a:
                                b.append(i[1])
                            self.scores.append(np.array(b))
        self.scores = np.array(self.scores)
        self.N = self.scores.shape[1]
        self.infogcn_alphas = np.array([1] * 10)
        self.mixformer_alphas = np.array([1] * 7)
        self.sttformer_alphas = np.array([1] * 4)
        self.ensemble_alphas = np.array([1] * 21)

    def adjust_alphas(self, mix_alphas, infogcn_alphas=None, mixformer_alphas=None, sttformer_alphas=None):
        assert len(mix_alphas) == len(self.ensemble_alphas)
        self.ensemble_alphas = np.array(mix_alphas)
        if infogcn_alphas is not None:
            assert len(infogcn_alphas) == len(self.infogcn_alphas)
            self.infogcn_alphas = np.array(infogcn_alphas)
        if mixformer_alphas is not None:
            assert len(mixformer_alphas) == len(self.mixformer_alphas)
            self.mixformer_alphas = np.array(mixformer_alphas)
        if sttformer_alphas is not None:
            assert len(sttformer_alphas) == len(self.sttformer_alphas)
            self.sttformer_alphas = np.array(sttformer_alphas)

    def forward(self, model):
        pred_score = np.zeros_like(self.scores)
        if model == 'ensemble_model':
            for i, _ in enumerate(self.ensemble_alphas):
                pred_score += self.scores[i] * self.ensemble_alphas[i]
        elif model == 'infogcn':
            for i, _ in enumerate(self.infogcn_alphas):
                pred_score += self.scores[i] * self.ensemble_alphas[i]
        elif model == 'mixformer':
            for i, _ in enumerate(self.mixformer_alphas):
                pred_score += self.scores[10 + i] * self.mixformer_alphas[i]
        elif model == 'sttformer':
            for i, _ in enumerate(self.sttformer_alphas):
                pred_score += self.scores[16 + i] * self.sttformer_alphas[i]
        else:
            raise ValueError('Unknown model')
        pred_score = pred_score.sum(axis=0)
        pred = pred_score.argmax(axis=-1)
        return pred

    def evaluate(self, label, model):
        pre = self.forward(model)
        acc = accuracy_score(label, pre)
        if model == 'ensemble_model':
            print(f'{self.datacase} acc:{acc}')
        elif model == 'infogcn':
            print(f'{self.datacase} acc:{acc}')
        elif model == 'mixformer':
            print(f'{self.datacase} acc:{acc}')
        elif model == 'sttformer':
            print(f'{self.datacase} acc{acc}')
        else:
            raise ValueError('Unknown model')
        return acc


if __name__ == '__main__':
    npz_data_v1 = np.load('./data/uav/MMVRAC_CSv1.npz')
    npz_data_v2 = np.load('data/uav/MMVRAC_CSv2.npz')
    label_v1 = np.where(npz_data_v1['y_test'] > 0)[1]
    label_v2 = np.where(npz_data_v2['y_test'] > 0)[1]
    evalModel_v1 = EvalModel('ensemble_results', 'CSv1')
    evalModel_v2 = EvalModel('ensemble_results', 'CSv2')

    evalModel_v1.adjust_alphas([0, 0.5, 2, 0.5, 0.5, 1, 0.5, 0.5, 1, 0.5,
                                1, 1, 1, 1, 0.5, 0.5, 0.5,
                                0.5, 1.5, 1.5, 1])
    evalModel_v2.adjust_alphas([1, 0, 0, 0, 1, 1.5, 1, 0.5, 0.5, 0.5,
                                2, 1.5, 1.5, 1.5, 1, 1, 1,
                                2, 1.5, 1.5, 1])
    evalModel_v1.evaluate(label_v1, 'ensemble_model')
    evalModel_v2.evaluate(label_v2,'ensemble_model')
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold


class DBLPEvaluator:

    def __init__(self, config):
        self.config = config
        root = 'explib/raw_data/dblp_processed/'
        fname = 'aid_label.csv'
        if config.use_mini:
            fname = '2011_mini_' + fname
        self.label = pd.read_csv(os.path.join(root, fname))

    def evaluate(self, embds):
        X = embds[self.label.aid]
        y = self.label.label
        nmi_score = self.cluster(X, y)
        acc_score = self.classification(X, y)

        metrics = {'NMI': nmi_score,
                   'ACC': acc_score}

        return metrics

    def classification(self, X, y):
        kf = KFold(3)
        scores = []
        for train_idx, test_idx in kf.split(X):
            model = LogisticRegressionCV()
            model.fit(X[train_idx], y[train_idx])
            pred_y = model.predict(X[test_idx])
            scores.append((pred_y == y[test_idx]).mean())
        return np.mean(scores)

    def cluster(self, X, y):
        # K-Means
        kmeans = KMeans(n_clusters=4)
        pred_y = kmeans.fit_predict(X)
        scores = NMI(y, pred_y)
        return scores

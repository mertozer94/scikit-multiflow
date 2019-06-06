import numpy as np
from array import array
import os
from skmultiflow.data import SEAGenerator, RandomTreeGenerator
from skmultiflow.anomaly_detection import HalfSpaceTrees
from skmultiflow.core.base import is_classifier
import pandas as pd
from sklearn.preprocessing import minmax_scale

from sklearn.metrics import roc_auc_score

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    test_hoeffding_tree("teast")

def test_hoeffding_tree(test_path):
    df = pd.read_csv('/home/mert/Downloads/data.csv', sep=',')

    Y_train = df["Target"].as_matrix()
    X_train = df.drop(labels=["Target"], axis=1).as_matrix()

    X_train = minmax_scale(X_train)

    total_lenght = 0
    total_ones = 0
    j = 0
    for i in range(0, 1):
        stream = SEAGenerator(random_state=1, noise_percentage=0.20)

        stream.prepare_for_use()

        learner = HalfSpaceTrees(dimensions=4, n_estimators=27, size_limit=70, anomaly_threshold=0.30, depth=20)

        cnt = 0
        max_samples = 100000
        predictions = array('i')
        proba_predictions = []
        wait_samples = 2000
        real_anom_cnt = 0
        match = 0
        pred_shape1 = []
        real_y = []
        while cnt < max_samples:
            X1, y1 = stream.next_sample()
            # X = X_train.iloc[i, :]
            # y = Y_train.iloc[i, :]

            X = X_train[j].reshape(1, 4)


            y = Y_train[j]

            if y == '\'Normal\'':

                y = [0]
            else :
                y = [1]

            # Test every n samples
            if (cnt % wait_samples == 0) and (cnt != 0):
                pre = learner.predict(X)[0]
                predictions.append(pre)
                prediction_sc = learner.predict_proba(X)[0]
                proba_predictions.append(prediction_sc)
                if(prediction_sc[0] < prediction_sc[1]):
                    pred_shape1.append(prediction_sc[0])
                else:
                    pred_shape1.append(prediction_sc[1])

                if y == [1]:
                    real_y.append(1)
                    real_anom_cnt = real_anom_cnt + 1
                    if pre == 1:
                        match = match + 1
                else:
                    real_y.append(0)
            learner.partial_fit(X, y)
            cnt += 1
            j = j + 1
        total_lenght += predictions.__len__()
        total_ones += predictions.count(1)

    print(roc_auc_score(real_y, pred_shape1))
    print("matching =", match)
    print("real pred count =", real_anom_cnt)
    print("predictions total length = ", total_lenght)
    print("we found anomaly count = ", total_ones)

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

def testa_naive_bayes(test_path):
    stream = SEAGenerator(random_state=1)
    stream.prepare_for_use()

    learner = HalfSpaceTrees(dimensions=3, n_estimators=25, size_limit=25, anomaly_threshold=0.5, depth=15)

    cnt = 0
    max_samples = 5000
    y_pred = array('i')
    X_batch = []
    y_batch = []
    y_proba = []
    wait_samples = 250

    while cnt < max_samples:
        X, y = stream.next_sample()
        X_batch.append(X[0])
        y_batch.append(y[0])
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred.append(learner.predict(X)[0])
            y_proba.append(learner.predict_proba(X)[0])

        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('i', [1, 1, 1, 0, 1, 1, 1, 0, 0, 1,
                                       1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                                       0, 0, 1, 1, 0, 0, 1, 0, 1, 1,
                                       1, 1, 0, 1, 0, 0, 1, 1, 1])

    assert np.alltrue(y_pred == expected_predictions)

    test_file = os.path.join(test_path, 'data_naive_bayes_proba.npy')
    y_proba_expected = np.load(test_file)
    assert np.allclose(y_proba, y_proba_expected)

    expected_info = 'NaiveBayes(nominal_attributes=None)'
    assert learner.get_info() == expected_info

    learner.reset()
    learner.fit(X=np.array(X_batch[:4500]), y=np.array(y_batch[:4500]))

    expected_score = 0.9378757515030061
    assert np.isclose(expected_score, learner.score(X=np.array(X_batch[4501:]),
                                                    y=np.array(y_batch[4501:])))

    assert is_classifier(learner)

    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray


if __name__ == "__main__":
    main()

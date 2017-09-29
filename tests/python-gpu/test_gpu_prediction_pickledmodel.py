from __future__ import print_function

import numpy as np
import unittest
import xgboost as xgb
from nose.plugins.attrib import attr
import sys

rng = np.random.RandomState(1994)

import _pickle as pickle
def save_obj(obj, name):
    # print("Saving %s" % name)
    with open(name, 'wb') as f:
        pickle.dump(obj=obj, file=f)
        # os.sync()


def load_obj(name):
    # print("Loading %s" % name)
    with open(name, 'rb') as f:
        return pickle.load(f)

@attr('gpu')
class TestGPUPredict(unittest.TestCase):
    def test_predict(self):
        iterations = 1
        np.random.seed(1)
        num_rows = 5000
        num_cols = 500
        dm = xgb.DMatrix(np.random.randn(num_rows, num_cols), label=[0, 1] * int(num_rows / 2))
        watchlist = [(dm, 'train')]
        res = {}
        param = {
            "objective": "binary:logistic",
            "predictor": "gpu_predictor",
            'eval_metric': 'auc',
        }
        bst = xgb.train(param, dm, iterations, evals=watchlist, evals_result=res)
        assert self.non_decreasing(res["train"]["auc"])

        # pickle model
        save_obj(bst,"bst.pkl")
        # delete model
        del bst
        # load model
        bst = load_obj("bst.pkl")

        # continue as before
        gpu_pred = bst.predict(dm, output_margin=True)
        print("Now try CPU")
        sys.stdout.flush()
        bst.set_param({"predictor": "cpu_predictor"})
        cpu_pred = bst.predict(dm, output_margin=True)
        np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)

    def test_predict_sklearn(self):
        iterations = 1
        np.random.seed(1)
        num_rows = 5000
        num_cols = 500
        X = np.random.randn(num_rows, num_cols)
        y= label=[0, 1] * int(num_rows / 2)

        from xgboost import XGBClassifier
        kwargs={}
        kwargs['tree_method'] = 'gpu_hist'
        kwargs['predictor'] = 'gpu_predictor'
        kwargs['silent'] = 0
        kwargs['objective'] = 'binary:logistic'

        model = XGBClassifier(**kwargs)
        model.fit(X,y)
        print(model)

        # pickle model
        save_obj(model,"model.pkl")
        # delete model
        del model
        # load model
        model = load_obj("model.pkl")

        # continue as before
        gpu_pred = model.predict(y, output_margin=True)
        print(gpu_pred)
        # ISSUE1: Doesn't use gpu_predictor.
        # ISSUE2: Also, no way to switch to cpu_predictor?
        #np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)

    def non_decreasing(self, L):
        return all((x - y) < 0.001 for x, y in zip(L, L[1:]))

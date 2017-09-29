from __future__ import print_function

import numpy as np
import unittest
import xgboost as xgb
from nose.plugins.attrib import attr
import os, sys
import time

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

num_rows = 5000
num_cols = 500
n_estimators = 1


def makeXy():
    np.random.seed(1)
    X = np.random.randn(num_rows, num_cols)
    y = [0, 1] * int(num_rows / 2)
    return X,y

def makeXtest():
    np.random.seed(1)
    Xtest = np.random.randn(num_rows, num_cols)
    return Xtest


@attr('gpu')
class TestGPUPredict(unittest.TestCase):
    def test_predict_nopickle(self):
        dm = xgb.DMatrix(np.random.randn(num_rows, num_cols), label=[0, 1] * int(num_rows / 2))
        watchlist = [(dm, 'train')]
        res = {}
        param = {
            "objective": "binary:logistic",
            "predictor": "gpu_predictor",
            'eval_metric': 'auc',
        }
        bst = xgb.train(param, dm, n_estimators, evals=watchlist, evals_result=res)
        assert self.non_decreasing(res["train"]["auc"])

        print("Before model.predict on GPU")
        sys.stdout.flush()
        tmp = time.time()
        gpu_pred = bst.predict(dm, output_margin=True)
        print(gpu_pred)
        print("non-zeroes: %d:" % (np.count_nonzero(gpu_pred)))
        print("GPU Time to predict = %g" % (time.time() - tmp))
        print("Before model.predict on CPU")
        sys.stdout.flush()
        bst.set_param({"predictor": "cpu_predictor"})
        tmp = time.time()
        cpu_pred = bst.predict(dm, output_margin=True)
        print(cpu_pred)
        print("non-zeroes: %d:" % (np.count_nonzero(cpu_pred)))
        print("CPU Time to predict = %g" % (time.time() - tmp))
        np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)

    def test_predict_nodm(self):
        dm = xgb.DMatrix(np.random.randn(num_rows, num_cols), label=[0, 1] * int(num_rows / 2))
        Xtest = makeXtest()

        watchlist = [(dm, 'train')]
        res = {}
        param = {
            "objective": "binary:logistic",
            "predictor": "gpu_predictor",
            'eval_metric': 'auc',
        }
        bst = xgb.train(param, dm, n_estimators, evals=watchlist, evals_result=res)
        assert self.non_decreasing(res["train"]["auc"])

        print("Before model.predict on GPU")
        sys.stdout.flush()
        tmp = time.time()
        dm_test = xgb.DMatrix(Xtest)
        gpu_pred = bst.predict(dm_test, output_margin=True)
        print(gpu_pred)
        print("non-zeroes: %d:" % (np.count_nonzero(gpu_pred)))
        print("GPU Time to predict = %g" % (time.time() - tmp))
        print("Before model.predict on CPU")
        sys.stdout.flush()
        bst.set_param({"predictor": "cpu_predictor"})
        tmp = time.time()
        cpu_pred = bst.predict(dm_test, output_margin=True)
        print(cpu_pred)
        print("non-zeroes: %d:" % (np.count_nonzero(cpu_pred)))
        print("CPU Time to predict = %g" % (time.time() - tmp))
        np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)

    def test_predict_pickle(self):
        dm = xgb.DMatrix(np.random.randn(num_rows, num_cols), label=[0, 1] * int(num_rows / 2))
        watchlist = [(dm, 'train')]
        res = {}
        param = {
            "objective": "binary:logistic",
            "predictor": "gpu_predictor",
            'eval_metric': 'auc',
        }
        bst = xgb.train(param, dm, n_estimators, evals=watchlist, evals_result=res)
        assert self.non_decreasing(res["train"]["auc"])

        # pickle model
        save_obj(bst,"bst.pkl")
        # delete model
        del bst
        # load model
        bst = load_obj("bst.pkl")
        os.remove("bst.pkl")

        # continue as before
        print("Before model.predict on GPU")
        sys.stdout.flush()
        tmp = time.time()
        gpu_pred = bst.predict(dm, output_margin=True)
        print(gpu_pred)
        print("non-zeroes: %d:" % (np.count_nonzero(gpu_pred)))
        print("GPU Time to predict = %g" % (time.time() - tmp))
        print("Before model.predict on CPU")
        sys.stdout.flush()
        bst.set_param({"predictor": "cpu_predictor"})
        tmp = time.time()
        cpu_pred = bst.predict(dm, output_margin=True)
        print(cpu_pred)
        print("non-zeroes: %d:" % (np.count_nonzero(cpu_pred)))
        print("CPU Time to predict = %g" % (time.time() - tmp))
        np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)

    def test_predict_sklearn_nopickle(self):
        X,y = makeXy()
        Xtest = makeXtest()

        from xgboost import XGBClassifier
        kwargs={}
        kwargs['tree_method'] = 'gpu_hist'
        kwargs['predictor'] = 'gpu_predictor'
        kwargs['silent'] = 0
        kwargs['objective'] = 'binary:logistic'

        model = XGBClassifier(n_estimators=n_estimators, **kwargs)
        model.fit(X,y)
        print(model)

        # continue as before
        print("Before model.predict")
        sys.stdout.flush()
        tmp = time.time()
        gpu_pred = model.predict(Xtest, output_margin=True)
        print(gpu_pred)
        print("non-zeroes: %d:" % (np.count_nonzero(gpu_pred)))
        print("Time to predict = %g" % (time.time() - tmp))
        # MAJOR issue: gpu predictions wrong  -- all zeros
        # ISSUE1: Doesn't use gpu_predictor.
        # ISSUE2: Also, no way to switch to cpu_predictor?
        #np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)

    def test_predict_sklearn_pickle(self):
        X,y = makeXy()
        Xtest = makeXtest()

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
        os.remove("model.pkl")

        # continue as before
        print("Before model.predict")
        sys.stdout.flush()
        tmp = time.time()
        gpu_pred = model.predict(Xtest, output_margin=True)
        print(gpu_pred)
        print("non-zeroes: %d:" % (np.count_nonzero(gpu_pred)))
        print("Time to predict = %g" % (time.time() - tmp))
        # ISSUE1: Doesn't use gpu_predictor.
        # ISSUE2: Also, no way to switch to cpu_predictor?
        #np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)


    # only run the below after the above
    def test_predict_sklearn_frompickle(self):
        Xtest = makeXtest()

        # load model
        model = load_obj("./tests_open/model_saved.pkl")

        # continue as before
        print("Before model.predict")
        sys.stdout.flush()
        tmp = time.time()
        gpu_pred = model.predict(Xtest, output_margin=True)
        print(gpu_pred)
        print("non-zeroes: %d:" % (np.count_nonzero(gpu_pred)))
        print("Time to predict = %g" % (time.time() - tmp))
        # ISSUE1: Doesn't use gpu_predictor.
        # ISSUE2: Also, no way to switch to cpu_predictor?
        #np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)

    def non_decreasing(self, L):
        return all((x - y) < 0.001 for x, y in zip(L, L[1:]))

import os, glob, tempfile
import numpy as np
from traitlets import (HasTraits, 
                       Integer, 
                       Unicode, 
                       Float, 
                       Integer, 
                       Instance, 
                       Dict, 
                       Bool,
                       default)
# Rpy

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri

from gaussian_methods import lasso_1se, parametric_method

# POSI selection
# since there are no p-values we just use
# marginal check, seeing if 0 in confidence interval

# in order to make speed tolerable,
# we compute POSI constant K on one instance only,
# similar to knockoffs 

class POSI90(parametric_method):

    method_name = Unicode("POSI")
    selectiveR_method = True
    selection_method = lasso_1se
    model_target = Unicode("selected")

    @classmethod
    def setup(cls, feature_cov, data_generating_mechanism, max_model_size=6, level=0.90):
        cls.feature_cov = feature_cov
        cls.data_generating_mechanism = data_generating_mechanism
        cls.noise = data_generating_mechanism.noise
        numpy2ri.activate()

        # see if we've factored this before

        have_POSI_K = False
        if not os.path.exists('.POSI_data'):
            os.mkdir('.POSI_data')
        posi_data = glob.glob('.POSI_data/*npz')
        for posi_file in posi_data:
            posi = np.load(posi_file)
            posi_f = posi['feature_cov']
            if ((posi_f.shape == feature_cov.shape) and
                np.allclose(posi_f, feature_cov) and 
                (posi['max_model_size'] == max_model_size) and 
                (posi['level'] == level)):
                have_POSI_K = True
                print('found POSI instance: %s' % posi)
                cls.POSI_K = float(posi['K'])

        if not have_POSI_K:
            print('simulating POSI constant')
            cls.POSI_K = float(POSI_instance(feature_cov, max_model_size, n=10*feature_cov.shape[0]))

        numpy2ri.deactivate()

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):
        parametric_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self._method = self.selection_method(X, Y, l_theory, l_min, l_1se, sigma_reid)

    def select(self):
        active_set = self._method.generate_pvalues()[0] # gives us selected variables at 1SE
        if len(active_set) > 0:
            numpy2ri.activate()
            rpy.r.assign("X", self.X[:, active_set])
            rpy.r.assign("Y", self.Y)
            rpy.r.assign("K", self.POSI_K)
            rpy.r('M = lm(Y ~ X - 1)')
            rpy.r('L = coef(M) - K * sqrt(diag(vcov(M)))')
            rpy.r('U = coef(M) + K * sqrt(diag(vcov(M)))')
            L = rpy.r('L')
            U = rpy.r('U')
            numpy2ri.deactivate()

            pre_select = np.nonzero((L > 0) + (U < 0))[0]
            selected = [active_set[i] for i in pre_select]
            return selected, active_set
        else:
            return [], []

    def generate_intervals(self):
        active_set = self._method.generate_pvalues()[0] # gives us selected variables at 1SE
        if len(active_set) > 0:
            numpy2ri.activate()
            rpy.r.assign("X", self.X[:, active_set])
            rpy.r.assign("Y", self.Y)
            rpy.r.assign("K", self.POSI_K)
            rpy.r('M = lm(Y ~ X - 1)')
            rpy.r('L = coef(M) - K * sqrt(diag(vcov(M)))')
            rpy.r('U = coef(M) + K * sqrt(diag(vcov(M)))')
            L = rpy.r('L')
            U = rpy.r('U')
            numpy2ri.deactivate()
            
            return active_set, L, U
        else:
            return [], [], []

    def generate_pvalues(self):
        raise NotImplementedError

    def point_estimator(self):
        raise NotImplementedError

POSI90.register()

class POSI80(POSI90):

    method_name = Unicode("POSI")
    selectiveR_method = True
    selection_method = lasso_1se
    model_target = Unicode("selected")

    @classmethod
    def setup(cls, feature_cov, data_generating_mechanism, max_model_size=6, level=0.80):
        cls.feature_cov = feature_cov
        cls.data_generating_mechanism = data_generating_mechanism
        cls.noise = data_generating_mechanism.noise
        numpy2ri.activate()

        # see if we've factored this before

        have_POSI_K = False
        if not os.path.exists('.POSI_data'):
            os.mkdir('.POSI_data')
        posi_data = glob.glob('.POSI_data/*npz')
        for posi_file in posi_data:
            posi = np.load(posi_file)
            posi_f = posi['feature_cov']
            if ((posi_f.shape == feature_cov.shape) and
                np.allclose(posi_f, feature_cov) and 
                (posi['max_model_size'] == max_model_size) and 
                (posi['level'] == level)):
                have_POSI_K = True
                print('found POSI instance: %s' % posi)
                cls.POSI_K = float(posi['K'])

        if not have_POSI_K:
            print('simulating POSI constant')
            cls.POSI_K = float(POSI_instance(feature_cov, max_model_size, n=10*feature_cov.shape[0]))

        numpy2ri.deactivate()

POSI80.register()

class POSI95(POSI90):

    method_name = Unicode("POSI")
    selectiveR_method = True
    selection_method = lasso_1se
    model_target = Unicode("selected")

    @classmethod
    def setup(cls, feature_cov, data_generating_mechanism, max_model_size=6, level=0.95):
        cls.feature_cov = feature_cov
        cls.data_generating_mechanism = data_generating_mechanism
        cls.noise = data_generating_mechanism.noise
        numpy2ri.activate()

        # see if we've factored this before

        have_POSI_K = False
        if not os.path.exists('.POSI_data'):
            os.mkdir('.POSI_data')
        posi_data = glob.glob('.POSI_data/*npz')
        for posi_file in posi_data:
            posi = np.load(posi_file)
            posi_f = posi['feature_cov']
            if ((posi_f.shape == feature_cov.shape) and
                np.allclose(posi_f, feature_cov) and 
                (posi['max_model_size'] == max_model_size) and 
                (posi['level'] == level)):
                have_POSI_K = True
                print('found POSI instance: %s' % posi)
                cls.POSI_K = float(posi['K'])

        if not have_POSI_K:
            print('simulating POSI constant')
            cls.POSI_K = float(POSI_instance(feature_cov, max_model_size, n=10*feature_cov.shape[0]))

        numpy2ri.deactivate()

POSI95.register()

def POSI_instance(feature_cov, max_model_size, n, level=np.array([0.8,0.9,0.95,0.99])):

    numpy2ri.activate()
    rpy.r.assign('Sigma', feature_cov)
    chol = np.linalg.cholesky(feature_cov)
    p = feature_cov.shape[0]
    X = np.random.standard_normal((n, p)).dot(chol.T)

    rpy.r.assign('X', X)
    rpy.r.assign('max_model_size', max_model_size)
    rpy.r.assign('level', level)
    rpy.r('''
    library(PoSI)
    posi_obj = PoSI(X, modelSZ=1:max_model_size)
    posi_K = summary(posi_obj, confidence=level)[,'K.PoSI']
    ''')
    K = rpy.r("posi_K")

    for i, l in enumerate(level):
        print(K[i])
        np.savez('.POSI_data/%s.npz' % (os.path.split(tempfile.mkstemp()[1])[1],),
                 K=K[i],
                 feature_cov=feature_cov,
                 level=l,
                 max_model_size=max_model_size)

    return K


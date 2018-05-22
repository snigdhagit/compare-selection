import numpy as np

from traitlets import (HasTraits, 
                       Integer, 
                       Unicode, 
                       Float, 
                       Integer, 
                       Instance, 
                       Dict, 
                       default, 
                       observe)

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri
rpy.r('library(glmnet)')

from instances import (equicor_instance, 
                       gaussian_instance, 
                       randomize_signs)

def low_signal(n=100, p=50, B=1000):
    sample_ = np.zeros(B)
    for b in range(B):
        X = np.random.standard_normal((n, p))
        Y = np.random.standard_normal(n)
        sample_[b] = np.fabs(X.T.dot(Y) / n).max()
    return np.median(sample_)

def high_signal(n=100, p=50, B=1000):
    sample_ = np.zeros(B)
    for b in range(B):
        X = np.random.standard_normal((n, p))
        Y = np.random.standard_normal(n)
        sample_[b] = np.fabs(X.T.dot(Y) / n).max()
    return np.percentile(sample_, 99.) + 0.25

def lam_CV(instance, B=100):
    sample_ = np.zeros((B, 2))
    numpy2ri.activate()
    for b in range(B):
        X, Y, _ = instance.generate()
        rpy.r.assign('x', X)
        rpy.r.assign('y', Y)
        rpy.r('y = as.numeric(y)')
        rpy.r('''
    G = cv.glmnet(x, y, standardize=FALSE, intercept=FALSE)
    lam.min = G$lambda.min
    lam.1se = G$lambda.1se
    ''')
        sample_[b] = (rpy.r('lam.min'), rpy.r('lam.1se'))
    numpy2ri.deactivate()
    return np.median(sample_, 0)

class liu_null(equicor_instance):

    n = Integer(100)
    p = Integer(50)
    s = Integer(5)
    instance_name = Unicode('null')
    signal = Float(0)
    penalty = Float(1.)

    def generate_X(self):

        n, p, s = self.n, self.p, self.s
        X = gaussian_instance(n=n, p=p, equicorrelated=False, rho=0.)[0]

        beta = np.zeros(p)
        beta[:s] = self.signal
        np.random.shuffle(beta)
        beta = randomize_signs(beta)

        X *= np.sqrt(n)
        return X

    def generate(self):
        (n, p, s) = (self.n,
                     self.p,
                     self.s)

        X = self.generate_X()

        beta = np.zeros(p)
        beta[:s] = self.signal
        np.random.shuffle(beta)
        Y = X.dot(beta) + np.random.standard_normal(n)
        return X, Y, beta

    @default('feature_cov')
    def _default_feature_cov(self):
        _feature_cov = np.identity(self.p)
        return _feature_cov

    @default('penalty')
    def _default_penalty(self):
        _penalty = np.sqrt(2 * np.log(self.p) / self.n)
        return _penalty

liu_null.register()

class liu_low(liu_null):

    instance_name = Unicode('low')

    @default('signal')
    def _default_signal(self):
        _signal = low_signal(n=self.n, p=self.p)
        return _signal

liu_low.register()

class liu_high(liu_null):

    instance_name = Unicode('high')

    @default('signal')
    def _default_signal(self):
        _signal = high_signal(n=self.n, p=self.p)
        return _signal

liu_high.register()

class liu_null_CV(liu_null):

    instance_name = Unicode('null_CV')

    @default('penalty')
    def _default_penalty(self):
        _penalty = lam_CV(self)[0]
        return _penalty
    
liu_null_CV.register()

class liu_low_CV(liu_low):

    instance_name = Unicode('low_CV')

    @default('penalty')
    def _default_penalty(self):
        _penalty = lam_CV(self)[0]
        return _penalty
    
liu_low_CV.register()

class liu_high_CV(liu_high):

    instance_name = Unicode('high_CV')

    @default('penalty')
    def _default_penalty(self):
        _penalty = lam_CV(self)[0]
        return _penalty
    
liu_high_CV.register()

class liu_null_1se(liu_null):

    instance_name = Unicode('null_1se')

    @default('penalty')
    def _default_penalty(self):
        _penalty = lam_CV(self)[1]
        return _penalty
    
liu_null_1se.register()

class liu_low_1se(liu_low):

    instance_name = Unicode('low_1se')

    @default('penalty')
    def _default_penalty(self):
        _penalty = lam_CV(self)[1]
        return _penalty
    
liu_low_1se.register()

class liu_high_1se(liu_high):

    instance_name = Unicode('high_1se')

    @default('penalty')
    def _default_penalty(self):
        _penalty = lam_CV(self)[1]
        return _penalty
    
liu_high_1se.register()

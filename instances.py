import numpy as np
import pandas as pd
from traitlets import (HasTraits, 
                       Integer, 
                       Unicode, 
                       Float, 
                       Integer, 
                       Instance, 
                       Dict, 
                       default, 
                       observe)

from selection.tests.instance import gaussian_instance

def randomize_signs(beta):
    return beta * (2 * np.random.binomial(1, 0.5, size=beta.shape) - 1)

data_instances = {}

class data_instance(HasTraits):

    distance_tol = Float(0)
    cor_thresh = Float(0.5)

    def generate(self):
        raise NotImplementedError('abstract method should return (X,Y,beta)')

    @classmethod
    def register(cls):
        data_instances[cls.__name__] = cls

    def discoveries(self, selected, truth):
        """
        A discovery is within a certain distance of a true signal
        """

        delta = np.fabs(np.subtract.outer(np.asarray(selected), np.asarray(truth))).min(1)

        return (delta <= self.distance_tol).sum()

class equicor_instance(data_instance):

    instance_name = Unicode('Exchangeable')
    n = Integer(500)
    p = Integer(200)
    s = Integer(20)
    rho = Float(0.0)
    l_theory = Float()
    feature_cov = Instance(np.ndarray)
    signal = Float(4.)

    @default('l_theory')
    def _default_l_theory(self):
        factor = 3

        nf = 0
        X = []
        self.fixed_l_theory = 0
        while True:
            X.append(self.generate_X())

            n, p = X[0].shape
            nf += n

            if nf > p * factor:
                break
        X = np.vstack(X)
        X /= np.sqrt((X**2).sum(0))[None, :]

        fixed_l_theory = np.fabs(X.T.dot(np.random.standard_normal((nf, 500)))).max(1).mean()
        return fixed_l_theory

    @observe('rho')
    def _observe_rho(self, change):
        rho = change['new']
        cor = rho
        tol = 0
        while cor >= self.cor_thresh:
            cor *= rho
            tol += 1
        self.distance_tol = tol

    @default('feature_cov')
    def _default_feature_cov(self):
        _feature_cov = np.ones((self.p, self.p)) * self.rho + (1 - self.rho) * np.identity(self.p)
        return self._feature_cov

    @property
    def params(self):
        df = pd.DataFrame([[getattr(self, n) for n in self.trait_names() if n != 'feature_cov']],
                          columns=[n for n in self.trait_names() if n != 'feature_cov'])
        return df

    def generate_X(self):

        (n, p, s, rho) = (self.n,
                          self.p,
                          self.s,
                          self.rho)

        X = gaussian_instance(n=n, p=p, equicorrelated=True, rho=rho, s=0)[0]
        X /= np.sqrt((X**2).sum(0))[None, :] 
        X *= np.sqrt(n)

        return X

    def generate(self):
        (n, p, s, rho) = (self.n,
                          self.p,
                          self.s,
                          self.rho)

        X = self.generate_X()

        beta = np.zeros(p)
        beta[:s] = self.signal / np.sqrt(n) # local alternatives
        np.random.shuffle(beta)
        beta = randomize_signs(beta)

        Y = X.dot(beta) + np.random.standard_normal(n)
        return X, Y, beta

    def relative_risk(self, point_estimate, true_parameter):

        delta = point_estimate - true_parameter
        num = np.sum(delta.T * (self.feature_cov.dot(delta)))
        den = np.sum(true_parameter.T * (self.feature_cov.dot(true_parameter)))

        return num / den

equicor_instance.register()

class mixed_instance(equicor_instance):

    instance_name = Unicode('Mixed')
    equicor_rho = Float(0.25)
    AR_weight = Float(0.5)

    def generate_X(self):

        (n, p, s, rho) = (self.n,
                          self.p,
                          self.s,
                          self.rho)

        X_equi = gaussian_instance(n=n, 
                                   p=p, 
                                   equicorrelated=True, 
                                   rho=self.equicor_rho)[0]
        X_AR = gaussian_instance(n=n, 
                                 p=p, 
                                 equicorrelated=False, 
                                 rho=rho)[0]

        X = np.sqrt(self.AR_weight) * X_AR + np.sqrt(1 - self.AR_weight) * X_equi
        X /= np.sqrt((X**2).mean(0))[None, :] 

        return X

    @default('feature_cov')
    def _default_feature_cov(self):
        _feature_cov = 0.5 * (self.rho**np.fabs(np.subtract.outer(np.arange(self.p), np.arange(self.p))) + 
                              np.ones((self.p, self.p)) * self.equicor_rho + (1 - self.equicor_rho) * np.identity(self.p))
        return _feature_cov

mixed_instance.register()


class AR_instance(equicor_instance):

    instance_name = Unicode('AR')

    def generate_X(self):

        n, p, s, rho = self.n, self.p, self.s, self.rho
        X = gaussian_instance(n=n, p=p, equicorrelated=False, rho=rho)[0]

        beta = np.zeros(p)
        beta[:s] = self.signal
        np.random.shuffle(beta)
        beta = randomize_signs(beta)

        X *= np.sqrt(n)
        return X

    @default('feature_cov')
    def _default_feature_cov(self):
        _feature_cov = self.rho**np.fabs(np.subtract.outer(np.arange(self.p), np.arange(self.p)))
        return _feature_cov

AR_instance.register()

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
        raise NotImplementedError('abstract method should return (X, Y, beta)')

    @classmethod
    def register(cls):
        data_instances[cls.__name__] = cls

    def discoveries(self, selected, truth):
        """
        A discovery is within a certain distance of a true signal
        """

        selected = np.asarray(selected)
        truth = np.asarray(truth)
        if selected.shape[0] > 0 and truth.shape[0] > 0:

            # if two or more selected are near one true signal, it will only be counted as one true discovery
            # here is how many true signals were detected -- for each truth, is there a selected within distance_tol?

            diff = np.fabs(np.subtract.outer(np.asarray(selected), np.asarray(truth)))
            num_true_discovered = (diff.min(0) <= self.distance_tol).sum()

            # it is possible that two true signals are very close so one selected find two "true" signals
            # but we should never have more discoveries than -- this should be rare

            num_select_discovered = (diff.min(1) <= self.distance_tol).sum()
            
            # this result is less than both the number of selected as well as the number of true

            return min(num_true_discovered, num_selected_discovered)
        else:
            return 0


class equicor_instance(data_instance):

    instance_name = Unicode('Exchangeable')
    n = Integer(500)
    p = Integer(200)
    s = Integer(20)
    rho = Float(0.0)
    l_theory = Float()
    feature_cov = Instance(np.ndarray)
    signal = Float(4.)
    noise = Float(1.)

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

        if not hasattr(self, "_beta"):
            beta = np.zeros(p)
            beta[:s] = self.signal / np.sqrt(n) # local alternatives
            np.random.shuffle(beta)
            beta = randomize_signs(beta)
            self._beta = beta * self.noise

        Y = X.dot(self._beta) + self.noise * np.random.standard_normal(n)
        return X, Y, self._beta

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

        X *= np.sqrt(n)
        return X

    @default('feature_cov')
    def _default_feature_cov(self):
        _feature_cov = self.rho**np.fabs(np.subtract.outer(np.arange(self.p), np.arange(self.p)))
        return _feature_cov

AR_instance.register()

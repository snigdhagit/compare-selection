import tempfile, os, glob
from scipy.stats import norm as ndist
from traitlets import (HasTraits, 
                       Integer, 
                       Unicode, 
                       Float, 
                       Integer, 
                       Instance, 
                       Dict, 
                       Bool,
                       default)

import numpy as np
import regreg.api as rr

from selectinf.algorithms.lasso import lasso, ROSI, ROSI_modelQ
from selectinf.algorithms.sqrt_lasso import choose_lambda
from selectinf.truncated.gaussian import truncated_gaussian_old as TG
from selectinf.randomized.lasso import lasso as random_lasso_method, form_targets
from selectinf.randomized.modelQ import modelQ as randomized_modelQ
from selectinf.randomized.randomization import randomization

from utils import BHfilter

from selectinf.base import restricted_estimator


# Rpy

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri

methods = {}

class generic_method(HasTraits):

    need_CV = False
    selectiveR_method = False
    wide_ok = True # ok for p>= n?

    # Traits

    q = Float(0.2)
    method_name = Unicode('Generic method')
    model_target = Unicode()

    @classmethod
    def setup(cls, feature_cov, data_generating_mechanism):
        cls.feature_cov = feature_cov
        cls.data_generating_mechanism = data_generating_mechanism
        cls.noise = data_generating_mechanism.noise

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):
        (self.X,
         self.Y,
         self.l_theory,
         self.l_min,
         self.l_1se,
         self.sigma_reid) = (X,
                             Y,
                             l_theory,
                             l_min,
                             l_1se,
                             sigma_reid)

    def select(self):
        raise NotImplementedError('abstract method')

    @classmethod
    def register(cls):
        methods[cls.__name__] = cls

    def selected_target(self, active, beta):
        C = self.feature_cov[active]
        Q = C[:,active]
        return np.linalg.inv(Q).dot(C.dot(beta))

    def full_target(self, active, beta):
        return beta[active]

    def get_target(self, active, beta):
        if self.model_target not in ['selected', 'full', 'debiased']:
            raise ValueError('Gaussian methods only have selected or full targets')
        if self.model_target in ['full', 'debiased']:
            return self.full_target(active, beta)
        else:
            return self.selected_target(active, beta)


class parametric_method(generic_method):

    confidence = Float(0.9)

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):
        generic_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self._fit = False

    def select(self):

        if not self._fit:
            self.method_instance.fit()
            self._fit = True

        active_set, pvalues = self.generate_pvalues()
        if len(pvalues) > 0:
            selected = [active_set[i] for i in BHfilter(pvalues, q=self.q)]
            return selected, active_set
        else:
            return [], active_set

    def generate_summary(self, compute_intervals=False): 
        raise NotImplementedError('abstract method, should return a data frame summary with "variable" denoting active set')

    def generate_pvalues(self):
        raise NotImplementedError('abstract method, should return (active_set, pvalues)')

    def generate_intervals(self):
        raise NotImplementedError('abstract method, should return (active_set, lower_limit, upper_limit)')

    def naive_pvalues(self, active_set):
        """
        selected model
        """

        numpy2ri.activate()

        rpy.r.assign("Y", self.Y)
        if self.model_target == 'selected':
            rpy.r.assign("X", self.X[:, active_set])
            rpy.r('pval = summary(lm(Y ~ X - 1))$coef[,4]')
            pval = np.asarray(rpy.r('pval'))
        else:
            n, p = self.X.shape
            if n > p:
                rpy.r.assign("X", self.X)
                rpy.r('pval = summary(lm(Y ~ X - 1))$coef[,4]')
                pval = np.asarray(rpy.r('pval'))
                pval = pval[active_set]
            else:
                return active_set, np.ones(len(active_set)) * np.nan

        numpy2ri.deactivate()

        return active_set, pval

    def naive_intervals(self, active_set):
        """
        selected model
        """
        
        numpy2ri.activate()
        if self.model_target == 'selected':
            rpy.r.assign("X", self.X[:, active_set])
        else:
            n, p = self.X.shape
            if n > p:
                rpy.r.assign("X", self.X)
            else:
                return (active_set, 
                        np.ones(len(active_set)) * np.nan,
                        np.ones(len(active_set)) * np.nan)

        rpy.r.assign("Y", self.Y)
        rpy.r.assign("level", self.confidence)
        rpy.r('CI = confint(lm(Y ~ X - 1), level=level)')
        CI = np.asarray(rpy.r('CI'))
        if self.model_target != 'selected':
            CI = CI[active_set]
        numpy2ri.deactivate()

        return active_set, CI[:, 0], CI[:, 1]

    def naive_estimator(self, active_set):
        """
        selected model
        """
        
        numpy2ri.activate()
        rpy.r.assign("Y", self.Y)
        if self.model_target == 'selected':
            rpy.r.assign("X", self.X[:, active_set])
            rpy.r('beta_hat = coef(lm(Y ~ X - 1))')
            beta_hat = np.asarray(rpy.r('beta_hat'))
        else:
            n, p = self.X.shape
            if n > p:
                rpy.r.assign("X", self.X)
                rpy.r('beta_hat = coef(lm(Y ~ X - 1))')
                beta_hat = np.asarray(rpy.r('beta_hat'))[active_set]
            else:
                return (active_set, 
                        np.ones(p) * np.nan)
        n, p = self.X.shape
        beta_full = np.zeros(p)
        beta_full[active_set] = beta_hat

        return active_set, beta_full

class ROSI_theory(parametric_method):

    sigma_estimator = Unicode('relaxed')
    method_name = Unicode("ROSI")
    lambda_choice = Unicode("theory")
    model_target = Unicode("debiased")
    dispersion = Float(0.)
    approximate_inverse = Unicode('BN')
    estimator = Unicode("OLS")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        parametric_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        n, p = X.shape
        self.lagrange = l_theory * np.ones(X.shape[1]) * self.noise

    method_name = Unicode("ROSI")

    """
    Force the use of the debiasing matrix.
    """

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            self._method_instance = ROSI.gaussian(self.X, 
                                                  self.Y, 
                                                  self.lagrange * np.sqrt(n), 
                                                  approximate_inverse=self.approximate_inverse)
        return self._method_instance

    def generate_summary(self, compute_intervals=False): 

        if not self._fit:
            self.method_instance.fit()
            self._fit = True

        X, Y, lagrange, L = self.X, self.Y, self.lagrange, self.method_instance
        n, p = X.shape

        if len(L.active) > 0:
            if self.sigma_estimator == 'reid' and n < p:
                dispersion = self.sigma_reid**2
            elif self.dispersion != 0:
                dispersion = self.dispersion
            else:
                dispersion = None
            S = L.summary(compute_intervals=compute_intervals, 
                          dispersion=dispersion, 
                          level=self.confidence)
            return S

    def generate_pvalues(self):
        S = self.generate_summary(compute_intervals=False)
        if S is not None:
            active_set = np.array(S['variable'])
            pvalues = np.asarray(S['pvalue'])
            return active_set, pvalues
        else:
            return [], []

    def generate_intervals(self): 
        S = self.generate_summary(compute_intervals=True)
        if S is not None:
            active_set = np.array(S['variable'])
            pvalues = np.asarray(S['pvalue'])
            lower, upper = (np.asarray(S['lower_confidence']), 
                            np.asarray(S['upper_confidence']))
            return active_set, lower, upper, pvalues
        else:
            return [], [], [], []

    def point_estimator(self):
        S = self.generate_summary(compute_intervals=False)
        n, p = self.X.shape
        beta_full = np.zeros(p)
        if S is not None:
            active_set = np.array(S['variable'])
            if self.estimator == "LASSO":
                beta_full[active_set] = S['lasso']
            elif self.estimator == "OLS":
                beta_full[active_set] = S['onestep']
            return active_set, beta_full
        else:
            return [], beta_full

ROSI_theory.register()

class ROSI_1se(ROSI_theory):

    method_name = Unicode("ROSI")
    need_CV = True

    """
    Force the use of the debiasing matrix.
    """

    lambda_choice = Unicode("1se")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        ROSI_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_1se * np.ones(X.shape[1])
ROSI_1se.register()

# Unrandomized selected

class lasso_theory(parametric_method):
    
    model_target = Unicode("selected")
    method_name = Unicode("Lee")
    estimator = Unicode("OLS")
    dispersion = Float(1.)
    
    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        parametric_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * self.noise

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            self._method_instance = lasso.gaussian(self.X, self.Y, self.lagrange * np.sqrt(n))
        return self._method_instance

    def generate_summary(self, compute_intervals=False): 

        if not self._fit:
            self.method_instance.fit()
            self._fit = True
            
        X, Y, lagrange, L = self.X, self.Y, self.lagrange, self.method_instance

        if len(L.active) > 0:
            S = L.summary(compute_intervals=compute_intervals,
                          alternative='onesided',
                          dispersion=self.dispersion)
            return S

    def generate_pvalues(self):
        S = self.generate_summary(compute_intervals=False)
        if S is not None:
            active_set = np.array(S['variable'])
            pvalues = np.asarray(S['pvalue'])
            return active_set, pvalues
        else:
            return [], []

    def generate_intervals(self): 
        S = self.generate_summary(compute_intervals=True)
        if S is not None:
            active_set = np.array(S['variable'])
            pvalues = np.asarray(S['pvalue'])
            lower, upper = (np.asarray(S['lower_confidence']), 
                            np.asarray(S['upper_confidence']))
            return active_set, lower, upper, pvalues
        else:
            return [], [], []

    def point_estimator(self):
        S = self.generate_summary(compute_intervals=False)
        n, p = self.X.shape
        beta_full = np.zeros(p)
        if S is not None:
            active_set = np.array(S['variable'])
            if self.estimator == "LASSO":
                beta_full[active_set] = S['lasso']
            elif self.estimator == "OLS":
                beta_full[active_set] = S['onestep']
            return active_set, beta_full
        else:
            return [], beta_full

lasso_theory.register()

class lasso_CV(lasso_theory):
    
    need_CV = True

    lambda_choice = Unicode("CV")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        lasso_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_min * np.ones(X.shape[1])

lasso_CV.register()

class lasso_1se(lasso_theory):
    
    need_CV = True

    lambda_choice = Unicode("1se")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        lasso_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_1se * np.ones(X.shape[1])

lasso_1se.register()

class ROSI_CV(ROSI_theory):

    method_name = Unicode("ROSI")
    need_CV = True
    """
    Force the use of the debiasing matrix.
    """

    lambda_choice = Unicode("1se")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        lasso_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_min * np.ones(X.shape[1])
ROSI_CV.register()

# Randomized selected

class randomized_lasso(parametric_method):

    method_name = Unicode("Randomized LASSO")
    model_target = Unicode("selected")
    lambda_choice = Unicode("theory")
    randomizer_scale = Float(1)
    use_MLE = Bool(False)
    ndraw = 10000
    burnin = 2000

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        parametric_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * self.noise

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            mean_diag = np.mean((self.X ** 2).sum(0))
            self._method_instance = random_lasso_method.gaussian(
                self.X,
                self.Y,
                feature_weights=self.lagrange * np.sqrt(n),
                ridge_term=np.std(self.Y) * np.sqrt(mean_diag) / np.sqrt(n),
                randomizer_scale=self.randomizer_scale * np.std(self.Y) * mean_diag)
        return self._method_instance

    def generate_summary(self, compute_intervals=False): 

        X, Y, lagrange, rand_lasso = self.X, self.Y, self.lagrange, self.method_instance
        n, p = X.shape

        if not self._fit:
            signs = self.method_instance.fit()
            self._fit = True

        signs = self.method_instance.selection_variable['sign']
        active_set = np.nonzero(signs)[0]

        active = signs != 0

        kwargs = {}
        if self.model_target == 'debiased':
            kwargs['penalty'] = rand_lasso.penalty
            
        (observed_target, 
         cov_target, 
         cov_target_score, 
         alternatives) = form_targets(self.model_target,
                                      rand_lasso.loglike,
                                      rand_lasso._W,
                                      active,
                                      **kwargs)

        if active.sum() > 0:
            if not self.use_MLE:
                S = rand_lasso.summary(observed_target, 
                                       cov_target, 
                                       cov_target_score, 
                                       alternatives,
                                       ndraw=self.ndraw,
                                       burnin=self.burnin,
                                       level=self.confidence,
                                       compute_intervals=compute_intervals)
            else:
                S = rand_lasso.selective_MLE(observed_target, 
                                             cov_target, 
                                             cov_target_score, 
                                             level=self.confidence)[0]
            return active_set, S
        else:
            return [], None

    def generate_pvalues(self, compute_intervals=False):
        active_set, S = self.generate_summary(compute_intervals=compute_intervals)
        if len(active_set) > 0:
            return active_set, S['pvalue']
        else:
            return [], []

    def generate_intervals(self): 
        active_set, S = self.generate_summary(compute_intervals=True)
        if len(active_set) > 0:
            return active_set, S['lower_confidence'], S['upper_confidence'], S['pvalue']
        else:
            return [], [], [], []

    def point_estimator(self):

        X, Y, lagrange, rand_lasso = self.X, self.Y, self.lagrange, self.method_instance
        n, p = X.shape

        if not self._fit:
            signs = self.method_instance.fit()
            self._fit = True

        signs = self.method_instance.selection_variable['sign']
        active_set = np.nonzero(signs)[0]

        active = signs != 0

        (observed_target, 
         cov_target, 
         cov_target_score, 
         alternatives) = form_targets(self.model_target,
                                      rand_lasso.loglike,
                                      rand_lasso._W,
                                      active)


        if active.sum() > 0:

            _, S = self.generate_summary(compute_intervals=False)
            beta_full = np.zeros(X.shape[1])
            if not self.use_MLE:
                beta_full[active] = S['target']
            else:
                beta_full[active] = S['MLE']
            return active_set, beta_full
        else:
            return [], np.zeros(p)

class randomized_lasso_CV(randomized_lasso):

    need_CV = True

    lambda_choice = Unicode("CV")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_min * np.ones(X.shape[1])

class randomized_lasso_1se(randomized_lasso):

    need_CV = True

    lambda_choice = Unicode("1se")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_1se * np.ones(X.shape[1])

randomized_lasso.register(), randomized_lasso_CV.register(), randomized_lasso_1se.register()


# Randomized full

class randomized_lasso_full(randomized_lasso):

    method_name = Unicode("Randomized One-step")
    model_target = Unicode('debiased')

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * self.noise

class randomized_lasso_full_CV(randomized_lasso_full):

    need_CV = True

    lambda_choice = Unicode("CV")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso_full.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_min * np.ones(X.shape[1])

class randomized_lasso_full_1se(randomized_lasso_full):

    need_CV = True

    lambda_choice = Unicode("1se")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso_full.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_1se * np.ones(X.shape[1])

randomized_lasso_full.register(), randomized_lasso_full_CV.register(), randomized_lasso_full_1se.register()

class data_splitting(parametric_method):

    method_name = Unicode('Data splitting')
    selection_frac = Float(0.5)
    model_target = Unicode("selected")
    lambda_choice = Unicode('theory')

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        parametric_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)

        self.lagrange = l_theory * np.ones(X.shape[1]) * self.noise

        n, p = self.X.shape
        n1 = int(self.selection_frac * n)
        X1, X2 = self.X1, self.X2 = self.X[:n1], self.X[n1:]
        Y1, Y2 = self.Y1, self.Y2 = self.Y[:n1], self.Y[n1:]

        pen = rr.weighted_l1norm(np.sqrt(n1) * self.lagrange, lagrange=1.)
        loss = rr.squared_error(X1, Y1)
        problem = rr.simple_problem(loss, pen)
        soln = problem.solve()

        self.active_set = np.nonzero(soln)[0]
        self.signs = np.sign(soln)[self.active_set]

        self._fit = True

    def generate_pvalues(self):

        X2, Y2 = self.X2[:,self.active_set], self.Y2
        if len(self.active_set) > 0 and len(self.active_set) < X2.shape[0]:
            s = len(self.active_set)
            X2i = np.linalg.inv(X2.T.dot(X2))
            beta2 = X2i.dot(X2.T.dot(Y2))
            resid2 = Y2 - X2.dot(beta2)
            n2 = X2.shape[0]
            sigma2 = np.sqrt((resid2**2).sum() / (n2 - s))
            Z2 = beta2 / np.sqrt(sigma2**2 * np.diag(X2i))
            signed_Z2 = self.signs * Z2
            pvalues = 1 - ndist.cdf(signed_Z2)
            return self.active_set, pvalues
        else:
            return [], []

    def generate_intervals(self):

        X2, Y2 = self.X2[:,self.active_set], self.Y2
        if len(self.active_set) > 0 and len(self.active_set) < X2.shape[0]:
            s = len(self.active_set)
            X2i = np.linalg.inv(X2.T.dot(X2))
            beta2 = X2i.dot(X2.T.dot(Y2))
            resid2 = Y2 - X2.dot(beta2)
            n2 = X2.shape[0]
            sigma2 = np.sqrt((resid2**2).sum() / (n2 - s))
            alpha = 1 - self.confidence
            Z_quant = ndist.ppf(1 - alpha / 2)
            upper = beta2 + Z_quant * np.sqrt(sigma2**2 * np.diag(X2i))
            lower = beta2 - Z_quant * np.sqrt(sigma2**2 * np.diag(X2i))
            Zval = np.fabs(beta2) / np.sqrt(sigma2**2 * np.diag(X2i))
            pval = 2 * ndist.sf(Zval)
            return self.active_set, lower, upper, pval
        else:
            return [], [], [], []
data_splitting.register()

class data_splitting_1se(data_splitting):

    lambda_choice = Unicode('1se')
    need_CV = True

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        parametric_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)

        self.lagrange = l_1se * np.ones(X.shape[1])

        n, p = self.X.shape
        n1 = int(self.selection_frac * n)
        X1, X2 = self.X1, self.X2 = self.X[:n1], self.X[n1:]
        Y1, Y2 = self.Y1, self.Y2 = self.Y[:n1], self.Y[n1:]

        pen = rr.weighted_l1norm(np.sqrt(n1) * self.lagrange, lagrange=1.)
        loss = rr.squared_error(X1, Y1)
        problem = rr.simple_problem(loss, pen)
        soln = problem.solve()

        self.active_set = np.nonzero(soln)[0]
        self.signs = np.sign(soln)[self.active_set]

        self._fit = True
data_splitting_1se.register()

class data_splitting_CV(data_splitting):

    method_name = Unicode('Data splitting')
    selection_frac = Float(0.5)
    model_target = Unicode("selected")
    lambda_choice = Unicode("CV")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        parametric_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)

        self.lagrange = l_min * np.ones(X.shape[1])

        n, p = self.X.shape
        n1 = int(self.selection_frac * n)
        X1, X2 = self.X1, self.X2 = self.X[:n1], self.X[n1:]
        Y1, Y2 = self.Y1, self.Y2 = self.Y[:n1], self.Y[n1:]

        pen = rr.weighted_l1norm(np.sqrt(n1) * self.lagrange, lagrange=1.)
        loss = rr.squared_error(X1, Y1)
        problem = rr.simple_problem(loss, pen)
        soln = problem.solve()

        self.active_set = np.nonzero(soln)[0]
        self.signs = np.sign(soln)[self.active_set]

        self._fit = True

data_splitting_CV.register()



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

from selection.algorithms.lasso import lasso, lasso_full, lasso_full_modelQ
from selection.algorithms.sqrt_lasso import choose_lambda
from selection.truncated.gaussian import truncated_gaussian_old as TG
from selection.randomized.lasso import lasso as random_lasso_method, form_targets
from selection.randomized.modelQ import modelQ as randomized_modelQ

from utils import BHfilter

from selection.randomized.base import restricted_estimator


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

    confidence = Float(0.95)

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

    def generate_pvalues(self):
        raise NotImplementedError('abstract method, should return (active_set, lower_limit, upper_limit)')

    def naive_pvalues(self, active_set):
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
                return active_set, np.ones(len(active_set)) * np.nan

        rpy.r.assign("Y", self.Y)
        rpy.r('pval = summary(lm(Y ~ X - 1))$coef[,4]')
        pval = np.asarray(rpy.r('pval'))
        if self.model_target != 'selected':
            pval = pval[active_set]
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
        if self.model_target == 'selected':
            rpy.r.assign("X", self.X[:, active_set])
        else:
            n, p = self.X.shape
            if n > p:
                rpy.r.assign("X", self.X)
            else:
                return (active_set, 
                        np.ones(p) * np.nan)
        rpy.r.assign("Y", self.Y)
        rpy.r('beta_hat = coef(lm(Y ~ X - 1))')
        beta_hat = np.asarray(rpy.r('beta_hat'))
        n, p = self.X.shape
        beta_full = np.zeros(p)
        beta_full[active_set] = beta_hat

        return active_set, beta_full

# Liu, Markovic, Tibs selection

class liu_theory(parametric_method):

    sigma_estimator = Unicode('relaxed')
    method_name = Unicode("Liu")
    lambda_choice = Unicode("theory")
    model_target = Unicode("full")
    dispersion = Float(0.)

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        parametric_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        n, p = X.shape
        if n < p:
            self.method_name = 'ROSI'
        self.lagrange = l_theory * np.ones(X.shape[1])

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            self._method_instance = lasso_full.gaussian(self.X, self.Y, self.lagrange * np.sqrt(n))
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
            pvalues = np.asarray(S['pval'])
            return active_set, pvalues
        else:
            return [], []

    def generate_intervals(self): 
        S = self.generate_summary(compute_intervals=True)
        if S is not None:
            active_set = np.array(S['variable'])
            lower, upper = (np.asarray(S['lower_confidence']), 
                            np.asarray(S['upper_confidence']))
            return active_set, lower, upper
        else:
            return [], [], []
liu_theory.register()

class liu_aggressive(liu_theory):

    lambda_choice = Unicode("aggressive")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        liu_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8

liu_aggressive.register()

class liu_modelQ_pop_aggressive(liu_aggressive):

    method_name = Unicode("Liu (ModelQ population)")

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            self._method_instance = lasso_full_modelQ(self.feature_cov * n, self.X, self.Y, self.lagrange * np.sqrt(n))
        return self._method_instance
liu_modelQ_pop_aggressive.register()

class liu_modelQ_semi_aggressive(liu_aggressive):

    method_name = Unicode("Liu (ModelQ semi-supervised)")

    B = 10000 # how many samples to use to estimate E[XX^T]

    @classmethod
    def setup(cls, feature_cov, data_generating_mechanism):
        cls.feature_cov = feature_cov
        cls.data_generating_mechanism = data_generating_mechanism
        cls.noise = data_generating_mechanism.noise
        cls._chol = np.linalg.cholesky(feature_cov)

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):

            # draw sample of X for semi-supervised method
            _chol = self._chol
            p = _chol.shape[0]
            Q = 0
            batch_size = int(self.B/10)
            for _ in range(10):
                X_semi = np.random.standard_normal((batch_size, p)).dot(_chol.T)
                Q += X_semi.T.dot(X_semi)
            Q += self.X.T.dot(self.X)
            Q /= (10 * batch_size + self.X.shape[0])
            n, p = self.X.shape
            self._method_instance = lasso_full_modelQ(Q * self.X.shape[0], self.X, self.Y, self.lagrange * np.sqrt(n))
        return self._method_instance
liu_modelQ_semi_aggressive.register()

class liu_sparseinv_aggressive(liu_aggressive):

    method_name = Unicode("ROSI")

    """
    Force the use of the debiasing matrix.
    """

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            self._method_instance = lasso_full.gaussian(self.X, self.Y, self.lagrange * np.sqrt(n))
            self._method_instance.sparse_inverse = True
        return self._method_instance
liu_sparseinv_aggressive.register()

class liu_aggressive_reid(liu_aggressive):

    sigma_estimator = Unicode('Reid')
    pass
liu_aggressive_reid.register()

class liu_CV(liu_theory):
            
    need_CV = True

    lambda_choice = Unicode("CV")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        liu_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_min * np.ones(X.shape[1])
liu_CV.register()

class liu_1se(liu_theory):
            
    need_CV = True

    lambda_choice = Unicode("1se")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        liu_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_1se * np.ones(X.shape[1])
liu_1se.register()

class liu_sparseinv_1se(liu_1se):

    method_name = Unicode("ROSI")

    """
    Force the use of the debiasing matrix.
    """

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            self._method_instance = lasso_full.gaussian(self.X, self.Y, self.lagrange * np.sqrt(n))
            self._method_instance.sparse_inverse = True
        return self._method_instance
liu_sparseinv_1se.register()

class liu_sparseinv_theory(liu_theory):

    method_name = Unicode("ROSI")

    """
    Force the use of the debiasing matrix.
    """

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            self._method_instance = lasso_full.gaussian(self.X, self.Y, self.lagrange * np.sqrt(n))
            self._method_instance.sparse_inverse = True
        return self._method_instance
liu_sparseinv_theory.register()

class liu_sparseinv_CV(liu_CV):

    method_name = Unicode("ROSI")

    """
    Force the use of the debiasing matrix.
    """

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            self._method_instance = lasso_full.gaussian(self.X, self.Y, self.lagrange * np.sqrt(n))
            self._method_instance.sparse_inverse = True
        return self._method_instance
liu_sparseinv_CV.register()

class liu_sparseinv_1se_known(liu_1se):

    method_name = Unicode("ROSI - known")
    dispersion = Float(1.)

    """
    Force the use of the debiasing matrix.
    """

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):
        liu_1se.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.dispersion = self.__class__.noise**2

    @property
    def method_instance(self):

        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            self._method_instance = lasso_full.gaussian(self.X, self.Y, self.lagrange * np.sqrt(n))
            self._method_instance.sparse_inverse = True
        return self._method_instance
liu_sparseinv_1se_known.register()

class liu_R_theory(liu_theory):

    selectiveR_method = True
    method_name = Unicode("Liu (R code)")

    def generate_pvalues(self):
        try:
            numpy2ri.activate()
            rpy.r.assign('X', self.X)
            rpy.r.assign('y', self.Y)
            rpy.r.assign('sigma_reid', self.sigma_reid)
            rpy.r('y = as.numeric(y)')

            rpy.r.assign('lam', self.lagrange[0])
            rpy.r('''
        p = ncol(X);
        n = nrow(X);

        sigma_est = 1.
        if (p >= n) { 
            sigma_est = sigma_reid
        } else {
            sigma_est = sigma(lm(y ~ X - 1))
        }

        penalty_factor = rep(1, p);
        lam = lam * sqrt(n);  # lambdas are passed a sqrt(n) free from python code
        soln = selectiveInference:::solve_problem_glmnet(X, 
                                                         y, 
                                                         lam/n, 
                                                         penalty_factor=penalty_factor, 
                                                         loss="ls")
        PVS = ROSI(X, 
                   y, 
                   soln,
                   lambda=lam, 
                   penalty_factor=penalty_factor, 
                   dispersion=sigma_est^2, 
                   family='gaussian',
                   solver="QP",
                   construct_ci=FALSE)
        active_vars=PVS$active_vars - 1 # for 0-based
        pvalues = PVS$pvalues
        ''')

            pvalues = np.asarray(rpy.r('pvalues'))
            active_set = np.asarray(rpy.r('active_vars'))
            numpy2ri.deactivate()
            if len(active_set) > 0:
                return active_set, pvalues
            else:
                return [], []
        except:
            return [np.nan], [np.nan] # some R failure occurred 
liu_R_theory.register()

class liu_R_aggressive(liu_R_theory):

    lambda_choice = Unicode('aggressive')

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        liu_R_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8
liu_R_aggressive.register()

class lee_full_R_theory(liu_theory):

    wide_OK = False # requires at least n>p
    method_name = Unicode("Lee (R code)")
    selectiveR_method = True

    def generate_pvalues(self):
        numpy2ri.activate()
        rpy.r.assign('x', self.X)
        rpy.r.assign('y', self.Y)
        rpy.r('y = as.numeric(y)')
        rpy.r.assign('sigma_reid', self.sigma_reid)
        rpy.r.assign('lam', self.lagrange[0])
        rpy.r('''
    sigma_est=sigma_reid
    n = nrow(x);
    gfit = glmnet(x, y, standardize=FALSE, intercept=FALSE)
    lam = lam / sqrt(n);  # lambdas are passed a sqrt(n) free from python code
    if (lam < max(abs(t(x) %*% y) / n)) {
        beta = coef(gfit, x=x, y=y, s=lam, exact=TRUE)[-1]
        out = fixedLassoInf(x, y, beta, lam*n, sigma=sigma_est, type='full', intercept=FALSE)
        active_vars=out$vars - 1 # for 0-based
        pvalues = out$pv
    } else {
        pvalues = NULL
        active_vars = numeric(0)
    }
    ''')

        pvalues = np.asarray(rpy.r('pvalues'))
        active_set = np.asarray(rpy.r('active_vars'))
        numpy2ri.deactivate()
        if len(active_set) > 0:
            return active_set, pvalues
        else:
            return [], []
lee_full_R_theory.register()

class lee_full_R_aggressive(lee_full_R_theory):

    lambda_choice = Unicode("aggressive")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        lee_full_R_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8
lee_full_R_aggressive.register()

# Unrandomized selected

class lee_theory(parametric_method):
    
    model_target = Unicode("selected")
    method_name = Unicode("Lee")
    estimator = Unicode("OLS")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        parametric_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1])

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
            S = L.summary(compute_intervals=compute_intervals, alternative='onesided')
            return S

    def generate_pvalues(self):
        S = self.generate_summary(compute_intervals=False)
        if S is not None:
            active_set = np.array(S['variable'])
            pvalues = np.asarray(S['pval'])
            return active_set, pvalues
        else:
            return [], []

    def generate_intervals(self): 
        S = self.generate_summary(compute_intervals=True)
        if S is not None:
            active_set = np.array(S['variable'])
            lower, upper = (np.asarray(S['lower_confidence']), 
                            np.asarray(S['upper_confidence']))
            return active_set, lower, upper
        else:
            return [], [], []

    def point_estimator(self):

        X, Y, lagrange, L = self.X, self.Y, self.lagrange, self.method_instance
        n, p = X.shape

        if not self._fit:
            L.fit()
            self._fit = True

        if len(L.active) > 0:
            beta_full = np.zeros(p)
            if self.estimator == "LASSO":
                beta_full[L.active] = L.soln
            elif self.estimator == "OLS":
                beta_full[L.active] = L.onestep_estimator
            else:
                raise ValueError('estimator must be "OLS" or "LASSO"')
            return L.active, beta_full
        else:
            return [], np.zeros(p)

lee_theory.register()

class lee_CV(lee_theory):
    
    need_CV = True

    lambda_choice = Unicode("CV")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        lee_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_min * np.ones(X.shape[1])

lee_CV.register()

class lee_1se(lee_theory):
    
    need_CV = True

    lambda_choice = Unicode("1se")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        lee_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_1se * np.ones(X.shape[1])

lee_1se.register()

class lee_aggressive(lee_theory):
    
    lambda_choice = Unicode("aggressive")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        lee_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = 0.8 * l_theory * np.ones(X.shape[1])

lee_aggressive.register()

class lee_weak(lee_theory):
    
    lambda_choice = Unicode("weak")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        lee_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = 2 * l_theory * np.ones(X.shape[1])

lee_weak.register()

class sqrt_lasso(parametric_method):

    method_name = Unicode('SqrtLASSO')
    kappa = Float(0.7)

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        parametric_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = self.kappa * choose_lambda(X)

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            self._method_instance = lasso.sqrt_lasso(self.X, self.Y, self.lagrange)
        return self._method_instance

    def generate_summary(self, compute_intervals=False): 

        X, Y, lagrange, L = self.X, self.Y, self.lagrange, self.method_instance
        n, p = X.shape
        X = X / np.sqrt(n)

        if len(L.active) > 0:
            S = L.summary(compute_intervals=compute_intervals, alternative='onesided')
            return S

    def generate_pvalues(self):
        S = self.generate_summary(compute_intervals=False)
        if S is not None:
            active_set = np.array(S['variable'])
            pvalues = np.asarray(S['pval'])
            return active_set, pvalues
        else:
            return [], []

    def generate_intervals(self): 
        S = self.generate_summary(compute_intervals=True)
        if S is not None:
            active_set = np.array(S['variable'])
            lower, upper = (np.asarray(S['lower_confidence']), 
                            np.asarray(S['upper_confidence']))
            return active_set, lower, upper
        else:
            return [], [], []
sqrt_lasso.register()

# Randomized selected

class randomized_lasso(parametric_method):

    method_name = Unicode("Randomized LASSO")
    model_target = Unicode("selected")
    lambda_choice = Unicode("theory")
    randomizer_scale = Float(1)
    confidence = Float(0.95)
    use_MLE = Bool(False)
    ndraw = 25000
    burnin = 5000

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        parametric_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1])

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            mean_diag = np.mean((self.X ** 2).sum(0))
            self._method_instance = random_lasso_method.gaussian(self.X,
                                                                 self.Y,
                                                                 feature_weights = self.lagrange * np.sqrt(n),
                                                                 ridge_term=np.std(self.Y) * np.sqrt(mean_diag) / np.sqrt(n),
                                                                 randomizer_scale=self.randomizer_scale * np.std(self.Y) * np.sqrt(n))
        return self._method_instance

    def generate_summary(self, compute_intervals=False): 

        X, Y, lagrange, rand_lasso = self.X, self.Y, self.lagrange, self.method_instance
        n, p = X.shape

        if not self._fit:
            signs = self.method_instance.fit()
            self._fit = True

        signs = rand_lasso.fit()
        active_set = np.nonzero(signs)[0]

        active = signs != 0

        # estimates sigma
        # JM: for transparency it's better not to have this digged down in the code
        X_active = X[:,active_set]
        resid = Y - X_active.dot(np.linalg.pinv(X_active).dot(Y))
        dispersion = np.sum(resid**2) / (n - active.sum())
        print("dispersion (sigma est for Python)", dispersion)

        kwargs = {}
        if self.model_target == 'debiased':
            kwargs['penalty'] = rand_lasso.penalty
            
        kwargs['dispersion'] = dispersion
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
                _, pvalues, intervals = rand_lasso.summary(observed_target, 
                                                           cov_target, 
                                                           cov_target_score, 
                                                           alternatives,
                                                           ndraw=self.ndraw,
                                                           burnin=self.burnin,
                                                           level=self.confidence,
                                                           compute_intervals=compute_intervals)
            else:
                (final_estimator, 
                 observed_info_mean, 
                 Z_scores, 
                 pvalues, 
                 intervals, 
                 ind_unbiased_estimator) = rand_lasso.selective_MLE(observed_target, 
                                                                    cov_target, 
                                                                    cov_target_score, 
                                                                    level=self.confidence)
            return active_set, pvalues, intervals
        else:
            return [], [], []

    def generate_pvalues(self, compute_intervals=False):
        active_set, pvalues, _ = self.generate_summary(compute_intervals=compute_intervals)
        if len(active_set) > 0:
            return active_set, pvalues
        else:
            return [], []

    def generate_intervals(self): 
        active_set, _, intervals = self.generate_summary(compute_intervals=True)
        if len(active_set) > 0:
            return active_set, intervals[:,0], intervals[:,1]
        else:
            return [], [], []

    def point_estimator(self):

        X, Y, lagrange, rand_lasso = self.X, self.Y, self.lagrange, self.method_instance
        n, p = X.shape

        if not self._fit:
            signs = self.method_instance.fit()
            self._fit = True

        signs = rand_lasso.fit()
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

            (final_estimator, 
             observed_info_mean, 
             Z_scores, 
             pvalues, 
             intervals, 
             ind_unbiased_estimator) = rand_lasso.selective_MLE(observed_target, 
                                                                cov_target, 
                                                                cov_target_score, 
                                                                level=self.confidence)
            beta_full = np.zeros(X.shape[1])
            beta_full[active] = final_estimator
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

# More aggressive lambda choice

class randomized_lasso_aggressive(randomized_lasso):

    lambda_choice = Unicode("aggressive")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8

class randomized_lasso_aggressive_half(randomized_lasso):

    lambda_choice = Unicode('aggressive')
    randomizer_scale = Float(0.5)


    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8

class randomized_lasso_weak_half(randomized_lasso):

    lambda_choice = Unicode('weak')
    randomizer_scale = Float(0.5)


    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 2.
randomized_lasso_weak_half.register()

class randomized_lasso_aggressive_quarter(randomized_lasso):

    randomizer_scale = Float(0.25)

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8

randomized_lasso_aggressive.register(), randomized_lasso_aggressive_half.register(), randomized_lasso_aggressive_quarter.register()

# Randomized selected smaller randomization

class randomized_lasso_half(randomized_lasso):

    randomizer_scale = Float(0.5)
    pass

class randomized_lasso_half_CV(randomized_lasso_CV):

    need_CV = True
    randomizer_scale = Float(0.5)
    pass

class randomized_lasso_half_1se(randomized_lasso_1se):

    need_CV = True
    randomizer_scale = Float(0.5)
    pass

class randomized_lasso_half_mle_1se(randomized_lasso_half_1se):

    method_name = Unicode("Randomized MLE")
    randomizer_scale = Float(1.0)
    use_MLE = Bool(True)
    pass
randomized_lasso_half_mle_1se.register()

randomized_lasso_half.register(), randomized_lasso_half_CV.register(), randomized_lasso_half_1se.register()


# Randomized sqrt selected

class randomized_sqrtlasso(randomized_lasso):

    method_name = Unicode("Randomized SqrtLASSO")
    model_target = Unicode("selected")
    randomizer_scale = Float(1)
    kappa = Float(0.7)

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            lagrange = np.ones(p) * choose_lambda(self.X) * self.kappa
            self._method_instance = random_lasso_method.gaussian(self.X,
                                                                 self.Y,
                                                                 lagrange,
                                                                 randomizer_scale=self.randomizer_scale * np.std(self.Y))
        return self._method_instance

    def generate_summary(self, compute_intervals=False):
        X, Y, rand_lasso = self.X, self.Y, self.method_instance
        n, p = X.shape
        X = X / np.sqrt(n)

        if not self._fit:
            self.method_instance.fit()
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

        _, pvalues, intervals = rand_lasso.summary(observed_target, 
                                                   cov_target, 
                                                   cov_target_score, 
                                                   alternatives,
                                                   ndraw=self.ndraw,
                                                   burnin=self.burnin,
                                                   level=self.confidence,
                                                   compute_intervals=compute_intervals)

        if len(pvalues) > 0:
            return active_set, pvalues, intervals
        else:
            return [], [], []


class randomized_sqrtlasso_half(randomized_sqrtlasso):

    randomizer_scale = Float(0.5)
    pass

randomized_sqrtlasso.register(), randomized_sqrtlasso_half.register()

class randomized_sqrtlasso_bigger(randomized_sqrtlasso):

    kappa = Float(0.8)
    pass

class randomized_sqrtlasso_bigger_half(randomized_sqrtlasso):

    kappa = Float(0.8)
    randomizer_scale = Float(0.5)
    pass

randomized_sqrtlasso_bigger.register(), randomized_sqrtlasso_bigger_half.register()

# Randomized full

class randomized_lasso_full(randomized_lasso):

    method_name = Unicode("Randomized One-step")
    model_target = Unicode('debiased')

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1])

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

# Randomized full smaller randomization

class randomized_lasso_full_half(randomized_lasso_full):

    randomizer_scale = Float(0.5)
    pass

class randomized_lasso_full_half_CV(randomized_lasso_full_CV):

    randomizer_scale = Float(0.5)
    pass

class randomized_lasso_full_half_1se(randomized_lasso_full_1se):

    need_CV = True

    randomizer_scale = Float(0.5)
    pass

class randomized_lasso_full_quarter_1se(randomized_lasso_full_half_1se):

    need_CV = True
    randomizer_scale = Float(0.25)
    pass

randomized_lasso_full_half.register(), randomized_lasso_full_half_CV.register(), randomized_lasso_full_half_1se.register(), randomized_lasso_full_quarter_1se.register()

# Aggressive choice of lambda

class randomized_lasso_full_aggressive(randomized_lasso_full):

    lambda_choice = Unicode("aggressive")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso_full.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8

class randomized_lasso_full_aggressive_half(randomized_lasso_full_aggressive):

    randomizer_scale = Float(0.5)
    pass

randomized_lasso_full_aggressive.register(), randomized_lasso_full_aggressive_half.register()

class randomized_lasso_R_theory(parametric_method):

    method_name = Unicode("Randomized LASSO (R code)")
    selectiveR_method = True

    def generate_pvalues(self, compute_intervals=False):
        self._fit = True
        numpy2ri.activate()
        rpy.r.assign('X', self.X)
        rpy.r.assign('y', self.Y)
        rpy.r('y = as.numeric(y)')
        rpy.r.assign('q', self.q)
        rpy.r.assign('lam', self.lagrange[0])
        rpy.r.assign("randomizer_scale", self.randomizer_scale)
        rpy.r.assign("compute_intervals", compute_intervals)
        rpy.r('''
        n = nrow(X)
        p = ncol(X)
        lam = lam * sqrt(n)
        mean_diag = mean(apply(X^2, 2, sum))
        ridge_term = sqrt(mean_diag) * sd(y) / sqrt(n)
        result = randomizedLasso(X, y, lam, ridge_term=ridge_term,
                                 noise_scale = randomizer_scale * sd(y) * sqrt(n), family='gaussian')
        active_set = result$active_set
        if (length(active_set)==0){
            active_set = -1
        } else{
            sigma_est = sigma(lm(y ~ X[,active_set] - 1))
            cat("sigma est for R", sigma_est,"\n")
            targets = selectiveInference:::compute_target(result, 'partial', sigma_est = sigma_est,
                                 construct_pvalues=rep(TRUE, length(active_set)), 
                                 construct_ci=rep(compute_intervals, length(active_set)))

            out = randomizedLassoInf(result,
                                 targets=targets,
                                 sampler = "norejection",
                                 level=0.9,
                                 burnin=1000,
                                 nsample=10000)
            active_set=active_set-1
            pvalues = out$pvalues
            intervals = out$ci
        }
        ''')

        active_set = np.asarray(rpy.r('active_set'), np.int)
        print(active_set)

        if active_set[0]==-1:
            numpy2ri.deactivate()
            return [], [], []

        pvalues = np.asarray(rpy.r('pvalues'))
        intervals = np.asarray(rpy.r('intervals'))
        numpy2ri.deactivate()
        if len(active_set) > 0:
            return active_set, pvalues
        else:
            return [], []

randomized_lasso_R_theory.register()

class data_splitting(parametric_method):

    method_name = Unicode('Data splitting')
    selection_frac = Float(0.5)
    model_target = Unicode("selected")
    lambda_choice = Unicode('theory')

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        parametric_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)

        self.lagrange = l_theory * np.ones(X.shape[1])

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
            return self.active_set, lower, upper
        else:
            return [], [], []
data_splitting.register()

class data_splitting_1se(data_splitting):

    lambda_choice = Unicode('1se')

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



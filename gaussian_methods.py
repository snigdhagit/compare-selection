import tempfile, os, glob
from traitlets import (HasTraits, 
                       Integer, 
                       Unicode, 
                       Float, 
                       Integer, 
                       Instance, 
                       Dict, 
                       Bool,
                       default, 
                       observe)

import numpy as np
import regreg.api as rr

from selection.algorithms.lasso import lasso, lasso_full, lasso_full_modelQ
from selection.algorithms.sqrt_lasso import choose_lambda
from selection.truncated.gaussian import truncated_gaussian_old as TG
from selection.randomized.lasso import highdim
from selection.randomized.modelQ import modelQ as randomized_modelQ

from utils import BHfilter

# Rpy

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri

methods = {}

class generic_method(HasTraits):

    selectiveR_method = False
    wide_ok = True # ok for p>= n?

    # Traits

    q = Float(0.2)
    method_name = Unicode('Generic method')
    model = Unicode()

    @classmethod
    def setup(cls, feature_cov):
        cls.feature_cov = feature_cov

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

# Knockoff selection

class knockoffs_mf(generic_method):

    method_name = Unicode('Knockoffs')
    knockoff_method = Unicode('Second order')
    model = Unicode("full")

    def select(self):
        try:
            numpy2ri.activate()
            rpy.r.assign('X', self.X)
            rpy.r.assign('Y', self.Y)
            rpy.r.assign('q', self.q)
            rpy.r('V=knockoff.filter(X, Y, fdr=q)$selected')
            rpy.r('if (length(V) > 0) {V = V-1}')
            V = rpy.r('V')
            numpy2ri.deactivate()
            return np.asarray(V, np.int), np.asarray(V, np.int)
        except:
            return [], []

knockoffs_mf.register()

class knockoffs_sigma(generic_method):

    factor_method = 'asdp'
    method_name = Unicode('Knockoffs')
    knockoff_method = Unicode("ModelX (asdp)")
    model = Unicode("full")

    @classmethod
    def setup(cls, feature_cov):

        cls.feature_cov = feature_cov
        numpy2ri.activate()

        # see if we've factored this before

        have_factorization = False
        if not os.path.exists('.knockoff_factorizations'):
            os.mkdir('.knockoff_factorizations')
        factors = glob.glob('.knockoff_factorizations/*npz')
        for factor_file in factors:
            factor = np.load(factor_file)
            feature_cov_f = factor['feature_cov']
            if ((feature_cov_f.shape == feature_cov.shape) and
                (factor['method'] == cls.factor_method) and
                np.allclose(feature_cov_f, feature_cov)):
                have_factorization = True
                print('found factorization: %s' % factor_file)
                cls.knockoff_chol = factor['knockoff_chol']

        if not have_factorization:
            print('doing factorization')
            cls.knockoff_chol = factor_knockoffs(feature_cov, cls.factor_method)

        numpy2ri.deactivate()

    def select(self):

        numpy2ri.activate()
        rpy.r.assign('chol_k', self.knockoff_chol)
        rpy.r('''
        knockoffs = function(X) {
           mu = rep(0, ncol(X))
           mu_k = X # sweep(X, 2, mu, "-") %*% SigmaInv_s
           X_k = mu_k + matrix(rnorm(ncol(X) * nrow(X)), nrow(X)) %*% 
            chol_k
           return(X_k)
        }
            ''')
        numpy2ri.deactivate()

        try:
            numpy2ri.activate()
            rpy.r.assign('X', self.X)
            rpy.r.assign('Y', self.Y)
            rpy.r.assign('q', self.q)
            rpy.r('V=knockoff.filter(X, Y, fdr=q, knockoffs=knockoffs)$selected')
            rpy.r('if (length(V) > 0) {V = V-1}')
            V = rpy.r('V')
            numpy2ri.deactivate()
            return np.asarray(V, np.int), np.asarray(V, np.int)
        except:
            return [], []

knockoffs_sigma.register()

def factor_knockoffs(feature_cov, method='asdp'):

    numpy2ri.activate()
    rpy.r.assign('Sigma', feature_cov)
    rpy.r.assign('method', method)
    rpy.r('''

    # Compute the Cholesky -- from create.gaussian

    diag_s = diag(switch(method, equi = create.solve_equi(Sigma), 
                  sdp = create.solve_sdp(Sigma), asdp = create.solve_asdp(Sigma)))
    if (is.null(dim(diag_s))) {
        diag_s = diag(diag_s, length(diag_s))
    }
    SigmaInv_s = solve(Sigma, diag_s)
    Sigma_k = 2 * diag_s - diag_s %*% SigmaInv_s
    chol_k = chol(Sigma_k)
    ''')
    knockoff_chol = np.asarray(rpy.r('chol_k'))
    SigmaInv_s = np.asarray(rpy.r('SigmaInv_s'))
    diag_s = np.asarray(rpy.r('diag_s'))
    np.savez('.knockoff_factorizations/%s.npz' % (os.path.split(tempfile.mkstemp()[1])[1],),
             method=method,
             feature_cov=feature_cov,
             knockoff_chol=knockoff_chol)

    return knockoff_chol

class knockoffs_sigma_equi(knockoffs_sigma):

    knockoff_method = Unicode('ModelX (equi)')
    factor_method = 'equi'

knockoffs_sigma_equi.register()

class knockoffs_orig(generic_method):

    wide_OK = False # requires at least n>p

    method_name = Unicode("Knockoffs")
    knockoff_method = Unicode('Candes & Barber')
    model = Unicode('full')

    def select(self):
        try:
            numpy2ri.activate()
            rpy.r.assign('X', self.X)
            rpy.r.assign('Y', self.Y)
            rpy.r.assign('q', self.q)
            rpy.r('V=knockoff.filter(X, Y, statistic=stat.glmnet_lambdadiff, fdr=q, knockoffs=create.fixed)$selected')
            rpy.r('if (length(V) > 0) {V = V-1}')
            V = rpy.r('V')
            numpy2ri.deactivate()
            V = np.asarray(V, np.int)
            return V, V
        except:
            return [], []

knockoffs_orig.register()

class knockoffs_fixed(generic_method):

    wide_OK = False # requires at least n>p

    method_name = Unicode("Knockoffs")
    knockoff_method = Unicode('Fixed')
    model = Unicode('full')

    def select(self):
        try:
            numpy2ri.activate()
            rpy.r.assign('X', self.X)
            rpy.r.assign('Y', self.Y)
            rpy.r.assign('q', self.q)
            rpy.r('V=knockoff.filter(X, Y, fdr=q, knockoffs=create.fixed)$selected')
            rpy.r('if (length(V) > 0) {V = V-1}')
            V = rpy.r('V')
            numpy2ri.deactivate()
            return np.asarray(V, np.int), np.asarray(V, np.int)
        except:
            return [], []

knockoffs_fixed.register()

# Liu, Markovic, Tibs selection

class parametric_method(generic_method):

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

    def generate_pvalues(self):
        # should return (active, pvalues)
        raise NotImplementedError('abstract method')

    def point_estimator(self):
        # should return (active, point_estimator)
        raise NotImplementedError('abstract method')
        
class liu_theory(parametric_method):

    sigma_estimator = Unicode('relaxed')
    method_name = Unicode("Liu")
    lambda_choice = Unicode("theory")
    model = Unicode("full")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        parametric_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1])

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            self._method_instance = lasso_full.gaussian(self.X, self.Y, self.lagrange * np.sqrt(n))
        return self._method_instance

    def generate_pvalues(self): 

        if not self._fit:
            self.method_instance.fit()
            self._fit = True

        X, Y, lagrange, L = self.X, self.Y, self.lagrange, self.method_instance
        n, p = X.shape

        if len(L.active) > 0:
            if self.sigma_estimator == 'reid' and n < p:
                dispersion = self.sigma_reid**2
            else:
                dispersion = None
            S = L.summary(compute_intervals=False, dispersion=dispersion)
            active_set = np.array(S['variable'])
            pvalues = np.asarray(S['pval'])
            return active_set, pvalues
        else:
            return [], []

liu_theory.register()

class liu_aggressive(liu_theory):

    lambda_choice = Unicode("aggressive")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        liu_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8

    def generate_pvalues(self):
        pval = liu_theory.generate_pvalues(self)
        return pval

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
    def setup(cls, feature_cov):
        cls.feature_cov = feature_cov
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

    method_name = Unicode("Liu (debiased)")

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
            
    lambda_choice = Unicode("CV")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        liu_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_min * np.ones(X.shape[1])
liu_CV.register()

class liu_1se(liu_theory):
            
    lambda_choice = Unicode("1se")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        liu_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_1se * np.ones(X.shape[1])
liu_1se.register()

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
        lam = lam / sqrt(n);  # lambdas are passed a sqrt(n) free from python code
        soln = selectiveInference:::solve_problem_glmnet(X, y, lam, penalty_factor=penalty_factor, loss="ls")
        PVS = selectiveInference:::inference_group_lasso(X, y, 
                                                         soln, groups=1:ncol(X), 
                                                         lambda=lam, penalty_factor=penalty_factor, 
                                                         sigma_est, loss="ls", algo="Q", 
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
    
    model = Unicode("selected")
    method_name = Unicode("Lee")
    estimator = Unicode("LASSO")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        parametric_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1])

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            self._method_instance = lasso.gaussian(self.X, self.Y, self.lagrange * np.sqrt(n))
        return self._method_instance

    def generate_pvalues(self):

        if not self._fit:
            self.method_instance.fit()
            self._fit = True

        X, Y, lagrange, L = self.X, self.Y, self.lagrange, self.method_instance

        if len(L.active) > 0:
            S = L.summary(compute_intervals=False, alternative='onesided')
            active_set = np.array(S['variable'])
            pvalues = np.asarray(S['pval'])
            return active_set, pvalues
        else:
            return [], []

    def point_estimator(self):
        X, Y, lagrange, L = self.X, self.Y, self.lagrange, self.method_instance
        n, p = X.shape
        beta_full = np.zeros(p)
        if self.estimator == "LASSO":
            beta_full[L.active] = L.soln
        else:
            beta_full[L.active] = L.onestep_estimator
        return L.active, beta_full

lee_theory.register()

class lee_CV(lee_theory):
    
    lambda_choice = Unicode("CV")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        lee_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_min * np.ones(X.shape[1])

lee_CV.register()

class lee_1se(lee_theory):
    
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
            self._method_instance = lasso.sqrt_lasso(self.X, self.Y, self.lagrange * np.sqrt(n))
        return self._method_instance

    def generate_pvalues(self):

        X, Y, lagrange, L = self.X, self.Y, self.lagrange, self.method_instance
        n, p = X.shape
        X = X / np.sqrt(n)

        if len(L.active) > 0:
            S = L.summary(compute_intervals=False, alternative='onesided')
            active_set = np.array(S['variable'])
            pvalues = np.asarray(S['pval'])
            return active_set, pvalues
        else:
            return [], []

# Randomized selected

class randomized_lasso(parametric_method):

    method_name = Unicode("Randomized LASSO")
    model = Unicode("selected")
    lambda_choice = Unicode("theory")
    randomizer_scale = Float(1)

    ndraw = 15000
    burnin = 2000

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        parametric_method.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1])

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            self._method_instance = highdim.gaussian(self.X,
                                                     self.Y,
                                                     self.lagrange * np.sqrt(n),
                                                     randomizer_scale=self.randomizer_scale * np.std(self.Y) * np.sqrt(n))
        return self._method_instance

    def generate_pvalues(self):
        X, Y, lagrange, rand_lasso = self.X, self.Y, self.lagrange, self.method_instance
        n, p = X.shape

        if not self._fit:
            signs = self.method_instance.fit()
            self._fit = True

        signs = rand_lasso.fit()
        active_set = np.nonzero(signs)[0]
        _, pvalues, _ = rand_lasso.summary(target=self.model,
                                           ndraw=self.ndraw,
                                           burnin=self.burnin,
                                           compute_intervals=False)
        if len(pvalues) > 0:
            return active_set, pvalues
        else:
            return [], []


class randomized_lasso_CV(randomized_lasso):

    lambda_choice = Unicode("CV")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_min * np.ones(X.shape[1])

class randomized_lasso_1se(randomized_lasso):

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

    randomizer_scale = Float(0.5)
    pass

class randomized_lasso_half_1se(randomized_lasso_1se):

    randomizer_scale = Float(0.5)
    pass

randomized_lasso_half.register(), randomized_lasso_half_CV.register(), randomized_lasso_half_1se.register()

# selective mle

class randomized_lasso_mle(randomized_lasso_aggressive_half):

    method_name = Unicode("Randomized MLE")
    randomizer_scale = Float(0.5)
    model = Unicode("selected")

#     @property
#     def method_instance(self):
#         if not hasattr(self, "_method_instance"):
#             n, p = self.X.shape
#             self._method_instance = randomized_modelQ(self.feature_cov * n,
#                                                       self.X,
#                                                       self.Y,
#                                                       self.lagrange * np.sqrt(n),
#                                                       randomizer_scale=self.randomizer_scale * np.std(self.Y) * np.sqrt(n))
#         return self._method_instance

    def generate_pvalues(self):
        X, Y, lagrange, rand_lasso = self.X, self.Y, self.lagrange, self.method_instance
        n, p = X.shape

        if not self._fit:
            signs = self.method_instance.fit()
            self._fit = True

        signs = rand_lasso.fit()
        active_set = np.nonzero(signs)[0]
        Z, pvalues = rand_lasso.selective_MLE(target=self.model, solve_args={'min_iter':1000, 'tol':1.e-12})[-3:-1]
        print(pvalues, 'pvalues')
        print(Z, 'Zvalues')
        if len(pvalues) > 0:
            return active_set, pvalues
        else:
            return [], []

randomized_lasso_mle.register()


# Using modelQ for randomized

class randomized_lasso_half_pop_1se(randomized_lasso_half_1se):

    method_name = Unicode("Randomized ModelQ (pop)")
    randomizer_scale = Float(0.5)
    nsample = 15000
    burnin = 2000

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            self._method_instance = randomized_modelQ(self.feature_cov * n,
                                                      self.X,
                                                      self.Y,
                                                      self.lagrange * np.sqrt(n),
                                                      randomizer_scale=self.randomizer_scale * np.std(self.Y) * np.sqrt(n))
        return self._method_instance

class randomized_lasso_half_semi_1se(randomized_lasso_half_1se):

    method_name = Unicode("Randomized ModelQ (semi-supervised)")
    randomizer_scale = Float(0.5)
    B = 10000
    nsample = 15000
    burnin = 2000

    @classmethod
    def setup(cls, feature_cov):
        cls.feature_cov = feature_cov
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
            self._method_instance = randomized_modelQ(Q * n,
                                                      self.X,
                                                      self.Y,
                                                      self.lagrange * np.sqrt(n),
                                                      randomizer_scale=self.randomizer_scale * np.std(self.Y) * np.sqrt(n))
        return self._method_instance

randomized_lasso_half_pop_1se.register(), randomized_lasso_half_semi_1se.register()

# Using modelQ for randomized

class randomized_lasso_half_pop_aggressive(randomized_lasso_aggressive_half):

    method_name = Unicode("Randomized ModelQ (pop)")

    randomizer_scale = Float(0.5)
    nsample = 15000
    burnin = 2000

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            self._method_instance = randomized_modelQ(self.feature_cov * n,
                                                      self.X,
                                                      self.Y,
                                                      self.lagrange * np.sqrt(n),
                                                      randomizer_scale=self.randomizer_scale * np.std(self.Y) * np.sqrt(n))
        return self._method_instance

class randomized_lasso_half_semi_aggressive(randomized_lasso_aggressive_half):

    method_name = Unicode("Randomized ModelQ (semi-supervised)")
    randomizer_scale = Float(0.25)

    B = 10000
    nsample = 15000
    burnin = 2000

    @classmethod
    def setup(cls, feature_cov):
        cls.feature_cov = feature_cov
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
            self._method_instance = randomized_modelQ(Q * n,
                                                      self.X,
                                                      self.Y,
                                                      self.lagrange * np.sqrt(n),
                                                      randomizer_scale=self.randomizer_scale * np.std(self.Y) * np.sqrt(n))
        return self._method_instance

randomized_lasso_half_pop_aggressive.register(), randomized_lasso_half_semi_aggressive.register()

# Randomized sqrt selected

class randomized_sqrtlasso(randomized_lasso):

    method_name = Unicode("Randomized SqrtLASSO")
    model = Unicode("selected")
    randomizer_scale = Float(1)
    kappa = Float(0.7)

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            lagrange = np.ones(p) * choose_lambda(self.X) * self.kappa
            self._method_instance = highdim.gaussian(self.X,
                                                     self.Y,
                                                     lagrange,
                                                     randomizer_scale=self.randomizer_scale * np.std(self.Y))
        return self._method_instance

    def generate_pvalues(self):
        X, Y, rand_lasso = self.X, self.Y, self.method_instance
        n, p = X.shape
        X = X / np.sqrt(n)

        if not self._fit:
            self.method_instance.fit()
            self._fit = True

        signs = self.method_instance.selection_variable['sign']
        active_set = np.nonzero(signs)[0]
        _, pvalues, _ = rand_lasso.summary(target="selected",
                                           ndraw=self.ndraw,
                                           burnin=self.burnin,
                                           compute_intervals=False)
        if len(pvalues) > 0:
            return active_set, pvalues
        else:
            return [], []


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

    model = Unicode('full')

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1])

class randomized_lasso_full_CV(randomized_lasso_full):

    lambda_choice = Unicode("CV")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso_full.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_min * np.ones(X.shape[1])

class randomized_lasso_full_1se(randomized_lasso_full):

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

    randomizer_scale = Float(0.5)
    pass

randomized_lasso_full_half.register(), randomized_lasso_full_half_CV.register(), randomized_lasso_full_half_1se.register()

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

class randomized_lasso_R_theory(randomized_lasso):

    method_name = Unicode("Randomized LASSO (R code)")
    selective_Rmethod = True

    def generate_pvalues(self):
        numpy2ri.activate()
        rpy.r.assign('X', self.X)
        rpy.r.assign('y', self.Y)
        rpy.r('y = as.numeric(y)')
        rpy.r.assign('q', self.q)
        rpy.r.assign('lam', self.lagrange[0])
        rpy.r('''
        n = nrow(X)
        p = ncol(X)
        lam = lam * sqrt(n)
        result = randomizedLasso(X, y, lam, ridge_term=sd(y) * sqrt(n), 
                                 noise_scale = sd(y) * 0.5 * sqrt(n), family='gaussian')
        active_set = result$active_set
        sigma_est = sigma(lm(y ~ X[,active_set] - 1))
        targets = selectiveInference:::compute_target(result, 'partial', sigma_est = sigma_est, 
                                 construct_pvalues=rep(TRUE, length(active_set)), 
                                 construct_ci=rep(FALSE, length(active_set)))
        out = randomizedLassoInf(result,
                                 targets=targets)
        pvalues = out$pvalues
        active_set = active_set - 1
        ''')

        pvalues = np.asarray(rpy.r('pvalues'))
        active_set = np.asarray(rpy.r('active_set'))
        numpy2ri.deactivate()
        if len(active_set) > 0:
            return active_set, pvalues
        else:
            return [], []
randomized_lasso_R_theory.register()

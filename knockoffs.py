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

from gaussian_methods import generic_method

# Knockoff selection

class knockoffs_mf(generic_method):

    method_name = Unicode('Knockoffs')
    knockoff_method = Unicode('Second order')
    model_target = Unicode("full")
    selectiveR_method = True

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
    model_target = Unicode("full")
    selectiveR_method = True

    @classmethod
    def setup(cls, feature_cov, data_generating_mechanism):

        cls.feature_cov = feature_cov
        cls.data_generating_mechanism = data_generating_mechanism
        cls.noise = data_generating_mechanism.noise
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
    selectiveR_method = True

knockoffs_sigma_equi.register()

class knockoffs_orig(generic_method):

    wide_OK = False # requires at least n>p

    method_name = Unicode("Knockoffs")
    knockoff_method = Unicode('Candes & Barber')
    model_target = Unicode('full')
    selectiveR_method = True

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
    model_target = Unicode('full')
    selectiveR_method = True

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

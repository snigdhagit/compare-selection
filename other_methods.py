
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
            raise ValueError('Liu does not work when n<p, use ROSI instead')
        self.lagrange = l_theory * np.ones(X.shape[1]) * self.noise

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            self._method_instance = ROSI.gaussian(self.X, self.Y, self.lagrange * np.sqrt(n), approximate_inverse=None)
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
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8 * self.noise

liu_aggressive.register()

class ROSI_modelQ_pop_aggressive(liu_aggressive):

    method_name = Unicode("Liu (ModelQ population)")

    @property
    def method_instance(self):
        if not hasattr(self, "_method_instance"):
            n, p = self.X.shape
            self._method_instance = ROSI_modelQ(self.feature_cov * n, self.X, self.Y, self.lagrange * np.sqrt(n))
        return self._method_instance
ROSI_modelQ_pop_aggressive.register()

class ROSI_modelQ_semi_aggressive(liu_aggressive):

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
            self._method_instance = ROSI_modelQ(Q * self.X.shape[0], self.X, self.Y, self.lagrange * np.sqrt(n))
        return self._method_instance
ROSI_modelQ_semi_aggressive.register()

class ROSI_aggressive(ROSI_theory):

    method_name = Unicode("ROSI")

    """
    Force the use of the debiasing matrix.
    """

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        ROSI_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8 * self.noise
ROSI_aggressive.register()

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


class lee_aggressive(lee_theory):
    
    lambda_choice = Unicode("aggressive")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        lee_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = 0.8 * l_theory * np.ones(X.shape[1]) * self.noise

lee_aggressive.register()

class lee_weak(lee_theory):
    
    lambda_choice = Unicode("weak")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        lee_theory.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = 2 * l_theory * np.ones(X.shape[1]) * self.noise

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

# More aggressive lambda choice

class randomized_lasso_aggressive(randomized_lasso):

    lambda_choice = Unicode("aggressive")

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8 * self.noise

class randomized_lasso_aggressive_half(randomized_lasso):

    lambda_choice = Unicode('aggressive')
    randomizer_scale = Float(0.5)

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8 * self.noise

class randomized_lasso_weak_half(randomized_lasso):

    lambda_choice = Unicode('weak')
    randomizer_scale = Float(0.5)

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 2. * self.noise
randomized_lasso_weak_half.register()

class randomized_lasso_aggressive_quarter(randomized_lasso_aggressive_half):

    randomizer_scale = Float(0.25)

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8 * self.noise

class randomized_lasso_aggressive_tenth(randomized_lasso_aggressive_half):

    randomizer_scale = Float(0.10)

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8 * self.noise

randomized_lasso_aggressive.register(), randomized_lasso_aggressive_half.register(), randomized_lasso_aggressive_quarter.register()
randomized_lasso_aggressive_tenth.register()

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


class randomized_lasso_1se_AR(randomized_lasso_1se):

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso_1se.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        n, p = X.shape

        ARrho = []
        for s in np.random.sample(100):
            Xr = X[int(s*n)]
            ARrho.append(np.corrcoef(Xr[1:], Xr[:-1])[0,1])
        ARrho = np.mean(ARrho) 
        print("AR parameter", ARrho)

        mean_diag = np.mean((X ** 2).sum(0))
        randomizer_scale = np.sqrt(mean_diag) * np.std(Y) * self.randomizer_scale

        ARcov = ARrho**(np.abs(np.subtract.outer(np.arange(p), np.arange(p)))) * randomizer_scale**2 
        self._randomizer = randomization.gaussian(ARcov)

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
            self._method_instance.randomizer = self._randomizer
        return self._method_instance
randomized_lasso_1se_AR.register()

class randomized_lasso_aggressive_AR(randomized_lasso_aggressive):

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso_aggressive.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        n, p = X.shape

        ARrho = []
        for s in np.random.sample(100):
            Xr = X[int(s*n)]
            ARrho.append(np.corrcoef(Xr[1:], Xr[:-1])[0,1])
        ARrho = np.mean(ARrho) 
        print("AR parameter", ARrho)

        mean_diag = np.mean((X ** 2).sum(0))
        randomizer_scale = np.sqrt(mean_diag) * np.std(Y) * self.randomizer_scale

        ARcov = ARrho**(np.abs(np.subtract.outer(np.arange(p), np.arange(p)))) * randomizer_scale**2 
        self._randomizer = randomization.gaussian(ARcov)

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
            self._method_instance.randomizer = self._randomizer
        return self._method_instance
randomized_lasso_aggressive_AR.register()

class randomized_lasso_AR(randomized_lasso):

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):

        randomized_lasso.__init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid)
        n, p = X.shape

        ARrho = []
        for s in np.random.sample(100):
            Xr = X[int(s*n)]
            ARrho.append(np.corrcoef(Xr[1:], Xr[:-1])[0,1])
        ARrho = np.mean(ARrho) 
        print("AR parameter", ARrho)

        mean_diag = np.mean((X ** 2).sum(0))
        randomizer_scale = np.sqrt(mean_diag) * np.std(Y) * self.randomizer_scale

        ARcov = ARrho**(np.abs(np.subtract.outer(np.arange(p), np.arange(p)))) * randomizer_scale**2 
        self._randomizer = randomization.gaussian(ARcov)

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
            self._method_instance.randomizer = self._randomizer
        return self._method_instance
randomized_lasso_AR.register()

class randomized_lasso_half_1se_AR(randomized_lasso_1se_AR):

    need_CV = True
    randomizer_scale = Float(0.5)
    pass
randomized_lasso_half_1se_AR.register()

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
        self.lagrange = l_theory * np.ones(X.shape[1]) * 0.8 * self.noise

class randomized_lasso_full_aggressive_half(randomized_lasso_full_aggressive):

    randomizer_scale = Float(0.5)
    pass

class randomized_lasso_full_aggressive_quarter(randomized_lasso_full_aggressive):

    randomizer_scale = Float(0.25)
    pass

randomized_lasso_full_aggressive.register(), randomized_lasso_full_aggressive_half.register()
randomized_lasso_full_aggressive_quarter.register()

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

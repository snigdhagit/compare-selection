class tuned_lasso(parametric_method):

    method_name = Unicode("Tuned LASSO")
    model_target = Unicode("full")

    def point_estimator(self):

        X, Y = self.X, self.Y
        X_new, Y_new = self.data_generating_mechanism.generate()[:2]
        n, p = X.shape

        active = np.zeros(p, np.bool)
        beta = np.zeros(p)
        return active, beta
tuned_lasso.register()

# selective mle

class randomized_lasso_mle(randomized_lasso_aggressive_half):

    method_name = Unicode("Randomized MLE")
    randomizer_scale = Float(0.5)
    model_target = Unicode("selected")

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

    def generate_pvalues(self):
        X, Y, lagrange, rand_lasso = self.X, self.Y, self.lagrange, self.method_instance
        n, p = X.shape

        if not self._fit:
            signs = self.method_instance.fit()
            self._fit = True

        signs = rand_lasso.fit()
        active_set = np.nonzero(signs)[0]
        Z, pvalues = rand_lasso.selective_MLE(target=self.model_target, 
                                              solve_args={'min_iter':1000, 'tol':1.e-12})[-3:-1]
        print(pvalues, 'pvalues')
        print(Z, 'Zvalues')
        if len(pvalues) > 0:
            return active_set, pvalues
        else:
            return [], []

randomized_lasso_mle.register()


# Using modelQ for randomized

class randomized_lasso_half_pop_1se(randomized_lasso_half_1se):

    model_target = Unicode("full")
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
    model_target = Unicode("full")
    randomizer_scale = Float(0.5)
    nsample = 10000
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
            self._method_instance = randomized_modelQ(Q * n,
                                                      self.X,
                                                      self.Y,
                                                      self.lagrange * np.sqrt(n),
                                                      randomizer_scale=self.randomizer_scale * np.std(self.Y) * np.sqrt(n))
        return self._method_instance

randomized_lasso_half_pop_aggressive.register(), randomized_lasso_half_semi_aggressive.register()

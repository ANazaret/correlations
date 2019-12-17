class GibbsSampler:
    def __init__(self, x, n_cat=5):
        self.n_obs, self.dim_obs = x.shape
        self.n_cat = n_cat
        self.x = x

        # Initialize the hyperparameters
        self.hyperparameters = {
            "dirich": np.array([10.0] * self.n_cat),
            "alpha_gamma": np.array([2] * self.dim_obs),
            "beta_gamma": np.array([2] * self.dim_obs),
        }

        # Initialize the latent variables
        self.latent_variables = dict()
        self.latent_variables["theta"] = np.random.dirichlet(
            self.hyperparameters["dirich"]
        )
        self.latent_variables["z"] = np.random.multinomial(
            1, [1 / self.n_cat] * n_cat, self.n_obs
        )
        self.latent_variables["lambda"] = np.random.uniform(
            self.x.min(axis=0), self.x.max(axis=0), (self.n_cat, self.dim_obs)
        )

        # We don't really need the constant but just in case
        log_dirichlet = gammaln(self.hyperparameters["dirich"].sum()) - np.sum(
            gammaln(self.hyperparameters["dirich"])
        )
        log_gamma = np.sum(
            np.log(self.hyperparameters["beta_gamma"])
            * self.hyperparameters["alpha_gamma"]
            - gammaln(self.hyperparameters["alpha_gamma"])
        )
        log_poisson = -sum(log_fact(xi) for xi in self.x.flatten())
        self.log_constant = log_dirichlet + log_gamma + log_poisson

    def log_density_x_cond_z(self, x, z):
        lambda_z = self.latent_variables["lambda"][z]
        log_proba = 0
        for xk, lk in zip(x, lambda_z):
            log_proba += np.log(lk) * xk - lk - log_fact(xk)
        return log_proba

    def sample_zi(self, i):
        log_densities = np.array(
            [self.log_density_x_cond_z(self.x[i], k) for k in range(self.n_cat)]
        )
        log_proportions = log_densities + np.log(self.latent_variables["theta"])
        log_normalizing_constant = np.logaddexp.reduce(log_proportions)
        log_proportions -= log_normalizing_constant
        proportions = np.exp(log_proportions)
        return np.random.multinomial(1, proportions)

    def sample_lambda_k(self, k):
        cluster = self.x[self.latent_variables["z"][:, k].astype(bool)]
        gamma_sample = np.random.gamma(
            self.hyperparameters["alpha_gamma"] + cluster.sum(axis=0),
            1 / (self.hyperparameters["beta_gamma"] + len(cluster)),
        )
        return gamma_sample

    def sample_theta(self):
        alpha = self.hyperparameters["dirich"]
        z_counts = self.latent_variables["z"].sum(axis=0)
        return np.random.dirichlet(alpha + z_counts)

    def evaluate_log_joint(self):
        # Log theta
        alpha = self.hyperparameters["dirich"]
        log_theta = (np.log(self.latent_variables["theta"]) * (alpha - 1)).sum()

        # Log lambda k
        lambdas = self.latent_variables["lambda"]
        log_lamb = np.sum(
            -self.hyperparameters["beta_gamma"] * lambdas
            + np.log(lambdas) * (self.hyperparameters["alpha_gamma"] - 1)
        )

        # Log z i
        theta = self.latent_variables["theta"]
        log_z = np.sum(np.log(theta) * self.latent_variables["z"])

        # Log x_i
        z = self.latent_variables["z"]
        lambdax = self.latent_variables["lambda"][z.argmax(axis=1)]
        log_x = np.sum(-lambdax + np.log(lambdax) * self.x)

        return log_x + log_z + log_lamb + log_theta  # + self.log_constant

    def one_epoch(self):
        for i in range(self.n_obs):
            self.latent_variables["z"][i] = self.sample_zi(i)
        self.latent_variables["theta"] = self.sample_theta()
        for k in range(self.n_cat):
            self.latent_variables["lambda"][k] = self.sample_lambda_k(k)

    def run_gibbs(self, n_epochs=100):
        hist = []
        for i in tqdm.tqdm(range(n_epochs)):
            self.one_epoch()
            hist.append(self.evaluate_log_joint())
        return hist
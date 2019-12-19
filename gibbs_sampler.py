import numpy as np
from scipy.stats import dirichlet, norm
import numpy_indexed as npi

class GibbsSampler:
    def __init__(
        self,
        num_cells,
        num_cell_types,
        num_genes,
        sigma,
        S,
        epsilon,
        graph,
        graph_reverse,
        alpha=None,
    ):
        self.num_cells = num_cells
        self.num_cell_types = num_cell_types
        self.num_genes = num_genes
        self.epsilon = epsilon
        self.graph = graph
        self.graph_reverse = graph_reverse
        self.epsilon_up = self.epsilon / self.graph.shape[1]
        self.sigma = sigma
        self.S = S
        self.Vi = None
        self.neighbors_cell_types = None
        if alpha is None:
            alpha = np.ones(num_cell_types)
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.pi = np.random.dirichlet(self.alpha)
        self.update_Z(
            np.random.choice(self.num_cell_types, p=self.pi, size=self.num_cells)
        )
        self.beta = np.random.normal(
            0, self.sigma, size=(self.num_cell_types, self.num_genes)
        )
        self.rho = np.random.normal(
            0,
            self.sigma,
            size=(self.num_cell_types, self.num_cell_types, self.num_genes),
        )
        self.log_joints = []

    def update_Z(self, Z):
        self.Z = Z.astype(int)

        # Count types
        self.n_i = np.zeros(self.num_cell_types)
        cell_types, counts = np.unique(self.Z, return_counts=True)
        self.n_i[cell_types] = counts

        # Reverse map each type l: list of cells having type l
        values, indices = npi.group_by(self.Z, np.arange(self.num_cells))
        self.Vi = [np.array([]) for _ in range(self.num_cell_types)]
        for i, v in enumerate(values):
            self.Vi[v] = indices[i]

        # For each cell v, number of neighbors of type k
        self.mvk = np.zeros((self.num_cells, self.num_cell_types))
        for i in range(self.num_cells):
            cell_types, counts = np.unique(self.Z[self.graph[i]], return_counts=True)
            self.mvk[i][cell_types] = counts

        # Count number of edge k->l (in l,k)
        self.ctype_ctype_counts = np.zeros((self.num_cell_types, self.num_cell_types))
        for l in range(self.num_cell_types):
            if len(self.Vi[l]):
                self.ctype_ctype_counts[l] = self.mvk[self.Vi[l]].sum(axis=0)

    def sample_pi(self):
        return np.random.dirichlet(self.alpha + self.n_i)

    def sample_beta(self, x):
        multipliers = self.sigma ** 2 / (self.S ** 2 + self.n_i * self.sigma ** 2)

        variances = (self.S ** 2) * multipliers
        alphas = np.array(
            [
                (x[self.Vi[i]].sum(axis=0) if len(self.Vi[i]) else 0)
                - self.epsilon_up
                * (self.rho[i, :, :] * self.ctype_ctype_counts[i, :, None]).sum(axis=0)
                for i in range(self.num_cell_types)
            ]
        )

        return np.random.normal(
            multipliers[:, None] * alphas, np.sqrt(variances)[:, None]
        )

    def sample_rho(self, x):
        self.mvk

        eta = 1.0 / (
            self.S ** 2 / self.sigma ** 2
            + self.epsilon_up ** 2
            * np.array(
                [
                    (self.mvk[self.Vi[i]] ** 2).sum(axis=0)
                    if len(self.Vi[i])
                    else np.zeros((self.num_cell_types))
                    for i in range(self.num_cell_types)
                ]
            )
        )

        t = self.epsilon_up * np.array(
            [
                [
                    np.sum(
                        (
                            self.mvk[:, k, None]
                            * (
                                x
                                - self.beta[i, None, :]
                                - self.epsilon_up
                                * np.sum(
                                    self.rho[i, self.Z[self.graph], :]
                                    * (self.Z[self.graph] != k)[:, :, None],
                                    axis=1,
                                )
                            )
                        )
                        * (self.Z == i)[:, None],
                        axis=0,
                    )
                    for k in range(self.num_cell_types)
                ]
                for i in range(self.num_cell_types)
            ]
        )

        #         for i in range(self.num_cell_types):
        #             mu[i] = self.epsilon_up * (
        #                 self.mvk[self.Vi[i]].sum(axis=0) if len(self.Vi[i]) else np.zeros((self.num_cell_types))
        #             )
        return np.random.normal(
            eta[:, :, None] * t, np.sqrt((self.S ** 2) * eta[:, :, None])
        )

    def log_pz(self, x, v, k):
        # logp(z_v = k) + constant
        term1 = (
            (
                self.beta[k]
                - x[v]
                + self.epsilon_up
                * self.rho[self.Z[v]][self.Z[self.graph[v]]].sum(axis=0)
            )
            ** 2
        ).sum()

        neighbors = self.graph_reverse[v]
        if len(neighbors):
            zw = self.Z[neighbors]
            old_zv = self.Z[v]
            self.Z[v] = k

            term2 = (
                (
                    self.beta[zw]
                    - x[neighbors]
                    + self.epsilon_up
                    * self.rho[zw[:, None], self.Z[self.graph[neighbors]], :].sum(
                        axis=1
                    )
                )
                ** 2
            ).sum()

            self.Z[v] = old_zv
        else:
            term2 = 0

        return term1 + term2

    def sample_Z(self, x):
        log_vals = np.array(
            [
                [
#                                         self._log_joint(
#                                             x, self.Z + (k - self.Z[i]) * (np.arange(self.num_cells) == i)
#                                         )
                    - self.log_pz(x, i, k) + np.log(self.pi[k])
                    for k in range(self.num_cell_types)
                ]
                for i in range(self.num_cells)
            ]
        )
        log_vals -= np.max(log_vals, axis=1)[:, None]
        probas = np.exp(log_vals) / np.sum(np.exp(log_vals), axis=1)[:, None]
        return np.array(
            [
                np.random.choice(np.arange(self.num_cell_types), p=probas[v])
                for v in range(self.num_cells)
            ]
        )

    """
        This function computes the log-joint for a specified Z, which is used while sampling Z. 
        To get the log-joint for the current value of the parameters, use log_joint
    """

    def _log_joint(self, x, Z):
        temp = np.sum(np.log(self.pi[Z]))
        loc = np.array(
            [
                self.beta[Z[v], :]
                + self.epsilon_up * np.sum(self.rho[Z[v], Z[self.graph[v]], :], axis=0)
                for v in range(self.num_cells)
            ]
        )
        return temp + np.sum(norm.logpdf(x, loc, self.S * np.ones(loc.shape)))

    def log_joint(self, x):
        return self._log_joint(x, self.Z)

    def train_(self, num_epochs, x):
        for _ in range(num_epochs):
            self.update_Z(self.sample_Z(x))
            self.pi = self.sample_pi()
            self.beta = self.sample_beta(x)
            self.rho = self.sample_rho(x)
            self.log_joints.append(self.log_joint(x))

    def train(self, num_epochs, x):
        for _ in range(num_epochs):
            self.update_Z(self.sample_Z(x))
            self.pi = self.sample_pi()
            self.beta = self.sample_beta(x)
            self.rho = self.sample_rho(x)
            self.log_joints.append(self.log_joint(x))

    def plot(self):
        plt.plot(np.arange(len(self.log_joints)), self.log_joints)

import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from scipy.stats import dirichlet, norm
import numpy_indexed as npi
from scipy.stats import mode
from sklearn.metrics import normalized_mutual_info_score

class GibbsSampler:
    def __init__(self, data_generator, sigma=10, S=1, alpha=None):
        self.n_cells = dg.n_cells
        self.n_cell_types = dg.n_cell_types
        self.n_genes = dg.n_genes
        self.epsilon = dg.epsilon
        self.graph = dg.graph
        self.graph_reversed = dg.graph_reversed
        self.dg = data_generator

        self.epsilon_up = self.epsilon / self.graph.shape[1]
        self.sigma = sigma
        self.S = S
        self.Vi = None
        self.neighbors_cell_types = None
        if alpha is None:
            alpha = np.ones(dg.n_cell_types)
        self.alpha = alpha
        self.cluster_truth = dg.cell_types
        self.reset()

    def reset(self):
        self.pi = np.random.dirichlet(self.alpha)
        self.Z = np.random.choice(self.n_cell_types, p=self.pi, size=self.n_cells)
        self.beta = np.random.normal(
            0, self.sigma, size=(self.n_cell_types, self.n_genes)
        )
        self.rho = np.random.normal(
            0, self.sigma, size=(self.n_cell_types, self.n_cell_types, self.n_genes)
        )
        self.update_intermediaries()
        self.log_joints = []
        self.cluster_score = []

    def update_intermediaries(self):
        # Count types
        self.n_i = np.zeros(self.n_cell_types)
        cell_types, counts = np.unique(self.Z, return_counts=True)
        self.n_i[cell_types] = counts

        # Reverse map each type l: list of cells having type l
        values, indices = npi.group_by(self.Z, np.arange(self.n_cells))
        self.Vi = [np.array([]) for _ in range(self.n_cell_types)]
        for i, v in enumerate(values):
            self.Vi[v] = indices[i]

        # For each cell v, number of neighbors of type k
        self.mvk = np.zeros((self.n_cells, self.n_cell_types))
        for i in range(self.n_cells):
            cell_types, counts = np.unique(self.Z[self.graph[i]], return_counts=True)
            self.mvk[i][cell_types] = counts

        # Count number of edge k->l (in l,k)
        self.ctype_ctype_counts = np.zeros((self.n_cell_types, self.n_cell_types))
        for l in range(self.n_cell_types):
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
                for i in range(self.n_cell_types)
            ]
        )

        return np.random.normal(
            multipliers[:, None] * alphas, np.sqrt(variances)[:, None]
        )

    def update_rho(self, x):
        eta = 1.0 / (
            self.S ** 2 / self.sigma ** 2
            + self.epsilon_up ** 2
            * np.array(
                [
                    (self.mvk[self.Vi[i]] ** 2).sum(axis=0)
                    if len(self.Vi[i])
                    else np.zeros((self.n_cell_types))
                    for i in range(self.n_cell_types)
                ]
            )
        )

        for k in range(self.n_cell_types):

            t = self.epsilon_up * np.array(
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
                    for i in range(self.n_cell_types)
                ]
            )
            # t: ij
            self.rho[:, k, :] = np.random.normal(
                eta[:, k, None] * t, np.sqrt((self.S ** 2) * eta[:, k, None])
            )

        #         for i in range(self.n_cell_types):
        #             mu[i] = self.epsilon_up * (
        #                 self.mvk[self.Vi[i]].sum(axis=0) if len(self.Vi[i]) else np.zeros((self.n_cell_types))
        #             )
        return self.rho

    def log_pz(self, x, v, k):
        # logp(z_v = k) + constant
        term1 = (
            (
                self.beta[k]
                - x[v]
                + self.epsilon_up
                * self.rho[k][self.Z[self.graph[v]]].sum(axis=0)
            )
            ** 2
        ).sum()

        neighbors = self.graph_reversed[v]
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

    def update_Z(self, x):
        for i in range(self.n_cells):
            log_vals = np.array(
                [
                    -self.log_pz(x, i, k) + np.log(self.pi[k])
                    for k in range(self.n_cell_types)
                ]
            )
            log_vals -= np.max(log_vals)
            probas = np.exp(log_vals) / np.sum(np.exp(log_vals))
            
            self.Z[i] = np.random.choice(np.arange(self.n_cell_types), p=probas)
        
    """
        This function computes the log-joint for a specified Z, which is used while sampling Z. 
        To get the log-joint for the current value of the parameters, use log_joint
    """

    def _log_joint(self, x, Z):
        temp = (
            dirichlet.logpdf(self.pi, self.alpha)
            + np.sum(np.log(self.pi[Z]))
            + np.sum(norm.logpdf(self.beta, 0, self.sigma))
            + np.sum(norm.logpdf(self.rho, 0, self.sigma))
        )
        loc = np.array(
            [
                self.beta[Z[v], :]
                + self.epsilon_up * np.sum(self.rho[Z[v], Z[self.graph[v]], :], axis=0)
                for v in range(self.n_cells)
            ]
        )
        return temp + np.sum(norm.logpdf(x, loc, self.S * np.ones(loc.shape)))

    def log_joint(self, x):
        return self._log_joint(x, self.Z)

    def cold_start(self, x, n_epochs=10, n_samples=15, seed=0):
        scores = []
        for i in range(n_samples):
            np.random.seed(seed + i)
            self.reset()
            self.train(n_epochs, x, False)
            scores.append((self.log_joint(x), i))
        _, i = max(scores)

        np.random.seed(seed + i)
        self.reset()
        self.train(n_epochs, x)

        return max(scores), min(scores)

    def train(self, num_epochs, x, log_scores=True):
        for ep in range(num_epochs):
            self.update_Z(x)
            self.update_intermediaries()
            self.pi = self.sample_pi()
            self.beta = self.sample_beta(x)
            self.update_rho(x)
            if log_scores and (ep + 1) % 5 == 0:
                self.log_joints.append(self.log_joint(x))
                self.cluster_score.append(
                    normalized_mutual_info_score(self.Z, self.cluster_truth)
                )

    def plot(self):
        plt.plot(np.arange(len(self.log_joints)), self.log_joints)

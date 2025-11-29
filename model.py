import numpy as np
import nlopt


class PoissonPPCA:
    """
    Cas simplifié du modèle de l'article. Ici, on ne prend pas compte des "covariates & offset". On limite la modélisation à Z = mu + BW.
    """

    def __init__(self, Y, q):
        """
        Initialisation du modèle.
        Inputs:
            - Y np.array, Matrices des variables observées [n, p]
            - q int, dimension des vartiables dans l'espace latent
        """
        self.Y = Y
        self.n, self.p = Y.shape
        self.q = q

        # Initialisation des paramètres

        self.M = np.zeros((self.n, q))
        self.S = np.ones((self.n, q)) * 0.5
        self.B = np.random.normal(0, 0.1, (self.p, q))
        self.mu = np.log(1 + Y.mean(axis=0))

    def compute_A(self):
        """
        Renvoie la matrice A de l'article qui correspond 'the matrix of conditional expectations'
        """
        MB = self.M @ self.B.T  # (n × p)
        SSBB = 0.5 * (self.S**2) @ (self.B**2).T
        return np.exp(self.mu + MB + SSBB)

    def objective(self, x, grad):
        """
        Calcule la "lower bound" de l'approche variationnelle.
        Inputs:
            - x, np.array, contient les paramètres (M, S, B et mu) concaténés et 'flattened'
            - grad, np.array, correspond au gradient
        Renvoie -Jp puisque l'optimiseur utilisé "nlopt" minimise la fonction en entrée, et on cherche à maximiser Jp la lower bound de notre côté.
        """
        self.unpack(x)

        A = self.compute_A()
        MB = self.M @ self.B.T
        linear = self.Y * (self.mu + MB)
        entropy = -0.5 * ((self.M**2 + self.S**2 - 2 * np.log(self.S) - 1).sum())

        J = linear.sum() - A.sum() + entropy

        # Si grad.size > 0 cela implique que nlopt a besoin d'updater le gradient
        if grad.size > 0:
            gM, gS, gB, gmu = self.gradients(A)
            grad[:] = self.pack(gM, gS, gB, gmu)

        return float(-J)

    def gradients(self, A):
        """
        Dans le cas de la loi Poisson, l'article expose une écriture explicite du gradient.
        """
        Y, B, M, S = self.Y, self.B, self.M, self.S

        R = Y - A

        gmu = R.sum(axis=0)

        A1 = (A.T @ (S**2)) * B.T
        gB = R.T @ M - A1.T

        gM = R @ B - M

        A2 = 2 * (A @ (B**2)) * S
        gS = (1 / S) - A2 - S

        return gM, gS, gB, gmu

    ### Pack et Unpack concatènent pour plus d'érgonomie et les dé-concaténent lors des calculs.
    def pack(self, M, S, B, mu):
        return np.concatenate([M.ravel(), S.ravel(), B.ravel(), mu.ravel()])

    def unpack(self, x):
        n, p, q = self.n, self.p, self.q
        idx = 0
        self.M = x[idx : idx + n * q].reshape(n, q)
        idx += n * q
        self.S = x[idx : idx + n * q].reshape(n, q)
        idx += n * q
        self.B = x[idx : idx + p * q].reshape(p, q)
        idx += p * q
        self.mu = x[idx : idx + p]
        idx += p

    def fit(self, max_iter=50):
        """Fit le modèle utilisant la méthode MMA décrite dans l'article à l'aide de la framework nlopt"""
        x0 = self.pack(self.M, self.S, self.B, self.mu)

        opt = nlopt.opt(nlopt.LD_MMA, len(x0))
        opt.set_min_objective(self.objective)
        opt.set_maxeval(max_iter)

        x_opt = opt.optimize(x0)
        self.unpack(x_opt)

        return self

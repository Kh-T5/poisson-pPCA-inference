import numpy as np
import nlopt


class PoissonPPCA:
    """
    Cas simplifié du modèle de l'article. Ici, on ne prend pas compte des "covariates & offset". On limite la modélisation à Z = mu + BW.
    """

    def __init__(self, Y, q, tol, max_iter, track_history):
        """
        Initialisation du modèle.
        Inputs:
            - Y np.array, Matrices des variables observées [n, p]
            - q int, dimension des vartiables dans l'espace latent
        """
        self.Y = Y

        # Dimensions du modèle
        self.n, self.p = Y.shape
        self.q = q

        # Params d'entraînement
        self.tol = tol
        self.max_iter = max_iter
        self.track_history = track_history
        self.history_J = []

        # Initialisation des paramètres
        self.M = np.zeros((self.n, q))
        self.S = np.ones((self.n, q)) * 0.5
        self.mu = np.log(1 + Y.mean(axis=0))

        # Initialiser B via SVD(Y) suivant l'initialisation de l'article pour garantir plus de stabilité
        U, s, Vt = np.linalg.svd(Y - Y.mean(0), full_matrices=False)
        self.B = Vt[:q].T  # (p, q) loadings initiaux

    def compute_A(self):
        """
        Renvoie la matrice A de l'article qui correspond 'the matrix of conditional expectations'
        """
        MB = self.M @ self.B.T  # [n, p]
        SSBB = 0.5 * (self.S**2) @ (self.B**2).T
        Z = self.mu + MB + SSBB
        Z_clip = np.clip(Z, -20, 20)
        return np.exp(Z_clip)

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
        S_clipped = np.clip(self.S, 1e-12, None)  # Éviter le log d'un S négatif ou nul

        linear = self.Y * (self.mu + MB)
        entropy = -0.5 * ((self.M**2 + S_clipped**2 - 2 * np.log(S_clipped) - 1).sum())

        J = linear.sum() - A.sum() + entropy
        if self.track_history:
            self.history_J.append(J)

        # Si grad.size > 0 cela implique que nlopt a besoin d'updater le gradient
        if grad.size > 0:
            gM, gS, gB, gmu = self.gradients(A)
            grad[:] = -self.pack(gM, gS, gB, gmu)

        return float(-J)

    def gradients(self, A):
        """
        Dans le cas de la loi Poisson, l'article expose une écriture explicite du gradient.
        """
        Y, B, M, S = self.Y, self.B, self.M, self.S
        S = np.clip(
            S, 1e-12, None
        )  # Clip S pour éviter les division par 0 / log négatif

        R = Y - A
        gmu = R.sum(axis=0)

        A1 = (A.T @ (S**2)) * B
        gB = R.T @ M - A1

        gM = R @ B - M

        A2 = 2 * (A @ (B**2)) * S
        gS = (1 / S) - A2 - S

        return gM, gS, gB, gmu

    ### Pack et Unpack concatènent pour plus d'érgonomie et les dé-concaténent lors des calculs.
    ### Ses fonctions sont adaptées à l'input et l'output de la librarie d'optimisation nlopt
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

    def fit(self, max_iter=None, tol=None):
        """Fit le modèle utilisant la méthode MMA décrite dans l'article à l'aide de la framework d'optimisation nlopt"""

        if max_iter is None:
            max_iter = self.max_iter
        if tol is None:
            tol = self.tol

        x0 = self.pack(self.M, self.S, self.B, self.mu)
        n, p, q = self.n, self.p, self.q

        # Init optimiseur & fonction objective
        opt = nlopt.opt(nlopt.LD_MMA, len(x0))
        opt.set_min_objective(self.objective)

        eps_s = 1e-6  # Valeur min
        start_S, end_S = n * q, 2 * n * q  # Emplacement de S dans l'array "packed"

        # Contraintes B : compris dans [-5, 5]
        start_B = 2 * n * q
        end_B = start_B + p * q

        # Contraintes M : compris dans [-6, 6]
        start_M = 0
        end_M = n * q

        # Vecteurs lower et upper bound pour l'opt
        lb = -np.inf * np.ones_like(x0)
        ub = np.inf * np.ones_like(x0)

        # Ajout de contraintes écart-type S positif à l'optimiseur / B et M pour éviter d'avoir J qui explose
        lb[start_S:end_S] = eps_s
        lb[start_B:end_B] = -5
        ub[start_B:end_B] = 5
        lb[start_M:end_M] = -6
        ub[start_M:end_M] = 6

        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)

        opt.set_maxeval(max_iter)  # nb max d'itérations
        opt.set_ftol_rel(tol)  # critère relatif
        opt.set_ftol_abs(tol)  # critère absolu

        x_opt = opt.optimize(x0)
        self.unpack(x_opt)

        self.final_J = -opt.last_optimum_value()
        self.result_code = opt.last_optimize_result()
        return self

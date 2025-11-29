from model import PoissonPPCA
import numpy as np
from config import q, p, n, max_iter, tol, Visu_lower_bound
from generate_dummy_data import simulate_poisson_lognormal
from visualisation import visualize_latent_space
import matplotlib.pyplot as plt


if __name__ == "__main__":
    Y, Z_true, W_true, mu_true, B_true = simulate_poisson_lognormal(n=n, p=p, q=q)

    model = PoissonPPCA(Y, q=q, tol=tol, max_iter=max_iter, track_history=True)
    model.fit()
    ### Print
    print("Var(P) =", np.var(model.M @ model.B.T))
    print("Nombre d'évaluations de J :", len(model.history_J))
    print("Dernières valeurs de J :", model.history_J[-5:])
    print("Code retour nlopt :", model.result_code)

    # Visualisation de la PCA sur l'espace latent trouvé par le modèle
    visualize_latent_space(model)

    # Visualisation de l'évolution de l'Objective durant l'optimisation
    if Visu_lower_bound:
        J = np.array(model.history_J)
        J_centered = J - J.max()

        n, p = model.n, model.p
        J_per_obs = J_centered / (n * p)

        plt.figure(figsize=(6, 5))
        plt.plot(J_per_obs)
        plt.xlabel("Évaluations de l'objectif")
        plt.ylabel("J_q centré")
        plt.title("Convergence normalisée de la borne J_q")
        plt.tight_layout()
        plt.show()

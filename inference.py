from model import PoissonPPCA
import numpy as np
from config import q, p, n, max_iter, tol, Visu_lower_bound
from generate_dummy_data import simulate_poisson_lognormal
from visualisation import visualize_latent_space

if __name__ == "__main__":
    Y, Z_true, W_true, mu_true, B_true = simulate_poisson_lognormal(n=n, p=p, q=q)

    model = PoissonPPCA(Y, q=q, tol=tol, max_iter=max_iter, track_history=True)
    model.fit()
    ### Print
    print("Var(P) =", np.var(model.M @ model.B.T))
    print("Nombre d'évaluations de J :", len(model.history_J))
    print("Dernières valeurs de J :", model.history_J[-5:])
    print("Code retour nlopt :", model.result_code)
    # Visu
    visualize_latent_space(model)

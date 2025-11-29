import numpy as np
import matplotlib.pyplot as plt


def visualize_latent_space(
    model,
    labels=None,
    title="Projection latente pPCA\nsur dataset dummy suivant\nune distribution poisson lognormal",
):
    """
    Visualise les individus dans l'espace latent via une PCA sur P = M B^T.

    Inputs:
        - model, PoissonPPCA déjà fit
        - labels, array-like ou None (Optionnel, étiquettes pour colorer les points).
        - title, str
    """
    P = model.M @ model.B.T
    P_centered = P - P.mean(axis=0, keepdims=True)  # Matrice centrée

    # Suivant la méthodologie de visualisation de l'article, on effectue une SVD sur P_centered pour avoir ensuite la PCA associée.
    U, S, Vt = np.linalg.svd(P_centered, full_matrices=False)

    scores = U[:, :2] * S[:2]  # scores résultat PCA

    # variance expliquée (Proportion de la variance attribuée à chaque valeur propre de la SVD)
    var_explained = (S**2) / (S**2).sum()
    pc1_var = var_explained[0] * 100
    pc2_var = var_explained[1] * 100

    plt.figure(figsize=(6, 5))
    if labels is None:
        plt.scatter(scores[:, 0], scores[:, 1], alpha=0.7)
    else:
        labels = np.array(labels)
        for lab in np.unique(labels):
            mask = labels == lab
            plt.scatter(scores[mask, 0], scores[mask, 1], label=str(lab), alpha=0.7)
        plt.legend()

    plt.axhline(0, linewidth=0.5)
    plt.axvline(0, linewidth=0.5)
    plt.xlabel(f"PC1 ({pc1_var:.1f}% var.)")
    plt.ylabel(f"PC2 ({pc2_var:.1f}% var.)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

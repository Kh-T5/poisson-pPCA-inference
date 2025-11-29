# poisson-pPCA-inference
This project implements a minimal Poisson probabilistic PCA using variational inference as described in Chiquet et al. (2018). The goal is to reproduce, on a dummy dataset, the core ideas of the paper: latent Gaussian factors, Poisson observations, and optimization of a variational lower bound by gradient descent.

Dependencies:
 - Python version: 3.10
 - RUN: pip install -r requirements.txt for packages dependencies.


- To run the project, run "python -m inference" in terminal / shell.
- Possibilities to play with model parameters in config.py

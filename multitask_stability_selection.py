import numpy as np

def stability_selection(model, alphas, n_bootstrap_iterations,
                        X, y, bootstrap_sets, seed):
    """
    Computes stability scores for a given model using stability selection.

    Args:
        model: A model instance (e.g., MuGLasso, SMuGLasso) with `set_params` and `fit` methods.
        alphas (array): Array of regularization parameter values to evaluate.
        n_bootstrap_iterations (int): Number of bootstrap iterations.
        X (ndarray): Feature matrix of shape (n_samples, n_features).
        y (ndarray): Target vector of shape (n_samples,).
        bootstrap_sets (list): List of bootstrap datasets (each is a list of indices).
        seed (int): Random seed for reproducibility.

    Returns:
        stability_scores (ndarray): Stability scores of shape (n_variables, n_alphas).
    """
    n_samples, n_variables = X.shape
    n_alphas = alphas.shape[0]
    rnd = np.random.RandomState(seed)

    # Initialize stability scores matrix
    stability_scores = np.zeros((n_variables, n_alphas))

    # Iterate over alphas
    for idx, alpha in enumerate(alphas):
        selected_variables = np.zeros((n_variables, n_bootstrap_iterations))

        # Iterate over bootstrap iterations
        for iteration in range(n_bootstrap_iterations):
            # Combine multiple bootstrap sets dynamically
            combined_bootstrap = np.concatenate([
                rnd.choice(bootstrap, size=len(bootstrap) // 2, replace=False)
                for bootstrap in bootstrap_sets
            ])

            # Subset the data based on the bootstrap indices
            X_train = X[combined_bootstrap, :]
            y_train = y[combined_bootstrap]

            # Set model parameters dynamically (assuming scikit-learn compatibility)
            model.set_params({'C': alpha}).fit(X_train, y_train)

            # Track selected variables based on the model coefficients
            selected_variables[:, iteration] = (np.abs(model.coef_) > 1e-4)

        # Compute stability scores for the current alpha
        stability_scores[:, idx] = selected_variables.mean(axis=1)

    return stability_scores

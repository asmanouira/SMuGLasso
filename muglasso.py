import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

# Importing the Cython-compiled functions from muglasso_fast
from muglasso_fast import bcd_fast_quadratic, bcd_fast_logistic

NO_SCREENING = 0
GAPSAFE_SEQ = 1
GAPSAFE = 2

class MuGLasso:
    """
    Multi-task Group Lasso (MuGLasso) model supporting both quadratic and logistic loss functions.
    - Quadratic loss for regression tasks (quantitative traits).
    - Logistic loss for classification tasks (binary qualitative traits).
    
    Parameters
    ----------
    loss : str, 'logistic' or 'quadratic'
        The loss function to use. 'logistic' for classification (binary), 'quadratic' for regression.
    tau : float, default=0.5
        The trade-off parameter between L1 and L2 regularization in group lasso.
    lambda2 : float, default=0
        L2 regularization strength.
    screen : int, default=GAPSAFE
        The screening rule to use (NO_SCREENING, GAPSAFE_SEQ, or GAPSAFE).
    max_iter : int, default=30000
        Maximum number of iterations for the optimization.
    f : int, default=10
        Frequency of applying the screening rule.
    eps : float, default=1e-4
        Accuracy tolerance for the dual gap.
    verbose : bool, default=False
        Whether to print optimization details.
    """

    def __init__(self, loss='quadratic', tau=0.5, lambda2=0, screen=GAPSAFE, 
                 max_iter=30000, f=10, eps=1e-4, verbose=False):
        if loss not in ['quadratic', 'logistic']:
            raise ValueError("Invalid loss function. Use 'quadratic' for regression or 'logistic' for classification.")
        self.loss = loss
        self.tau = tau
        self.lambda2 = lambda2
        self.screen = screen
        self.max_iter = max_iter
        self.f = f
        self.eps = eps
        self.verbose = verbose
        self.betas = None
        self.gaps = None
        self.n_iters = None
        self.screening_sizes_features = None

    def fit(self, X, y, size_groups, omega, lambdas=None, beta_init=None,
            gap_active_warm_start=False, strong_active_warm_start=True):
        """
        Fit the MuGLasso model to the data using the specified loss function.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input data matrix.
        y : ndarray, shape (n_samples,)
            The target values. Binary for logistic loss, continuous for quadratic loss.
        size_groups : ndarray, shape (n_groups,)
            Sizes of the different groups of features.
        omega : ndarray, shape (n_groups,)
            Weights for the different groups of features.
        lambdas : ndarray, optional
            A sequence of lambda regularization values.
        beta_init : ndarray, optional
            Initial values for the regression coefficients.
        gap_active_warm_start : bool, optional
            Use active warm start based on the duality gap.
        strong_active_warm_start : bool, optional
            Use strong active warm start based on previous solutions.
        
        Returns
        -------
        self : object
            Fitted MuGLasso instance.
        """
        if self.loss == 'quadratic':
            # Call the quadratic loss solver from Cython
            self.betas, self.gaps, self.n_iters, self.screening_sizes_features = bcd_fast_quadratic(
                X, y, size_groups, omega, lambdas, self.tau, self.lambda2, beta_init,
                self.screen, self.max_iter, self.f, self.eps, gap_active_warm_start,
                strong_active_warm_start, self.verbose
            )
        elif self.loss == 'logistic':
            # Call the logistic loss solver from Cython
            self.betas, self.gaps, self.n_iters, self.screening_sizes_features = bcd_fast_logistic(
                X, y, size_groups, omega, lambdas, self.tau, self.lambda2, beta_init,
                self.screen, self.max_iter, self.f, self.eps, gap_active_warm_start,
                strong_active_warm_start, self.verbose
            )

        return self

    def get_params(self):
        """Return the learned parameters."""
        return {
            'betas': self.betas,
            'gaps': self.gaps,
            'n_iters': self.n_iters,
            'screening_sizes_features': self.screening_sizes_features
        }

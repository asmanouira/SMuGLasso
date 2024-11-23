import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from smuglasso_fast import elasticnet_path_logistic, elasticnet_path_quadratic

NO_SCREENING = 0
GAPSAFE_SEQ = 1
GAPSAFE = 2

class SMuGLasso:
    """
    Safe ElasticNet (SMuGLasso) model supporting both quadratic and logistic loss functions.
    - Quadratic loss for regression tasks (quantitative traits).
    - Logistic loss for classification tasks (binary qualitative traits).
    
    Parameters
    ----------
    loss : str, 'logistic' or 'quadratic'
        The loss function to use. 'logistic' for classification (binary), 'quadratic' for regression.
    alpha : float, default=1.0
        The mixing parameter for L1 (Lasso) and L2 (Ridge) regularization (ElasticNet).
        `alpha = 1.0` means Lasso, `alpha = 0.0` means Ridge.
    lambda1 : float, default=1.0
        The L1 regularization strength.
    lambda2 : float, default=0.0
        The L2 regularization strength.
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

    def __init__(self, loss='quadratic', alpha=1.0, lambda1=1.0, lambda2=0.0,
                 screen=GAPSAFE, max_iter=30000, f=10, eps=1e-4, verbose=False):
        if loss not in ['quadratic', 'logistic']:
            raise ValueError("Invalid loss function. Use 'quadratic' for regression or 'logistic' for classification.")
        self.loss = loss
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.screen = screen
        self.max_iter = max_iter
        self.f = f
        self.eps = eps
        self.verbose = verbose
        self.betas = None
        self.gaps = None
        self.n_iters = None
        self.screening_sizes_groups = None
        self.screening_sizes_features = None

    def fit(self, X, y, lambdas=None, beta_init=None, gap_active_warm_start=False, strong_active_warm_start=True):
        """
        Fit the SMuGLasso model to the data using the specified loss function.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input data matrix.
        y : ndarray, shape (n_samples,)
            The target values. Binary for logistic loss, continuous for quadratic loss.
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
            Fitted SMuGLasso instance.
        """
        if self.loss == 'quadratic':
            # Call the Cython quadratic loss solver (ElasticNet for regression)
            self.betas, self.gaps, self.n_iters, self.screening_sizes_features = elasticnet_path_quadratic(
                X, y, lambdas=lambdas, beta_init=beta_init, alpha=self.alpha,
                lambda1=self.lambda1, lambda2=self.lambda2, screen=self.screen,
                max_iter=self.max_iter, f=self.f, eps=self.eps,
                gap_active_warm_start=gap_active_warm_start,
                strong_active_warm_start=strong_active_warm_start,
                verbose=self.verbose
            )
        elif self.loss == 'logistic':
            # Call the Cython logistic loss solver (ElasticNet for classification)
            self.betas, self.gaps, self.n_iters, self.screening_sizes_features = elasticnet_path_logistic(
                X, y, lambdas=lambdas, beta_init=beta_init, alpha=self.alpha,
                lambda1=self.lambda1, lambda2=self.lambda2, screen=self.screen,
                max_iter=self.max_iter, f=self.f, eps=self.eps,
                gap_active_warm_start=gap_active_warm_start,
                strong_active_warm_start=strong_active_warm_start,
                verbose=self.verbose
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

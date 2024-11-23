# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
from libc.math cimport fabs, sqrt, exp, log
from scipy.linalg.cython_blas cimport daxpy, ddot, dnrm2, dscal
import numpy as np
cimport numpy as np
cimport cython

cdef:
    int inc = 1
    int NO_SCREENING = 0
    int GAPSAFE_SEQ = 1
    int GAPSAFE = 2


cdef inline double logistic_loss(double z) nogil:
    """ Logistic loss component """
    if z > 0:
        return log(1 + exp(-z))
    else:
        return log(1 + exp(z)) - z


cdef inline double logistic_gradient(double z) nogil:
    """ Gradient of the logistic loss function """
    return -1.0 / (1.0 + exp(z))


cdef double primal_value_logistic(int n_samples, int n_features,
                                  double * residual, double * beta,
                                  double lambda1, double lambda2) nogil:
    cdef:
        double l1_norm = 0.
        double l2_norm = 0.
        double fval = 0.
        int i = 0

    # L1 and L2 norms (ElasticNet penalty)
    for i in range(n_features):
        l1_norm += fabs(beta[i])
        l2_norm += beta[i] ** 2

    # Logistic loss
    for i in range(n_samples):
        fval += logistic_loss(residual[i])

    fval += lambda1 * l1_norm + 0.5 * lambda2 * l2_norm
    return fval


cdef double primal_value_quadratic(int n_samples, int n_features,
                                   double * residual, double * beta,
                                   double lambda1, double lambda2) nogil:
    cdef:
        double l1_norm = 0.
        double l2_norm = 0.
        double fval = 0.
        int i = 0

    # L1 and L2 norms (ElasticNet penalty)
    for i in range(n_features):
        l1_norm += fabs(beta[i])
        l2_norm += beta[i] ** 2

    # Quadratic loss
    residual_norm2 = dnrm2(&n_samples, residual, &inc) ** 2
    fval = 0.5 * residual_norm2 + lambda1 * l1_norm + 0.5 * lambda2 * l2_norm
    return fval


cdef double dual_gap_logistic(int n_samples, int n_features,
                              double * residual, double * beta, double * y,
                              double lambda1, double lambda2, double dual_scale) nogil:
    cdef:
        double dobj = 0.
        double pobj = primal_value_logistic(n_samples, n_features, residual, beta, lambda1, lambda2)
        double Ry = ddot(&n_samples, residual, &inc, y, &inc)

    # Dual objective (simplified)
    dobj = -0.5 * lambda1 ** 2 * dnrm2(&n_samples, residual, &inc) ** 2 / dual_scale + lambda1 * Ry / dual_scale
    return pobj - dobj


cdef double dual_gap_quadratic(int n_samples, int n_features,
                               double * residual, double * beta, double * y,
                               double lambda1, double lambda2, double dual_scale) nogil:
    cdef:
        double dobj = 0.
        double pobj = primal_value_quadratic(n_samples, n_features, residual, beta, lambda1, lambda2)
        double Ry = ddot(&n_samples, residual, &inc, y, &inc)

    # Dual objective (simplified)
    dobj = -0.5 * lambda1 ** 2 * dnrm2(&n_samples, residual, &inc) ** 2 / dual_scale + lambda1 * Ry / dual_scale
    return pobj - dobj


def bcd_fast_logistic(double[::1, :] X, double[::1] y, double[::1] beta,
                      double[::1] XTR, double[::1] residual, double dual_scale,
                      int n_samples, int n_features, double lambda1, double lambda2,
                      int max_iter, int f, double tol, int screen, 
                      int[::1] disabled_features, int strong_warm_start=0):
    """
    Block Coordinate Descent for ElasticNet with logistic loss.
    """

    cdef:
        int n_iter = 0
        double gap_t = 666.
        double double_tmp = 0.

    for n_iter in range(max_iter):

        # Apply dual scaling and compute the dual gap at every `f` iterations
        if f != 0 and n_iter % f == 0:
            residual_norm2 = dnrm2(&n_samples, &residual[0], &inc) ** 2
            gap_t = dual_gap_logistic(n_samples, n_features, &residual[0], &beta[0], &y[0], lambda1, lambda2, dual_scale)

            if gap_t <= tol:
                break

        # Proximal step for each feature (ElasticNet)
        for j in range(n_features):
            if disabled_features[j] == 1:
                continue

            double_tmp = ddot(&n_samples, &X[0, j], &inc, &residual[0], &inc)
            XTR[j] = double_tmp - lambda2 * beta[j]
            beta[j] = ST(lambda1, beta[j] + XTR[j])

    return (dual_scale, gap_t, n_iter)


def bcd_fast_quadratic(double[::1, :] X, double[::1] y, double[::1] beta,
                       double[::1] XTR, double[::1] residual, double dual_scale,
                       int n_samples, int n_features, double lambda1, double lambda2,
                       int max_iter, int f, double tol, int screen, 
                       int[::1] disabled_features, int strong_warm_start=0):
    """
    Block Coordinate Descent for ElasticNet with quadratic loss.
    """

    cdef:
        int n_iter = 0
        double gap_t = 666.
        double double_tmp = 0.

    for n_iter in range(max_iter):

        # Apply dual scaling and compute the dual gap at every `f` iterations
        if f != 0 and n_iter % f == 0:
            residual_norm2 = dnrm2(&n_samples, &residual[0], &inc) ** 2
            gap_t = dual_gap_quadratic(n_samples, n_features, &residual[0], &beta[0], &y[0], lambda1, lambda2, dual_scale)

            if gap_t <= tol:
                break

        # Proximal step for each feature (ElasticNet)
        for j in range(n_features):
            if disabled_features[j] == 1:
                continue

            double_tmp = ddot(&n_samples, &X[0, j], &inc, &residual[0], &inc)
            XTR[j] = double_tmp - lambda2 * beta[j]
            beta[j] = ST(lambda1, beta[j] + XTR[j])

    return (dual_scale, gap_t, n_iter)


cdef double ST(double u, double x) nogil:
    """ Soft thresholding operator for ElasticNet """
    return fsign(x) * fmax(fabs(x) - u, 0.)


cdef inline double fsign(double f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0

'''
SQ-UKF utility functions for implementations of multiple methods
'''

from collections import namedtuple
import numpy as np
from numpy import ma
from scipy import linalg

from ..SKLearn_utils import array2d
from ..KF_utils import _last_dims


# sigma points w/ associated weights, as a row
SP = namedtuple(
    'SP',
    ['points', 'weights_mean', 'weights_cov']
)


# mean & covariance, as a row
Moments = namedtuple('Moments', ['mean', 'cov'])


def _reconstruct_covariances(covariance2s):
    '''Reconstruct covariance matrices w/ their cholesky factors'''
    
    if len(covariance2s.shape) == 2:
        covariance2s = covariance2s[np.newaxis, :, :]

    T = covariance2s.shape[0]
    covariances = np.zeros(covariance2s.shape)

    for t in range(T):
        M = covariance2s[t]
        covariances[t] = M.T.dot(M)

    return covariances


def cholupdate(A2, X, weight):
    '''Calculate chol(A + w.x.x')

    Params
    ----------
    A2 : [n_dim, n_dim] array
        A = A2.T.dot(A2) for A positive definite, symmetric
    X : [n_dim] or [n_vec, n_dim] array of vector(s) to be used for x
        If X has 2 dimensions, then each row is added in turn.
    weight : float
        weight to be multiplied to each x.x'
        If negative (downdate), use sign(weight) * sqrt(abs(weight)) instead of sqrt(weight).

    Returns
    -------
    A2 : [n_dim, n_dim array]
        cholesky factor of updated matrix

    Notes
    -----    
    function [L] = cholupdate(L,x)
        p = length(x);
        x = x';
        for k=1:p
            r = sqrt(L(k,k)^2 + x(k)^2);
            c = r / L(k, k);
            s = x(k) / L(k, k);
            L(k, k) = r;
            L(k,k+1:p) = (L(k,k+1:p) + s*x(k+1:p)) / c;
            x(k+1:p) = c*x(k+1:p) - s*L(k, k+1:p);
        end
    end
    '''
    
    # make copies
    X = X.copy()
    A2 = A2.copy()

    # standardize input shape
    if len(X.shape) == 1:
        X = X[np.newaxis, :]
    n_vec, n_dim = X.shape

    # take sign of weight into account
    sign, weight = np.sign(weight), np.sqrt(np.abs(weight))
    X = weight * X

    for i in range(n_vec):
        x = X[i, :]
        for k in range(n_dim):
            r_squared = A2[k, k] ** 2 + sign * x[k] ** 2
            r = 0.0 if r_squared < 0 else np.sqrt(r_squared)
            c = r / A2[k, k]
            s = x[k] / A2[k, k]
            A2[k, k] = r
            A2[k, k + 1:] = (A2[k, k + 1:] + sign * s * x[k + 1:]) / c
            x[k + 1:] = c * x[k + 1:] - s * A2[k, k + 1:]
    return A2


def qr(A):
    '''Get cholesky factor transpose of A
    Returns the square upper triangular matrix of A QR-decomposition.
    '''
    
    N, L = A.shape
    if not N >= L:
        raise ValueError("Number of columns must exceed number of rows")
    Q, R = linalg.qr(A)
    return R[:L, :L]


def points2moments(points, sigma2_noise=None):
    '''
    Params
    ------
    points : [2 * n_dim_state + 1, n_dim_state] SP
    sigma2_noise : [n_dim_state, n_dim_state] array - for additive case only

    Returns
    -------
    moments : [n_dim_state] Moments
        mean & square-root covariance
    '''
    
    (points, weights_mu, weights_sigma) = points
    mu = points.T.dot(weights_mu)

    # make points to perform QR factorization on. Each column is one data point.
    qr_points = [
        np.sign(weights_sigma)[np.newaxis, :]
        * np.sqrt(np.abs(weights_sigma))[np.newaxis, :]
        * (points.T - mu[:, np.newaxis])
    ]
    if sigma2_noise is not None:
        qr_points.append(sigma2_noise)
    sigma2 = qr(np.hstack(qr_points).T)
    sigma2 = cholupdate(sigma2, points[0] - mu, weights_sigma[0])
    return Moments(mu.ravel(), sigma2)


def moments2points(moments, alpha=None, beta=None, kappa=None):
    '''
    Params
    ------
    moments : [n_dim] Moments
    alpha : float
        Spread of SP. Typically 1e-3.
    beta : float
        Used to incorporate prior knowledge of the distribution of the state.
        2 is optimal is the state is normally distributed.
    kappa : float

    Returns
    -------
    points : [2*n_dim+1, n_dim] SP
    '''
    
    (mu, sigma2) = moments
    n_dim = len(mu)
    mu = array2d(mu, dtype=float)

    if alpha is None:
      alpha = 1.0
    if beta is None:
      beta = 0.0
    if kappa is None:
      kappa = 3.0 - n_dim

    sigma2 = sigma2.T

    # calculate scaling factor for all off-center points
    lamda = (alpha * alpha) * (n_dim + kappa) - n_dim
    gamma = n_dim + lamda

    # calculate the SP, as a stack of columns :
    #   mu
    #   mu + each column of sigma2 * sqrt(gamma)
    #   mu - each column of sigma2 * sqrt(gamma)
    # Each column of points is one of these.
    points = np.tile(mu.T, (1, 2 * n_dim + 1))
    points[:, 1:(n_dim + 1)] += sigma2 * np.sqrt(gamma)
    points[:, (n_dim + 1):] -= sigma2 * np.sqrt(gamma)

    # calculate associated weights
    weights_mean = np.ones(2 * n_dim + 1)
    weights_mean[0] = lamda / gamma
    weights_mean[1:] = 0.5 / gamma
    weights_cov = np.copy(weights_mean)
    weights_cov[0] = lamda / gamma + (1 - alpha * alpha + beta)

    return SP(points.T, weights_mean, weights_cov)


def _unscented_transform(points, f=None, sigma2_noise=None, params=None):
    '''
    Params
    ------
    points : [n_points, n_dim_1] SP
    f : [n_dim_1, n_dim_3] -> [n_dim_2] transition function
    sigma2_noise : [n_dim_2, n_dim_2] array - for additive case only
    params : [n_dim_params] array - ONLY for dual estimation

    Returns
    -------
    points_pred : [n_points, n_dim_2] SP
        transformed by f, same weights remaining
    moments_pred : [n_dim_2] Moments
        associated to points_pred
    '''
    
    n_points, n_dim_state = points.points.shape
    (points, weights_mean, weights_covariance) = points

    # propagate points through f. Each column is a sample point
    if f is not None:
        if params is None:
            points_pred = [f(points[i]) for i in range(n_points)]
        else:
            points_pred = [f(params, points[i]) for i in range(n_points)]
    else:
        points_pred = points

    # make each row a predicted point
    points_pred = np.vstack(points_pred)
    points_pred = SP(points_pred, weights_mean, weights_covariance)

    # calculate approximate mean, covariance
    moments_pred = points2moments(
        points_pred, sigma2_noise=sigma2_noise
    )

    return (points_pred, moments_pred)


def _unscented_correct(cross_sigma, moments_pred, obs_moments_pred, y):
    '''Correct predicted state estimates with an observation

    Params
    ------
    cross_sigma : [n_dim_state, n_dim_obs] array
        cross-covariance between t state & t obs | [0, t-1] obs
    moments_pred : [n_dim_state] Moments
        mean & covariance of t state | [0, t-1] obs
    obs_moments_pred : [n_dim_obs] Moments
        mean & covariance of t obs | [0, t-1] obs
    y : [n_dim_obs] array
        t obs

    Returns
    -------
    moments_filt : [n_dim_state] Moments
        mean & covariance of t state | [0, t] obs
    '''
    
    mu_pred, sigma2_pred = moments_pred
    obs_mu_pred, obs_sigma2_pred = obs_moments_pred

    n_dim_state = len(mu_pred)
    n_dim_obs = len(obs_mu_pred)

    if not np.any(ma.getmask(y)):
        
        # K = (cross_sigma / obs_sigma2_pred.T) / obs_sigma2_pred
        K = linalg.lstsq(obs_sigma2_pred, cross_sigma.T)[0]
        K = linalg.lstsq(obs_sigma2_pred.T, K)[0]
        K = K.T

        # correct mu, sigma
        mu_filt = mu_pred + K.dot(y - obs_mu_pred)
        U = K.dot(obs_sigma2_pred)
        sigma2_filt = cholupdate(sigma2_pred, U.T, -1.0)
        
    else:
        # no corrections to be made
        mu_filt = mu_pred
        sigma2_filt = sigma2_pred

    return Moments(mu_filt, sigma2_filt)


def unscented_filter_predict(transition_function, points_state,
                             sigma2_transition=None
                             params=None):
    '''Prediction of t+1 state distribution
    Using SP for t state | [0, t] obs, calculate predicted SP for t+1 state, associated mean & covariance.

    Params
    ------
    transition_function : function
    points_state : [2*n_dim_state+1, n_dim_state] SP
    sigma2_transition : [n_dim_state, n_dim_state] array
        covariance of additive noise in transitioning from t to t+1. If missing, assume noise is not additive.
    params : [n_dim_params] array - ONLY for dual estimation

    Returns
    -------
    points_pred : [2*n_dim_state+1, n_dim_state] SP
        for t+1 state | [0, t] obs - these points have not been "standardized" by UT yet.
    moments_pred : [n_dim_state] Moments
        mean & covariance associated to points_pred
    '''
    
    assert sigma2_transition is not None, \
        "Your system can't be noiseless and noise needs to be additive"
    (points_pred, moments_pred) = _unscented_transform(
        points_state, transition_function,
        sigma2_noise=sigma2_transition, params=params
        )
    
    return (points_pred, moments_pred)


def unscented_filter_correct(observation_function, moments_pred,
                             points_pred, observation,
                             sigma2_observation=None
                             params=None):
    '''Integration of t obs to correct predicted t state estimates (mean & covariance)

    Params
    ------
    observation_function : function
    moments_pred : [n_dim_state] Moments
        mean and covariance of t state | [0, t-1] obs
    points_pred : [2*n_dim_state+1, n_dim_state] SP
    observation : [n_dim_state] array
        t obs. If masked, treated as missing.
    sigma2_observation : [n_dim_obs, n_dim_obs] array
        covariance of additive noise in t obs. If missing, noise is assumed to be additive.
    params : [n_dim_params] array - ONLY for dual estimation

    Returns
    -------
    moments_filt : [n_dim_state] Moments
        mean & covariance of t state | [0, t] obs
    '''
    
    # calculate E[y_t | y_{0:t-1}] & Var(y_t | y_{0:t-1})
    (obs_points_pred, obs_moments_pred) = (
        _unscented_transform(
            points_pred, observation_function,
            sigma2_noise=sigma2_observation,
            params=params
        )
    )

    # calculate Cov(x_t, y_t | y_{0:t-1})
    sigma_pair = (
        ((points_pred.points - moments_pred.mean).T)
        .dot(np.diag(points_pred.weights_mean))
        .dot(obs_points_pred.points - obs_moments_pred.mean)
    )

    # calculate E[x_t | y_{0:t}] & Var(x_t | y_{0:t})
    moments_filt = _unscented_correct(sigma_pair, moments_pred, obs_moments_pred, observation)
    return moments_filt


def _additive_unscented_filter(mu_0, sigma_0, f, g, Q, R, Y):
    '''SQ-UKF w/ additive (zero-mean) transition & obs noises

    Params
    ------
    mu_0 : [n_dim_state] array
        mean of initial state distribution
    sigma_0 : [n_dim_state, n_dim_state] array
        covariance of initial state distribution
    f : function or [T-1] array of functions
        state transition function(s)
    g : function or [T] array of functions
        observation function(s)
    Q : [n_dim_state, n_dim_state] array
        transition noise covariance matrix
    R : [n_dim_state, n_dim_state] array
        observation noise covariance matrix
    Y : [T] array
        [0,T-1] obs

    Returns
    -------
    mu_filt : [T, n_dim_state] array
        mu_filt[t] = mean of t state | [0, t] obs
    sigma_filt : [T, n_dim_state, n_dim_state] array
        sigma_filt[t] = covariance of t state | [0, t] obs
    '''
    
    # extract size of key components
    T = Y.shape[0]
    n_dim_state = Q.shape[-1]
    n_dim_obs = R.shape[-1]

    # construct container for results
    mu_filt = np.zeros((T, n_dim_state))
    sigma2_filt = np.zeros((T, n_dim_state, n_dim_state))
    
    Q2 = linalg.cholesky(Q)
    R2 = linalg.cholesky(R)

    for t in range(T):
        # SP for P(x_{t-1} | y_{0:t-1})
        if t == 0:
            mu, sigma2 = mu_0, linalg.cholesky(sigma_0)
        else:
            mu, sigma2 = mu_filt[t - 1], sigma2_filt[t - 1]

        points_state = moments2points(Moments(mu, sigma2))

        # calculate E[x_t | y_{0:t-1}] & Var(x_t | y_{0:t-1})
        if t == 0:
            points_pred = points_state
            moments_pred = points2moments(points_pred)
        else:
            transition_function = _last_dims(f, t - 1, ndims=1)[0]
            (_, moments_pred) = (
                unscented_filter_predict(
                    transition_function, points_state, sigma2_transition=Q2,
                    params=None
                    )
                )
            points_pred = moments2points(moments_pred)

        # calculate E[x_t | y_{0:t}] & Var(x_t | y_{0:t})
        observation_function = _last_dims(g, t, ndims=1)[0]
        mu_filt[t], sigma2_filt[t] = (
            unscented_filter_correct(
                observation_function, moments_pred, points_pred,
                Y[t], sigma2_observation=R2,
                params=None
                )
            )
            
    return (mu_filt, sigma2_filt)


def _additive_unscented_smoother(mu_filt, sigma2_filt, f, Q):
    '''SQ-UKS w/ additive (zero-mean) transition & obs noises

    Params
    ------
    mu_filt : [T, n_dim_state] array
        mu_filt[t] = mean of t state | [0, t] obs
    sigma2_filt : [T, n_dim_state, n_dim_state] array
        sigma_filt[t] = covariance of t state | [0, t] obs
    f : function or [T-1] array of functions
        state transition function(s)
    Q : [n_dim_state, n_dim_state] array
        transition noise covariance matrix

    Returns
    -------
    mu_smooth : [T, n_dim_state] array
        mu_smooth[t] = mean of t state | [0, T-1] obs
    sigma2_smooth : [T, n_dim_state, n_dim_state] array
        sigma_smooth[t] = covariance of t state | [0, T-1] obs
    '''
    
    # extract size of key parts of problem
    T, n_dim_state = mu_filt.shape

    # instantiate containers for results
    mu_smooth = np.zeros(mu_filt.shape)
    sigma2_smooth = np.zeros(sigma2_filt.shape)
    mu_smooth[-1], sigma2_smooth[-1] = mu_filt[-1], sigma2_filt[-1]
    
    Q2 = linalg.cholesky(Q)

    for t in reversed(range(T - 1)):
        # SP for state
        mu = mu_filt[t]
        sigma2 = sigma2_filt[t]

        moments_state = Moments(mu, sigma2)
        points_state = moments2points(moments_state)

        # calculate E[x_{t+1} | y_{0:t}], Var(x_{t+1} | y_{0:t})
        f_t = _last_dims(f, t, ndims=1)[0]
        (points_pred, moments_pred) = (
            _unscented_transform(points_state, f_t, sigma2_noise=Q2)
        )

        # calculate Cov(x_{t+1}, x_t | y_{0:t-1})
        sigma_pair = (
            (points_pred.points - moments_pred.mean).T
            .dot(np.diag(points_pred.weights_covariance))
            .dot(points_state.points - moments_state.mean).T
        )

        # compute smoothed mean & covariance
        smoother_gain = linalg.lstsq(moments_pred.covariance.T, sigma_pair.T)[0]
        smoother_gain = linalg.lstsq(moments_pred.covariance, smoother_gain)[0]
        smoother_gain = smoother_gain.T

        mu_smooth[t] = (
            mu_filt[t]
            + smoother_gain
              .dot(mu_smooth[t + 1] - moments_pred.mean)
        )
        
        U = cholupdate(moments_pred.covariance, sigma2_smooth[t + 1], -1.0)
        
        sigma2_smooth[t] = (
            cholupdate(sigma2_filt[t], smoother_gain.dot(U.T).T, -1.0)
        )

    return (mu_smooth, sigma2_smooth)

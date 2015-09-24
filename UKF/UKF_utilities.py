'''
UKF utility functions for implementations of multiple methods
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


def points2moments(points, sigma_noise=None):
    '''
    Params
    ------
    points : [2 * n_dim_state + 1, n_dim_state] SP
    sigma_noise : [n_dim_state, n_dim_state] array - for additive case only

    Returns
    -------
    moments : [n_dim_state] Moments
    '''
    
    (points, weights_mu, weights_sigma) = points
    mu = points.T.dot(weights_mu)
    points_diff = points.T - mu[:, np.newaxis]
    sigma = points_diff.dot(np.diag(weights_sigma)).dot(points_diff.T)
    
    # additive noise covariance array 
    if sigma_noise is not None:
        sigma = sigma + sigma_noise
    return Moments(mu.ravel(), sigma)


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
    
    (mu, sigma) = moments
    n_dim = len(mu)
    mu = array2d(mu, dtype=float)

    if alpha is None:
      alpha = 1.0
    if beta is None:
      beta = 0.0
    if kappa is None:
      kappa = 3.0 - n_dim

    # compute sqrt(sigma)
    sigma2 = linalg.cholesky(sigma).T

    # calculate scaling factor for all off-center points
    lamda = (alpha * alpha) * (n_dim + kappa) - n_dim
    gamma = n_dim + lamda

    # calculate the SP, as a stack of columns :
    #   mu
    #   mu + each column of sigma2 * sqrt(gamma)
    #   mu - each column of sigma2 * sqrt(gamma)
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


def unscented_transform(points, f=None, points_noise=None, sigma_noise=None):
    '''
    Params
    ------
    points : [n_points, n_dim_state] SP
    f : transition function
    points_noise : [n_points, n_dim_state] array
        noise or exogeneous input
    sigma_noise : [n_dim_state, n_dim_state] array

    Returns
    -------
    points_pred : [n_points, n_dim_state] SP
        transformed by f, same weights remaining
    moments_pred : [n_dim_state] Moments
        associated to points_pred
    '''
    
    n_points, n_dim_state = points.points.shape
    (points, weights_mean, weights_covariance) = points

    # propagate points through f
    if f is not None:
        if points_noise is None:
            points_pred = [f(points[i]) for i in range(n_points)]
        else:
            points_noise = points_noise.points
            points_pred = [f(points[i], points_noise[i]) for i in range(n_points)]
    else:
        points_pred = points

    # make each row a predicted point
    points_pred = np.vstack(points_pred)
    points_pred = SP(points_pred, weights_mean, weights_covariance)

    # calculate approximate mean & covariance
    moments_pred = points2moments(points_pred, sigma_noise)

    return (points_pred, moments_pred)


def unscented_correct(cross_sigma, moments_pred, obs_moments_pred, y):
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
    
    mu_pred, sigma_pred = moments_pred
    obs_mu_pred, obs_sigma_pred = obs_moments_pred

    n_dim_state = len(mu_pred)
    n_dim_obs = len(obs_mu_pred)

    if not np.any(ma.getmask(y)):
        # calculate Kalman gain
        K = cross_sigma.dot(linalg.pinv(obs_sigma_pred))

        # correct mu, sigma
        mu_filt = mu_pred + K.dot(y - obs_mu_pred)
        sigma_filt = sigma_pred - K.dot(cross_sigma.T)
        
    else:
        # no corrections to be made
        mu_filt = mu_pred
        sigma_filt = sigma_pred
        
    return Moments(mu_filt, sigma_filt)


def augmented_points(momentses):
    '''augmented state representation w/ original state & noise variables concatenated

    Params
    ------
    momentses : list of Moments

    Returns
    -------
    pointses : list of Points
        SP for each element of momentses
    '''
    
    # stack everything together
    means, covariances = zip(*momentses)
    mu_aug = np.concatenate(means)
    sigma_aug = linalg.block_diag(*covariances)
    moments_aug = Moments(mu_aug, sigma_aug)

    # turn augmented representation into SP
    points_aug = moments2points(moments_aug)

    # unstack everything
    dims = [len(m) for m in means]
    result = []
    start = 0
    for i in range(len(dims)):
        end = start + dims[i]
        part = SP(
            points_aug.points[:, start:end],
            points_aug.weights_mean,
            points_aug.weights_covariance
        )
        result.append(part)
        start = end

    # return
    return result


def augmented_unscented_filter_points(mean_state, covariance_state,
                                      covariance_transition,
                                      covariance_observation):
    '''Extraction of various SP from augmented state representation
    Pre-processing step before predicting and updating in augmented UKF.

    Params
    ------
    mean_state : [n_dim_state] array
        mean of t state | [0, t] obs
    covariance_state : [n_dim_state, n_dim_state] array
        covariance of t state | [0, t] obs
    covariance_transition : [n_dim_state, n_dim_state] array
        covariance of zero-mean noise resulting from transitioning from timestep t to t+1
    covariance_observation : [n_dim_obs, n_dim_obs] array
        covariance of zero-mean noise resulting from t+1 obs

    Returns
    -------
    points_state : [2 * n_dim_state + 1, n_dim_state] SP
        SP for t state
    points_transition : [2 * n_dim_state + 1, n_dim_state] SP
        SP for transition noise from t to t+1
    points_observation : [2 * n_dim_state + 1, n_dim_obs] SP
        SP for t obs noise
    '''
    
    # get size of dimensions
    n_dim_state = covariance_state.shape[0]
    n_dim_obs = covariance_observation.shape[0]

    # extract SP using augmented representation
    state_moments = Moments(mean_state, covariance_state)
    transition_noise_moments = (
        Moments(np.zeros(n_dim_state), covariance_transition)
    )
    observation_noise_moments = (
        Moments(np.zeros(n_dim_obs), covariance_observation)
    )

    (points_state, points_transition, points_observation) = (
        augmented_points([
            state_moments,
            transition_noise_moments,
            observation_noise_moments
        ])
    )
    return (points_state, points_transition, points_observation)


def unscented_filter_predict(transition_function, points_state,
                             points_transition=None,
                             sigma_transition=None,
                             params=None):
    '''Prediction of t+1 state distribution
    Using SP for t state | [0, t] obs, calculate predicted SP for t+1 state, associated mean & covariance.

    Params
    ------
    transition_function : function
    points_state : [2*n_dim_state+1, n_dim_state] SP
    points_transition : [2*n_dim_state+1, n_dim_state] SP
        If not, assume that noise is additive.
    sigma_transition : [n_dim_state, n_dim_state] array
        covariance of additive noise in transitioning from t to t+1. If missing, assume noise is not additive.
    params : [n_dim_params] array - ONLY for dual estimation

    Returns
    -------
    points_pred : [2*n_dim_state+1, n_dim_state] SP
        for t+1 state | [0, t] obs - these points have not been "standardized" by UT yet.
    moments_pred : [n_dim_state] Moments
        mean & covariance associated to points_pred
    '''
    
    assert points_transition is not None or sigma_transition is not None, \
        "Your system can't be noiseless"
    (points_pred, moments_pred) = (
        unscented_transform(
            points_state, transition_function,
            points_noise=points_transition, sigma_noise=sigma_transition,
            params=params
        )
    )
    return (points_pred, moments_pred)


def unscented_filter_correct(observation_function, moments_pred,
                             points_pred, observation,
                             points_observation=None,
                             sigma_observation=None,
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
    points_observation : [2*n_dim_state, n_dim_obs] SP
        If not, noise is assumed to be additive.
    sigma_observation : [n_dim_obs, n_dim_obs] array
        covariance of additive noise in t obs. If missing, noise is assumed to be additive.
    params : [n_dim_params] array - ONLY for dual estimation

    Returns
    -------
    moments_filt : [n_dim_state] Moments
        mean & covariance of t state | [0, t] obs
    '''
    
    # calculate E[y_t | y_{0:t-1}] & Var(y_t | y_{0:t-1})
    (obs_points_pred, obs_moments_pred) = (
        unscented_transform(
            points_pred, observation_function,
            points_noise=points_observation, sigma_noise=sigma_observation,
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
    moments_filt = unscented_correct(sigma_pair, moments_pred, obs_moments_pred, observation)
    return moments_filt


def augmented_unscented_filter(mu_0, sigma_0, f, g, Q, R, Y):
    '''UKF w/ arbitrary (zero-mean) transition & obs noises

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
    sigma_filt = np.zeros((T, n_dim_state, n_dim_state))

    for t in range(T):
        if t == 0:
            mu, sigma = mu_0, sigma_0
        else:
            mu, sigma = mu_filt[t - 1], sigma_filt[t - 1]

        # SP for augmented representation
        (points_state, points_transition, points_observation) = (
            augmented_unscented_filter_points(mu, sigma, Q, R)
        )

        # calculate E[x_t | y_{0:t-1}], Var(x_t | y_{0:t-1}) & SP (for P(x_t | y_{0:t-1}))
        if t == 0:
            points_pred = points_state
            moments_pred = points2moments(points_pred)
        else:
            transition_function = _last_dims(f, t - 1, ndims=1)[0]
            (points_pred, moments_pred) = (
                unscented_filter_predict(
                    transition_function, points_state,
                    points_transition=points_transition
                )
            )

        # calculate E[y_t | y_{0:t-1}] & Var(y_t | y_{0:t-1})
        observation_function = _last_dims(g, t, ndims=1)[0]
        mu_filt[t], sigma_filt[t] = (
            unscented_filter_correct(
                observation_function, moments_pred, points_pred,
                Y[t], points_observation=points_observation
            )
        )

    return (mu_filt, sigma_filt)


def augmented_unscented_smoother(mu_filt, sigma_filt, f, Q):
    '''UKS w/ arbitrary (zero-mean) transition & obs noises

    Params
    ------
    mu_filt : [T, n_dim_state] array
        mu_filt[t] = mean of t state | [0, t] obs
    sigma_filt : [T, n_dim_state, n_dim_state] array
        sigma_filt[t] = covariance of t state | [0, t] obs
    f : function or [T-1] array of functions
        state transition function(s)
    Q : [n_dim_state, n_dim_state] array
        transition noise covariance matrix

    Returns
    -------
    mu_smooth : [T, n_dim_state] array
        mu_smooth[t] = mean of t state | [0, T-1] obs
    sigma_smooth : [T, n_dim_state, n_dim_state] array
        sigma_smooth[t] = covariance of t state | [0, T-1] obs
    '''
    
    # extract size of key parts of problem
    T, n_dim_state = mu_filt.shape

    # instantiate containers for results
    mu_smooth = np.zeros(mu_filt.shape)
    sigma_smooth = np.zeros(sigma_filt.shape)
    mu_smooth[-1], sigma_smooth[-1] = mu_filt[-1], sigma_filt[-1]

    for t in reversed(range(T - 1)):
        # SP for state & transition noise
        mu = mu_filt[t]
        sigma = sigma_filt[t]

        moments_state = Moments(mu, sigma)
        moments_transition_noise = Moments(np.zeros(n_dim_state), Q)
        (points_state, points_transition) = (
            augmented_points([moments_state, moments_transition_noise])
        )

        # calculate E[x_{t+1} | y_{0:t}] & Var(x_{t+1} | y_{0:t})
        f_t = _last_dims(f, t, ndims=1)[0]
        (points_pred, moments_pred) = unscented_transform(
            points_state, f_t, points_noise=points_transition
        )

        # calculate Cov(x_{t+1}, x_t | y_{0:t-1})
        sigma_pair = (
            (points_pred.points - moments_pred.mean).T
            .dot(np.diag(points_pred.weights_covariance))
            .dot(points_state.points - moments_state.mean).T
        )

        # compute smoothed mean & covariance
        smoother_gain = sigma_pair.dot(linalg.pinv(moments_pred.covariance))
        mu_smooth[t] = (
            mu_filt[t]
            + smoother_gain
              .dot(mu_smooth[t + 1] - moments_pred.mean)
        )
        sigma_smooth[t] = (
            sigma_filt[t]
            + smoother_gain
              .dot(sigma_smooth[t + 1] - moments_pred.covariance)
              .dot(smoother_gain.T)
        )

    return (mu_smooth, sigma_smooth)


def additive_unscented_filter(mu_0, sigma_0, f, g, Q, R, Y):
    '''UKF w/ additive (zero-mean) transition & obs noises

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
    sigma_filt = np.zeros((T, n_dim_state, n_dim_state))

    for t in range(T):
        # SP for P(x_{t-1} | y_{0:t-1})
        if t == 0:
            mu, sigma = mu_0, sigma_0
        else:
            mu, sigma = mu_filt[t - 1], sigma_filt[t - 1]

        points_state = moments2points(Moments(mu, sigma))

        # calculate E[x_t | y_{0:t-1}] & Var(x_t | y_{0:t-1})
        if t == 0:
            points_pred = points_state
            moments_pred = points2moments(points_pred)
        else:
            transition_function = _last_dims(f, t - 1, ndims=1)[0]
            (_, moments_pred) = (
                unscented_filter_predict(
                    transition_function, points_state, sigma_transition=Q,
                    params=None
                )
            )
            points_pred = moments2points(moments_pred)

        # calculate E[x_t | y_{0:t}] & Var(x_t | y_{0:t})
        observation_function = _last_dims(g, t, ndims=1)[0]
        mu_filt[t], sigma_filt[t] = (
            unscented_filter_correct(
                observation_function, moments_pred, points_pred,
                Y[t], sigma_observation=R,
                params=None
            )
        )

    return (mu_filt, sigma_filt)


def additive_unscented_smoother(mu_filt, sigma_filt, f, Q):
    '''UKS w/ additive (zero-mean) transition & obs noises

    Params
    ------
    mu_filt : [T, n_dim_state] array
        mu_filt[t] = mean of t state | [0, t] obs
    sigma_filt : [T, n_dim_state, n_dim_state] array
        sigma_filt[t] = covariance of t state | [0, t] obs
    f : function or [T-1] array of functions
        state transition function(s)
    Q : [n_dim_state, n_dim_state] array
        transition noise covariance matrix

    Returns
    -------
    mu_smooth : [T, n_dim_state] array
        mu_smooth[t] = mean of t state | [0, T-1] obs
    sigma_smooth : [T, n_dim_state, n_dim_state] array
        sigma_smooth[t] = covariance of t state | [0, T-1] obs
    '''
    
    # extract size of key parts of problem
    T, n_dim_state = mu_filt.shape

    # instantiate containers for results
    mu_smooth = np.zeros(mu_filt.shape)
    sigma_smooth = np.zeros(sigma_filt.shape)
    mu_smooth[-1], sigma_smooth[-1] = mu_filt[-1], sigma_filt[-1]

    for t in reversed(range(T - 1)):
        # SP for state
        mu = mu_filt[t]
        sigma = sigma_filt[t]

        moments_state = Moments(mu, sigma)
        points_state = moments2points(moments_state)

        # calculate E[x_{t+1} | y_{0:t}], Var(x_{t+1} | y_{0:t})
        f_t = _last_dims(f, t, ndims=1)[0]
        (points_pred, moments_pred) = (
            unscented_transform(points_state, f_t, sigma_noise=Q)
        )

        # calculate Cov(x_{t+1}, x_t | y_{0:t-1})
        sigma_pair = (
            (points_pred.points - moments_pred.mean).T
            .dot(np.diag(points_pred.weights_covariance))
            .dot(points_state.points - moments_state.mean).T
        )

        # compute smoothed mean & covariance
        smoother_gain = sigma_pair.dot(linalg.pinv(moments_pred.covariance))
        mu_smooth[t] = (
            mu_filt[t]
            + smoother_gain
              .dot(mu_smooth[t + 1] - moments_pred.mean)
        )
        sigma_smooth[t] = (
            sigma_filt[t]
            + smoother_gain
              .dot(sigma_smooth[t + 1] - moments_pred.covariance)
              .dot(smoother_gain.T)
        )

    return (mu_smooth, sigma_smooth)

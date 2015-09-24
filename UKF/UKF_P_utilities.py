'''
UKF utility functions for parameter estimation implementations
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


# covariance matrices for transition & observation noises
Noises = namedtuple('Noises', ['transition', 'obs'])


def noise_ff(lambda, sigma):
    '''Approx. exponential decaying weighting on past data
    
    Params
    ------
    lambda : float
        forgetting factor
    sigma : [n_dim_state, n_dim_state] array
        t state or obs covariance matrix
    
    Returns
    -------
    noise_covariance : [n_dim_state, n_dim_state] array
        t noise covariance matrix
    '''
    
    assert lambda <= 1 and lambda > 0, \
        "No authorized bounds for forgetting factor"
    
    noise_covariance = (1-1./lambda)*sigma
    
    return noise_covariance


def noise_rm(alpha, prev_noise_cov, corrected_state_mean, estimated_state_mean):
    '''Robbins-Monro stochastic approx. scheme for estimating innovations
    
    Params
    ------
    alpha : float
        Robbins-Monro parameter
    prev_noise_cov : [n_dim_state, n_dim_state] array
        t-1 transition or obs noise covariance matrix
    corrected_state_mean : [n_dim_state] array
        t corrected mean vector
    estimated_state_mean : [n_dim_state] array
        t estimated mean vector
    
    Returns
    -------
    noise_cov : [n_dim_state, n_dim_state] array
        t noise covariance matrix
    '''
    
    mean_diff = corrected_state_mean - estimated_state_mean
    noise_cov = alpha*mean_diff.dot(mean_diff.T)
    noise_cov += (1-alpha)*prev_noise_cov
     
    return noise_cov
    
    
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


def unscented_transform_P(moments, sigma_noise=None):
    '''
    Params
    ------
    moments : [n_dim_state] Moments
    sigma_noise : [n_dim_state, n_dim_state] array

    Returns
    -------
    points_pred : [2*n_points+1, n_dim_state] points
    moments_pred : [n_dim_state] Moments
        associated to points_pred
    '''
    
    n_points, n_dim_state = moments.mean.shape
    (weights_mean, weights_covariance) = moments
    
    mu = weights_mean
    sigma = weights_covariance + sigma_noise

    moments_pred = Moments(mu, sigma)
    points_pred = moments2points(moments_pred)
    
    # make each row a predicted point
    points_pred = np.vstack(points_pred)

    return (points_pred, moments_pred)


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
            points_pred = [f(points[i], points_noise[i]) for i in range(n_points)]
    else:
        points_pred = points

    # make each row a predicted point
    points_pred = np.vstack(points_pred)
    points_pred = SP(points_pred, weights_mean, weights_covariance)

    # calculate approximate mean & covariance
    moments_pred = points2moments(points_pred, sigma_noise=sigma_noise)

    return (points_pred, moments_pred)


def unscented_correct(cross_sigma, moments_pred, obs_moments_pred, y,
                      noise_process, noise_parameter, noises):
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
    noise_process : ff or rm
        transition & obs noises stochastic process
    noise_parameter : float
        forgetting factor or Robbins-Monro parameter
    noises : [n_dim_state] Noises
        t-1 transition & obs noises covariance matrices

    Returns
    -------
    moments_filt : [n_dim_state] Moments
        mean & covariance of t state | [0, t] obs
    updated_noises : [n_dim_state] Noises
        updated t transition & obs noises covariance matrices
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

    # update state noise covariance matrix
    if noise_process == ff:
        sigma_trans_noise = noise_ff(noise_parameter, sigma_filt)
        sigma_obs_noise = noise_ff(noise_parameter, obs_sigma_pred)
        
    elif noise_process == rm:
        assert corrected_mean is not None
        assert estimated_mean is not None
        sigma_trans_noise = noise_rm(noise_parameter, noises.transition, mu_filt, mu_pred)
        sigma_obs_noise = noise_rm(noise_parameter, noises.obs, mu_filt, mu_pred)
        
    else:
        assert noise_process == ff or noise_process == rm, \
            "Following noise process is unknown: " % noise_process
    
    return (Moments(mu_filt, sigma_filt), Noises(sigma_trans_noise, sigma_obs_noise))
    
    
def unscented_filter_predict_P(moments_state, noises=None):
    '''Prediction of t+1 state distribution
    Using SP for t state | [0, t] obs, calculate predicted SP for t+1 state, associated mean & covariance.

    Params
    ------
    moments_state : [n_dim_state] Moments
    noises : [n_dim_state] Noises
        noises.transition : covariance of additive noise in transitioning from t to t+1.
        If missing, assume noise is not additive.

    Returns
    -------
    points_pred : [2*n_dim_state+1, n_dim_state] SP
        for t+1 state | [0, t] obs - these points have not been "standardized" by UT yet.
    moments_pred : [n_dim_state] Moments
        mean & covariance associated to points_pred
    '''
    
    assert noises.transition is not None, \
        "Your system can't be noiseless and params noise needs being additive"
        
    (points_pred, moments_pred) = (
        unscented_transform_P(moments_state, noises.transition)
    )
    return (points_pred, moments_pred)


def unscented_filter_correct(observation_function, moments_pred,
                             points_pred, observation,
                             points_observation=None,
                             noise_process, noise_parameter,
                             noises=None):
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
    noise_process : ff or rm
        transition & obs noises stochastic process
    noise_parameter : float
        forgetting factor or Robbins-Monro parameter
    noises : [n_dim_state] Noises
        t-1 transition & obs noises covariance matrices

    Returns
    -------
    moments_filt : [n_dim_state] Moments
        mean & covariance of t state | [0, t] obs
    updated_noises : [n_dim_state] Noises
        updated t transition & obs noises covariance matrices
    '''
    
    assert noises.obs is not None, \
        "Your system can't be noiseless and observation noise needs being additive"
    
    # calculate E[y_t | y_{0:t-1}] & Var(y_t | y_{0:t-1})
    (obs_points_pred, obs_moments_pred) = (
        unscented_transform(
            points_pred, observation_function,
            points_noise=points_observation, sigma_noise=noises.obs
        )
    )

    # calculate Cov(x_t, y_t | y_{0:t-1})
    sigma_pair = (
        ((points_pred.points - moments_pred.mean).T)
        .dot(np.diag(points_pred.weights_mean))
        .dot(obs_points_pred.points - obs_moments_pred.mean)
    )

    # calculate E[x_t | y_{0:t}], Var(x_t | y_{0:t}) & update transition & obs noises
    (moments_filt, updated_noises) = (
        unscented_correct(
            sigma_pair, moments_pred, obs_moments_pred, observation,
            noise_process, noise_parameter, noises
        )
    )
    
    return (moments_filt, updated_noises)


def additive_unscented_filter_P(mu_0, sigma_0, g, X, Y, noise_process, noise_parameter, n_dim_obs):
    '''UKF w/ additive (zero-mean) stochastic transition & obs noises

    Params
    ------
    mu_0 : [n_dim_state] array
        mean of initial state distribution
    sigma_0 : [n_dim_state, n_dim_state] array
        covariance of initial state distribution
    g : function or [T] array of functions
        observation function(s)
    X : [T] array
        [0,T-1] obs
    Y : [T] array
        [0,T-1] obs
    noise_process : ff or rm
        transition & obs noises stochastic process
    noise_parameter : float
        forgetting factor or Robbins-Monro parameter
    n_dim_obs : int

    Returns
    -------
    mu_filt : [T, n_dim_state] array
        mu_filt[t] = mean of t state | [0, t] obs
    sigma_filt : [T, n_dim_state, n_dim_state] array
        sigma_filt[t] = covariance of t state | [0, t] obs
    '''
    
    # extract size of key components
    assert X.shape[0] == Y.shape[0], \
        "both observation arrays need to have same first dim"
    
    T = Y.shape[0]
    n_dim_state = mu_0.shape[0]

    # construct container for results
    mu_filt = np.zeros((T, n_dim_state))
    sigma_filt = np.zeros((T, n_dim_state, n_dim_state))

    for t in range(T):
        # SP for P(x_{t-1} | y_{0:t-1})
        if t == 0:            
            moments_state = Moments(mu_0, sigma_0)
            
            if noise_process == ff:
                buffer_sigma_trans_noise = noise_ff(noise_parameter, moments_state.cov)
                buffer_sigma_obs_noise = noise_ff(noise_parameter, np.eye(n_dim_obs))
                
            elif noise_process == rm:
                buffer_sigma_trans_noise = (
                    noise_rm(
                        noise_parameter, np.eye(n_dim_state),
                        np.zeros(n_dim_state), np.zeros(n_dim_state)
                    )
                )
                buffer_sigma_obs_noise = (
                    noise_rm(
                        noise_parameter, np.eye(n_dim_state),
                        np.zeros(n_dim_state), np.zeros(n_dim_state)
                    )
                )
              
            else:
                assert noise_process == ff or noise_process == rm, \
                    "Following noise process is unknown: " % noise_process
            
            buffer_noises = Noises(buffer_sigma_trans_noise, buffer_sigma_obs_noise)
            
        else:
            moments_state = Moments(mu_filt[t - 1], sigma_filt[t - 1])
        
        (points_pred, moments_pred) = (
            unscented_filter_predict_P(
                moments_state,
                noises=buffer_noises)
            )
        )

        # calculate E[x_t | y_{0:t}] & Var(x_t | y_{0:t})
        observation_function = _last_dims(g, t, ndims=1)[0]
        (moments_filt, buffer_noises) = (
            unscented_filter_correct(
                observation_function, moments_pred, points_pred,
                Y[t], points_observation=X[t], noise_process, noise_parameter,
                noises=buffer_noises
            )
        )
            
        mu_filt[t], sigma_filt[t] = moments_filt.mean, moments_filt.cov

    return (mu_filt, sigma_filt)

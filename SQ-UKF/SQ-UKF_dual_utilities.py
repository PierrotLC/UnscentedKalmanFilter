'''
SQ-UKF utility functions for dual estimation implementation
'''

import numpy as np
from scipy import linalg

from ..KF_utils import _last_dims

from SQ-UKF_utilities import unscented_filter_predict as UF_state_predict, \
    unscented_filter_correct as UF_state_correct, \
    Moments, moments2points, points2moments
from SQ-UKF_P_utilities import unscented_filter_predict_P as UF_params_predict, \
    unscented_filter_correct as UF_params_correct
    

def additive_unscented_filter_dual(w_mu_0, w_sigma_0, x_mu_0, x_sigma_0,
                                   f, g, Q, R, Y, noise_process, noise_parameter):
    '''SQ-UKF w/ additive (zero-mean) parameters, transition & obs noises

    Params
    ------
    w_mu_0 : [n_dim_state] array
        mean of initial paramaters distribution
    w_sigma_0 : [n_dim_state, n_dim_state] array
        covariance of initial parameters distribution
    x_mu_0 : [n_dim_state] array
        mean of initial state distribution
    x_sigma_0 : [n_dim_state, n_dim_state] array
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
    noise_process : ff or rm
        transition & obs noises stochastic process
    noise_parameter : float
        forgetting factor or Robbins-Monro parameter

    Returns
    -------
    x_mu_filt : [T, n_dim_state] array
        mu_filt[t] = mean of t state | [0, t] obs
    x_sigma_filt : [T, n_dim_state, n_dim_state] array
        sigma_filt[t] = covariance of t state | [0, t] obs
    w_mu_filt : [T, n_dim_state] array
        mu_filt[t] = mean of t parameters | [0, t] obs
    w_sigma_filt : [T, n_dim_state, n_dim_state] array
        sigma_filt[t] = covariance of t parameters | [0, t] obs
    '''
    
    T = Y.shape[0]
    n_dim_state = x_mu_0.shape[0]
    n_dim_param = w_mu_0.shape[0]
    n_dim_obs = R.shape[-1]

    # construct container for results
    x_mu_filt = np.zeros((T, n_dim_state))
    x_sigma2_filt = np.zeros((T, n_dim_state, n_dim_state))
    w_mu_filt = np.zeros((T, n_dim_param))
    w_sigma2_filt = np.zeros((T, n_dim_param, n_dim_param))
    
    Q2 = linalg.cholesky(Q)
    R2 = linalg.cholesky(R)

    for t in range(T):
        # SP for P(w_{t-1} | y_{0:t-1}) & P(x_{t-1} | y_{0:t-1})
        if t == 0:
            w_mu, w_sigma2 = w_mu_0, linalg.cholesky(w_sigma_0)
                
            assert noise_process == rm, \
                "parameters noise must be processed by Robbins-Monro during filtering"
            buffer_noise = (
                noise_rm(
                    noise_parameter, np.eye(n_dim_param),
                    np.zeros(n_dim_param), np.zeros(n_dim_param)
                )
            )
                
            x_mu, x_sigma2 = x_mu_0, linalg.cholesky(x_sigma_0)
            
        else:
            w_mu, w_sigma2 = w_mu_filt[t - 1], w_sigma2_filt[t - 1]
            x_mu, x_sigma2 = x_mu_filt[t - 1], x_sigma2_filt[t - 1]
        
        '''t-1 state being fixed, estimate t parameters'''
        w_moments_state = Moments(w_mu, w_sigma2)

        noise = diag_noise_rm(buffer_noise, moments_state.cov)
        (w_points_pred, w_moments_pred) = (
            UF_params_predict(
                w_moments_state, sigma2_transition=noise,
                noise_process, noise_parameter)
            )
        )

        # calculate E[w_t | y_{0:t}] & Var(w_t | y_{0:t})
        observation_function = _last_dims(g, t, ndims=1)[0]
        (w_moments_filt, buffer_noise) = (
            UF_params_correct(
                observation_function, w_moments_pred, w_points_pred,
                Y[t], points_observation=x_mu, noise_process, noise_parameter,
                sigma2_observation=R2
            )
        )
            
        w_mu_filt[t], w_sigma2_filt[t] = w_moments_filt
        
        '''once t parameters estimated, estimate t state'''
        points_state = moments2points(Moments(x_mu, x_sigma2))
        
        # calculate E[x_t | y_{0:t-1}] & Var(x_t | y_{0:t-1})
        if t == 0:
            x_points_pred = points_state
            x_moments_pred = points2moments(x_points_pred)
        else:
            transition_function = _last_dims(f, t - 1, ndims=1)[0]
            (_, x_moments_pred) = (
                UF_state_predict(
                    transition_function, points_state, sigma2_transition=Q2,
                    params=w_mu_filt[t]
                    )
                )
            x_points_pred = moments2points(x_moments_pred)

        # calculate E[x_t | y_{0:t}] & Var(x_t | y_{0:t})
        observation_function = _last_dims(g, t, ndims=1)[0]
        x_mu_filt[t], x_sigma2_filt[t] = (
            UF_state_correct(
                observation_function, x_moments_pred, x_points_pred,
                Y[t], sigma2_observation=R2, params=w_mu_filt[t]
                )
            )
        
        return (x_mu_filt, x_sigma2_filt, w_mu_filt, w_sigma2_filt)

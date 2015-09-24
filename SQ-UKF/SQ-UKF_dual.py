'''
SQ-UKF implementations for Dual Estimation

w/ additive parameters, state & observation noises
'''

import numpy as np

from ..SKLearn_utils import array1d, array2d, get_params, preprocess_arguments, check_random_state
from ..KF_utils import _determine_dimensionality, _last_dims
from .SQ_UKF_dual_utilities import additive_unscented_filter_dual


class UnscentedMixin(object):
    '''Methods for SQ-UKF implementations'''
    
    def __init__(self, transition_functions=None, observation_functions=None,
            transition_covariance=None, observation_covariance=None,
            initial_state_mean=None, initial_state_covariance=None,
            initial_params_mean=None, initial_params_covariance=None,
            n_dim_state=None, n_dim_obs=None, n_dim_params=None, random_state=None):

        # size of state & obs spaces
        n_dim_state = _determine_dimensionality(
            [(transition_covariance, array2d, -2),
             (initial_state_covariance, array2d, -2),
             (initial_state_mean, array1d, -1)],
            n_dim_state
        )
        n_dim_obs = _determine_dimensionality(
            [(observation_covariance, array2d, -2)],
            n_dim_obs
        )
        n_dim_params = _determine_dimensionality(
            [(initial_params_covariance, array2d, -2),
             (initial_params_mean, array1d, -1)],
            n_dim_state
        )

        # set params
        self.transition_functions = transition_functions
        self.observation_functions = observation_functions
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs
        self.n_dim_params = n_dim_params
        self.random_state = random_state


    def _initialize_parameters(self):
        '''Retrieve params if they exist, else replace w/ defaults'''

        arguments = get_params(self)
        defaults = self._default_parameters()
        converters = self._converters()

        processed = preprocess_arguments([arguments, defaults], converters)
        return (
            processed['transition_functions'],
            processed['observation_functions'],
            processed['transition_covariance'],
            processed['observation_covariance'],
            processed['initial_state_mean'],
            processed['initial_state_covariance']
        )


    def _parse_observations(self, obs):
        '''Convert obs to their expected format'''
        
        obs = ma.atleast_2d(obs)
        if obs.shape[0] == 1 and obs.shape[1] > 1:
            obs = obs.T
        return obs


    def _converters(self):
        return {
            'transition_functions': array1d,
            'observation_functions': array1d,
            'transition_covariance': array2d,
            'observation_covariance': array2d,
            'initial_state_mean': array1d,
            'initial_state_covariance': array2d,
            'initial_params_mean': array1d,
            'initial_params_covariance': array2d,
            'n_dim_state': int,
            'n_dim_obs': int,
            'n_dim_params': int,
            'random_state': check_random_state,
        }
    
    
    
    class AdditiveUnscentedKalmanFilter_dual(UnscentedMixin):
    '''Implements SQ-UKF w/ additive noise

        x_0       w/   (x_mu_0, x_sigma_0)
        w_0       w/   (w_mu_0, w_sigma_0)
        w_{t+1}   =    w_t + (0, P)
        x_{t+1}   =    f_t(w_t, x_t) + (0, Q)
        y_{t}     =    g_t(w_t, x_t) + (0, R)

    where :
        w : parameters to update, viewed as state update w/ stochastic P noise process
        x : non-observed state to estimate
        y : observation

    Complexity O(2*Tn^3) w/ :
        T number of timesteps - or n_timesteps,
        n state space size.

    Params
    ------
    transition_functions : function or [T-1] array of functions
        transition_functions[t] as f_t
    observation_functions : function or [T] array of functions
        observation_functions[t] as g_t
    transition_covariance : [n_dim_state, n_dim_state] array
        transition noise covariance matrix as Q
    observation_covariance : [n_dim_obs, n_dim_obs] array
        observation noise covariance matrix as R
    initial_state_mean : [n_dim_state] array
        mean of initial state distribution as x_mu_0
    initial_state_covariance : [n_dim_state, n_dim_state] array
        covariance of initial state distribution as x_sigma_0
    initial_params_mean : [n_dim_state] array
        mean of initial state distribution as w_mu_0
    initial_params_covariance : [n_dim_state, n_dim_state] array
        covariance of initial state distribution as w_sigma_0
    n_dim_state: optional integer
        state space dimensionality - necessary when you do not specify initial values
        for transition_covariance, initial_state_mean or initial_state_covariance.
    n_dim_obs: optional integer
        obs space dimensionality - necessary when you do not specify initial values
        for observation_covariance.
    n_dim_params : optional integer
        params space dimensionality - necessary when you do not specify initial values
        for initial_params_mean or initial_params_covariance.
    random_state : optional int or RandomState
        seed for random sample generation
    '''
    
    
     def filter(self, Y, noise_parameter):
        '''Run SQ-UKF
        Robbins-Monro process for parameters noises should be preferred.
        
        Params
        ------
        Y : [n_timesteps, n_dim_state] array
            Y[t] = t obs.
            If Y is a masked array and any Y[t] is masked, obs is assumed missing and ignored.
        noise_parameter : float
            Robbins-Monro parameter for processing noises

        Returns
        -------
        filtered_state_means : [n_timesteps, n_dim_state] array
            filtered_state_means[t] = mean of t state distribution | [0, t] obs
        filtered_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
            filtered_state_covariances[t] = covariance of t state distribution | [0, t] obs
        '''
        
        Y = self._parse_observations(Y)
        
        (transition_functions, observation_functions,
         transition_covariance, observation_covariance,
         initial_state_mean, initial_state_covariance,
         initial_params_mean, initial_params_covariance) = (
            self._initialize_parameters()
        )

        n_timesteps = Y.shape[0]

        # run square root filter
        (filtered_state_means, x_sigma2_filt, filtered_params_means, w_sigma2_filt) = (
            additive_unscented_filter_dual( initial_params_mean, initial_params_covariance,
                                           initial_state_mean, initial_state_covariance,
                                           transition_functions, observation_functions,
                                           transition_covariance, observation_covariance,
                                           Y, rm, noise_parameter)
        )
        
        # reconstruct covariance matrices
        filtered_state_covariances, filtered_params_covariances = (
            np.zeros(x_sigma2_filt.shape), np.zeros(w_sigma2_filt.shape)
        )
        
        for t in range(n_timesteps):
            filtered_state_covariances[t] = x_sigma2_filt[t].T.dot(x_sigma2_filt[t])
            filtered_params_covariances[t] = w_sigma2_filt[t].T.dot(w_sigma2_filt[t])

        return (filtered_state_means, filtered_state_covariances, filtered_params_means, filtered_params_covariances)
        
        
    def _default_parameters(self):
        return {
            'transition_functions': lambda state: state,
            'observation_functions': lambda state: state,
            'transition_covariance': np.eye(self.n_dim_state),
            'observation_covariance': np.eye(self.n_dim_obs),
            'initial_state_mean': np.zeros(self.n_dim_state),
            'initial_state_covariance': np.eye(self.n_dim_state),
            'initial_params_mean': np.zeros(self.n_dim_params),
            'initial_params_covariance': np.eye(self.n_dim_params),
            'random_state': 0,
        }
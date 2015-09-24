'''
SQ-UKF implementations w/ additive noise

Methods : filtering, tracking & smoothing
'''

import numpy as np
from numpy import ma

from ..SKLearn_utils import array1d, array2d, get_params, preprocess_arguments, check_random_state
from ..KF_utils import _determine_dimensionality, _last_dims, _arg_or_default
from .SQ_UKF_utilities import _reconstruct_covariances, Moments, moments2points, unscented_filter_predict, unscented_filter_correct, \
    _additive_unscented_filter, _additive_unscented_smoother


class UnscentedMixin(object):
    '''Methods for SQ-UKF implementations'''
    
    def __init__(self, transition_functions=None, observation_functions=None,
            transition_covariance=None, observation_covariance=None,
            initial_state_mean=None, initial_state_covariance=None,
            n_dim_state=None, n_dim_obs=None, random_state=None):

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

        # set params
        self.transition_functions = transition_functions
        self.observation_functions = observation_functions
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs
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
            'n_dim_state': int,
            'n_dim_obs': int,
            'random_state': check_random_state,
        }



class AdditiveUnscentedKalmanFilter(UnscentedMixin):
    '''Implements SQ-UKF w/ additive noise

        x_0       w/   (mu_0, sigma_0)
        x_{t+1}   =    f_t(x_t) + (0, Q)
        y_{t}     =    g_t(x_t) + (0, R)


    Less general than the general-noise UKF, the Additive version is more computationally
    efficient with complexity O(Tn^3) w/ :
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
        mean of initial state distribution as mu_0
    initial_state_covariance : [n_dim_state, n_dim_state] array
        covariance of initial state distribution as sigma_0
    n_dim_state: optional integer
        state space dimensionality - necessary when you do not specify initial values
        for transition_covariance, initial_state_mean or initial_state_covariance.
    n_dim_obs: optional integer
        obs space dimensionality - necessary when you do not specify initial values
        for observation_covariance.
    random_state : optional int or RandomState
        seed for random sample generation
    '''
  
  
    def filter(self, Y):
        '''Run SQ-UKF
        
        Params
        ------
        Y : [n_timesteps, n_dim_state] array
            Y[t] = t obs.
            If Y is a masked array and any Y[t] is masked, obs is assumed missing and ignored.

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
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        n_timesteps = Y.shape[0]

        # run square root filter
        (filtered_state_means, sigma2_filt) = (
            _additive_unscented_filter(
                initial_state_mean, initial_state_covariance,
                transition_functions, observation_functions,
                transition_covariance, observation_covariance,
                Y
            )
        )
        
        # reconstruct covariance matrices
        filtered_state_covariances = np.zeros(sigma2_filt.shape)
        for t in range(n_timesteps):
            filtered_state_covariances[t] = sigma2_filt[t].T.dot(sigma2_filt[t])

        return (filtered_state_means, filtered_state_covariances)


    def filter_update(self,
                      filtered_state_mean, filtered_state_covariance,
                      observation=None,
                      transition_function=None, transition_covariance=None,
                      observation_function=None, observation_covariance=None):
        '''Update of a SQ-KF state estimation

        One-step update to estimate t+1 state | t+1 obs with t state estimation | [0...t] obs.
        To track an object with streaming observations.

        Params
        ------
        filtered_state_mean : [n_dim_state] array
            mean for estimated t state | [1...t] obs
        filtered_state_covariance : [n_dim_state, n_dim_state] array
            covariance for estimated t state | [1...t] obs
        observation : [n_dim_obs] array or None
            t+1 obs.
            If observation is a masked array and any of observation's components are masked
            or if observation is None, then observation is treated as a missing observation.
        transition_function : optional function
            state transition function from t to t+1. If unspecified, self.transition_functions is used.
        transition_covariance : optional [n_dim_state, n_dim_state] array
            state transition noise covariance from t to t+1. If unspecified, self.transition_covariance is used.
        observation_function : optional, function
            t+1 obs function.  If unspecified, self.observation_functions is used.
        observation_covariance : optional [n_dim_obs, n_dim_obs] array
            t+1 obs noise covariance. If unspecified, self.observation_covariance is used.

        Returns
        -------
        next_filtered_state_mean : [n_dim_state] array
            mean estimatation for t+1 state | [1...t+1] obs
        next_filtered_state_covariance : [n_dim_state, n_dim_state] array
            covariance estimation for t+1 state | [1...t+1] obs
        '''
        
        # initialize params
        (transition_functions, observation_functions,
         transition_covariance, observation_covariance,
         _, _) = (
            self._initialize_parameters()
        )

        def default_function(f, arr):
            if f is None:
                assert len(arr) == 1
                f = arr[0]
            return f

        transition_function = default_function(
            transition_function, transition_functions
        )
        observation_function = default_function(
            observation_function, observation_functions
        )
        transition_covariance = _arg_or_default(
            transition_covariance, transition_cov,
            2, "transition_covariance"
        )
        observation_covariance = _arg_or_default(
            observation_covariance, observation_cov,
            2, "observation_covariance"
        )

        # make a masked observation if necessary
        if observation is None:
            n_dim_obs = observation_covariance.shape[0]
            observation = np.ma.array(np.zeros(n_dim_obs))
            observation.mask = True
        else:
            observation = np.ma.asarray(observation)

        # preprocess covariance matrices
        filtered_state_covariance2 = linalg.cholesky(filtered_state_covariance)
        transition_covariance2 = linalg.cholesky(transition_covariance)
        observation_covariance2 = linalg.cholesky(observation_covariance)

        # make sigma points
        moments_state = Moments(filtered_state_mean, filtered_state_covariance2)
        points_state = moments2points(moments_state)

        # predict
        (_, moments_pred) = (
            unscented_filter_predict(
                transition_function, points_state,
                sigma2_transition=transition_covariance2
            )
        )
        points_pred = moments2points(moments_pred)

        # correct
        (next_filtered_state_mean, next_filtered_state_covariance2) = (
            unscented_filter_correct(
                observation_function, moments_pred, points_pred,
                observation, sigma2_observation=observation_covariance2
            )
        )

        next_filtered_state_covariance = (
            _reconstruct_covariances(next_filtered_state_covariance2)
        )

        return (next_filtered_state_mean, next_filtered_state_covariance)


    def sample(self, n_timesteps, random_state=None, observations=None):
        '''Sampling from SQ-UKF model

        Params
        ------
        n_timesteps : int
        random_state : optional int or Random
            random number generator as rng
        observations : [n_dim_obs, n_timesteps] array
        '''
        
        (transition_functions, observation_functions,
         transition_covariance, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        n_dim_state = transition_covariance.shape[-1]
        n_dim_obs = observation_covariance.shape[-1]

        # instantiating rng
        if random_state is None:
            rng = check_random_state(self.random_state)
        else:
            rng = check_random_state(random_state)

        # sampling from multivariate normal w/ SQ-UKF updates
        x = np.zeros((n_timesteps, n_dim_state))
        
        for t in range(n_timesteps):  
            if t == 0:
                x[0] = rng.multivariate_normal(
                    initial_state_mean, initial_state_covariance
                )
                filtered_state_mean = initial_state_mean
                filtered_state_covariance = initial_state_covariance
            
            else:
                transition_function = (
                    _last_dims(transition_functions, t - 1, ndims=1)[0]
                )
                observation_function = (
                    _last_dims(observation_functions, t, ndims=1)[0]
                )
                observation = (
                    _last_dims(observations, t, ndims=1)[0]
                )
                
                '''update prior t distribution w/ UKF'''
                (filtered_state_mean, filtered_state_covariance) = (
                    AdditiveUnscentedKalmanFilter.filter_update(
                        self, filtered_state_mean, filtered_state_covariance,
                        observation=observation,
                        transition_function=transition_function,
                        transition_covariance=transition_covariance,
                        observation_function=observation_function,
                        observation_covariance=observation_covariance
                    )
                )
              
                x[t] = rng.multivariate_normal(
                    filtered_state_mean, filtered_state_covariance
                )

        return (x)
    

    def smooth(self, Y):
        '''Run SQ-UKS

        Params
        ------
        Y : [T, n_dim_state] array
            Y[t] =  t obs
            If Y is a masked array and any Y[t] is masked, obs is assumed missing and ignored.

        Returns
        -------
        smoothed_state_means : [T, n_dim_state] array
            filtered_state_means[t] = mean of t state distribution | [0, T-1] obs
        smoothed_state_covariances : [T, n_dim_state, n_dim_state] array
            filtered_state_covariances[t] = covariance of t state distribution | [0, T-1] obs
        '''
        
        Y = self._parse_observations(Y)

        (transition_functions, observation_functions,
         transition_covariance, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        n_timesteps = Y.shape[0]

        # run filter, then smooth
        (filtered_state_means, sigma2_filt) = (
            _additive_unscented_filter(
                initial_state_mean, initial_state_covariance,
                transition_functions, observation_functions,
                transition_covariance, observation_covariance,
                Y
            )
        )
        (smoothed_state_means, sigma2_smooth) = (
            _additive_unscented_smoother(
                filtered_state_means, sigma2_filt,
                transition_functions, transition_covariance
            )
        )

        # reconstruction covariance matrices
        smoothed_state_covariances = np.zeros(sigma2_smooth.shape)
        for t in range(n_timesteps):
            smoothed_state_covariances[t] = (
                sigma2_smooth[t].T.dot(sigma2_smooth[t])
            )

        return (smoothed_state_means, smoothed_state_covariances)
    
    
    def _default_parameters(self):
        return {
            'transition_functions': lambda state: state,
            'observation_functions': lambda state: state,
            'transition_covariance': np.eye(self.n_dim_state),
            'observation_covariance': np.eye(self.n_dim_obs),
            'initial_state_mean': np.zeros(self.n_dim_state),
            'initial_state_covariance': np.eye(self.n_dim_state),
            'random_state': 0,
        }
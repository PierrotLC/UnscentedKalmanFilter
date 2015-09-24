'''
SQ-UKF implementations for Parameter Estimation

w/ additive parameters & observation noises
'''

import numpy as np
from numpy import ma

from ..SKLearn_utils import array1d, array2d, get_params, preprocess_arguments, check_random_state
from ..KF_utils import _determine_dimensionality, _last_dims, _arg_or_default
from .SQ_UKF_P_utilities import noise_ff, _reconstruct_covariances, Moments, moments2points, \
    unscented_filter_predict_P, unscented_filter_correct, _additive_unscented_filter_P


class UnscentedMixin(object):
    '''Methods for SQ-UKF_P implementations'''
    
    def __init__(self, observation_functions=None, observation_covariance=None,
            initial_state_mean=None, initial_state_covariance=None,
            n_dim_state=None, n_dim_obs=None, random_state=None):

        # size of state & obs spaces
        n_dim_state = _determine_dimensionality(
            [(initial_state_covariance, array2d, -2),
             (initial_state_mean, array1d, -1)],
            n_dim_state
        )
        n_dim_obs = _determine_dimensionality(
            [(observation_covariance, array2d, -2)],
            n_dim_obs
        )

        # set params
        self.observation_functions = observation_functions
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
            processed['observation_functions'],
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
            'observation_functions': array1d,
            'observation_covariance': array2d,
            'initial_state_mean': array1d,
            'initial_state_covariance': array2d,
            'n_dim_state': int,
            'n_dim_obs': int,
            'random_state': check_random_state,
        }



class AdditiveUnscentedKalmanFilter_P(UnscentedMixin):
    '''Implements SQ-UKF_P w/ additive noise

        w_0       w/   (mu_0, sigma_0)
        w_{t+1}   =    w_t + (0, Q)
        y_{t}     =    g_t(x_t, w_t) + (0, R)

    where :
        w : parameters to update, viewed as state update
        x : observation as noise point or exogeneous input
        y : observation

    The UKF Additive version is imposed by parameters estimation conditions.
    Complexity of O(Tn^3) w/ :
        T number of timesteps - or n_timesteps,
        n state space size.
        
    Parameters noise Q is processed either by forgetting factor (RLS-like) or Robbins-Monro process.
    Observations noise R is fixed.

    Params
    ------
    observation_functions : function or [T] array of functions
        observation_functions[t] as g_t
    observation_covariance : [n_dim_obs, n_dim_obs] array
        observation noise covariance matrix as R
    initial_state_mean : [n_dim_state] array
        mean of initial state distribution as mu_0
    initial_state_covariance : [n_dim_state, n_dim_state] array
        covariance of initial state distribution as sigma_0
    n_dim_state: optional integer
        state space dimensionality - necessary when you do not specify initial values
        for initial_state_mean or initial_state_covariance.
    n_dim_obs: optional integer
        obs space dimensionality - necessary when you do not specify initial values
        for observation_covariance.
    random_state : optional int or RandomState
        seed for random sample generation
    '''
  
  
    def filter(self, X, Y, noise_parameter):
        '''Run SQ-UKF
        Robbins-Monro process for parameters noises should be preferred.
        
        Params
        ------
        Y : [n_timesteps, n_dim_state] array
            Y[t] = t obs.
            If Y is a masked array and any Y[t] is masked, obs is assumed missing and ignored.
        X : [n_timesteps, n_dim_state] array
            X[t] = t obs as noise point or exogeneous input
            If X is a masked array and any X[t] is masked, obs is assumed missing and ignored.
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
        X = self._parse_observations(X)

        (observation_functions, observation_covariance,
         initial_state_mean, initial_state_covariance,
         _, .n_dim_obs) = (
            self._initialize_parameters()
        )

        n_timesteps = Y.shape[0]

        # run square root filter
        (filtered_state_means, sigma2_filt) = (
            _additive_unscented_filter_P(
                initial_state_mean, initial_state_covariance,
                observation_functions, observation_covariance,
                X, Y, rm, noise_parameter
            )
        )
        
        # reconstruct covariance matrices
        filtered_state_covariances = np.zeros(sigma2_filt.shape)
        for t in range(n_timesteps):
            filtered_state_covariances[t] = sigma2_filt[t].T.dot(sigma2_filt[t])

        return (filtered_state_means, filtered_state_covariances)


    def filter_update_P(self,
                      filtered_state_mean, filtered_state_covariance,
                      observation_X=None, observation_Y=None,
                      observation_function=None, observation_covariance=None,
                      noise_parameter):
        '''Update of a parameter estimation
        Forgetting factor process for parameters noise should be preferred.

        One-step update to estimate t+1 state | t+1 obs with t state estimation | [0...t] obs.
        To track an object with streaming observations.

        Params
        ------
        filtered_state_mean : [n_dim_state] array
            mean for estimated t state | [1...t] obs
        filtered_state_covariance : [n_dim_state, n_dim_state] array
            covariance for estimated t state | [1...t] obs
        observation_X : [n_dim_obs] array or None
            t+1 obs as noise point or exogeneous input.
            If observation is a masked array and any of observation's components are masked
            or if observation is None, then observation is treated as a missing observation.
        observation_Y : [n_dim_obs] array or None
            t+1 obs.
            If observation is a masked array and any of observation's components are masked
            or if observation is None, then observation is treated as a missing observation.
        observation_function : optional function
            t+1 obs function.  If unspecified, self.observation_functions is used.
        observation_covariance : optional [n_dim_obs, n_dim_obs] array
            t+1 obs noise covariance. If unspecified, self.observation_covariance is used.
        noise_parameter : float
            forgetting factor for processing noises

        Returns
        -------
        next_filtered_state_mean : [n_dim_state] array
            mean estimatation for t+1 state | [1...t+1] obs
        next_filtered_state_covariance : [n_dim_state, n_dim_state] array
            covariance estimation for t+1 state | [1...t+1] obs
        '''
        
        # initialize params
        (observation_functions, observation_covariance,
         _, _) = (
            self._initialize_parameters()
        )

        def default_function(f, arr):
            if f is None:
                assert len(arr) == 1
                f = arr[0]
            return f

        observation_function = default_function(
            observation_function, observation_functions
        )
        observation_covariance = _arg_or_default(
            observation_covariance, observation_cov,
            2, "observation_covariance"
        )

        # make a masked observation if necessary
        if observation_X is None:
            n_dim_obs = estimated_observation_covariance.shape[0]
            observation_X = np.ma.array(np.zeros(n_dim_obs))
            observation_X.mask = True
        else:
            observation_X = np.ma.asarray(observation_X)
            
        if observation_Y is None:
            n_dim_obs = estimated_observation_covariance.shape[0]
            observation_Y = np.ma.array(np.zeros(n_dim_obs))
            observation_Y.mask = True
        else:
            observation_Y = np.ma.asarray(observation_Y)

        # preprocess covariance matrices
        filtered_state_covariance2 = linalg.cholesky(filtered_state_covariance)
        observation_covariance2 = linalg.cholesky(observation_covariance)

        # make sigma points
        moments_state = Moments(filtered_state_mean, filtered_state_covariance2)
        points_state = moments2points(moments_state)

        # predict
        (_, moments_pred) = (
            unscented_filter_predict_P(
                moments_state,
                sigma2_transition=None,
                ff, noise_parameter
            )
        )
        points_pred = moments2points(moments_pred)

        # correct
        (next_filtered_state_moments, _) = (
            unscented_filter_correct(
                observation_function, moments_pred, points_pred,
                observation_Y, points_observation=observation_X,
                ff, noise_parameter,
                sigma2_observation=observation_covariance2
            )
        )

        next_filtered_state_covariance = (
            _reconstruct_covariances(next_filtered_state_moments.cov)
        )

        return (next_filtered_state_moments.mean, next_filtered_state_covariance)
            
        
    def sample(self, n_timesteps, random_state=None, observations_X=None, observations_Y=None,
               noise_parameter):
        '''Sampling from SQ-UKF model

        Params
        ------
        n_timesteps : int
        random_state : optional int or Random
            random number generator as rng
        observations_X : [n_dim_obs, n_timesteps] array
        observations_Y : [n_dim_obs, n_timesteps] array
        noise_parameter : float
            forgetting factor for processing noises
        '''
        
        (observation_functions, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        n_dim_state = initial_state_mean.shape[0]
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
                observation_function = (
                    _last_dims(observation_functions, t, ndims=1)[0]
                )
                observation_X, observation_Y = (
                    _last_dims(observations_X, t, ndims=1)[0],
                    _last_dims(observations_Y, t, ndims=1)[0]
                )
                
                '''update prior t distribution w/ UKF'''
                (filtered_state_mean, filtered_state_covariance) = (
                    AdditiveUnscentedKalmanFilter_P.filter_update_P(
                        self, filtered_state_mean, filtered_state_covariance,
                        observation_X=observation_X, observation_Y=observation_Y,
                        observation_function=observation_function,
                        observation_covariance=observation_covariance,
                        noise_parameter
                    )
                )
              
                x[t] = rng.multivariate_normal(
                    filtered_state_mean, filtered_state_covariance
                )

        return (x)
        
        
    def _default_parameters(self):
        return {
            'observation_functions': lambda state: state,
            'observation_covariance': np.eye(self.n_dim_obs),
            'initial_state_mean': np.zeros(self.n_dim_state),
            'initial_state_covariance': np.eye(self.n_dim_state),
            'random_state': 0,
        }
        
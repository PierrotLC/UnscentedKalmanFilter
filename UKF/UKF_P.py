'''
UKF implementations for Parameter Estimation

w/ additive parameters & observation noises
'''

import numpy as np
from numpy import ma

from ..SKLearn_utils import array1d, array2d, get_params, preprocess_arguments, check_random_state
from ..KF_utils import _determine_dimensionality
from .UKF_P_utilities import Moments, Noises, noise_ff, moments2points, \
    unscented_filter_predict_P, unscented_filter_correct, additive_unscented_filter_P


class UnscentedMixin(object):
    '''Methods shared by UKF_P implementations'''
    
    def __init__(self, observation_functions=None,
            initial_state_mean=None, initial_state_covariance=None,
            n_dim_state=None, n_dim_obs=None, random_state=None):

        # size of state & obs spaces
        n_dim_state = _determine_dimensionality(
            [(initial_state_covariance, array2d, -2),
             (initial_state_mean, array1d, -1)],
            n_dim_state
        )

        # set params
        self.observation_functions = observation_functions
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
            'initial_state_mean': array1d,
            'initial_state_covariance': array2d,
            'n_dim_state': int,
            'n_dim_obs': int,
            'random_state': check_random_state,
        }



class AdditiveUnscentedKalmanFilter_P(UnscentedMixin):
    '''Implements UKF w/ additive noise

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
        
    Parameters & obs noises Q & R are processed either by forgetting factor (RLS-like) or Robbins-Monro process.

    Params
    ------
    observation_functions : function or [T] array of functions
        observation_functions[t] as g_t
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
    '''


    def filter(self, X, Y, noise_parameter):
        '''Run UKF
        Robbins-Monro process for parameter & observation noises should be preferred.
        
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

        (observation_functions,
         initial_state_mean, initial_state_covariance, _, n_dim_obs) = (
            self._initialize_parameters()
        )

        (filtered_state_means, filtered_state_covariances) = (
            additive_unscented_filter_P(
                initial_state_mean, initial_state_covariance,
                observation_functions,
                X, Y,
                rm, noise_parameter, n_dim_obs
            )
        )

        return (filtered_state_means, filtered_state_covariances)


    def filter_update_P(self,
                      filtered_state_mean, filtered_state_covariance,
                      observation_X=None, observation_Y=None,
                      estimated_observation_covariance,
                      observation_function=None,
                      noise_parameter):
        '''Update of a parameter estimation
        Forgetting factor process for parameter & observation noises should be preferred.

        One-step update to estimate t+1 state | t+1 obs with t state estimation | [0...t] obs.
        To track an object with streaming observations.

        Params
        ------
        filtered_state_mean : [n_dim_state] array
            mean for corrected t state | [1...t] obs
        filtered_state_covariance : [n_dim_state, n_dim_state] array
            covariance for corrected t state | [1...t] obs
        observation_X : [n_dim_obs] array or None
            t+1 obs as noise point or exogeneous input.
            If observation is a masked array and any of observation's components are masked
            or if observation is None, then observation is treated as a missing observation.
        observation_Y : [n_dim_obs] array or None
            t+1 obs.
            If observation is a masked array and any of observation's components are masked
            or if observation is None, then observation is treated as a missing observation.
        estimated_observation_covariance : [n_dim_obs, n_dim_obs] array
            covariance for estimated t obs | [1...t] obs
        observation_function : optional function
            t+1 obs function.  If unspecified, self.observation_functions is used.
        noise_parameter : float
            forgetting factor for processing noises

        Returns
        -------
        next_filtered_state_mean : [n_dim_state] array
            mean estimation for t+1 state | [1...t+1] obs
        next_filtered_state_covariance : [n_dim_state, n_dim_state] array
            covariance estimation for t+1 state | [1...t+1] obs
        '''
        
        # initialize params
        (observation_functions, _, _) = (
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
        
        # calculate covariance matrices for parameters & observation noises
        transition_covariance = noise_ff(noise_parameter, filtered_state_covariance)
        observation_covariance = noise_ff(noise_parameter, estimated_observation_covariance)

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

        # make sigma points
        moments_state = Moments(filtered_state_mean, filtered_state_covariance)
        points_state = moments2points(moments_state)

        # predict
        (_, moments_pred) = (
            unscented_filter_predict_P(
                moments_state,
                noises=Noises(transition_covariance, observation_covariance)
            )
        )
        points_pred = moments2points(moments_pred)

        # correct
        (next_filtered_state_moments, _) = (
            unscented_filter_correct(
                observation_function, moments_pred, points_pred,
                observation_Y, points_observation=observation_X,
                ff, noise_parameter,
                noises=Noises(transition_covariance, observation_covariance)
            )
        )

        return (next_filtered_state_moments.mean, next_filtered_state_moments.cov)


    def sample(self, n_timesteps, random_state=None, observations_X=None, observations_Y=None,
               estimated_observation_covariance, noise_parameter):
        '''Sampling from UKF model

        Params
        ------
        n_timesteps : int
        random_state : optional int or Random
            random number generator as rng
        observations_X : [n_dim_obs, n_timesteps] array
        observations_Y : [n_dim_obs, n_timesteps] array
        estimated_observation_covariance : [n_dim_obs, n_dim_obs] array
            covariance for estimated t obs | [1...t] obs
        noise_parameter : float
            forgetting factor for processing noises
        '''
        
        (observation_functions,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        n_dim_state = initial_state_mean.shape[0]
        n_dim_obs = estimated_observation_covariance.shape[-1]

        # instantiating rng
        if random_state is None:
            rng = check_random_state(self.random_state)
        else:
            rng = check_random_state(random_state)

        # sampling from multivariate normal w/ UKF updates
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
                        estimated_observation_covariance,
                        observation_function=observation_function,
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
            'initial_state_mean': np.zeros(self.n_dim_state),
            'initial_state_covariance': np.eye(self.n_dim_state),
            'random_state': 0,
        }

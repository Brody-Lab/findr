# This file contains python adaptations of the original code by Thomas Luo
import numpy as np
import math as math
from jax import lax, random, numpy as jnp
from scipy.stats import norm, bootstrap
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

BIN_WIDTH = 0.01 # in seconds
MAX_TIME_STEPS = 100 # in bins
Array = Any
PRNGKey = Any

def mask_sequences(sequence_batch: Array, lengths: Array) -> Array:
  """Sets positions beyond the length of each sequence to 0."""
  return sequence_batch * (
    lengths[:, None] > jnp.arange(sequence_batch.shape[1])[None])

def causalgaussian(sigma):
  """Returns a causal gaussian filter with standard deviation sigma."""
  dt = BIN_WIDTH
  maxtimesteps = MAX_TIME_STEPS
  h = norm.pdf(np.linspace(0.0, dt*maxtimesteps-dt, num=maxtimesteps), loc=0, scale=sigma)
  w = 1. / (np.convolve(np.ones(maxtimesteps)*dt, h)[:maxtimesteps])
  return h, w

def smooth(y):
  """Smooths a vector y with a causal gaussian filter."""
  h, w = causalgaussian(0.1)
  x = np.convolve(y, h, mode='full')
  return np.array([w[t]*x[t] for t in range(len(y))])

def radial_basis_functions(D, N, begins_at_0=False, ends_at_0=False):
  """
    Unitary radial basis functions
  
    Each of `D` radial basis functions are evaluated at `N` values.
    The basis functions are orthogonalized and constrained to have unit norm.
  
    Args:
      D: (integer) number of basis functions
      N : (integer) number of values for which the basis functions are evaluated
      begins_at_0 : (bool, optional) whether the output of the basis functions are set to be 0 for the first element
      ends_at_0 : (bool, optional) whether the output of the basis functions are set to be 0 for the last element
  
    Returns:
      Phi : values of the unitary radial basis functions
      Phiraw : values of the non-unitary radial basis functions
  """
  if begins_at_0:
    if ends_at_0:
      delta_centers = N/(D+3)
    else:
      delta_centers = N/(D+1)
  else:
    if ends_at_0:
      delta_centers = N/(D+1)
    else:
      delta_centers = N/(D-1)
  firstcenter = 1+2*delta_centers if begins_at_0 else 1
  lastcenter = N-2*delta_centers if ends_at_0 else N
  centers = np.linspace(firstcenter, lastcenter, D)
  x = np.linspace(1,N,N)
  y = np.reshape(x, (N,1)) - np.reshape(centers, (1,np.size(centers)))
  y = y*math.pi/delta_centers/2
  y = np.minimum(math.pi,y)
  y = np.maximum(-math.pi,y)
  Phiraw = (np.cos(y) + 1)/2
  if (not begins_at_0) and (not ends_at_0) and (D % 2 == 0):
    Phi = np.hstack((np.ones((N,1)),Phiraw))
  else:
    Phi = Phiraw
  U, S, Vh = np.linalg.svd(Phiraw)
  Phi = U[:,0:D]
  return (Phi, Phiraw)

def radial_basis_functions_v2(D, x, begins_at_0=False, ends_at_0=False):
  """
    Unitary radial basis functions
  
    Each of `D` radial basis functions are evaluated at `N` values.
    The basis functions are orthogonalized and constrained to have unit norm.
  
    Args:
      D: (integer) number of basis functions
      x : (real vector) input to the raised cosine function
      begins_at_0 : (bool, optional) whether the output of the basis functions are set to be 0 for the first element
      ends_at_0 : (bool, optional) whether the output of the basis functions are set to be 0 for the last element
  
    Returns:
      Phi : values of the unitary radial basis functions
      Phiraw : values of the non-unitary radial basis functions
  """
  delta_x = x[-1] - x[0]
  if begins_at_0:
    if ends_at_0:
      delta_centers = delta_x / (D+3)
    else:
      delta_centers = delta_x / (D+1)
    if D == 1:
      centers = x[0] + 2*delta_centers + np.arange(1) * delta_centers
    else:
      centers = x[0] + 2*delta_centers + np.arange(D) * delta_centers
  else:
    if ends_at_0:
      delta_centers = delta_x / (D+1)
    else:
      delta_centers = delta_x / (D-1)
    if D == 1:
      centers = x[0] + np.arange(1) * delta_centers
    else:
      centers = x[0] + np.arange(D) * delta_centers
  omega = math.pi / delta_centers / 2
  t = np.reshape(x, (-1, 1)) - np.reshape(centers, (1, -1))
  Phiraw = (np.cos(np.maximum(-math.pi, np.minimum(math.pi, omega*t))) + 1)/2
  if (not begins_at_0) and (not ends_at_0):
    t_left = x - (centers[0] - delta_centers)
    lefttail = (np.cos(np.maximum(-math.pi, np.minimum(math.pi, omega*t_left))) + 1)/2
    t_right = x - (centers[-1] + delta_centers)
    righttail = (np.cos(np.maximum(-math.pi, np.minimum(math.pi, omega*t_right))) + 1)/2
    Phiraw[:,0] += lefttail
    Phiraw[:,-1] += righttail
    indices = x < centers[0] + 2/delta_centers
    deviations = 2.0 - np.sum(Phiraw, axis=1)
    Phiraw[indices,0] += deviations[indices]

  U, S, Vh = np.linalg.svd(Phiraw)
  Phi = U[:,0:D]
  return (Phi, Phiraw)

def infer_baseline_across_trials(
  trial_start_times:Array, 
  spikes_across_trials: Array, 
  train_indices, 
  valid_indices,
  n_splits: int
) -> Array:
  """
    Infers the baseline firing rate across trials.

    Args:
      trial_start_times: (real vector) the start times of each trial
      spikes_across_trials: (real tensor) the spikes across trials
      train_indices: (list of lists) the indices of the training set
      valid_indices: (list of lists) the indices of the validation set
      n_splits: (integer) the number of splits

    Returns:
      baseline_across_trials: (real tensor) the inferred baseline firing rate
  """
  baseline_across_trials = np.zeros((spikes_across_trials.shape + (n_splits,)))
  
  regs_set = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
  num_basis_set = [4, 5, 6, 7, 8, 9, 10]

  a = 9999/(np.max(trial_start_times) - np.min(trial_start_times))
  time_indx = np.round(a * (trial_start_times - np.min(trial_start_times))).astype(int)
  assert len(np.unique(time_indx)) == trial_start_times.shape[0]

  Phis = []
  for num_basis in num_basis_set:
      _, Phi = radial_basis_functions(num_basis, 10000)
      Phis.append(Phi)

  for k in range(n_splits):
      for neuron in range(spikes_across_trials.shape[-1]):
          y = spikes_across_trials[:, 0, neuron]
          y_train = y[np.sort(train_indices[k])]
          y_valid = y[np.sort(valid_indices[k])]

          min_mse = 1e6
          best_reg = -1
          best_num_basis = -1
          best_num_basis_indx = -1
          for i, num_basis in enumerate(num_basis_set):
              for s in regs_set:
                  Phi = Phis[i]
                  Phi_train = Phi[time_indx[np.sort(train_indices[k])],:]
                  Phi_val = Phi[time_indx[np.sort(valid_indices[k])],:]
                  w = np.linalg.pinv(
                    Phi_train.T @ Phi_train + s*np.diag(np.ones(Phi_train.shape[-1]))
                  ) @ Phi_train.T @ y_train
                  x_valid = Phi_val @ w
                  mse = np.mean((x_valid - y_valid)**2)
                  if mse < min_mse:
                      min_mse = mse
                      best_reg = s
                      best_num_basis = num_basis
                      best_num_basis_indx = i
          Phi = Phis[best_num_basis_indx]
          Phi_train = Phi[time_indx[np.sort(train_indices[k])],:]
          w = np.linalg.pinv(
            Phi_train.T @ Phi_train + best_reg*np.diag(np.ones(Phi_train.shape[-1]))
          ) @ Phi_train.T @ y_train
          baseline_across_trials[:, 0, neuron, k] = Phi[time_indx,:] @ w
  return baseline_across_trials

def infer_baseline(
  spikes: Array,
  lengths: Array,
  baseline_across_trials: Array,
  train_indices, 
  valid_indices,
  n_splits: int
):
  """
    Infers the baseline firing rate.

    Args:
      spikes: (real tensor) the spikes
      lengths: (integer vector) the lengths of each trial
      baseline_across_trials: (real tensor) the inferred baseline firing rate across trials
      train_indices: (list of lists) the indices of the training set
      valid_indices: (list of lists) the indices of the validation set
      n_splits: (integer) the number of splits

    Returns:
      baseline: (real tensor) the inferred baseline firing rate
  """
  dt = BIN_WIDTH
  regs_set = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
  num_basis_set = [5, 6, 7, 8, 9, 10]

  Phis = []
  X_trains = []
  Xs = []
  eta = 0.12
  x_eval = np.arange(spikes.shape[1])
  x_eval = np.arcsinh(eta * x_eval)
  for num_basis in num_basis_set:
      _, Phi = radial_basis_functions_v2(num_basis, x_eval)
      Phis.append(Phi)

  baseline = np.zeros((spikes.shape + (n_splits,)))
  for k in range(n_splits):    
      for neuron in range(spikes.shape[-1]):
          y_train = spikes[np.sort(train_indices[k]), :, neuron]
          y_valid = spikes[np.sort(valid_indices[k]), :, neuron]
          
          len_train = lengths[np.sort(train_indices[k])]
          len_valid = lengths[np.sort(valid_indices[k])]

          y_train_ = []
          for i in range(y_train.shape[0]):
              y_train_.append(y_train[i, :len_train[i]])
          y_train_ = np.hstack(y_train_)

          y_valid_ = []
          for i in range(y_valid.shape[0]):
              y_valid_.append(y_valid[i, :len_valid[i]])
          y_valid_ = np.hstack(y_valid_)

          y_ = []
          for i in range(spikes.shape[0]):
              y_.append(spikes[i, :lengths[i], neuron])
          y_ = np.hstack(y_)

          baseline_across_trials_train_ = []
          for i in range(y_train.shape[0]):
            baseline_across_trials_train_.append(
              np.tile(
                baseline_across_trials[np.sort(train_indices[k])[i], 0, neuron, k]*dt,
                len_train[i]
              )
            )
          baseline_across_trials_train_ = np.hstack(baseline_across_trials_train_)

          baseline_across_trials_ = []
          for i in range(spikes.shape[0]):
            baseline_across_trials_.append(
              np.tile(
                baseline_across_trials[i, 0, neuron, k]*dt,
                lengths[i]
              )
            )
          baseline_across_trials_ = np.hstack(baseline_across_trials_)

          min_mse = 1e6
          best_reg = -1
          best_num_basis = -1
          best_num_basis_indx = -1
          for ii, num_basis in enumerate(num_basis_set):
              for s in regs_set:
                  Phi = Phis[ii]            
                  X_train = np.zeros((y_train_.shape[0], num_basis))
                  X = np.zeros((y_.shape[0], num_basis))

                  j = 0
                  for i in range(y_train.shape[0]):
                      Ti = len_train[i]
                      X_train[j:j+Ti,:] = Phi[:Ti,:]
                      j=j+Ti

                  j = 0
                  for i in range(spikes.shape[0]):
                      Ti = lengths[i]
                      X[j:j+Ti,:] = Phi[:Ti,:]
                      j=j+Ti

                  w = np.linalg.pinv(
                    X_train.T @ X_train + s*np.diag(np.ones(X.shape[-1]))
                    ) @ (X_train.T @ (y_train_ - baseline_across_trials_train_))
                  x_ = X @ w + baseline_across_trials_
                  x = np.zeros_like(spikes[:,:,0])
                  j = 0
                  for i in range(spikes.shape[0]):
                      Ti = lengths[i]
                      x[i,:Ti] = x_[j:j+Ti]
                      j=j+Ti
                  x_valid = x[np.sort(valid_indices[k]), :]
                  mse = np.mean(
                    np.hstack(
                      [
                        (y_valid[trial, :len_valid[trial]] - \
                        x_valid[trial, :len_valid[trial]])**2 for trial in range(y_valid.shape[0])
                      ]
                    )
                  )
                  if mse < min_mse:
                    min_mse = mse
                    best_reg = s
                    best_num_basis = num_basis
                    best_num_basis_indx = ii
          Phi = Phis[best_num_basis_indx]
          X_train = np.zeros((y_train_.shape[0], best_num_basis))
          X = np.zeros((y_.shape[0], best_num_basis))

          j = 0
          for i in range(y_train.shape[0]):
              Ti = len_train[i]
              X_train[j:j+Ti,:] = Phi[:Ti,:]
              j=j+Ti

          j = 0
          for i in range(spikes.shape[0]):
              Ti = lengths[i]
              X[j:j+Ti,:] = Phi[:Ti,:]
              j=j+Ti
          w = np.linalg.pinv(
            X_train.T @ X_train + best_reg*np.diag(np.ones(X.shape[-1]))
            ) @ (X_train.T @ (y_train_ - baseline_across_trials_train_))
          x_ = X @ w + baseline_across_trials_
          x = np.zeros_like(spikes[:,:,0]).astype(float)
          j = 0
          for i in range(spikes.shape[0]):
              Ti = lengths[i]
              x[i,:Ti] = x_[j:j+Ti]
              j=j+Ti
          baseline[:, :, neuron, k] = x/dt
  return baseline

def generate_smoothed_spikes(spikes, lengths):
  """
    Smooths the spikes with a causal gaussian filter.

    Args:
      spikes: (real tensor) the spikes
      lengths: (integer vector) the lengths of each trial
    
    Returns:
      smoothed_spikes: (real tensor) the smoothed spikes
  """

  smoothed_spikes = np.zeros_like(spikes).astype(np.float32)
  for trial in range(smoothed_spikes.shape[0]):
    for neuron in range(smoothed_spikes.shape[2]):
      smoothed_spikes[trial, :lengths[trial], neuron] = smooth(spikes[trial, :lengths[trial], neuron])
  return smoothed_spikes

def generate_psths(spikes, lengths, pokedR):
  """
    Generates the PSTHs conditioned on left and right choices of the animal.

    Args:
      spikes: (real tensor) the spikes
      lengths: (integer vector) the lengths of each trial
      pokedR: (boolean vector) the choices of the animal
    
    Returns:
      right_observed_psth: (real matrix) the observed PSTH for right choices
      left_observed_psth: (real matrix) the observed PSTH for left choices
      right_psth_ci_low: (real matrix) the lower bound of the confidence interval for right choices
      right_psth_ci_high: (real matrix) the upper bound of the confidence interval for right choices
      left_psth_ci_low: (real matrix) the lower bound of the confidence interval for left choices
      left_psth_ci_high: (real matrix) the upper bound of the confidence interval for left choices
  """
  smoothed_spikes = generate_smoothed_spikes(spikes, lengths)
  mask_ = mask_sequences(np.ones_like(smoothed_spikes[:,:,0]), lengths)
  smoothed_spikes[mask_ == 0] = np.nan
    
  right_observed_psth = np.nanmean(smoothed_spikes[pokedR, :, :], axis=0)
  left_observed_psth = np.nanmean(smoothed_spikes[~pokedR, :, :], axis=0)
    
  right_psth_ci_low = np.zeros(right_observed_psth.shape)
  left_psth_ci_low = np.zeros(left_observed_psth.shape)
    
  right_psth_ci_high = np.zeros(right_observed_psth.shape)
  left_psth_ci_high = np.zeros(left_observed_psth.shape)
    
  num_neurons = smoothed_spikes.shape[2]
  num_timebins = smoothed_spikes.shape[1]
    
  for neuron in range(num_neurons):
    for timebin in range(num_timebins):
      right_smoothed = smoothed_spikes[pokedR, timebin, neuron]
      right_data = (right_smoothed[~np.isnan(right_smoothed)],)
      right_res = bootstrap(right_data, np.mean, n_resamples=1000, confidence_level=0.95)
      right_psth_ci_low[timebin, neuron] = right_res.confidence_interval.low
      right_psth_ci_high[timebin, neuron] = right_res.confidence_interval.high
            
      left_smoothed = smoothed_spikes[~pokedR, timebin, neuron]
      left_data = (left_smoothed[~np.isnan(left_smoothed)],)
      left_res = bootstrap(left_data, np.mean, n_resamples=1000, confidence_level=0.95)
      left_psth_ci_low[timebin, neuron] = left_res.confidence_interval.low
      left_psth_ci_high[timebin, neuron] = left_res.confidence_interval.high
    
  return right_observed_psth, left_observed_psth, right_psth_ci_low, right_psth_ci_high, left_psth_ci_low, left_psth_ci_high
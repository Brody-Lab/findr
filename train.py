import functools

from absl import app
from absl import flags
from absl import logging

import jax
import numpy as np
#import pandas as pd
#from juliacall import Main as jl
from typing import Any, Callable, Dict, Tuple, Sequence
from jax.nn import softplus
from jax import lax, random, numpy as jnp
from jax.scipy.special import gammaln
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax import serialization
from flax.training import train_state, checkpoints
from flax import traverse_util
from flax import struct
from sklearn.model_selection import KFold
import optax
import ml_collections
import models
import utils
import os

BIN_WIDTH = 0.01 # in seconds
SMALL_CONSTANT = 1e-5
Array = Any
FLAGS = flags.FLAGS
PRNGKey = Any

def create_train_state(
  rng: PRNGKey, 
  config: ml_collections.ConfigDict, 
  learning_rate_fn,
  xs,
  ckptdir = None
):  
  """
    Creates an initial `TrainState`.

    Args:
      rng: jax.random.PRNGKey, random key.
      config: ml_collections.ConfigDict, configuration parameters.
      learning_rate_fn: function, learning rate schedule.
      xs: dict, dataset.
      ckptdir: str, path to the checkpoint directory.

    Returns:
      state: train_state.TrainState, the initial state of the model.
  """
  key_1, key_2, key_3 = random.split(rng, 3)
  model = models.FINDR(
    alpha = config.alpha,
    noise_level = config.noise_level,
    features_prior = config.features_prior,
    features_posterior = config.features_posterior,
    non_task_related_gru_size = config.non_task_related_gru_size, 
    task_related_latent_size = config.task_related_latent_size,
    inference_network_size = config.inference_network_size,
    num_neurons = xs['spikes'].shape[-1],
    constrain_prior = config.constrain_prior
  )
  if ckptdir is not None:
    raw_restored = checkpoints.restore_checkpoint(
      ckpt_dir=ckptdir, 
      target=None, 
      parallel=False
    )
    params = freeze(raw_restored['model']['params'])
  else:
    params = model.init(
      key_2,
      xs['spikes'], 
      xs['externalinputs'],
      xs['baselineinputs'],
      xs['lengths'], 
      key_3
    )['params']
  
  tx = optax.chain(
    optax.clip_by_global_norm(5.0),
    optax.sgd(learning_rate_fn, config.momentum)
  )
  state = train_state.TrainState.create(
    apply_fn=model.apply, params=params, tx=tx
  )
  return state

@functools.partial(jax.jit, static_argnums=[3, 4, 5, 6, 7, 10])
def apply_model(
  state: train_state.TrainState, 
  batch, 
  loss_weights: Array,
  alpha: float, 
  beta_coeff: float,
  beta_inc_rate: float,
  lossw_inc_rate: float,
  l2_coeff: float,
  beta_counter: int,
  lossw_counter:int, 
  learning_rate_fn,
  rng: PRNGKey
):
  ntrgru_mask = traverse_util.path_aware_map(
    lambda path, _: 1 if 'non_task_related_gru' in path
    else 2, state.params
  )

  ortho_mask = traverse_util.path_aware_map(
    lambda path, _: 1 if 'task_related_latents_to_neurons' in path
    else 2, state.params
  )

  """Computes gradients and loss for a single batch."""
  def loss_fn(params): 
    logrates, z, b, mu, mu_theta, mu_phi, std = state.apply_fn(
      {'params': params},
      batch['spikes'], 
      batch['externalinputs'], 
      batch['baselineinputs'],
      batch['lengths'], 
      rng
    )
    nll_loss = neg_poisson_log_likelihood(
      logrates, 
      batch['spikes'],
      batch['lengths'],
      loss_weights,
      lossw_inc_rate,
      lossw_counter
    )
    kld_loss = beta_coeff * beta(beta_inc_rate, beta_counter) * kl_divergence(
      alpha, 
      mu_theta, 
      mu_phi, 
      std,
      batch['lengths'],
      loss_weights,
      lossw_inc_rate,
      lossw_counter
    )
    loss = nll_loss + kld_loss
    loss += sum(
      l2_loss(w, alpha=l2_coeff) if label==1 
      else l2_loss(w, alpha=1e-7)
      for label, w in zip(jax.tree_leaves(ntrgru_mask), jax.tree_leaves(params))
    )
    return loss, (nll_loss, kld_loss, logrates)
  
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (nll, kld, logrates)), grads = grad_fn(state.params)

  return grads, loss, nll, kld

@jax.jit
def update_model(
  state: train_state.TrainState, 
  grads
):
  return state.apply_gradients(grads=grads)

def train_epoch(
  state: train_state.TrainState,
  train_ds, 
  loss_weights: Array,
  config: ml_collections.ConfigDict, 
  beta_counter: int,
  lossw_counter: int,
  learning_rate_fn,
  rng: PRNGKey
):
  """
    Train for a single epoch.

    Args:
      state: train_state.TrainState, the current state of the model.
      train_ds: dict, training dataset.
      loss_weights: jnp.ndarray, loss weights.
      config: ml_collections.ConfigDict, configuration parameters.
      beta_counter: int, counter for the beta coefficient.
      lossw_counter: int, counter for the loss weights.
      learning_rate_fn: function, learning rate schedule.
      rng: jax.random.PRNGKey, random key.
    
    Returns:
      state: train_state.TrainState, the updated state of the model.
      train_loss: float, training loss.
      train_nll: float, training negative log-likelihood.
      train_kld: float, training
  """
  key_1, key_2 = random.split(rng, 2)
  train_ds_size = len(train_ds['externalinputs'])
  steps_per_epoch = train_ds_size // config.batch_size

  perms = random.permutation(key_1, len(train_ds['externalinputs']))
  perms = perms[:steps_per_epoch * config.batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, config.batch_size))

  epoch_loss = []
  epoch_nll = []
  epoch_kld = []
  for perm in perms:
    key_2, key_3 = random.split(key_2, 2)
    batch_spikes = train_ds['spikes'][perm, ...]
    batch_inputs = train_ds['externalinputs'][perm, ...]
    batch_lengths = train_ds['lengths'][perm, ...]
    batch_baseinputs = train_ds['baselineinputs'][perm, ...]
    batch = {
    'spikes': batch_spikes,
    'externalinputs': batch_inputs,
    'lengths': batch_lengths,
    'baselineinputs': batch_baseinputs
    }
    grads, loss, nll, kld = apply_model(
      state, 
      batch, 
      loss_weights,
      config.alpha,
      config.beta,
      config.beta_inc_rate, 
      config.lossw_inc_rate,
      config.l2_coeff,
      beta_counter,
      lossw_counter, 
      learning_rate_fn,
      key_3
    )
    state = update_model(state, grads)
    epoch_loss.append(loss)
    epoch_nll.append(nll)
    epoch_kld.append(kld)
  train_loss = np.mean(epoch_loss)
  train_nll = np.mean(epoch_nll)
  train_kld = np.mean(epoch_kld)
  return state, train_loss, train_nll, train_kld

def train_and_evaluate(
  config: ml_collections.ConfigDict,
  datapath: str,
  workdir: str,
  randseedpath: str = None
) -> train_state.TrainState:
  """
    Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the checkpoints are saved in.
    Returns:
      The train state (which includes the `.params`).
  """
  train_ds, val_ds, test_ds, ds, perms = get_datasets(
    datapath, 
    workdir,
    randseedpath,
    k_cv=config.k_cv, 
    n_splits=config.n_splits,
    baseline_fit=config.baseline_fit
  )
  rng = random.PRNGKey(1) # this is the random seed that we use to initialize the model

  key_1, key_2 = random.split(rng, 2)
  train_ds_size = len(train_ds['externalinputs'])
  steps_per_epoch = train_ds_size // config.batch_size
  print('steps_per_epoch ', steps_per_epoch)
  learning_rate_fn = create_learning_rate_fn(config, steps_per_epoch)
  state = create_train_state(key_1, config, learning_rate_fn, test_ds)
  best_state = state
  
  # the epoch around which the coefficient to the KL divergence term reaches 0.99
  annealing_epochs = np.floor(np.log(0.01)/np.log(config.beta_inc_rate)).astype(int)

  # train only the first 0.3s
  early_loss_weights = -1. * jnp.ones_like(train_ds['spikes'][0,:,0])
  early_loss_weights = early_loss_weights.at[:30].set(0.)
  
  # train only the first 0.5s
  middle_loss_weights = -1. * jnp.ones_like(train_ds['spikes'][0,:,0])
  middle_loss_weights = middle_loss_weights.at[:50].set(0.)

  # train all data points
  late_loss_weights = jnp.zeros_like(train_ds['spikes'][0,:,0])

  train_losses = []
  train_nlls = []
  train_klds = []

  val_losses = []
  val_nlls = []
  val_klds = []

  test_losses = []
  test_nlls = []
  test_klds = []

  for epoch in range(1, config.num_epochs + 1):
    key_2, key_3, key_4, key_5 = random.split(key_2, 4)
    if epoch < config.earlymiddle_epochs/3:
      state, train_loss, train_nll, train_kld = train_epoch(
        state, 
        train_ds, 
        early_loss_weights,
        config,
        0,
        1e6,
        learning_rate_fn,
        key_3
      )
    elif epoch < config.earlymiddle_epochs:
      state, train_loss, train_nll, train_kld = train_epoch(
        state, 
        train_ds, 
        middle_loss_weights,
        config, 
        0,
        1e6,
        learning_rate_fn,
        key_3
      )
    else:
      state, train_loss, train_nll, train_kld = train_epoch(
        state, 
        train_ds, 
        late_loss_weights,
        config, 
        epoch - config.earlymiddle_epochs,
        epoch - config.earlymiddle_epochs,
        learning_rate_fn,
        key_3
      )

    _, val_loss, val_nll, val_kld = apply_model(
      state, 
      val_ds,
      late_loss_weights,
      config.alpha,
      config.beta,
      config.beta_inc_rate,
      config.lossw_inc_rate,
      config.l2_coeff,  
      1e6,
      0,
      learning_rate_fn,
      key_4
    )

    _, test_loss, test_nll, test_kld = apply_model(
      state, 
      test_ds,
      late_loss_weights,
      config.alpha,
      config.beta,
      config.beta_inc_rate,
      config.lossw_inc_rate,
      config.l2_coeff,  
      1e6, 
      0,
      learning_rate_fn,
      key_5
    )
    
    if epoch > (config.earlymiddle_epochs + annealing_epochs):
      if val_losses[(config.earlymiddle_epochs + annealing_epochs):]:
        if val_loss < min(val_losses[(config.earlymiddle_epochs + annealing_epochs):]):
          best_state = state

    logging.info(
      '%s: %d, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f' % (
        'epoch', epoch,
        'train_loss', train_loss,
        'train_nll',  train_nll,
        'train_kld',  train_kld,
        'val_loss', val_loss, 
        'val_nll', val_nll, 
        'val_kld', val_kld,
        'test_loss', test_loss,
        'test_nll', test_nll, 
        'test_kld', test_kld
      )
    )

    train_losses.append(train_loss.item())
    train_nlls.append(train_nll.item())
    train_klds.append(train_kld.item())
    val_losses.append(val_loss.item())
    val_nlls.append(val_nll.item())
    val_klds.append(val_kld.item())
    test_losses.append(test_loss.item())
    test_nlls.append(test_nll.item())
    test_klds.append(test_kld.item())

    losses = {
      'train_losses': train_losses, 
      'train_nlls': train_nlls, 
      'train_klds': train_klds, 
      'val_losses': val_losses, 
      'val_nlls': val_nlls, 
      'val_klds': val_klds,
      'test_losses': test_losses, 
      'test_nlls': test_nlls, 
      'test_klds': test_klds
    }

    ckpt = {
      'model': best_state, 
      'config': config.to_dict(), 
      'losses': losses, 
      'perms': perms
    }

    checkpoints.save_checkpoint(
      ckpt_dir=FLAGS.workdir, 
      target=ckpt, 
      step=int(state.step), 
      keep=1,
      overwrite=True
    )

  return best_state

def get_datasets(datapath, workdir, randseedpath=None, k_cv=1, n_splits=5, baseline_fit=True):
  """
    Returns train, validation, and test datasets.

    Args:
      datapath: str, path to the data file.
      workdir: str, path to the directory where the checkpoints are saved.
      randseedpath: str, path to the random seed file.
      k_cv: int, the cross-validation fold.
      n_splits: int, the number of splits.
      baseline_fit: bool, whether to fit the baseline or not.
    
    Returns:
      train_ds: dict, training dataset.
      val_ds: dict, validation dataset.
      test_ds: dict, test dataset.
      ds: dict, concatenated dataset.
      concat_ds: jnp.ndarray, concatenated indices.
  """
  if randseedpath:
    df = pd.read_csv(randseedpath)
    one_hot = np.array(
      [True if session_id in datapath else False for session_id in df['session_id'].values.astype(str)]
    )
    if np.any(one_hot):
      random_seed = df['random_state'].values.astype(int)[one_hot][0]
    else:
      random_seed = 17
  else:
    random_seed = 17
  dt = BIN_WIDTH
  data = np.load(datapath)

  # we need to check if the data has external inputs as 'externalinputs' or 'clicks'
  try:
    externalinputs = data['externalinputs']
  except(KeyError):
    externalinputs = data['clicks']

  # we need to check if the data has has keyword 'choices' or not
  try:
    choices = data['choices']
    haschoices = True
  except(KeyError):
    choices = 0
    haschoices = False
  
  # we need to check if the data has keyword 'times'
  try:
    times = data['times']
  except(KeyError):
    times = np.arange(0, data['spikes'].shape[0])

  kf = KFold(n_splits = n_splits, random_state=random_seed, shuffle=True)
  train_valid_indices = []
  test_indices = []
  np.random.seed(seed=random_seed)
  for i, (train_valid_index, test_index) in enumerate(kf.split(data['spikes'])):
    train_valid_indices.append(np.random.permutation(train_valid_index))
    test_indices.append(np.random.permutation(test_index))

  train_indices = []
  valid_indices = []
  for i in range(n_splits-1):
    train_indices.append(train_valid_indices[i][~np.isin(train_valid_indices[i], test_indices[i+1])])
    valid_indices.append(train_valid_indices[i][np.isin(train_valid_indices[i], test_indices[i+1])])
  train_indices.append(train_valid_indices[-1][~np.isin(train_valid_indices[-1], test_indices[0])])
  valid_indices.append(train_valid_indices[-1][np.isin(train_valid_indices[-1], test_indices[0])])

  baselinepath = workdir.rsplit('/', 1)[0]

  if os.path.exists(baselinepath + '/spGLM_baseline.npy'):
    baselines = np.load(baselinepath + '/spGLM_baseline.npy')
    baseline = baselines[:,:,:,k_cv-1]
  elif os.path.exists(baselinepath + '/tzl_baseline.npy'):
    baseline = np.load(baselinepath + '/tzl_baseline.npy')
  elif os.path.exists(baselinepath + '/baseline.npy'):
    baselines = np.load(baselinepath + '/baseline.npy')
    baseline = baselines[:,:,:,k_cv-1]
  else:
    os.makedirs(baselinepath, exist_ok=True)
    spikes_across_trials1 = np.zeros((data['spikes'].shape[0], 1, data['spikes'].shape[2]))
    for trial in range(data['spikes'].shape[0]):
      spikes_across_trials1[trial, 0, :] = np.mean(
        data['spikes'][trial, :data['lengths'][trial], :], 
        axis=0
      )/dt

    try:
      baseline_hz = data['baseline_hz']
      spikes_across_trials = np.reshape(
        baseline_hz, 
        (baseline_hz.shape[0], 1, baseline_hz.shape[1])
      )
      alpha = np.mean(spikes_across_trials1, axis=0)[0, :] / np.mean(spikes_across_trials, axis=0)[0, :]
      spikes_across_trials = spikes_across_trials * alpha
    except(KeyError):
      spikes_across_trials = spikes_across_trials1
    
    if baseline_fit:
      baselines_across_trials = utils.infer_baseline_across_trials(
        times,
        spikes_across_trials, 
        train_indices, 
        valid_indices,
        n_splits
      )

      baselines = utils.infer_baseline(
        data['spikes'],
        data['lengths'],
        baselines_across_trials,
        train_indices, 
        valid_indices,
        n_splits
      )
      baseline = baselines[:,:,:,k_cv-1]
      np.save(baselinepath + '/baseline.npy', baselines)
    else:
      baselines = np.stack([np.tile(
        np.mean(
          spikes_across_trials[train_valid_indices[k_cv-1], :, :], axis=0
        )[np.newaxis, :, :], 
        (
          data['spikes'].shape[0], data['spikes'].shape[1], 1
        )
      ) for k in range(n_splits)], axis=3)
      baseline = baselines[:,:,:,k_cv-1]
      np.save(baselinepath + '/baseline.npy', baselines)

  train_ds = {
    'spikes': data['spikes'][train_indices[k_cv-1],:,:],
    'externalinputs': externalinputs[train_indices[k_cv-1],:,:], 
    'lengths':data['lengths'][train_indices[k_cv-1]], 
    'baselineinputs': baseline[train_indices[k_cv-1],:,:],
    'choices': data['choices'][train_indices[k_cv-1]] if haschoices else 0
  }
  
  val_ds = {
    'spikes': data['spikes'][valid_indices[k_cv-1],:,:],
    'externalinputs': externalinputs[valid_indices[k_cv-1],:,:], 
    'lengths':data['lengths'][valid_indices[k_cv-1]], 
    'baselineinputs': baseline[valid_indices[k_cv-1],:,:],
    'choices': data['choices'][valid_indices[k_cv-1]] if haschoices else 0
  }
  
  test_ds = {
    'spikes': data['spikes'][test_indices[k_cv-1],:,:],
    'externalinputs': externalinputs[test_indices[k_cv-1],:,:], 
    'lengths':data['lengths'][test_indices[k_cv-1]], 
    'baselineinputs': baseline[test_indices[k_cv-1],:,:],
    'choices': data['choices'][test_indices[k_cv-1]] if haschoices else 0
  }
  
  ds = {
    'spikes': jnp.concatenate(
        [
          train_ds['spikes'], 
          val_ds['spikes'], 
          test_ds['spikes']
        ], 0
      ),
    'externalinputs': jnp.concatenate(
        [
          train_ds['externalinputs'], 
          val_ds['externalinputs'], 
          test_ds['externalinputs']
        ], 0
      ),
    'lengths': jnp.concatenate(
        [
          train_ds['lengths'], 
          val_ds['lengths'], 
          test_ds['lengths']
        ], 0
      ),
    'baselineinputs': jnp.concatenate(
        [
          train_ds['baselineinputs'], 
          val_ds['baselineinputs'], 
          test_ds['baselineinputs']
        ], 0
      ),
    'choices': jnp.concatenate(
        [
          train_ds['choices'], 
          val_ds['choices'], 
          test_ds['choices']
        ], 0
      ) if haschoices else 0
  }

  concat_ds = jnp.concatenate(
    [
      train_indices[k_cv-1], 
      valid_indices[k_cv-1], 
      test_indices[k_cv-1]
    ], 0
  )

  return train_ds, val_ds, test_ds, ds, concat_ds

def beta(beta_inc_rate:float, counter: int) -> float:
  return 1 - beta_inc_rate**counter

def l2_loss(x, alpha):
    return alpha * (x ** 2).mean()

def kl_divergence(
  alpha: float, 
  mu_theta: Array, 
  mu_phi: Array, 
  std: Array, 
  lengths: Array,
  loss_weights: Array,
  lossw_inc_rate: float,
  counter: int
) -> float:
  """Calculates KL divergence between two Gaussians."""
  cov = std ** 2
  m = jnp.square(mu_theta - mu_phi) / cov
  kld = jnp.sum(m, axis=-1)
  kld_masked = 0.5 * alpha * jnp.sum(
    utils.mask_sequences(kld, lengths)
    ) / jnp.sum(
    utils.mask_sequences(kld, lengths) > 0
    )
  return kld_masked

def neg_poisson_log_likelihood(
  logrates: Array, 
  spikes: Array, 
  lengths: Array,
  loss_weights: Array,
  lossw_inc_rate: float,
  counter: int
) -> float:
  """Calculates Poisson negative log likelihood given rates and spikes.
  formula: -log(e^(-r) / n! * r^n)
          = r - n*log(r) + log(n!)
  """
  dt = BIN_WIDTH
  rates = softplus(logrates) + SMALL_CONSTANT
  result = dt*rates - spikes * jnp.log(dt*rates) + gammaln(spikes + 1.0)
  nll = jnp.sum(result, axis=-1)
  weights = beta(lossw_inc_rate, counter) * loss_weights + 1.
  weights = weights / jnp.sum(weights) * 100.
  masked_nll = jnp.sum(
    utils.mask_sequences(nll, lengths) * weights
  ) / jnp.sum((utils.mask_sequences(nll, lengths) > 0))
  return masked_nll

def create_learning_rate_fn(config, steps_per_epoch):
  """Creates learning rate schedule."""
  num_iter = config.num_epochs // (config.cosine_epochs + config.warmup_epochs)
  cosine_kwargs = [{
    'init_value': 0.,
    'peak_value': config.base_learning_rate,
    'warmup_steps': config.warmup_epochs * steps_per_epoch,
    'decay_steps': (config.cosine_mult_by ** i) * config.cosine_epochs * steps_per_epoch,
    'end_value': 0.
  } for i in range(num_iter)]
  
  schedule_fn = optax.sgdr_schedule(cosine_kwargs)
  return schedule_fn

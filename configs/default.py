"""Default Hyperparameter configuration."""

import ml_collections
import numpy as np

def get_config(config_id):
  """Set of hyperparameters"""
  k_cv_set = [1, 2, 3, 4, 5] # 5-fold cross-validation
  learning_rates = 10**np.linspace(-2, -0.5, num=5)
  features_set = [30, 50, 100]
  net_size = [50, 100, 200]

  hps = []
  for i in range(len(k_cv_set)):
    for j in range(len(learning_rates)):
      for k in range(len(features_set)):
        for l in range(len(net_size)):
          hps.append(
            (
              k_cv_set[i], 
              learning_rates[j], 
              features_set[k],
              net_size[l]
            )
          )

  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.k_cv = hps[int(config_id)-1][0]
  config.base_learning_rate = hps[int(config_id)-1][1]
  config.features_prior = [hps[int(config_id)-1][2]]
  config.features_posterior = [hps[int(config_id)-1][2]]
  config.inference_network_size = hps[int(config_id)-1][3]
  config.beta = 2.
  config.noise_level = 1.
  config.cosine_epochs = 190
  config.alpha = 0.1 # \Delta t / \tau : \Delta t = 10ms = 0.01s
  config.task_related_latent_size = 2
  config.non_task_related_gru_size = 0
  config.n_splits = len(k_cv_set)
  config.l2_coeff = 1e-4
  config.cosine_mult_by = 2
  config.warmup_epochs = 10
  config.batch_size = 25
  config.num_epochs = 3000
  config.momentum = 0.9
  config.beta_inc_rate = 0.99 # decreasing this value increases beta faster
  config.lossw_inc_rate = 1.  # decreasing this value increases lossw faster
  config.earlymiddle_epochs = 0
  config.baseline_fit = True
  config.constrain_prior = False

  return config

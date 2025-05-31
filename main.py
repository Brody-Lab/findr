from absl import app
from absl import flags
from absl import logging
import ml_collections
import numpy as np
import optuna
import train

FLAGS = flags.FLAGS

flags.DEFINE_string('datapath', None, 'Path to data.')
flags.DEFINE_string('workdir', None, 'Directory to store model fits.')

def get_config(k_cv, num_epochs, learning_rate, feature_size, net_size):
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.k_cv = k_cv
  config.base_learning_rate = learning_rate
  config.features_prior = [feature_size]
  config.features_posterior = [feature_size]
  config.inference_network_size = net_size
  config.beta = 2.
  config.noise_level = 1.
  config.cosine_epochs = 190
  config.alpha = 0.1 # \Delta t / \tau : \Delta t = 10ms = 0.01s
  config.task_related_latent_size = 2
  config.n_splits = 5 # 5-fold cross-validation
  config.l2_coeff = 1e-4
  config.cosine_mult_by = 2
  config.warmup_epochs = 10
  config.batch_size = 25
  config.num_epochs = num_epochs
  config.momentum = 0.9
  config.beta_inc_rate = 0.99 # decreasing this value increases beta faster
  config.lossw_inc_rate = 1.  # decreasing this value increases lossw faster
  config.earlymiddle_epochs = 0
  config.baseline_fit = True
  config.constrain_prior = False

  return config

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  def objective(trial):
      learning_rate = trial.suggest_categorical('learning_rate', 10**np.linspace(-2, -0.5, num=5))
      feature_size = trial.suggest_categorical('feature_size', [30, 50, 100])
      net_size = trial.suggest_categorical('net_size', [50, 100, 200])
      config = get_config(
        k_cv=1, 
        num_epochs=1500, 
        learning_rate=learning_rate,
        feature_size=feature_size,
        net_size=net_size
      )
      return train.train_and_evaluate(config, FLAGS.datapath, FLAGS.workdir, ckpt_save=False)

  study = optuna.create_study()
  study.optimize(objective, n_trials=10)
  config = get_config(
    k_cv=1, 
    num_epochs=3000,
    learning_rate=study.best_params['learning_rate'],
    feature_size=study.best_params['feature_size'],
    net_size=study.best_params['net_size']
  )
  train.train_and_evaluate(config, FLAGS.datapath, FLAGS.workdir, ckpt_save=True)
  logging.info('Best hyperparameters: %s', study.best_params)

if __name__ == '__main__':
  flags.mark_flags_as_required(['datapath', 'workdir'])
  app.run(main)
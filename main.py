from absl import app
from absl import flags
from absl import logging
import jax
from ml_collections import config_flags

import train

FLAGS = flags.FLAGS

flags.DEFINE_string('datapath', None, 'Path to data.')
flags.DEFINE_string('workdir', None, 'Directory to store model fits.')

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train.train_and_evaluate(FLAGS.config, FLAGS.datapath, FLAGS.workdir)

if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'datapath', 'workdir'])
  app.run(main)
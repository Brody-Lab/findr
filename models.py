import functools
import flax
import jax
from jax import lax, random, numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union
from mycells import MyGRUCell, GNSDECell, Constrained_GNSDECell, NTRGRUCell, PosteriorSDECell, Flow, Constrained_Flow
from utils import mask_sequences

Array = Any
PRNGKey = Any

@jax.vmap
def flip_sequences(inputs: Array, lengths: Array) -> Array:
  """Flips a sequence of inputs along the time dimension.
  This function can be used to prepare inputs for the reverse direction of a
  bidirectional GRU. It solves the issue that, when naively flipping multiple
  padded sequences stored in a matrix, the first elements would be padding
  values for those sequences that were padded. This function keeps the padding
  at the end, while flipping the rest of the elements.
  Example:
  ```python
  inputs = [[1, 0, 0],
            [2, 3, 0],
            [4, 5, 6]]
  lengths = [1, 2, 3]
  flip_sequences(inputs, lengths) = [[1, 0, 0],
                                     [3, 2, 0],
                                     [6, 5, 4]]
  ```
  Args:
    inputs: An array of input IDs <int>[batch_size, seq_length].
    lengths: The length of each sequence <int>[batch_size].
  Returns:
    An ndarray with the flipped inputs.
  """
  # Note: since this function is vmapped, the code below is effectively for
  # a single example.
  max_length = inputs.shape[0]
  return jnp.flip(jnp.roll(inputs, max_length - lengths, axis=0), axis=0)

class PosteriorSDE(nn.Module):
  """The gated neural SDE (gnSDE) model for the posterior process."""
  features: Sequence[int]
  alpha: float = 1.0
  noise_level: float = 1.0

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x, external_inputs, noise):
    gnsde_state, (y, mu_phi, mu, std) = PosteriorSDECell(self.features, self.alpha, self.noise_level)(
      carry, 
      x, 
      external_inputs,
      noise
    )
    return gnsde_state, (y, mu_phi, mu, std)
  
  @staticmethod
  def initialize_carry(
    rng: PRNGKey, 
    batch_size: int, 
    hidden_size: int, 
    init_fn=initializers.zeros_init()
  ):
    # Use a dummy key since the default state init fn is just zeros.
    return PosteriorSDECell.initialize_carry(rng, (batch_size,), hidden_size, init_fn)

class GNSDE(nn.Module):
  """The gated neural SDE (gnSDE) model for the prior process."""
  features: Sequence[int]
  alpha: float = 1.0

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x, noise):
    gnsde_state, (y, mu_theta, mu) = GNSDECell(self.features, self.alpha)(
      carry,
      x,
      noise
    )
    return gnsde_state, (y, mu_theta, mu)
  
  @staticmethod
  def initialize_carry(
    rng: PRNGKey, 
    batch_size: int, 
    hidden_size: int, 
    init_fn=initializers.zeros_init()
  ):
    # Use a dummy key since the default state init fn is just zeros.
    return GNSDECell.initialize_carry(rng, (batch_size,), hidden_size, init_fn)

class Constrained_GNSDE(nn.Module):
  """The model for the prior process."""
  features: Sequence[int]
  alpha: float = 1.0

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x, noise):
    gnsde_state, (y, mu_theta, mu) = Constrained_GNSDECell(self.features, self.alpha)(
      carry,
      x,
      noise
    )
    return gnsde_state, (y, mu_theta, mu)
  
  @staticmethod
  def initialize_carry(
    rng: PRNGKey, 
    batch_size: int, 
    hidden_size: int, 
    init_fn=initializers.normal(stddev=0.01)
  ):
    # Use a dummy key since the default state init fn is just zeros.
    return Constrained_GNSDECell.initialize_carry(rng, (batch_size,), hidden_size, init_fn)

class NTRGRU(nn.Module):
  """A simple unidirectional GRU that takes no input."""

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    basegru_state, y = NTRGRUCell()(carry, x)
    return basegru_state, y
  
  @staticmethod
  def initialize_carry(
    rng: PRNGKey, 
    batch_size: int, 
    hidden_size: int, 
    init_fn=initializers.ones_init()
  ):
    # Use a dummy key since the default state init fn is just zeros.
    return NTRGRUCell.initialize_carry(rng, (batch_size,), hidden_size, init_fn)

class Reduce(nn.Module):
  num_latents: int

  @nn.compact
  def __call__(self, z):
    z = nn.Dense(
      self.num_latents, 
      use_bias=False, 
      name='rnn_to_latents'
    )(z) # maybe dropout?
    return z

class InitialState(nn.Module):
  num_latents: int

  @nn.compact
  def __call__(self):
    z0 = self.param(
      'init_state', 
      lambda rng, shape: jnp.zeros(shape), 
      (self.num_latents,)
    )
    return z0

class SimpleGRU(nn.Module):
  """A simple unidirectional GRU."""

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x, y, mask):
    new_carry, z = MyGRUCell()(carry, x, y)
    def select_carried_state(new_state, old_state):
      return jnp.where(mask[:, None], old_state, new_state)
    carried_state = select_carried_state(new_carry, carry)
    return carried_state, z

  @staticmethod
  def initialize_carry(
    rng: PRNGKey, 
    batch_size: int, 
    hidden_size: int, 
    init_fn=initializers.zeros_init()
  ):
    # Use a dummy key since the default state init fn is just zeros.
    return nn.GRUCell.initialize_carry(
        rng, (batch_size,), hidden_size, init_fn)

class SimpleBiGRU(nn.Module):
  """A simple bi-directional GRU."""
  hidden_size: int

  def setup(self):
    self.forward_gru = SimpleGRU()
    self.backward_gru = SimpleGRU()

  def __call__(self, spike_inputs, external_inputs, trial_lengths, rng):
    key_1, key_2 = random.split(rng, 2)
    batch_size = spike_inputs.shape[0]
    mask = mask_sequences(
      jnp.sum(external_inputs, axis=-1), 
      trial_lengths
    )

    # Forward GRU.
    initial_state = SimpleGRU.initialize_carry(
      key_1, 
      batch_size, 
      self.hidden_size
    )
    _, forward_outputs = self.forward_gru(
      initial_state, 
      spike_inputs, 
      external_inputs, 
      mask
    )

    # Backward GRU.
    reversed_spike_inputs = flip_sequences(
      spike_inputs, 
      trial_lengths
    )
    reversed_external_inputs = flip_sequences(
      external_inputs, 
      trial_lengths
    )
    reversed_mask = mask_sequences(
      jnp.sum(reversed_external_inputs, axis=-1), 
      trial_lengths
    )
    initial_state = SimpleGRU.initialize_carry(
      key_2, 
      batch_size, 
      self.hidden_size
    )
    _, backward_outputs = self.backward_gru(
      initial_state, 
      reversed_spike_inputs, 
      reversed_external_inputs,
      reversed_mask
    )
    backward_outputs = flip_sequences(
      backward_outputs, 
      trial_lengths
    )

    # Concatenate the forward and backward representations.
    outputs = jnp.concatenate(
      [forward_outputs, backward_outputs], -1
    )
    return outputs

class FINDR(nn.Module):
  """FINDR."""
  features_prior: Sequence[int]         # structure of the gnSDE network for the prior process
  features_posterior: Sequence[int]     # structure of the gnSDE network for the posterior process
  task_related_latent_size: int         # task related firing rate within a trial
  non_task_related_gru_size: int        # RNN that generates non-task related latent
  inference_network_size: int           # RNN that feeds past and future spikes and external inputs into the posterior
  num_neurons: int
  alpha: float = 1.0
  noise_level: float = 1.0
  constrain_prior: bool = False

  def setup(self):
    self.inference_network = SimpleBiGRU(
      hidden_size=self.inference_network_size
    )
    self.posterior_process = PosteriorSDE(
      features=self.features_posterior,
      alpha=self.alpha,
      noise_level=self.noise_level
    )

    if self.constrain_prior:
      self.prior_process = Constrained_Flow(
        features=self.features_prior,
        alpha=self.alpha
      )
    else:
      self.prior_process = Flow(
        features=self.features_prior,
        alpha=self.alpha
      )
    
    if self.non_task_related_gru_size == 0:
      self.non_task_related_gru = 0
    else:
      self.non_task_related_gru = NTRGRU()

    self.task_related_latents_to_neurons = nn.Dense(
      self.num_neurons,
      name='task_related_latents_to_neurons',
      use_bias=False
    )

    self.non_task_related_latents_to_neurons = nn.Dense(
      self.num_neurons,
      name='non_task_related_latents_to_neurons'
    )
    
    self.gru_initial_state = InitialState(
      num_latents=self.non_task_related_gru_size
    )

  def __call__(
    self, 
    spike_inputs, 
    external_inputs,
    baseline_inputs,  
    trial_lengths, 
    rng
  ) -> Array:
    key_1, key_2, key_3, key_4, key_5, = random.split(rng, 5)
    batch_size = len(trial_lengths)
    hs = self.inference_network(
      spike_inputs, 
      external_inputs,
      trial_lengths,
      key_1
    )
    carry_dl = self.posterior_process.initialize_carry(
      key_2, 
      batch_size, 
      self.task_related_latent_size
    ) # task-related latent
    noise_prior = random.normal(
      key_3, 
      hs.shape[:-1] + (self.task_related_latent_size,)
    )
    noise_posterior = random.normal(
      key_4, 
      hs.shape[:-1] + (self.task_related_latent_size,)
    )
    _, (z, mu_phi, mu, std) = self.posterior_process(
      carry_dl, 
      hs,
      external_inputs,
      noise_posterior
    )
    mu_theta = self.prior_process(
      z, 
      external_inputs
    )
    if self.non_task_related_gru_size != 0:
      carry_ndl = self.gru_initial_state() * self.non_task_related_gru.initialize_carry(
        key_5, 
        batch_size, 
        self.non_task_related_gru_size
      ) # non-task-related latent
      _, b = self.non_task_related_gru(
        carry_ndl,
        external_inputs
      )
      logrates = self.task_related_latents_to_neurons(z) + \
        self.non_task_related_latents_to_neurons(b) + baseline_inputs
    else:
      b = 0
      logrates = self.task_related_latents_to_neurons(z) + baseline_inputs

    return logrates, z, b, mu, mu_theta, mu_phi, std

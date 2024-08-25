from functools import partial   # pylint: disable=g-importing-member
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

from flax import linen as nn
from flax.linen.activation import relu
from flax.linen.activation import sigmoid
from flax.linen.activation import tanh
from flax.linen.activation import silu
from jax.nn import softplus
from flax.linen.dtypes import promote_dtype
from flax.linen import initializers
from flax.linen.linear import Conv
from flax.linen.linear import default_kernel_init
from flax.linen.linear import Dense
from flax.linen.linear import PrecisionLike
from flax.linen.module import compact
from flax.linen.module import Module
from jax import numpy as jnp
from jax import random
import numpy as np

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any

class MyGRUCell(nn.Module):
  """A GRU cell that gets spike trains and external inputs.

  Attributes:
    gate_fn: activation function used for gates (default: sigmoid)
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: initializers.orthogonal()).
    bias_init: initializer for the bias parameters (default: initializers.zeros_init())
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      default_kernel_init)
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      initializers.orthogonal())
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.normal(1e-3)
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, carry, spikes_inputs, external_inputs):
    """Gated recurrent unit (GRU) cell.

    Args:
      carry: the hidden state of the GRU cell,
        initialized using `GRUCell.initialize_carry`.
      spikes_inputs/external_inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.

    Returns:
      A tuple with the new carry and the output.
    """
    h = carry
    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = partial(Dense,
                      features=hidden_features,
                      use_bias=True,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.recurrent_kernel_init,
                      bias_init=self.bias_init)
    dense_ix = partial(Dense,
                      features=hidden_features,
                      use_bias=False,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.kernel_init,
                      bias_init=self.bias_init)
    dense_ic = partial(Dense,
                      features=hidden_features,
                      use_bias=False,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.kernel_init,
                      bias_init=self.bias_init)
    r = self.gate_fn(dense_ic(name='irc')(external_inputs) + \
                     dense_ix(name='irx')(spikes_inputs) + \
                     dense_h(name='hr')(h))
    z = self.gate_fn(dense_ic(name='izc')(external_inputs) + \
                     dense_ix(name='izx')(spikes_inputs) + \
                     dense_h(name='hz')(h))
    # add bias because the linear transformations aren't directly summed.
    n = self.activation_fn(dense_ic(name='inc')(external_inputs) + \
                           dense_ix(name='inx')(spikes_inputs) + \
                           r * dense_h(name='hn')(h))
    new_h = (1. - z) * n + z * h
    return new_h, new_h

class NTRGRUCell(nn.Module):
  """A non-task-related (NTR) GRU cell that captures nuisance fluctuations 
    in firing rate that are not dependent on task variables.

  Attributes:
    gate_fn: activation function used for gates (default: sigmoid)
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: initializers.orthogonal()).
    bias_init: initializer for the bias parameters (default: initializers.zeros_init())
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      default_kernel_init)
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      initializers.orthogonal())
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.normal(1e-3)
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, carry, inputs):
    """Gated recurrent unit (GRU) cell.

    Args:
      carry: the hidden state of the GRU cell,
        initialized using `GRUCell.initialize_carry`.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.

    Returns:
      A tuple with the new carry and the output.
    """
    h = carry
    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = partial(Dense,
                      features=hidden_features,
                      use_bias=True,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.recurrent_kernel_init,
                      bias_init=self.bias_init)
    r = self.gate_fn(dense_h(name='hr')(h))
    z = self.gate_fn(dense_h(name='hz')(h))
    # add bias because the linear transformations aren't directly summed.
    n = self.activation_fn(r * dense_h(name='hn')(h))
    new_h = (1. - z) * n + z * h
    return new_h, new_h

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.ones_init()):
    """Initialize the RNN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given GNSDE cell.
    """
    mem_shape = batch_dims + (size,)
    return init_fn(rng, mem_shape)

class PosteriorSDECell(nn.Module):
  """A gated neural SDE cell that models the posterior process. The drift function is 
    parametrized by a gated feedforward neural network (FNN). 

  Attributes:
    features: the number of hidden units in the hidden layers of the FNN.
    alpha: the effective time step used for Euler-Maruyama integration of the SDE.
    gate_fn: activation function used for gates (default: sigmoid).
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: normalization proposed in Kim, Can, Krishnamurthy).
    bias_init: initializer for the bias parameters.
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  features: Sequence[int] # is possible to default to []?
  alpha: float = 1.0 # float32?
  noise_level: float = 1.0
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      default_kernel_init)
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      initializers.variance_scaling(scale=1.414, mode="fan_in", distribution="truncated_normal"))
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.normal(1e-3)
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, carry, rnn_inputs, external_inputs, noise):
    """The gnSDE cell for the posterior process.
    Args:
      carry: the hidden state of the PosteriorSDE cell,
        initialized using `NODECell.initialize_carry`.
      rnn_inputs/external_inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.
      noise: an ndarray with the standard normal noise for the current time step.
    Returns:
      A tuple with the new carry and the output.
    """
    h = carry
    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = partial(Dense,
                      features=hidden_features,
                      use_bias=True,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.recurrent_kernel_init,
                      bias_init=self.bias_init)
    dense_i = partial(Dense,
                      features=hidden_features,
                      use_bias=False,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.kernel_init,
                      bias_init=self.bias_init)
    
    z = self.gate_fn(
      dense_h(name='hz')(h) + \
      dense_i(name='izr')(rnn_inputs) + \
      dense_i(name='izc')(external_inputs)
    )
    for i, feat in enumerate(self.features):
      if i == 0:
        h = dense_h(features=feat, name=f'layer_{i}_h')(h) + \
            dense_i(features=feat, name=f'layer_{i}_ir')(rnn_inputs) + \
            dense_i(features=feat, name=f'layer_{i}_ic')(external_inputs)
        h = silu(h)
      else:
        h = dense_h(features=feat, name=f'layer_{i}')(h)
        h = silu(h)
    if self.features:
      h = dense_h(features=hidden_features, name='layer_fin')(h)
    else:
      h = dense_h(features=hidden_features, name='layer_0_h')(h)
    h = tanh(h) # final nonlinearity
    logvar = self.param('logvar', lambda rng, shape: jnp.zeros(shape), (hidden_features,))
    std = self.noise_level * sigmoid(logvar) * jnp.ones_like(noise)
    mu = (1. - self.alpha * z) * carry + self.alpha * z * h
    new_h = mu + jnp.sqrt(self.alpha) * std * noise 
    mu_phi = z * (-carry + h)
    return new_h, ( new_h, mu_phi, mu, std )

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros_init()):
    """Initialize the PosteriorSDE cell carry.
    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given GNSDE cell.
    """
    mem_shape = batch_dims + (size,)
    return init_fn(rng, mem_shape)

class Flow(nn.Module):
  """The drift function of the prior process used in the training procedure.
    It is parametrized by a gated feedforward neural network (FNN). 

  Attributes:
    features: the number of hidden units in the hidden layers of the FNN.
    alpha: the effective time step used for Euler-Maruyama integration of the SDE.
    gate_fn: activation function used for gates (default: sigmoid).
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: normalization proposed in Kim, Can, Krishnamurthy).
    bias_init: initializer for the bias parameters.
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  features: Sequence[int]
  alpha: float = 1.0
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      default_kernel_init)
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      initializers.variance_scaling(scale=1.414, mode="fan_in", distribution="truncated_normal"))
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.normal(1e-3)
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, h, x):
    new_h = h
    hidden_features = h.shape[-1]
    dense_h = partial(Dense,
                      features=hidden_features,
                      use_bias=True,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.recurrent_kernel_init,
                      bias_init=self.bias_init)
    dense_i = partial(Dense,
                      features=hidden_features,
                      use_bias=False,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.kernel_init,
                      bias_init=self.bias_init)
    
    z = self.gate_fn(
      dense_h(name='hz')(new_h) + \
      dense_i(name='izx')(x)
    )
    for i, feat in enumerate(self.features):
      if i == 0:
        new_h = dense_h(features=feat, name=f'layer_{i}_h')(new_h) + \
                dense_i(features=feat, name=f'layer_{i}_i')(x)
        new_h = silu(new_h)
      else:
        new_h = dense_h(features=feat, name=f'layer_{i}')(new_h)
        new_h = silu(new_h)
    new_h = dense_h(features=hidden_features, name='layer_fin')(new_h)
    new_h = tanh(new_h) # final nonlinearity
    mu_theta = z * (-h + new_h)
    return mu_theta

class GNSDECell(nn.Module):
  """A gated neural SDE cell that models the prior process. This module is
  only used in the generative mode of FINDR which samples from prior.

  Attributes:
    features: the number of hidden units in the hidden layers of the FNN.
    alpha: the effective time step used for Euler-Maruyama integration of the SDE.
    gate_fn: activation function used for gates (default: sigmoid).
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: normalization proposed in Kim, Can, Krishnamurthy).
    bias_init: initializer for the bias parameters.
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  features: Sequence[int] # is possible to default to []?
  alpha: float = 1.0 # float32?
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      default_kernel_init)
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      initializers.variance_scaling(scale=1.414, mode="fan_in", distribution="truncated_normal"))
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.normal(1e-3)
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, carry, inputs, siglogvar_noise):
    """The gnSDE cell for the prior process.

    Args:
      carry: the hidden state of the GNSDE cell,
        initialized using `NSDECell.initialize_carry`.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.
      siglogvar_noise: an ndarray with the noise for the current time step.

    Returns:
      A tuple with the new carry and the output.
    """
    h = carry
    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = partial(Dense,
                      features=hidden_features,
                      use_bias=True,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.recurrent_kernel_init,
                      bias_init=self.bias_init)
    dense_i = partial(Dense,
                      features=hidden_features,
                      use_bias=False,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.kernel_init,
                      bias_init=self.bias_init)
    
    z = self.gate_fn(
      dense_h(name='hz')(h) + \
      dense_i(name='izx')(inputs)
    )
    for i, feat in enumerate(self.features):
      if i == 0:
        h = dense_h(features=feat, name=f'layer_{i}_h')(h) + \
            dense_i(features=feat, name=f'layer_{i}_i')(inputs)
        h = silu(h)
      else:
        h = dense_h(features=feat, name=f'layer_{i}')(h)
        h = silu(h)
    h = dense_h(features=hidden_features, name='layer_fin')(h)
    h = tanh(h) # final nonlinearity
    mu = (1. - self.alpha * z) * carry + self.alpha * z * h
    new_h = mu + jnp.sqrt(self.alpha) * siglogvar_noise
    mu_theta = z * (-carry + h)
    return new_h, (new_h, mu_theta, mu)

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros_init()):
    """Initialize the GNSDE cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given GNSDE cell.
    """
    mem_shape = batch_dims + (size,)
    return init_fn(rng, mem_shape)

class Constrained_GNSDECell(nn.Module):
  """A cell that models the prior process.

  Attributes:
    features: the number of hidden units in the hidden layers of the FNN.
    alpha: the effective time step used for Euler-Maruyama integration of the SDE.
    gate_fn: activation function used for gates (default: sigmoid).
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: normalization proposed in Kim, Can, Krishnamurthy).
    bias_init: initializer for the bias parameters.
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  features: Sequence[int] # is possible to default to []?
  alpha: float = 1.0 # float32?
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      default_kernel_init)
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      initializers.variance_scaling(scale=1.414, mode="fan_in", distribution="truncated_normal"))
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.normal(1e-3)
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, carry, inputs, siglogvar_noise):
    """The cell for the prior process.

    Args:
      carry: the hidden state of the GNSDE cell,
        initialized using `NSDECell.initialize_carry`.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.
      siglogvar_noise: an ndarray with the noise for the current time step.

    Returns:
      A tuple with the new carry and the output.
    """
    new_h = carry
    hidden_features = carry.shape[-1]
    dense_i = partial(
      Dense,
      features=hidden_features,
      use_bias=False,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      kernel_init=self.kernel_init,
      bias_init=self.bias_init
    )

    theta = self.param(
      'theta', 
      lambda rng, 
      shape: jnp.zeros(shape), 
      (1,)
    )

    r = self.param(
      'r', 
      lambda rng, 
      shape: jnp.ones(shape), 
      (1,)
    )

    scale = self.param(
      'scale', 
      lambda rng, 
      shape: jnp.ones(shape), 
      (1,)
    )

    fp_loc = self.param(
      'fp', 
      lambda rng, 
      shape: jnp.ones(shape), 
      (1,)
    )

    sigma_squared = self.param(
      'sigma_squared', 
      lambda rng, 
      shape: jnp.zeros(shape), 
      (1,)
    )

    V = jnp.array([[1., jnp.sin(theta[0])], [0., jnp.cos(theta[0])]])
    L = jnp.array([[0., 0.], [0., -softplus(r[0])]])
    V_inv = jnp.array([[1., -jnp.sin(theta[0]) / jnp.cos(theta[0])], [0., 1. / jnp.cos(theta[0])]])
    M = V @ L @ V_inv
    z_linear = carry @ M.T
    fp = jnp.array([fp_loc[0], 0.])
    z_sp = -softplus(scale[0]) * jnp.exp( -(carry - fp)**2 / (0.2*sigmoid(sigma_squared[0])) ) * (carry - fp) + \
      -softplus(scale[0]) * jnp.exp( -(carry + fp)**2 / (0.2*sigmoid(sigma_squared[0])) ) * (carry + fp)
    z = z_linear + z_sp
    mu_theta = z + dense_i(name='ix')(inputs)
    mu = carry + self.alpha * mu_theta
    new_h = mu + jnp.sqrt(self.alpha) * siglogvar_noise
    return new_h, (new_h, mu_theta, mu)

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros_init()):
    """Initialize the cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given GNSDE cell.
    """
    mem_shape = batch_dims + (size,)
    return init_fn(rng, mem_shape)

class Constrained_Flow(nn.Module):
  """The drift function of the prior process used in the training procedure.
    It contains dynamics proposed in previous hypotheses as special cases -- works for 2D only.

  Attributes:
    features: the number of hidden units in the hidden layers of the FNN.
    alpha: the effective time step used for Euler-Maruyama integration of the SDE.
    gate_fn: activation function used for gates (default: sigmoid).
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: normalization proposed in Kim, Can, Krishnamurthy).
    bias_init: initializer for the bias parameters.
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  features: Sequence[int]
  alpha: float = 1.0
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      default_kernel_init)
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      initializers.variance_scaling(scale=1.414, mode="fan_in", distribution="truncated_normal"))
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.normal(1e-3)
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, h, x):
    hidden_features = h.shape[-1]
    dense_i = partial(
      Dense,
      features=hidden_features,
      use_bias=False,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      kernel_init=self.kernel_init,
      bias_init=self.bias_init
    )

    theta = self.param(
      'theta', 
      lambda rng, 
      shape: jnp.zeros(shape), 
      (1,)
    )

    r = self.param(
      'r', 
      lambda rng, 
      shape: jnp.ones(shape), 
      (1,)
    )

    scale = self.param(
      'scale', 
      lambda rng, 
      shape: jnp.ones(shape), 
      (1,)
    )

    fp_loc = self.param(
      'fp', 
      lambda rng, 
      shape: jnp.ones(shape), 
      (1,)
    )

    sigma_squared = self.param(
      'sigma_squared', 
      lambda rng, 
      shape: jnp.zeros(shape), 
      (1,)
    )

    V = jnp.array(
      [
        [1., jnp.sin(theta[0])], 
        [0., jnp.cos(theta[0])]
      ]
    )

    L = jnp.array(
      [
        [0., 0.], 
        [0., -softplus(r[0])]
      ]
    )

    V_inv = jnp.array(
      [
        [1., -jnp.sin(theta[0]) / jnp.cos(theta[0])], 
        [0., 1. / jnp.cos(theta[0])]
      ]
    )

    M = V @ L @ V_inv
    z_linear = h @ M.T
    fp = jnp.array([fp_loc[0], 0.])
    z_sp = -softplus(scale[0]) * jnp.exp( -(h - fp)**2 / (0.2*sigmoid(sigma_squared[0])) ) * (h - fp) + \
      -softplus(scale[0]) * jnp.exp( -(h + fp)**2 / (0.2*sigmoid(sigma_squared[0])) ) * (h + fp)
    z = z_linear + z_sp
    mu_theta = z + dense_i(name='ix')(x)
    return mu_theta
# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Demo model training functions and utilities.

Much of this code was adapted directly from the Neural Network Potentials
notebook of JAX MD (see github/google/jax-md).

"""

from absl import logging
logging.set_verbosity(logging.INFO)

import tensorflow as tf
import tensorflow_datasets as tfds
import flatland.dataset

import jax
from jax import random, vmap, jit, grad
import jax.numpy as jnp
import numpy as np

from jax import lax

from flatland.log import TrainingLogger

import optax

import haiku as hk
from typing import Callable, List, Iterator, Tuple, Any, Dict  #Callable, Tuple, TextIO, Dict, Any, Optional
from jax_md import nn

from jax_md import energy, util, space, partition, simulate, quantity

from functools import partial
from jax_md.energy import _canonicalize_node_state

key = random.PRNGKey(0)

ExampleStream = Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]

Array = util.Array

PyTree = Any
Box = space.Box
DisplacementFn = space.DisplacementFn
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList


def demo_example_stream(batch_size: int, split: str) -> ExampleStream:

  ds = tfds.load('flatland_mock', split=split)
  assert isinstance(ds, tf.data.Dataset)

  ds = ds.cache().repeat().batch(batch_size)

  # HACK: The plan is to re-work the source dataset to produce things
  # with a (length, dimensions) shape coming in to remove this from
  # the processing stream.
  def reshape(x, y):
    return jnp.ravel(jnp.array([x, y]), order="F").reshape(10, 2)

  for example in tfds.as_numpy(ds):

    positions_x = jnp.array(example["structure_x"])
    positions_y = jnp.array(example["structure_y"])
    positions = vmap(reshape)(positions_x, positions_y)

    energies = jnp.array(example["structure_energy"])

    # Sure, let's pretend for now since these are random anyway
    forces = positions

    yield positions, energies, forces


# Only slight modification from github/google/jax-md
def graph_network_neighbor_list(
    network,
    polymer_length: int,
    polymer_dimensions: int,
    displacement_fn: DisplacementFn,
    box_size: Box,
    r_cutoff: float,
    dr_threshold: float,
    nodes: Array = None,
    n_recurrences: int = 2,
    mlp_sizes: Tuple[int, ...] = (64, 64),
    mlp_kwargs: Dict[str, Any] = None
) -> Tuple[NeighborFn, nn.InitFn, Callable[[PyTree, Array, NeighborList],
                                           Array]]:
  """Convenience wrapper around EnergyGraphNet model using neighbor lists.

  Args:
    network: The class name of a network to initialize.
    displacement_fn: Function to compute displacement between two positions.
    box_size: The size of the simulation volume, used to construct neighbor
      list.
    r_cutoff: A floating point cutoff; Edges will be added to the graph
      for pairs of particles whose separation is smaller than the cutoff.
    dr_threshold: A floating point number specifying a "halo" radius that we use
      for neighbor list construction. See `neighbor_list` for details.
    nodes: None or an ndarray of shape `[N, node_dim]` specifying the state
      of the nodes. If None this is set to the zeroes vector. Often, for a
      system with multiple species, this could be the species id.
    n_recurrences: The number of steps of message passing in the graph network.
    mlp_sizes: A tuple specifying the layer-widths for the fully-connected
      networks used to update the states in the graph network.
    mlp_kwargs: A dict specifying args for the fully-connected networks used to
      update the states in the graph network.

  Returns:
    A pair of functions. An `params = init_fn(key, R)` that instantiates the
    model parameters and an `E = apply_fn(params, R)` that computes the energy
    for a particular state.

  """

  nodes = _canonicalize_node_state(nodes)

  @hk.without_apply_rng
  @hk.transform
  def model(R, neighbor, **kwargs):
    N = R.shape[0]

    d = partial(displacement_fn, **kwargs)
    d = space.map_neighbor(d)
    R_neigh = R[neighbor.idx]
    dR = d(R, R_neigh)

    if 'nodes' in kwargs:
      _nodes = _canonicalize_node_state(kwargs['nodes'])
    else:
      _nodes = jnp.zeros((N, 1), R.dtype) if nodes is None else nodes

    _globals = jnp.zeros((1,), R.dtype)

    dr_2 = space.square_distance(dR)
    edge_idx = jnp.where(dr_2 < r_cutoff**2, neighbor.idx, N)

    net = network(n_recurrences=n_recurrences,
                  mlp_sizes=mlp_sizes,
                  mlp_kwargs=mlp_kwargs,
                  polymer_length=polymer_length,
                  polymer_dimensions=polymer_dimensions)

    return net(nn.GraphTuple(_nodes, dR, _globals, edge_idx))  # pytype: disable=wrong-arg-count

  neighbor_fn = partition.neighbor_list(displacement_fn,
                                        box_size,
                                        r_cutoff,
                                        dr_threshold,
                                        mask_self=False)

  init_fn, apply_fns = model.init, model.apply

  return neighbor_fn, init_fn, apply_fns


class OrigamiNet(hk.Module):
  """Extends jax-md EnergyGraphNet to also learn to min structure energy.

  See github/google/jax-md/jax-md/energy.py for the original implementation.

  """

  def __init__(self,
               polymer_length: int,
               polymer_dimensions: int,
               n_recurrences: int,
               mlp_sizes: Tuple[int, ...],
               mlp_kwargs: Dict[str, Any] = None,
               name: str = 'Energy'):
    super().__init__(name=name)

    if mlp_kwargs is None:
      mlp_kwargs = {
          'w_init': hk.initializers.VarianceScaling(),
          'b_init': hk.initializers.VarianceScaling(0.1),
          'activation': jax.nn.softplus
      }

    self._graph_net = nn.GraphNetEncoder(n_recurrences, mlp_sizes, mlp_kwargs)

    structure_size = polymer_length * polymer_dimensions
    self._decoder = hk.nets.MLP(output_sizes=mlp_sizes + (structure_size + 1,),
                                activate_final=False,
                                name='GlobalDecoder',
                                **mlp_kwargs)

  def __call__(self, graph: nn.GraphTuple) -> jnp.ndarray:
    """Produce energy and structure predictions."""
    output = self._graph_net(graph)
    return self._decoder(output.globals)


def _configure_losses(key, batch_size: int, polymer_length: int,
                      polymer_dimensions: int, box_size: float,
                      example_stream_fn: Callable):
  """Configure losses needed for the demo structure solver."""

  # Very temporary patch
  positions, energies, forces = example_stream_fn(batch_size=batch_size,
                                                  split="train").__next__()

  _, polymer_length, polymer_dimensions = positions.shape

  displacement, shift = space.periodic(box_size)

  neighbor_fn, init_fn, origami_fn = graph_network_neighbor_list(
      network=OrigamiNet,
      polymer_length=polymer_length,
      polymer_dimensions=polymer_dimensions,
      displacement_fn=displacement,
      box_size=box_size,
      r_cutoff=3.0,
      dr_threshold=0.0)

  neighbor = neighbor_fn(positions[0], extra_capacity=6)

  @jit
  def energy_fn(params, R):
    """Predict an energy value for an input structure."""
    _neighbor = neighbor_fn(R, neighbor)
    # Consider the first element of the array to be the energy prediction
    return origami_fn(params, R, _neighbor)[0]

  @jit
  def structure_energy_fn(params, R):
    """Predict a structure then an energy value for that structure."""
    _neighbor = neighbor_fn(R, neighbor)
    # Consider the 2nd through the last of the array to be position predictions
    structure = origami_fn(params, R, _neighbor)[1:]
    polymer_shape = (polymer_length, polymer_dimensions)
    return energy_fn(params, jnp.reshape(structure, polymer_shape))

  # Vectorize each of the energy functions
  vmap_energy_fn = vmap(energy_fn, (None, 0))
  vmap_structure_energy_fn = vmap(structure_energy_fn, (None, 0))

  grad_fn = grad(energy_fn, argnums=1)
  force_fn = lambda params, R, **kwargs: -grad_fn(params, R)
  vectorized_force_fn = vmap(force_fn, (None, 0))

  key, _ = random.split(key)
  params = init_fn(key, positions[0], neighbor)

  @jit
  def energy_loss(params, R, energy_targets):
    return jnp.mean((vmap_energy_fn(params, R) - energy_targets)**2)

  @jit
  def force_loss(params, R, force_targets):
    dforces = vectorized_force_fn(params, R) - force_targets
    return jnp.mean(jnp.sum(dforces**2, axis=(1, 2)))

  @jit
  def structure_energy_loss(params, R, energy_targets):

    # The difference in energy between the predicted structure and the
    # input structure.
    predicted_structure_energies = vmap_structure_energy_fn(params, R)
    energy_difference = predicted_structure_energies - energy_targets

    return jnp.mean(energy_difference)

  @jit
  def loss(params, R, targets):
    e = energy_loss(params, R, targets[0])
    f = force_loss(params, R, targets[1])
    se = structure_energy_loss(params, R, targets[0])
    return e + f + se

  def error_fn(params, positions, energies):
    return float(jnp.sqrt(energy_loss(params, positions, energies)))

  return loss, error_fn, params


def configure_update_step(learning_rate: float, loss: Callable):
  """Configure an optax training update step."""

  opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))

  @jit
  def _update_step(params, opt_state, positions, labels):
    updates, opt_state = opt.update(
        grad(loss)(params, positions, labels), opt_state)
    return optax.apply_updates(params, updates), opt_state

  @jit
  def update_step(params_and_opt_state, batches):

    def inner_update(params_and_opt_state, batch):
      params, opt_state = params_and_opt_state
      b_xs, b_labels = batch

      return _update_step(params, opt_state, b_xs, b_labels), 0

    return lax.scan(inner_update, params_and_opt_state, batches)[0]

  return update_step, opt


def train_demo_solver(num_training_steps: int, training_log_every: int,
                      batch_size: int):
  """Training wrapper for training a demo structure solver.
  
  Note:
  * The current form of this only predicts energies. The current plan
  is to use this as pre-training for a subsequent step to predict
  structures that are of lower energy.
  * The initial purpose of this is to ensure integration between the
  data generation steps of Flatland and at least one specific way of
  using the data (that will be the one provided in the structure_solver
  notebook).

  """

  test_batch = next(demo_example_stream(batch_size=batch_size, split="test"))
  test_positions, test_energies, test_forces = test_batch
  _, polymer_length, polymer_dimensions = test_positions.shape

  loss, error_fn, params = _configure_losses(
      key,
      batch_size=16,
      box_size=10.862,
      polymer_length=polymer_length,
      polymer_dimensions=polymer_dimensions,
      example_stream_fn=demo_example_stream)

  update_step, opt = configure_update_step(learning_rate=1e-3, loss=loss)

  opt_state = opt.init(params)

  tl = TrainingLogger(log_every=training_log_every,
                      num_training_steps=num_training_steps)

  logging.info("Beginning training run.")

  train_example_iterator = demo_example_stream(batch_size, split="train")

  for step_num, stream in enumerate(train_example_iterator):

    positions, energies, forces = stream

    # ============
    # Hack
    positions = jnp.expand_dims(positions, axis=0)
    energies = jnp.expand_dims(energies, axis=0)
    forces = jnp.expand_dims(forces, axis=0)
    # ============

    if step_num >= num_training_steps:
      logging.info("Finished training")
      break

    logging.debug("Performing training step %s" % step_num)

    if step_num % training_log_every == 0:

      train_error = error_fn(params, positions[0], energies[0])
      test_error = error_fn(params, test_positions, test_energies)

      tl.log(test_error=test_error, train_error=train_error, step_num=step_num)

    packed_update_args = (params, opt_state)
    packed_update_batch = (positions, (energies, forces))

    params, opt_state = update_step(params_and_opt_state=packed_update_args,
                                    batches=packed_update_batch)

  return params

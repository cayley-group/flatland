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

import optax
from jax_md import energy, space, simulate, quantity
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds
import flatland.dataset

import numpy as np
import jax.numpy as jnp
from jax import vmap, jit, grad, random, lax

from typing import Iterator, Tuple
from collections.abc import Callable

from flatland.log import TrainingLogger

key = random.PRNGKey(0)

ExampleStream = Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]

def demo_example_stream(batch_size: int, split: str) -> ExampleStream:

  ds = tfds.load('flatland_mock', split=split)
  assert isinstance(ds, tf.data.Dataset)

  ds = ds.cache().repeat().batch(batch_size)

  # The source dataset can be reworked so this isn't necessary
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


def _configure_losses(key, batch_size: int, box_size: float,
                      example_stream_fn: Callable):
  """Configure losses needed for the demo structure solver."""

  # Very temporary patch
  positions, energies, forces = example_stream_fn(
      batch_size=batch_size, split="train").__next__()

  displacement, shift = space.periodic(box_size)

  neighbor_fn, init_fn, energy_fn = energy.graph_network_neighbor_list(
    displacement, box_size, r_cutoff=3.0, dr_threshold=0.0)

  neighbor = neighbor_fn(positions[0], extra_capacity=6)

  @jit
  def train_energy_fn(params, R):
    _neighbor = neighbor_fn(R, neighbor)
    return energy_fn(params, R, _neighbor)

  vectorized_energy_fn = vmap(train_energy_fn, (None, 0))

  grad_fn = grad(train_energy_fn, argnums=1)
  force_fn = lambda params, R, **kwargs: -grad_fn(params, R)
  vectorized_force_fn = vmap(force_fn, (None, 0))

  key, _ = random.split(key)
  params = init_fn(key, positions[0], neighbor)

  @jit
  def energy_loss(params, R, energy_targets):
    return jnp.mean((vectorized_energy_fn(params, R) - energy_targets) ** 2)

  @jit
  def force_loss(params, R, force_targets):
    dforces = vectorized_force_fn(params, R) - force_targets
    return jnp.mean(jnp.sum(dforces ** 2, axis=(1, 2)))

  @jit
  def loss(params, R, targets):
    return energy_loss(params, R, targets[0]) + force_loss(params, R, targets[1])

  def error_fn(params, positions, energies):
    return float(jnp.sqrt(energy_loss(params, positions, energies)))

  return loss, error_fn, params


def configure_update_step(learning_rate: float, loss: Callable):
  """Configure an optax training update step."""

  opt = optax.chain(optax.clip_by_global_norm(1.0),
                    optax.adam(learning_rate))

  @jit
  def _update_step(params, opt_state, positions, labels):
    updates, opt_state = opt.update(grad(loss)(params, positions, labels),
                                    opt_state)
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

  loss, error_fn, params = _configure_losses(key, batch_size=16, box_size=10.862,
                                             example_stream_fn=demo_example_stream)

  update_step, opt = configure_update_step(learning_rate=1e-3, loss=loss)

  opt_state = opt.init(params)

  tl = TrainingLogger(
    log_every=training_log_every,
    num_training_steps=num_training_steps)

  logging.info("Beginning training run.")

  test_batch = demo_example_stream(batch_size=batch_size, split="test").__next__()
  test_positions, test_energies, test_forces = test_batch

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

      tl.log(test_error=test_error,
             train_error=train_error,
             step_num=step_num)

    packed_update_args = (params, opt_state)
    packed_update_batch = (positions, (energies, forces))

    params, opt_state = update_step(
        params_and_opt_state=packed_update_args,
        batches=packed_update_batch)

  return params

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
"""Physics simulation utilities."""

import jax.numpy as jnp
from jax import random
from jax import jit, grad, vmap, value_and_grad
from jax import lax
from jax import ops
import numpy as np
from typing import Tuple

from matplotlib import pyplot as plt

#from jax.config import config
#config.update("jax_enable_x64", True)

from jax_md import space, smap, energy, minimize, quantity, simulate, partition

from functools import partial
import time

PolymerLayout = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


def make_linear_layout(polymer_length: int, box_size: float) -> PolymerLayout:
  """Compute the coordinates and bonds of a linear polymer layout.

  Args:
    polymer_length: The length of the input polymer.
    box_size: The x and y dimensions of a square simulation domain.

  Returns:
    positions, bonds: A tuple of positions and bonds.

  """

  positions = np.zeros((polymer_length, 2))

  for i in range(polymer_length):
    x = box_size / 2
    y = box_size / 4 + 0.5 * i * box_size / polymer_length
    positions[i] = [x, y]

  bonds = jnp.array([[i, i + 1] for i in range(polymer_length - 1)],
                    dtype=jnp.int64)

  bond_lengths = [0] * len(bonds)
  for i, _ in enumerate(bond_lengths):
    bond_lengths[i] = positions[i + 1][1] - positions[i][1]

  return jnp.array(positions), jnp.array(bonds), jnp.array(bond_lengths)

# ========================================
# From Jax MD (github/google/jax-md), below

def run_brownian(energy_fn,
                 R_init,
                 shift,
                 key,
                 num_steps,
                 kT,
                 dt=0.00001,
                 gamma=0.1) -> jnp.ndarray:
  """Simulate Brownian motion."""

  init, apply = simulate.brownian(energy_fn, shift, dt=dt, kT=kT, gamma=gamma)

  apply = jit(apply)

  @jit
  def scan_fn(state, current_step):
    # Dynamically pass r0 to apply, which passes it on to energy_fn
    return apply(state), 0

  key, split = random.split(key)
  state = init(split, R_init)
  state, _ = lax.scan(scan_fn, state, jnp.arange(num_steps))

  return state.position

def harmonic_morse(dr, D0=5.0, alpha=5.0, r0=1.0, k=50.0, **kwargs):
  """Compute the harmonic morse potential for a pair of particles at distance `dr`."""
  U = jnp.where(
      dr < r0, 0.5 * k * (dr - r0)**2 - D0,
      D0 * (jnp.exp(-2. * alpha * (dr - r0)) - 2. * jnp.exp(-alpha *
                                                            (dr - r0))))
  return jnp.array(U, dtype=dr.dtype)

def harmonic_morse_pair(displacement_or_metric,
                        species=None,
                        D0=5.0,
                        alpha=10.0,
                        r0=1.0,
                        k=50.0):
  """The harmonic morse function over all pairs of particles in a system."""

  # Initialize various parameters of the harmonic morse function
  D0 = jnp.array(D0, dtype=jnp.float32)
  alpha = jnp.array(alpha, dtype=jnp.float32)
  r0 = jnp.array(r0, dtype=jnp.float32)
  k = jnp.array(k, dtype=jnp.float32)

  # Pass the harmonic morse function defined above along with its parameters and a
  # displacement/metric function.
  return smap.pair(
      harmonic_morse,
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      species=species,
      D0=D0,
      alpha=alpha,
      r0=r0,
      k=k)

def bistable_spring(dr, r0=1.0, a2=2, a4=5, **kwargs):
  return (a4 * (dr - r0)**4 - a2 * (dr - r0)**2)

def bistable_spring_bond(displacement_or_metric,
                         bond,
                         bond_type=None,
                         r0=1,
                         a2=2,
                         a4=5):
  """Convenience wrapper to compute energy of particles bonded by springs."""
  r0 = jnp.array(r0, jnp.float32)
  a2 = jnp.array(a2, jnp.float32)
  a4 = jnp.array(a4, jnp.float32)
  return smap.bond(
      bistable_spring,
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      bond,
      bond_type,
      r0=r0,
      a2=a2,
      a4=a4)


# From Jax MD (github/google/jax-md), above
# =================

def simulate_polymer_brownian(key,
                              polymer,
                              box_size,
                              brownian_steps=10,
                              perturb_kt=200):
  """Given a polymer, simulate it with Brownian motion"""

  key, subkey = random.split(key)

  positions, bonds, bond_lengths = make_linear_layout(
      polymer_length=len(polymer), box_size=box_size)

  displacement, shift = space.periodic(box_size)

  # HACK: Still working on making these simulations physically sensible.
  # Until then this will hard-code some values assuming there are only
  # three species as a more formal function for generating this may not
  # be necessary at this point.
  r0_species_matrix = jnp.array([[1.0, 1.0, 0.5], [1.0, 1.0, 1.0],
                                 [0.5, 1.0, 1.0]])

  energy_fn = harmonic_morse_pair(displacement,
                                  D0=0.,
                                  alpha=10.0,
                                  k=1.0,
                                  species=polymer,
                                  r0=r0_species_matrix)

  bond_energy_fn = bistable_spring_bond(displacement, bonds, r0=bond_lengths)

  def combined_energy_fn(positions):
    return energy_fn(positions) + 2 * bond_energy_fn(positions)

  positions = run_brownian(combined_energy_fn,
                           positions,
                           shift,
                           key=subkey,
                           num_steps=brownian_steps,
                           kT=perturb_kt)

  return positions, combined_energy_fn

def plot_system(positions, box_size, species=None, ms=10, bonds=[]):
  """Plot a bonded polymer system."""

  for b in bonds:

    idx0 = b[0]
    idx1 = b[1]

    coord_a = positions[idx0]
    coord_b = positions[idx1]

    start = [coord_a[0], coord_b[0]]
    end = [coord_a[1], coord_b[1]]

    plt.plot(start, end,
             linewidth=4,
             color="lightgrey")

  if species is None:

    positions_x = positions[:, 0]
    positions_y = positions[:, 1]

    plt.plot(positions_x, positions_y, 'o', markersize=ms)

  else:

    for ii in range(jnp.amax(species) + 1):

      temp = positions[species==ii]
      temp_x = temp[:, 0]
      temp_y = temp[:, 1]

      plt.plot(temp_x, temp_y, 'o', markersize=ms)

  plt.xlim([0, box_size])
  plt.ylim([0, box_size])
  plt.xticks([], [])
  plt.yticks([], [])

  plot_xy_dim = 1.5 * plt.gcf().get_size_inches()[1]

  plt.gcf().set_size_inches(plot_xy_dim, plot_xy_dim)

  plt.tight_layout()

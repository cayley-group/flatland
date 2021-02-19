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
"""Data generation wrapper."""

import tempfile

from absl import logging
logging.set_verbosity(logging.INFO)

import jax.numpy as jnp
from jax import random

from flatland import physics
import numpy as np

import pickle

key = random.PRNGKey(0)


def compute_distances(positions: jnp.ndarray) -> np.ndarray:

  length = positions.shape[0]

  distances = np.zeros(shape=(length, length))

  for i in range(length):
    for j in range(length):
      distances[i][j] = jnp.linalg.norm(positions[i] - positions[j])

  return distances


def _compute_angle(p0: jnp.ndarray, p1: jnp.ndarray,
                   p2: jnp.ndarray) -> jnp.float32:
  """Compute the angle centered at `p1` between the other two points."""

  a = p1 - p0
  da = np.linalg.norm(a)

  b = p1 - p2
  db = np.linalg.norm(b)

  c = p0 - p2
  dc = np.linalg.norm(c)

  x = (da**2 + db**2 - dc**2) / (2.0 * da * db)

  angle = jnp.arccos(x) * 180.0 / jnp.pi

  return min(angle, 360 - angle)


def compute_bond_angles(positions: jnp.ndarray) -> jnp.ndarray:

  angles = np.zeros(shape=(len(positions) - 2,))

  for i in range(len(positions) - 2):
    angles[i] = _compute_angle(positions[i], positions[i + 1], positions[i + 2])

  return angles


def compile_dataset_for_population(key, population) -> str:

  dataset = [{} for _ in range(population.shape[0])]

  report_every = 1

  logging.info("Generating examples for polymer population of size %s" %
               population.shape[0])

  for i, polymer in enumerate(population):

    positions, energy_fn = physics.simulate_polymer_brownian(key=key,
                                                             polymer=polymer,
                                                             box_size=6.8)

    positions_x = positions[:, 0]
    positions_y = positions[:, 1]

    angles = compute_bond_angles(positions)
    distances = compute_distances(positions).flatten()

    energy = energy_fn(positions)

    example = {
        "aa_sequence": polymer,
        "structure_energy": energy,
        "structure_x": positions_x,
        "structure_y": positions_y,
        "solved_angles": angles,
        "solved_distances": distances
    }

    dataset[i] = example

    if i % report_every == 0:
      logging.info("Finished processing %s examples." % (i + 1))

  logging.info("Finished generating examples, writing to disk.")

  name = None
  with tempfile.NamedTemporaryFile() as out_file:
    pickle.dump(dataset, out_file)
    name = out_file.name

  return name

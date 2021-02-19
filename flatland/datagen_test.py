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
"""Tests of datagen wrapper."""

from jax import vmap, random, jit, random
import jax.numpy as jnp
from flatland import evolution as evo
from flatland import datagen

def test_compile_dataset_for_population():

  alphabet_size = 3
  population_size = 2
  genome_length = 10
  mutation_rate = 0.15
  num_generations = 2

  def fitness_mean_value_target(v, target_value=1.0):
    return 1 - jnp.abs(jnp.mean(v) - target_value)/target_value

  @jit
  def batched_fitness_mean_value_target(population):
    return vmap(fitness_mean_value_target)(population)

  key = random.PRNGKey(1)

  _, _, population = evo.evolve_with_mutation(
    fitness_fn=batched_fitness_mean_value_target,
    num_generations=num_generations,
    pop_size=population_size,
    genome_length=genome_length,
    mutation_rate=mutation_rate,
    alphabet_size=alphabet_size,
    keep_full_population_history=True,
    key=key)

  dataset_path = datagen.compile_dataset_for_population(key, population)
  assert isinstance(dataset_path, str)
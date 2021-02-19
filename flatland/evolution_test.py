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
"""Tests of polymer evolutionary optimization tools."""

import jax.numpy as jnp
from jax import vmap, jit, random
from flatland import evolution as evo


def test_types():

  key = random.PRNGKey(0)

  pop = evo.generate_population(key=key)

  assert isinstance(pop, evo.Population)

  pop2 = evo.mutate_population(
      key=random.split(key)[0],
      population=pop,
  )

  assert isinstance(pop2, evo.Population)

  fitnesses = random.uniform(key, (pop2.shape[0],))

  pop3 = evo.resample_population(key=random.split(key)[0],
                                 population=pop2,
                                 fitnesses=fitnesses)

  assert isinstance(pop3, evo.Population)


def test_evolve_with_mutation_runs():
  """Evolve a population of polymers whose mean value is 1.0.
  
  E.g. [0,1,2,1,0,1,2,1,0,1,2,1,0]
  """

  alphabet_size = 3
  population_size = 10
  genome_length = 5
  mutation_rate = 0.15
  num_generations = 10

  def fitness_mean_value_target(v, target_value=1.0):
    return 1 - jnp.abs(jnp.mean(v) - target_value) / target_value

  @jit
  def batched_fitness_mean_value_target(population):
    return vmap(fitness_mean_value_target)(population)

  res = evo.evolve_with_mutation(fitness_fn=batched_fitness_mean_value_target,
                                 num_generations=num_generations,
                                 pop_size=population_size,
                                 genome_length=genome_length,
                                 mutation_rate=mutation_rate,
                                 alphabet_size=alphabet_size,
                                 keep_full_population_history=True,
                                 key=random.PRNGKey(1))

  fitness_history, population_history, population = res

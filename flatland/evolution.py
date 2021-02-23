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
"""Evolutionary optimization of polymers.

This module performs a simple form of evolutionary optimization designed
specifically for one-dimensional integer-encoded polymers of variable-
sized alphabets and lengths.

"""

import jax.numpy as jnp
from jax import random, jit, vmap
from matplotlib import pyplot
import numpy as np

from collections import namedtuple

from typing import Tuple

key = random.PRNGKey(0)

# ======
# For now, just keep the typing simple. But it would be good to formalize
# each of these as they are interfaces with other parts of the code.
# Also, I'm not clear how to provide types
Fitnesses = jnp.ndarray
FitnessHistory = jnp.ndarray
PopulationHistory = np.ndarray
Population = jnp.ndarray
EvolutionResult = Tuple[FitnessHistory, PopulationHistory, Population]
# ======


def generate_population(key: jnp.ndarray,
                        length: int = 10,
                        pop_size: int = 10,
                        alphabet_size: int = 4) -> Population:
  """Generate a polymer population.

  Args:
    key: A random number generator key e.g. produced via
      jax.random.PRNGKey.
    length: The length of the polymers in the population. Here we
      consider a population whose polymers are all of the same length.
    pop_size: The size of the population (number of members).
    alphabet_size: The size of the alphabet from which polymer
      elements are sampled.

  Returns:
    Population: A population of polymers.

  """
  return random.randint(minval=0,
                        maxval=alphabet_size,
                        shape=(pop_size, length),
                        key=key)


def mutate_population(key: jnp.ndarray,
                      population: Population,
                      mutation_rate: float = 0.15,
                      alphabet_size: int = 4) -> Population:
  """Mutate a polymer population.

  Here we convert the total mutation rate into mutation rates for each
  member of an alphabet and use this, combined with the non-mutation
  rate, to sample mutations. These mutations are simply members of the
  alphabet that are added to the existing values followed by a modulo
  operation for the size of the alphabet.
  
  Example:
    alphabet_size = 4, pop = [[0,1,2,3]], mutation = [[0,0,1,2]],
    new_population = [[0,1,3,1]]

  Args:
    key: A random number generator key e.g. produced via
      jax.random.PRNGKey.
    population: A polymer population.
    mutation_rate: The combined rate of mutation over all forms
      of mutation. E.g. 15% that it would be one of 1->2 or 2->3.
    alphabet_size: The size of the alphabet from which polymer
      elements are sampled.

  Returns:
    Population: A population of polymers.

  """
  individual_p_mutation = mutation_rate / alphabet_size
  # Lazily double-counts self-transitions as a type of mutation
  # in the interest of prototyping
  p_no_mutation = (1 - mutation_rate)
  mutation_probs = [individual_p_mutation for _ in range(alphabet_size)]
  mutation_probs = [p_no_mutation] + mutation_probs

  mutation = random.choice(key,
                           a=jnp.array(range(alphabet_size + 1)),
                           shape=population.shape,
                           p=jnp.array(mutation_probs))

  return jnp.mod(population + mutation, alphabet_size - 1)


def resample_population(key: jnp.ndarray, population: Population,
                        fitnesses: Fitnesses) -> Population:
  """Re-sample the members of a polymer population.

  Here we re-sample population members to be retained for a subsequent
  generation with random sampling weighted according to the fitness of
  each member.

  Args:
    key: A random number generator key e.g. produced via 
      jax.random.PRNGKey.
    population: A population of polymers.
    fitnesses: The size of the population (number of members).
    alphabet_size: The size of the alphabet from which polymer elements
      are sampled.

  Returns:
    Population: A population of polymers.

  """

  retain_members = random.choice(key=key,
                                 a=jnp.array(
                                     [i for i in range(population.shape[0])]),
                                 shape=(population.shape[0],),
                                 p=fitnesses,
                                 replace=True)

  return jnp.stack([population[i] for i in retain_members])


SimulationConfig = namedtuple("SimulationConfig", [
    "keep_full_population_history", "num_generations", "pop_size",
    "genome_length", "report_every", "mutation_rate", "alphabet_size"
])


def fitness_mean_value_target(v, target_value=1.0):
  return 1 - jnp.abs(jnp.mean(v) - target_value) / target_value


@jit
def batched_fitness_mean_value_target(population):
  return vmap(fitness_mean_value_target)(population)


def evolve_with_mutation(key,
                         fitness_fn,
                         keep_full_population_history=False,
                         num_generations=100,
                         pop_size=100,
                         genome_length=10,
                         report_every=10,
                         mutation_rate=0.15,
                         alphabet_size=4) -> EvolutionResult:

  base_mutation_rate = mutation_rate

  fitness_history = np.zeros(shape=(num_generations,))

  population_history = None
  if keep_full_population_history:
    population_history = np.zeros(shape=(num_generations, pop_size,
                                         genome_length))

  population = generate_population(key=key,
                                   length=genome_length,
                                   pop_size=pop_size)

  for i in range(num_generations):

    if keep_full_population_history:
      # Could keep track at any point in this loop, making sure
      # at least fitness_history and population_history are in
      # sync.
      population_history[i] = population

    fitnesses = fitness_fn(population)
    fitness_history[i] = jnp.mean(fitnesses)
    max_fitness = jnp.max(fitnesses)
    min_fitness = jnp.min(fitnesses)
    epsilon = 0.000000000000001
    norm_fitnesses = (fitnesses - min_fitness +
                      epsilon) / (max_fitness - min_fitness + epsilon)

    if i % report_every == 0.0:
      print("Current average fitness: %s" % jnp.mean(fitnesses))

    key, _ = random.split(key)
    population = resample_population(key=key,
                                     population=population,
                                     fitnesses=norm_fitnesses)

    mutation_rate = base_mutation_rate * (1 - jnp.mean(norm_fitnesses) / 2)

    key, _ = random.split(key)
    population = mutate_population(key, population, mutation_rate=mutation_rate)

  return fitness_history, population_history, population

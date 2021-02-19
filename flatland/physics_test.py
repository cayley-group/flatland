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
"""Tests of physics simulation utilities."""

from jax import random
from flatland import physics


def test_simulate_polymer_brownian():

  key = random.PRNGKey(0)

  polymer = random.randint(key, minval=0, maxval=4, shape=(10,))

  key, _ = random.split(key)

  positions, energy_fn = physics.simulate_polymer_brownian(
    key=key, polymer=polymer, box_size=6.8)

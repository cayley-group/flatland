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
"""Tests of logging and related."""

from flatland.log import TrainingLogger


def test_training_logger():

  log_every = 10
  num_training_steps = 100

  tl = TrainingLogger(
    log_every=log_every,
    num_training_steps=num_training_steps
  )

  for i in range(num_training_steps):
    if i % log_every == 0:
      tl.log(0.001, 0.0001, i)
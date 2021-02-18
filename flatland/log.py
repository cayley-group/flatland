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
"""Logging and related utilities."""

import logging


class TrainingLogger(object):
  """A custom logger for model training."""

  def __init__(self, log_every: int, num_training_steps: int):
    """Initialize the training logger.

    Notes:
    * Construct fixed-size lists for storing loss values for
    both training and testing.

    """

    self.log_every = log_every
    self.history_size = int(num_training_steps / log_every)
    self.test_history = [None for _ in range(self.history_size)]
    self.train_history = [None for _ in range(self.history_size)]

  def log(self, test_error: float, train_error: float, step_num: int):
    """Safely record error values for `record_every`-stepped array."""

    if step_num % self.log_every != 0:
      msg = "Trying to log error history not at a multiple of `log_every`."
      raise Exception(msg)

    write_idx = int(step_num / self.log_every)

    self.test_history[write_idx] = {"error": test_error,
                                    "step": step_num}
    self.train_history[write_idx] = {"error": train_error,
                                     "step": step_num}

    test_log_msg = "Test error is: %s" % (test_error)
    train_log_msg = "Train error is: %s" % (train_error)
    preamble = '=====\nAt step %s' % step_num

    logging.info("\n".join([preamble, test_log_msg, train_log_msg]))

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
"""Tests of Tensorflow Datasets dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds
from flatland import dataset


def test_can_load_example_batch():

  ds = tfds.load('flatland_mock', split="train")
  assert isinstance(ds, tf.data.Dataset)

  ds = ds.take(1).cache().repeat()
  for example in tfds.as_numpy(ds):
    break

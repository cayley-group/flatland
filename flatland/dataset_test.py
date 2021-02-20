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

import os
import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
from flatland import dataset


def test_can_load_mock_example():

  ds = tfds.load('flatland_mock', split="train")
  assert isinstance(ds, tf.data.Dataset)

  ds = ds.take(1).cache().repeat()
  for example in tfds.as_numpy(ds):
    break

def test_can_load_base_example():
  """Test an example can be loaded for the base dataset.
  
  This test requires FLATLAND_REQUESTER_PAYS_PROJECT_ID
  to be set in the test environment (command line or ci).

  """

  ds = tfds.load('flatland_base', split="train")
  assert isinstance(ds, tf.data.Dataset)

  ds = ds.cache().repeat()
  for example in tfds.as_numpy(ds):
    break

def test_flatland_base_e2e():

  if "FLATLAND_TEST_BUCKET" not in os.environ:
    msg = "Please specify a GCS bucket for testing via FLATLAND_TEST_BUCKET"
    raise Exception(msg)

  timestamp = int(datetime.datetime.utcnow().timestamp())

  # Override the default location of the production Flatland dataset
  # with a bucket that can be used for testing. This might even be one
  # that is newly created
  class TestFlatlandBase(dataset.FlatlandBase):

    VERSION = tfds.core.Version('0.1.%s' % timestamp)

    def sim_bucket_name(self):
      return os.environ["FLATLAND_TEST_BUCKET"]

  base = TestFlatlandBase()

  # Do a mock run for all shards for all of train/test/validation
  for split, split_size in base._sim_shard_sizes().items():
    for shard_id in range(split_size):
      base.simulate_dataset(split=split,
                            shard_id=shard_id)

  base.download_and_prepare()

  ds = base.as_dataset(split="train")

  ds = ds.cache().repeat()
  for example in tfds.as_numpy(ds):
    break

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
"""Flatland pre-training dataset for polymer structure meta-optimization."""

import os

from absl import logging

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import Iterator, Tuple

from jax import random

from flatland import utils
from flatland import evolution as evo
from flatland import datagen

_DESCRIPTION = """
The Flatland environment can be used to evolve datasets of the form used by current
state-of-the-art polymer structure prediction systems. This data includes input polymer
sequences, some form of evolutionary history, solved structures, and derived distance
matrices and bond angles. From a conventional perspective, this data presents an
opportunity to debug complex solver systems and to make it easier to get started in
this area of research.

From a more interesting perspective, we position this data to support next-generation
solver development including (1) learning meta-optimizers and (2) leveraging novel
forms of data such as strucutre-encoding (or "holography") datasets.

For the former, we provide solved structures with position and energy values -
thereby enabling pre-training on the problem of learned energy potentials before
transfer-learning to the problem of predicting position updates that minimize energy
(i.e. a domain-specific learned optimizer).

Simulated compound-polymer interaction affinities are also included for solved
structures. These may be used as model inputs and/or intermediate predictions to
condition latent representations.

Simulations performed here are in two dimensions over an alphabet of three (instead
of 21 in the case of protein polymers) with simplified evolutionary conditions.
More sophisticated simulations may be performed by way of the Flatland framework
directly (https://github.com/cayley-group/flatland).
"""

_CITATION = """
@misc{flatland2021,
    title={Flatland: A simulation environment for simplifying the development of 
           polymer structure prediction tools.},
    author={Christopher Beitel},
    year={2021},
    howpublished={\\url{https://github.com/cayley-group/flatland},
                  \\url{https://doi.org/10.5281/zenodo.4536540}},
    archivePrefix={Zenodo},
}
"""


def get_destination_blob_path(filename, test_train_validation, dataset_name,
                              dataset_version):
  """Return a blob path."""
  ttv = ["test", "train", "validation"]
  assert test_train_validation in ttv
  return os.path.join(dataset_name, dataset_version, test_train_validation,
                      filename)


def _assert_valid_split_name(split):
  assert split in ["train", "test", "validation"]


def _build_dataset_info(builder):
  return tfds.core.DatasetInfo(
      builder=builder,
      description=_DESCRIPTION,
      features=tfds.features.FeaturesDict({
          'aa_sequence':
              tfds.features.Sequence(tf.int32),
          'alignments':
              tfds.features.Sequence(tfds.features.Sequence(tf.int32)),
          'compound_affinity':
              tfds.features.Sequence(tf.float32),
          'solved_distances':
              tfds.features.Sequence(tf.float32),
          'solved_angles':
              tfds.features.Sequence(tf.float32),
          'structure_x':
              tfds.features.Sequence(tf.float32),
          'structure_y':
              tfds.features.Sequence(tf.float32),
          'structure_energy':
              tf.float32,
      }),
      supervised_keys=None,
      homepage='https://github.com/cayley-group/flatland',
      citation=_CITATION,
  )


def _build_split_generators(train_paths, test_paths, validation_paths):
  return [
      tfds.core.SplitGenerator(
          name=tfds.Split.TRAIN,
          gen_kwargs={'filepaths': train_paths},
      ),
      tfds.core.SplitGenerator(
          name=tfds.Split.TEST,
          gen_kwargs={'filepaths': test_paths},
      ),
      tfds.core.SplitGenerator(
          name=tfds.Split.VALIDATION,
          gen_kwargs={'filepaths': validation_paths},
      ),
  ]


class FlatlandMock(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for mock/randomly-generated Flatland dataset.

  This dataset is of the same shape as the rest of the Flatland datasets
  but where all values are randomly generated. This is useful for testing
  within the Flatland project but developers might also find it convenient
  to work with at the earliest stages of model development - e.g. when
  seeing if a new model can at least over-fit on a small random dataset.

  """

  VERSION = tfds.core.Version('0.0.1')

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return _build_dataset_info(builder=self)

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    return _build_split_generators(train_paths=None,
                                   test_paths=None,
                                   validation_paths=None)

  def simulation_config(self):
    return evo.SimulationConfig(alphabet_size=3,
                                pop_size=2,
                                genome_length=10,
                                mutation_rate=0.15,
                                num_generations=2,
                                keep_full_population_history=True,
                                report_every=10)

  def _generate_examples(self, filepaths) -> Iterator[Tuple[str, dict]]:
    """Generator of examples for each split."""

    logging.debug("Processing filepaths: %s" % filepaths)

    cfg = self.simulation_config()
    alphabet_size = cfg.alphabet_size
    polymer_length = cfg.genome_length
    num_alignments = 5
    num_examples = 1000
    num_compounds = 100

    aa_shape = (polymer_length,)
    alignments_shape = (
        num_alignments,
        polymer_length,
    )
    compounds_shape = (num_compounds,)

    distances_shape = (polymer_length**2,)

    for i in range(num_examples):
      yield str(i), {
          'aa_sequence': np.random.randint(0, alphabet_size, aa_shape),
          'alignments': np.random.randint(0, alphabet_size, alignments_shape),
          'compound_affinity': np.random.random(compounds_shape),
          'solved_distances': np.random.random(distances_shape),
          'solved_angles': np.random.random((polymer_length - 1,)),
          'structure_x': np.random.random((polymer_length,)),
          'structure_y': np.random.random((polymer_length,)),
          'structure_energy': np.random.random()
      }


class FlatlandBase(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for `flatland` dataset."""

  VERSION = tfds.core.Version('0.0.1')

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return _build_dataset_info(builder=self)

  def simulation_config(self):
    # To be converted to a namedtuple that is tested for integration with
    # the evolution portion of the code.
    return evo.SimulationConfig(alphabet_size=3,
                                pop_size=2,
                                genome_length=10,
                                mutation_rate=0.15,
                                num_generations=2,
                                keep_full_population_history=True,
                                report_every=10)

  def _sim_shard_sizes(self):
    return {"test": 1, "train": 1, "validation": 1}

  def _sim_shard_paths(self, split):

    assert split in ["test", "train", "validation"]
    num_sim_shards = self._sim_shard_sizes()[split]
    sim_paths = [None for _ in range(num_sim_shards)]

    for i in range(num_sim_shards):

      sim_path = get_destination_blob_path(filename="sim-%s.pkl" % i,
                                           test_train_validation=split,
                                           dataset_name=self.name,
                                           dataset_version=str(self.VERSION))

      sim_paths[i] = sim_path

    return sim_paths

  def _train_sim_paths(self):
    return self._sim_shard_paths("train")

  def _test_sim_paths(self):
    return self._sim_shard_paths("test")

  def _validation_sim_paths(self):
    return self._sim_shard_paths("validation")

  def simulate_dataset(self, split, shard_id):
    """Simulate one shard of the dataset."""

    # Make sure each of our simulation runs will be different by virtue of
    # using a different PRNG key.
    _assert_valid_split_name(split)
    split_to_int = {"train": 1, "test": 2, "validation": 3}
    key = random.PRNGKey(split_to_int[split] * shard_id)

    shard_sizes = self._sim_shard_sizes()
    assert split in shard_sizes.keys()
    assert shard_id < shard_sizes[split]

    bucket_name = self.sim_bucket_name()

    path_lookup = {
        "train": self._train_sim_paths,
        "test": self._test_sim_paths,
        "validation": self._validation_sim_paths
    }

    paths = path_lookup[split]()
    destination_path = paths[shard_id]

    key, subkey = random.split(key)

    _, _, population = evo.evolve_with_mutation(
        key=key,
        fitness_fn=evo.batched_fitness_mean_value_target,
        **dict(self.simulation_config()._asdict()))

    local_dataset_path = datagen.compile_dataset_for_population(
        subkey, population)

    utils.upload_blob(bucket_name=bucket_name,
                      source_file_name=local_dataset_path,
                      destination_blob_name=destination_path,
                      user_project_id=utils.get_requester_project())

  def sim_bucket_name(self):
    """The remote bucket in which to store the simulated dataset.
    
    To generate your own version of this or a related dataset, simply
    sub-class this object and provide here the name of your own GCS
    bucket. Then, when the simulation or example generation steps run,
    data will either be written or read from your bucket (respectively).

    """
    return "cg-pub"

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""

    # Data is coming from a requester-pays GCS bucket so we need a
    # a special DL manager.
    del dl_manager

    local_tmp_dir = "/tmp"  # Should allow the user to customize this.

    bucket_name = self.sim_bucket_name()
    requester_project = utils.get_requester_project()

    train_paths_remote = self._train_sim_paths()
    #utils.ensure_paths_exist(train_paths_remote)
    train_paths_local = utils.download_files_requester_pays(
        bucket_name=bucket_name,
        paths=train_paths_remote,
        requester_project=requester_project,
        local_tmp_dir=local_tmp_dir)

    test_paths_remote = self._test_sim_paths()
    #utils.ensure_paths_exist(test_paths_remote)
    test_paths_local = utils.download_files_requester_pays(
        bucket_name=bucket_name,
        paths=test_paths_remote,
        requester_project=requester_project,
        local_tmp_dir=local_tmp_dir)

    validation_paths_remote = self._validation_sim_paths()
    #utils.ensure_paths_exist(validation_paths_remote)
    validation_paths_local = utils.download_files_requester_pays(
        bucket_name=bucket_name,
        paths=validation_paths_remote,
        requester_project=requester_project,
        local_tmp_dir=local_tmp_dir)

    return _build_split_generators(train_paths=train_paths_local,
                                   test_paths=test_paths_local,
                                   validation_paths=validation_paths_local)

  def _generate_examples(self, filepaths) -> Iterator[Tuple[str, dict]]:
    """Generator of examples for each split.
    
    For now, generate dummy data of the same shape we intend to generate.
    """

    logging.debug("Processing filepaths: %s" % filepaths)

    alphabet_size = 4
    polymer_length = 10
    num_alignments = 5
    num_examples = 1000
    num_compounds = 100

    aa_shape = (polymer_length,)
    alignments_shape = (
        num_alignments,
        polymer_length,
    )
    compounds_shape = (num_compounds,)

    # One option would be to only pack a subset of the distance
    # matrix in the example tfrecord if only a subset will be used
    # during training. For now, this.
    distances_shape = (polymer_length**2,)

    for i in range(num_examples):
      yield str(i), {
          'aa_sequence': np.random.randint(0, alphabet_size, aa_shape),
          'alignments': np.random.randint(0, alphabet_size, alignments_shape),
          'compound_affinity': np.random.random(compounds_shape),
          'solved_distances': np.random.random(distances_shape),
          'solved_angles': np.random.random((polymer_length - 1,)),
          'structure_x': np.random.random((polymer_length,)),
          'structure_y': np.random.random((polymer_length,)),
          'structure_energy': np.random.random()
      }

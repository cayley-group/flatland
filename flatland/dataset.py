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

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import Iterator, Tuple

import os

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

_DATA_DIR = tfds.core.as_path('gs://flatland-public/small')


class FlatlandMock(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for `flatland` dataset."""

  VERSION = tfds.core.Version('0.0.1')

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
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
        supervised_keys=None,  # e.g. ('image', 'label')
        homepage='https://github.com/cayley-group/flatland',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    del dl_manager

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={'filepath': os.path.join(_DATA_DIR, 'train')},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={'filepath': os.path.join(_DATA_DIR, 'validation')},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={'filepath': os.path.join(_DATA_DIR, 'test')},
        ),
    ]

  def _generate_examples(self, filepath) -> Iterator[Tuple[str, dict]]:
    """Generator of examples for each split.
    
    For now, generate dummy data of the same shape we intend to generate.
    """

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

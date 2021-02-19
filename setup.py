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

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

setuptools.setup(
    name="cg-flatland",
    version="0.0.1",
    author="The Cayley Group",
    author_email="contact@cayleygroup.org",
    description="Easier development of polymer structure prediction tools.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cayleygroup/flatland",
    license="Apache 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "tensorflow_datasets>=4.2.0",
        "tensorflow>=2.4.0",
        "optax>=0.0.2",
        "jax-md",
        "absl-py",
        "matplotlib",
        "jax"  # May need to include as an extra since the relevant Jax version
        # should depend on the version of CUDA installed?
    ],
    extras_require={
        #'tensorflow': ['tensorflow'], # In the future tensorflow import here
        # because host may already have tensorflow-gpu?
        "test": ["pytest", "yapf", "jupyter", "pytest-cov"],
    },
    keywords="protein folding",
)

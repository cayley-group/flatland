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
"""Test a Jupyter notebook runs without error."""

import subprocess
import argparse
import sys
import tempfile
import json
import nbformat
import os

from nbconvert.preprocessors import ExecutePreprocessor

def _run(command):
  
  process = subprocess.Popen(
    command, stdout=subprocess.PIPE
  )

  output = []

  for line in process.stdout:
    line = line.decode("utf-8")
    sys.stdout.write(line)
    output.append(line)

  return output


def get_a_jupyter_kernel_name():
  """Get the name of the first available Jupyter kernelspec (if any)."""

  command = ["jupyter", "kernelspec", "list", "--json"]
  output = _run(command)
  output = "".join(output)

  data = json.loads(output)

  if data and "kernelspecs" in data:
    for key, value in data["kernelspecs"].items():
      return key

  # If we didn't find any, return None
  return None


def test_run(notebook_path):
  """Adapted from: https://www.blog.pythonlibrary.org/2018/10/16/testing-jupyter-notebooks/"""

  kernel_name = get_a_jupyter_kernel_name()

  output_path = tempfile.NamedTemporaryFile().name

  nb_name, _ = os.path.splitext(os.path.basename(notebook_path))
  dirname = os.path.dirname(notebook_path)

  with open(notebook_path) as f:
    nb = nbformat.read(f, as_version=4)

  proc = ExecutePreprocessor(timeout=600, kernel_name=kernel_name)
  proc.allow_errors = True

  proc.preprocess(nb, {'metadata': {'path': '/'}})

  with open(output_path, mode='wt') as f:
    nbformat.write(nb, f)

  errors = []
  for cell in nb.cells:
    if 'outputs' in cell:
      for output in cell['outputs']:
        if output.output_type == 'error':
          errors.append(output)

  return nb, errors


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Test run a Jupyter notebook.')

  parser.add_argument('--path', type=str, default=None,
                      required=True,
                      help='Path to the Jupyter notebook to test-run.')

  args = parser.parse_args()

  nb, errors = test_run(str(args.path))
  
  if errors:
    raise Exception(errors)

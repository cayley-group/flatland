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


def test_run(path):

  kernel_name = get_a_jupyter_kernel_name()

  temp_output = tempfile.NamedTemporaryFile()

  command = ["jupyter", "nbconvert", "--to", "notebook",
             "--execute", "--ExecutePreprocessor.kernel_name='%s'" % kernel_name,
             "--output", temp_output.name, path]

  output = _run(command)

  print(output)
  
  # TODO: Check output for errors in the way described in 
  # https://www.blog.pythonlibrary.org/2018/10/16/testing-jupyter-notebooks/
  has_error = False

  return has_error, output


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Test run a Jupyter notebook.')

  parser.add_argument('--path', type=str, default=None,
                      required=True,
                      help='Path to the Jupyter notebook to test-run.')

  args = parser.parse_args()

  has_error, output = test_run(str(args.path))
  
  if has_error:
    raise Exception(output)

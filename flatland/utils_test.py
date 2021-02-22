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
"""Tests of utils."""

import os

from flatland import utils


def test_get_requester_project():

  test_project_name = "hello_world"

  os.environ["FLATLAND_REQUESTER_PAYS_PROJECT_ID"] = test_project_name

  project_name = utils.get_requester_project()

  assert project_name == test_project_name


def test_download_files_requester_pays():
  pass

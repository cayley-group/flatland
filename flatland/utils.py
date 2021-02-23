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
"""General utilities."""

import os
import logging

from typing import List

from google.cloud import storage


def upload_blob(user_project_id, bucket_name, source_file_name,
                destination_blob_name):
  """Uploads a file to the bucket.
  
  Args:
    user_project_id: A project to bill for requester-pays operations.

  """

  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name, user_project=user_project_id)
  blob = bucket.blob(destination_blob_name)

  blob.upload_from_filename(source_file_name)

  logging.info("File {} uploaded to {}.".format(source_file_name,
                                                destination_blob_name))


def get_requester_project():

  env_tag = "FLATLAND_REQUESTER_PAYS_PROJECT_ID"

  env = os.environ

  if env_tag not in os.environ:
    msg = "The environment variable %s must be set " % env_tag
    msg += "to a valid GCP project ID. Setting this variable "
    msg += "as such acknowledges that the specified project will "
    msg += "be billed for any expenses incurred by accessing the "
    msg += "referenced GCP resource."
    raise Exception(msg)

  return os.environ[env_tag]


def download_file_requester_pays(bucket_name: str, project_id: str,
                                 source_blob_name: str,
                                 destination_file_name: str):
  """Download file using specified project as the requester."""

  storage_client = storage.Client()

  bucket = storage_client.bucket(bucket_name, user_project=project_id)
  blob = bucket.blob(source_blob_name)
  blob.download_to_filename(destination_file_name)

  logging.info(
      "Blob {} downloaded to {} using a requester-pays request.".format(
          source_blob_name, destination_file_name))


def download_files_requester_pays(bucket_name: str, paths: List[str],
                                  requester_project: str, local_tmp_dir: str):
  """Download a collection of files from a requester-pays GCS bucket.

  Notable here is that `local_tmp_dir` specifies a root tmp data directory
  and files will be downloaded to sub-directory paths of this that match
  those of the source paths.

  Args:
    bucket_name: The name of the GCS bucket from which to download.
    paths: A list of paths within the GCS bucket to objects to download.
    requester_project: The project to bill for access.
    local_tmp_dir: The root local directory to which to download files.

  """

  local_file_paths = [None for _ in paths]

  for i, path in enumerate(paths):

    if path[0] == "/":
      path = path[1:]

    local_path = os.path.join(local_tmp_dir, path)

    dir_path, filename = os.path.split(local_path)
    os.makedirs(dir_path, exist_ok=True)

    download_file_requester_pays(bucket_name=bucket_name,
                                 project_id=requester_project,
                                 source_blob_name=path,
                                 destination_file_name=local_path)

    local_file_paths[i] = local_path

  return local_file_paths

from __future__ import unicode_literals
import hashlib
import io
import logging
import time
import requests
import os.path

URL_BASE = "https://www.bitballoon.com/api/v1"


def get_sha1(path):
    with io.open(path, "rb") as file_in:
        return hashlib.sha1(file_in.read()).hexdigest()

def with_base(*url_parts):
    return "/".join([URL_BASE] + list(url_parts))


def compute_file_prefix_length(files):
    indexes = [f for f in files if f.endswith("index.html")]
    if indexes:
        index = min(indexes, key=len)
        return index.find("index.html")
    elif len(files) > 1:
        common_prefix = os.path.commonprefix(files)
        return len(common_prefix)
    else:
        dir, filename = os.path.split(files[0])
        return len(dir) + 1


def preprocess_files(files):
    strip_len = compute_file_prefix_length(files)

    for path in files:
        yield path, path[strip_len:], get_sha1(path)



class BitBalloon(object):
    def __init__(self, access_token, site_id, email_address):
        self.access_token = access_token
        self.site_id = site_id
        self.email_address = email_address

        self.headers = {"User-Agent": "({})".format(self.email_address)}
        self.params = {"access_token": self.access_token}
        self.logger = logging.getLogger(__name__)

    def list_files(self):
        r = requests.get("/".join([URL_BASE, "sites", self.site_id, "files"]), headers=self.headers, params=self.params)
        r.raise_for_status()
        return r.json()

    def update_file_data(self, page_data, remote_path, deploy=False):
        """
        :param page_data: The binary/str page data
        :param remote_path: Server-side path
        :return: Response data
        """
        url = "/".join([URL_BASE, "sites", self.site_id, "files", remote_path])
        r = requests.put(url, page_data, headers=self.headers, params=self.params)
        r.raise_for_status()

        return_data = {"upload": r.json()}

        if deploy:
            return_data["deploy"] = self.deploy(return_data["upload"]["deploy_id"])

        return return_data

    def deploy(self, deploy_id):
        url = "/".join([URL_BASE, "sites/{site_id}/deploys/{deploy_id}/restore".format(site_id=self.site_id,
                                                                                        deploy_id=deploy_id)])
        r = requests.post(url, headers=self.headers, params=self.params)
        r.raise_for_status()
        return r.json()

    def deploy_files(self, files):
        remote_name_to_sha1 = dict()
        remote_name_to_local = dict()
        sha1_to_remote = dict()
        for local_path, remote_path, sha1 in preprocess_files(files):
            remote_name_to_sha1[remote_path] = sha1
            remote_name_to_local[remote_path] = local_path
            sha1_to_remote[sha1] = remote_path

        request_data = {"files": remote_name_to_sha1}
        self.logger.info("Deploy request: %s", request_data)

        url = with_base("sites/{site_id}/deploys".format(site_id=self.site_id))

        # step 1: post the list of files and SHA1
        r = requests.post(url, headers=self.headers, params=self.params, json=request_data)
        self.logger.info("Response: %s", r.text)
        r.raise_for_status()
        self.logger.info("Deploy response: {}".format(r.json()))
        deploy_data = r.json()
        deploy_id = deploy_data["id"]
        required_hashes = deploy_data["required"]

        # step 2: upload the required files
        for required_hash in required_hashes:
            remote_path = sha1_to_remote[required_hash]
            local_path = remote_name_to_local[remote_path]

            url = with_base("deploys/{deploy_id}/files/{filename}".format(deploy_id=deploy_id, filename=remote_path))
            with io.open(local_path, "rb") as file_in:
                byte_data = file_in.read()
                headers = {"content-type": "application/octet-stream"}
                headers.update(self.headers)
                r = requests.put(url, headers=headers, params=self.params, data=byte_data)
                r.raise_for_status()
                self.logger.info("Response to uploading {}: {}".format(remote_path, r.json()))

        time.sleep(5)
        url = with_base("deploys/{deploy_id}".format(deploy_id=deploy_id))
        r = requests.get(url, headers=self.headers, params=self.params)
        r.raise_for_status()
        self.logger.info("Deploy status: %s", r.json())

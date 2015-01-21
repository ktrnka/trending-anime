from __future__ import unicode_literals
import requests

URL_BASE = "https://www.bitballoon.com/api/v1"


class BitBalloon(object):
    def __init__(self, access_token, site_id, email_address):
        self.access_token = access_token
        self.site_id = site_id
        self.email_address = email_address

        self.headers = {"User-Agent": "({})".format(self.email_address)}
        self.params = {"access_token": self.access_token}

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
        url = "/".join([URL_BASE, "/sites/{site_id}/deploys/{deploy_id}/restore".format(site_id=self.site_id,
                                                                                        deploy_id=deploy_id)])
        r = requests.post(url, headers=self.headers, params=self.params)
        r.raise_for_status()
        return r.json()

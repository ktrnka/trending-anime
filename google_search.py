import logging
import requests

__author__ = 'keith'


class SearchEngine(object):
    """Wrap Google Custom Search Engine requests"""

    def __init__(self, api_key, cx, mongo_db, max_requests=10):
        self.api_key = api_key
        self.cx = cx
        self.url = "https://www.googleapis.com/customsearch/v1"

        self.max_requests = max_requests
        self.requests = 0
        self.logger = logging.getLogger(__name__)
        self.db = mongo_db

    def get_link(self, series_name):
        cached = self.db["links"].find_one({"title": series_name})
        if cached:
            self.logger.debug("Found cached object in database: %s", cached)
            return cached["url"]

        if self.requests >= self.max_requests:
            return None

        try:
            r = requests.get(self.url, params={"key": self.api_key, "cx": self.cx, "q": series_name})
            self.logger.info("Request URL: {}".format(r.url))
            r.raise_for_status()
        except (requests.exceptions.HTTPError, requests.exceptions.SSLError):
            self.logger.exception("Failed to query API, skipping further requests")
            self.requests = self.max_requests
            return None

        self.requests += 1
        data = r.json()
        first_result = data["items"][0]["link"]
        self.logger.info("Link for {}: {}".format(series_name, first_result))
        self.db["links"].insert({"title": series_name, "url": first_result})
        return first_result
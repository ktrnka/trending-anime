from __future__ import unicode_literals
import json
import logging
import collections
import requests
import datetime

__author__ = 'keith'


def refine_query(initial_query):
    """Tweak the search string for higher accuracy"""
    return "{} {}".format(initial_query, datetime.datetime.now().year)


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

        self.stats = collections.Counter()

    def get_link(self, series_name):
        cached = self.db["links"].find_one({"title": series_name})
        if cached:
            self.logger.debug("Found cached object in database: %s", cached)
            self.stats["cached search"] += 1
            return cached["url"]

        if self.requests >= self.max_requests:
            self.stats["skipped, no more requests"] += 1
            return None

        try:
            r = requests.get(self.url, params={"key": self.api_key, "cx": self.cx, "q": refine_query(series_name)})
            self.logger.debug("Request URL: {}".format(r.url))
            r.raise_for_status()
        except (requests.exceptions.HTTPError, requests.exceptions.SSLError):
            self.logger.exception("Failed to query API, skipping further requests")
            self.requests = self.max_requests
            self.stats["search exception"] += 1
            return None

        self.requests += 1
        data = r.json()

        # not found, but maybe another day
        if int(data["searchInformation"]["totalResults"]) == 0:
            self.stats["search string not found"] += 1
            return None

        try:
            first_result = data["items"][0]["link"]
        except KeyError:
            self.logger.exception("Bad search result from %s: %s", series_name, json.dumps(data))
        self.logger.debug("Link for {}: {}".format(series_name, first_result))
        self.db["links"].insert({"title": series_name, "url": first_result})
        self.stats["search success"] += 1
        return first_result

    def log_summary(self):
        for reason, count in self.stats.most_common():
            self.logger.info("%s: %d", reason, count)
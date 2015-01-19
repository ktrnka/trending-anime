from __future__ import unicode_literals

import argparse
import collections
import io
import json
import logging
import sys
import os
import re
import datetime

import requests
import pymongo
import string

BASIC_PATTERN = re.compile(r"\[([^]]+)\] (.+) - (\d+)\s*(?:\[([\d]+p)\])?.(\w+)")
MD5_PATTERN = re.compile(r"\s*\[[0-9A-F]+\]{6,}\s*", re.IGNORECASE)
NORMALIZATION_MAP = {ord(c): None for c in "/:;._ -'\"!,~()"}


class Templates(object):
    """Dict of templates"""
    def __init__(self, template_dir):
        self.templates = dict()

        for filename in os.listdir(template_dir):
            name = os.path.splitext(os.path.basename(filename))[0]
            filename = os.path.join(template_dir, filename)

            with io.open(filename, "r", encoding="UTF-8") as template_in:
                self.templates[name] = string.Template(template_in.read())

    def sub(self, template_name, **kwargs):
        return self.templates[template_name].substitute(**kwargs)


class SearchEngine(object):
    """Wrap Google Custom Search Engine requests"""
    def __init__(self, mongo_db, max_requests=10):
        self.api_key = "***REMOVED***"
        self.cx = "***REMOVED***"
        self.url = "https://www.googleapis.com/customsearch/v1"

        self.max_requests = max_requests
        self.requests = 0
        self.logger = logging.getLogger(__name__)
        self.db = mongo_db

    def get_link(self, series_name):
        cached = self.db["links"].find_one({"title": series_name})
        if cached:
            self.logger.info("Found cached object in database: %s", cached)
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

    def linkify(self, series_name):
        url = self.get_link(series_name)
        if url:
            return '<a href="{}">{}</a>'.format(url, series_name)
        return series_name


def format_row(series_names, downloads, max_downloads, search_engine, html_templates):
    best_name = series_names.most_common(1)[0][0]
    return html_templates.sub("row",
                              series_name=search_engine.linkify(best_name),
                              value="{:,}".format(downloads),
                              bar_width=int(520. * downloads / max_downloads))


def get_title_key(title):
    return title.lower().translate(NORMALIZATION_MAP)


class Episode(object):
    def __init__(self, series, episode, sub_group, resolution, file_type):
        self.series = series
        self.episode = int(episode)
        self.sub_group = sub_group
        self.resolution = resolution
        self.file_type = file_type

    @staticmethod
    def from_name(title):
        title_cleaned = MD5_PATTERN.sub("", title)
        title_cleaned = title_cleaned.translate({0x2012: "-"})
        m = BASIC_PATTERN.match(title_cleaned)
        if m:
            return Episode(m.group(2), m.group(3), m.group(1), m.group(4), m.group(5))
        else:
            return None

    def __repr__(self):
        return "[{}] {} - {} [{}].{}".format(self.sub_group, self.series, self.episode, self.resolution, self.file_type)


def _parse_size(size_string):
    parts = size_string.split()
    if not parts:
        return -1

    if parts[1] == "MiB":
        return float(parts[0])
    elif parts[1] == "GiB":
        return float(parts[0]) * 1024

    return -1


class Torrent(object):
    def __init__(self):
        self.title = None
        self.url = None
        self.seeders = 0
        self.leechers = 0
        self.downloads = 0
        self.size = 0

    @classmethod
    def from_json(cls, data):
        torrent = Torrent()
        torrent.title = data["title"]["text"]
        torrent.url = data["title"]["href"]

        try:
            torrent.seeders = int(data["seeders"])
        except ValueError:
            torrent.seeders = 0

        try:
            torrent.leechers = int(data["leechers"])
        except ValueError:
            torrent.leechers = 0
        torrent.downloads = int(data["downloads"])
        torrent.size = _parse_size(data["size"])
        return torrent


def load(endpoint):
    logger = logging.getLogger(__name__)
    torrents = []
    with io.open(endpoint, "rb") as json_in:
        data = json.load(json_in)

        for torrent in data["results"]["collection1"]:
            torrents.append(Torrent.from_json(torrent))
    logger.info("Loaded %d torrents", len(torrents))

    return torrents


def make_table(series_downloads, spelling_map, html_templates):
    mongo_client = pymongo.MongoClient("***REMOVED***")
    search_engine = SearchEngine(mongo_client["anime-trends"])
    top_series = series_downloads.most_common()
    data_entries = [format_row(spelling_map[series], downloads, top_series[0][1], search_engine, html_templates) for
                    series, downloads in top_series]
    return "\n".join(data_entries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--template_dir", default="templates", help="Dir of templates")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Verbose logging")
    parser.add_argument("endpoint", help="Json file or web location to fetch data")
    parser.add_argument("output", help="Output webpage")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    logger = logging.getLogger(__name__)

    templates = Templates(args.template_dir)

    torrents = load(args.endpoint)

    resolutions = collections.Counter()
    spelling_map = collections.defaultdict(collections.Counter)

    series_downloads = collections.Counter()
    episode_numbers = collections.defaultdict(set)

    who_subs = collections.defaultdict(collections.Counter)

    parse_fail = collections.Counter()

    success = collections.Counter()

    for torrent in torrents:
        episode = Episode.from_name(torrent.title)
        if episode:
            episode_key = get_title_key(episode.series)

            resolutions[episode.resolution] += torrent.downloads

            spelling_map[episode_key][episode.series] += torrent.downloads

            series_downloads[episode_key] += torrent.downloads
            episode_numbers[episode_key].add(episode.episode)
            who_subs[episode_key][episode.sub_group] += torrent.downloads
            success[True] += torrent.downloads
        elif torrent.size > 1000 or "OVA" in torrent.title:
            # success[False] += torrent.downloads
            pass
        else:
            parse_fail[torrent.title] += torrent.downloads
            success[False] += torrent.downloads

    logger.debug("Parsed {:.1f}% of downloads".format(100. * success[True] / (success[True] + success[False])))
    logger.debug("Failed to parse %s", parse_fail.most_common(40))

    for series in series_downloads.iterkeys():
        if episode_numbers[series]:
            series_downloads[series] /= len(episode_numbers[series])

    for series, downloads in series_downloads.most_common():
        best_title, _ = spelling_map[series].most_common(1)[0]
        # print "{}: {:,} downloads per episode".format(best_title, downloads)
        # print "\tEpisodes {}".format(", ".join(unicode(i) for i in sorted(episode_numbers[series])))
        #
        # if len(spelling_map[series]) > 1:
        # print "\tSpellings: {}".format(", ".join(t for t, v in spelling_map[series].most_common()))
        #
        # print "\tSub groups: {}".format(", ".join(g for g, _ in who_subs[series].most_common()))

    with io.open(args.output, "w", encoding="UTF-8") as html_out:
        table_data = make_table(series_downloads, spelling_map, templates)
        html_out.write(templates.sub("main",
                                     refreshed_timestamp=datetime.datetime.now().strftime("%A, %B %d"),
                                     table_body=table_data))


if __name__ == "__main__":
    sys.exit(main())
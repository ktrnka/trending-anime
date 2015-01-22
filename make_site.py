from __future__ import unicode_literals
import ConfigParser
import argparse
import collections
import io
import json
import logging
import sys
import shutil
import datetime
import string
import bitballoon

import os
import re
import requests
import pymongo

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


def format_row(series, top_series, search_engine, html_templates):
    best_name = series.spelling_counts.most_common(1)[0][0]
    return html_templates.sub("row",
                              series_name=search_engine.linkify(best_name),
                              value="{:,}".format(series.num_downloads),
                              bar_width=int(520. * series.num_downloads / top_series.num_downloads))


def get_title_key(title):
    return title.lower().translate(NORMALIZATION_MAP)


class ParsedTorrent(object):
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
            return ParsedTorrent(m.group(2), m.group(3), m.group(1), m.group(4), m.group(5))
        else:
            return None

    def __repr__(self):
        return "[{}] {} - {} [{}].{}".format(self.sub_group, self.series, self.episode, self.resolution, self.file_type)


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
        torrent.size = Torrent._parse_nyaa_size_string(data["size"])
        return torrent

    @staticmethod
    def _parse_nyaa_size_string(size_string):
        parts = size_string.split()
        if not parts:
            return -1

        if parts[1] == "MiB":
            return float(parts[0])
        elif parts[1] == "GiB":
            return float(parts[0]) * 1024

        return -1


class Series(object):
    def __init__(self):
        self.episodes = set()
        self.sub_groups = set()
        self.num_downloads = 0
        self.spelling_counts = collections.Counter()

    def get_season(self):
        assert False




def parse_timestamp(timestamp):
    """Parse a timestamp like Fri Jan 02 2015 22:04:11 GMT+0000 (UTC)"""
    timestamp = timestamp.replace(" GMT+0000 (UTC)", "")
    return datetime.datetime.strptime(timestamp, "%a %b %d %Y %H:%M:%S")


def load(endpoint):
    logger = logging.getLogger(__name__)
    torrents = []
    if endpoint.startswith("https:"):
        r = requests.get(endpoint)
        r.raise_for_status()
        data = r.json()
    else:
        with io.open(endpoint, "rb") as json_in:
            data = json.load(json_in)

    for torrent in data["results"]["collection1"]:
        torrents.append(Torrent.from_json(torrent))
    logger.info("Loaded %d torrents", len(torrents))

    return parse_timestamp(data["lastsuccess"]), torrents


def make_table_body(series, html_templates, search_engine):
    top_series = sorted(series.itervalues(), key=lambda s: s.num_downloads, reverse=True)
    data_entries = [format_row(anime, top_series[0], search_engine, html_templates) for
                    anime in top_series]
    return "\n".join(data_entries)


def process_torrents(torrents):
    """

    :param torrents:
    :return: Mapping of normalized series names to Series objects
    """
    logger = logging.getLogger(__name__)

    success_counts = collections.Counter()
    parse_fail = collections.Counter()

    animes = dict()

    for torrent in torrents:
        episode = ParsedTorrent.from_name(torrent.title)
        if episode:
            episode_key = get_title_key(episode.series)
            if episode_key in animes:
                anime = animes[episode_key]
            else:
                anime = Series()
                animes[episode_key] = anime

            anime.spelling_counts[episode.series] += torrent.downloads
            anime.num_downloads += torrent.downloads
            anime.episodes.add(episode.episode)
            anime.sub_groups.add(episode.sub_group)

            success_counts[True] += torrent.downloads
        elif torrent.size > 1000 or "OVA" in torrent.title:
            pass
        else:
            parse_fail[torrent.title] += torrent.downloads
            success_counts[False] += torrent.downloads
    logger.debug("Parsed {:.1f}% of downloads".format(100. * success_counts[True] / (success_counts[True] + success_counts[False])))
    logger.debug("Failed to parse %s", parse_fail.most_common(40))

    return animes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--template_dir", default="templates", help="Dir of templates")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Verbose logging")
    parser.add_argument("--style-file", default="over9000.css", help="CSS style")
    parser.add_argument("config", help="Config file")
    parser.add_argument("output", help="Output filename or 'bitballoon' to upload to bitballoon")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # load config
    config = ConfigParser.RawConfigParser()
    config.read([args.config])

    # load templates
    templates = Templates(args.template_dir)

    # load torrent list
    data_date, torrents = load(config.get("kimono", "endpoint"))

    # mongodb, google search
    mongo_client = pymongo.MongoClient(config.get("mongo", "uri"))
    search_engine = SearchEngine(config.get("google", "api_key"),
                                 config.get("google", "cx"),
                                 mongo_client["anime-trends"])

    bb = bitballoon.BitBalloon(config.get("bitballoon", "access_key"),
                               config.get("bitballoon", "site_id"),
                               config.get("bitballoon", "email"))

    animes = process_torrents(torrents)

    for anime in animes.itervalues():
        if anime.episodes:
            anime.num_downloads /= len(anime.episodes)

    animes = collections.Counter({k: v for k, v in animes.iteritems() if v.num_downloads > 1000})

    table_data = make_table_body(animes, templates, search_engine)
    html_data = templates.sub("main",
                              refreshed_timestamp=data_date.strftime("%A, %B %d"),
                              table_body=table_data)

    if args.output == "bitballoon":
        bb.update_file_data(html_data.encode("UTF-8"), "index.html", deploy=True)
    else:
        with io.open(args.output, "w", encoding="UTF-8") as html_out:
            html_out.write(html_data)

        dest_dir = os.path.dirname(args.output)
        if dest_dir:
            shutil.copy(args.style_file, os.path.join(dest_dir, args.style_file))


if __name__ == "__main__":
    sys.exit(main())
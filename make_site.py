from __future__ import unicode_literals
import ConfigParser
import argparse
import collections
import io
import json
import logging
import pprint
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
SEASONS = ["Winter season", "Spring season", "Summer season", "Fall season", "Long running show"]
SEASON_IMAGES = ["winter.svg", "spring.svg", "summer.svg", "fall.svg", "infinity.svg"]

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


def format_row(index, series, top_series, html_templates):
    extras = []
    alternate_names = series.get_alternate_names()
    if alternate_names:
        extras.append("Alternate titles: {}".format(", ".join(alternate_names)))

    extras.append("Sub groups: {}".format(", ".join(series.get_sub_groups())))

    extras.append(", ".join("Episode {}: {:,}".format(ep, count) for ep, count in series.get_episode_counts()))

    extras.append(", ".join("Episode {}: {}".format(ep, date) for ep, date in series.episode_dates.iteritems()))

    season_images = "".join(html_templates.sub("season_image", image=SEASON_IMAGES[season], season=SEASONS[season]) for season in series.get_seasons())

    return html_templates.sub("row",
                              id="row_{}".format(index),
                              season_images=season_images,
                              extra="<br>".join(extras),
                              series_name=series.get_link(),
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
        self.release_date = None

    @classmethod
    def from_json(cls, data):
        """Parse Json from either endpoint"""
        torrent = Torrent()

        if "title" in data:
            torrent.title = data["title"]["text"]
            torrent.url = data["title"]["href"]
        elif "name" in data:
            torrent.title = data["name"]
            torrent.url = None

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

        if "date" in data:
            torrent.release_date = Torrent._parse_nyaa_date(data["date"])

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

    @staticmethod
    def _parse_nyaa_date(date_string):
        """Parse a date like 2015-01-21, 18:42 UTC"""
        return datetime.datetime.strptime(date_string, "%Y-%m-%d, %H:%M %Z")


def date_to_season(date):
    """Return season as a number from 0-3"""
    if date.month <= 3:
        return 0
    elif date.month <= 6:
        return 1
    elif date.month <= 9:
        return 2
    else:
        return 3


class Series(object):
    def __init__(self):
        self.num_downloads = 0
        self.spelling_counts = collections.Counter()
        self.url = None
        self.episode_counts = collections.Counter()
        self.sub_group_counts = collections.Counter()
        self.episode_dates = dict()

    def add_torrent(self, torrent, parsed_torrent):
        self.spelling_counts[parsed_torrent.series] += torrent.downloads
        self.num_downloads += torrent.downloads

        self.episode_counts[parsed_torrent.episode] += torrent.downloads
        self.sub_group_counts[parsed_torrent.sub_group] += torrent.downloads

    def add_release_date_torrent(self, torrent, parsed_torrent):
        logger = logging.getLogger(__name__)
        if not torrent.release_date:
            return

        if parsed_torrent.episode in self.episode_dates:
            if torrent.downloads > 1000 and torrent.release_date < self.episode_dates[parsed_torrent.episode]:
                logger.info("Updating release date of %s, episode %d from %s tp %s", self.get_name(), parsed_torrent.episode, self.episode_dates[parsed_torrent.episode], torrent.release_date)
                self.episode_dates[parsed_torrent.episode] = torrent.release_date
        else:
            self.episode_dates[parsed_torrent.episode] = torrent.release_date

    def get_name(self):
        """Get the best name for this anime"""
        assert len(self.spelling_counts) > 0
        return self.spelling_counts.most_common(1)[0][0]

    def get_alternate_names(self):
        assert len(self.spelling_counts) > 0
        return [name for name, _ in self.spelling_counts.most_common()[1:]]

    def get_link(self):
        name = self.get_name()
        if self.url:
            return '<a href="{}">{}</a>'.format(self.url, name)
        return name

    def normalize_counts(self):
        if self.episode_counts:
            self.num_downloads /= len(self.episode_counts)

    def __repr__(self):
        return "{} {:,} DL".format(self.get_name(), self.num_downloads)

    def get_episode_counts(self):
        return [(ep, self.episode_counts[ep]) for ep in sorted(self.episode_counts.iterkeys())]

    def get_sub_groups(self):
        return [name for name, _ in self.sub_group_counts.most_common()]

    def merge(self, other):
        logger = logging.getLogger(__name__)
        logger.info("Merging by URL %s and %s", self, other)
        self.num_downloads += other.num_downloads
        self.spelling_counts.update(other.spelling_counts)
        self.episode_counts.update(other.episode_counts)
        self.sub_group_counts.update(other.sub_group_counts)

        for episode, release_date in other.episode_dates.iteritems():
            if episode in self.episode_dates:
                self.episode_dates[episode] = min(self.episode_dates[episode], release_date)
            else:
                self.episode_dates[episode] = release_date

    def get_seasons(self):
        if not self.episode_dates:
            return []

        max_episode = max(self.episode_dates.iterkeys())

        computed_dates = dict()
        for episode in xrange(1, max_episode + 1):
            if episode in self.episode_dates:
                computed_dates[episode] = self.episode_dates[episode]
            else:
                estimated_dates = [d + datetime.timedelta((episode - e) * 7) for e, d in self.episode_dates.iteritems()]
                computed_dates[episode] = min(estimated_dates)

        season_counts = collections.Counter(date_to_season(d) for d in computed_dates.itervalues())
        if len(season_counts) > 2:
            return [4]
        else:
            return sorted(season_counts.iterkeys())


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


def make_table_body(series, html_templates):
    top_series = sorted(series, key=lambda s: s.num_downloads, reverse=True)
    data_entries = [format_row(i, anime, top_series[0], html_templates) for i, anime in enumerate(top_series)]
    return "\n".join(data_entries)


def process_torrents(torrents, release_date_torrents):
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

            anime.add_torrent(torrent, episode)

            success_counts[True] += torrent.downloads
        elif torrent.size > 1000 or "OVA" in torrent.title:
            pass
        else:
            parse_fail[torrent.title] += torrent.downloads
            success_counts[False] += torrent.downloads
    logger.debug("Parsed {:.1f}% of downloads".format(100. * success_counts[True] / (success_counts[True] + success_counts[False])))
    logger.debug("Failed to parse %s", parse_fail.most_common(40))

    # add release dates when possible
    for torrent in release_date_torrents:
        episode = ParsedTorrent.from_name(torrent.title)
        if episode:
            episode_key = get_title_key(episode.series)
            if episode_key in animes:
                animes[episode_key].add_release_date_torrent(torrent, episode)

    return animes


def update_mal_link(anime, search_engine):
    anime.url = search_engine.get_link(anime.get_name())


def merge_by_link(animes):
    logger = logging.getLogger(__name__)
    filtered_anime = []

    url_map = dict()
    for anime in animes:
        if not anime.url:
            logger.info("No url, can't deduplicate %s", anime.get_name())
            filtered_anime.append(anime)
        elif anime.url in url_map:
            url_map[anime.url].merge(anime)
        else:
            url_map[anime.url] = anime

    return filtered_anime + url_map.values()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--template_dir", default="templates", help="Dir of templates")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Verbose logging")
    parser.add_argument("--style-file", default="res/over9000.css", help="CSS style")
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

    # load release dates
    release_data_date, release_date_torrents = load(config.get("kimono", "release_date_endpoint"))

    # mongodb, google search
    mongo_client = pymongo.MongoClient(config.get("mongo", "uri"))
    search_engine = SearchEngine(config.get("google", "api_key"),
                                 config.get("google", "cx"),
                                 mongo_client["anime-trends"])

    bb = bitballoon.BitBalloon(config.get("bitballoon", "access_key"),
                               config.get("bitballoon", "site_id"),
                               config.get("bitballoon", "email"))

    animes = process_torrents(torrents, release_date_torrents)

    for anime in animes.itervalues():
        anime.normalize_counts()

    animes = collections.Counter({k: v for k, v in animes.iteritems() if v.num_downloads > 1000})

    for anime in animes.itervalues():
        update_mal_link(anime, search_engine)

    animes = merge_by_link(animes.values())

    table_data = make_table_body(animes, templates)
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
            for filename in [args.style_file] + ["res/" + f for f in SEASON_IMAGES]:
                shutil.copy(filename, os.path.join(dest_dir, os.path.basename(filename)))


if __name__ == "__main__":
    sys.exit(main())
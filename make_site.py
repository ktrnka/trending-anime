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
import math
# import matplotlib.pyplot
import numpy
import scipy.optimize
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
MONGO_TIME = "%Y/%m/%d %H:%M"
SEC_IN_DAY = 60. * 60 * 24

ACCURACY_TABLE = {"3": 84.60400510425866, "4": 86.6924739899726, "5": 91.60210393760404, "6": 92.09773148280004, "7": 92.90524222602687, "8": 95.61050273545129, "9": 97.96066608319538, "10": 97.94706928696975}
ACCURACY_TABLE = {int(k): v for k, v in ACCURACY_TABLE.iteritems()}

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


def format_episode(episode, count, release_date, download_estimate, retention_rate):
    release_string = "???"
    if release_date:
        release_string = release_date.strftime("%Y-%m-%d")

    retention_string = "???"
    if retention_rate:
        retention_string = "{:.1f}%".format(retention_rate)

    download_estimate_string = "???"
    if download_estimate and math.isnan(download_estimate.prediction):
        download_estimate_string = "NaN"
    elif download_estimate:
        download_estimate_string = "{:,} +/- {:,}".format(int(download_estimate.prediction), int(download_estimate.prediction * (1. - download_estimate.confidence / 100) / 2))
    return """
<td>
Episode {}<br>
Released {}<br>
Current DLs: {:,}<br>
DLs at 7d: {}<br>
{} of previous ep
</td>
    """.format(episode, release_string, count, download_estimate_string, retention_string)


def format_row(index, series, top_series, html_templates):
    extras = []
    alternate_names = series.get_alternate_names()
    if alternate_names:
        extras.append("Alternate titles: {}".format(", ".join(alternate_names)))

    extras.append("Sub groups: {}".format(", ".join(series.get_sub_groups())))

    episode_counts = series.get_episode_counts()
    retention_rates = series.compute_retention()
    download_estimates = series.estimate_downloads(7)
    episode_cells = [format_episode(episode, count, series.episodes[episode].get_release_date(), download_estimates.get(episode), retention_rates.get(episode)) for episode, count in episode_counts[-4:]]
    extras.append("<table width='100%'><tr>{}</tr></table>".format("".join(episode_cells)))

    # extras.append(", ".join("Episode {}: {:,}".format(ep, count) for ep, count in series.get_episode_counts()))

    # extras.append(", ".join("Episode {}: {}".format(ep, date.strftime("%Y-%m-%d")) for ep, date in series.episode_dates.iteritems()))

    # retention_rates = series.compute_retention()
    # if retention_rates:
    #     retention_rates = sorted(retention_rates.iteritems(), key=lambda x: x[0])
    #     extras.append(", ".join("Episode {}: {:.1f}% viewer retention".format(rate[0], rate[1]) for rate in retention_rates))

    season_images = "".join(html_templates.sub("season_image", image=SEASON_IMAGES[season], season=SEASONS[season]) for season in series.get_seasons())

    return html_templates.sub("row",
                              id="row_{}".format(index),
                              season_images=season_images,
                              extra="<br>".join(extras),
                              series_name=series.get_link(),
                              value="{:,}".format(series.get_score()),
                              bar_width=int(100 * series.get_score() / top_series.get_score()))


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


def median(data):
    data = sorted(data)
    return data[len(data) / 2]


def get_accuracy(num_datapoints):
    if num_datapoints in ACCURACY_TABLE:
        return ACCURACY_TABLE[num_datapoints]

    if num_datapoints < min(ACCURACY_TABLE.iterkeys()):
        return min(ACCURACY_TABLE.itervalues())

    if num_datapoints > max(ACCURACY_TABLE.iterkeys()):
        return max(ACCURACY_TABLE.itervalues())

    return -1


def first_positive(datapoints, key):
    for i, point in enumerate(datapoints):
        if key(point) > 0:
            return i
    return -1


class Episode(object):
    def __init__(self):
        self.downloads_current = 0
        self.downloads_history = dict()
        self.downloads_estimate = 0
        self.release_date = None

    def update(self, other):
        """

        :type other: Episode
        """
        self.downloads_current += other.downloads_current
        self.downloads_history.update(other.downloads_history)
        self.downloads_estimate += other.downloads_estimate

        if other.release_date and (self.release_date is None or self.release_date > other.release_date):
            self.release_date = other.release_date

    def get_release_date(self):
        if self.release_date:
            return self.release_date

        earliest_date = min(self.downloads_history.iterkeys())
        if earliest_date > datetime.datetime(2015, 1, 25):
            return earliest_date - datetime.timedelta(0.5)

        return None

    def update_release_date(self, release_date):
        if not self.release_date or self.release_date > release_date:
            self.release_date = release_date

    def downloads_history_to_mongo(self):
        return {history_date.strftime(MONGO_TIME): str(count) for history_date, count in self.downloads_history.iteritems()}


class Series(object):
    def __init__(self):
        self.num_downloads = 0
        self.spelling_counts = collections.Counter()
        self.url = None
        self.sub_group_counts = collections.Counter()
        self.score = None

        self.episodes = collections.defaultdict(Episode)

    def clean_download_history(self):
        logger = logging.getLogger(__name__)
        for episode in self.episodes.itervalues():
            dates = sorted(episode.downloads_history.iteritems(), key=lambda p: p[0])
            good_dates = []
            previous_count = None
            for date in dates:
                if not previous_count or previous_count <= date[1]:
                    good_dates.append(date)
                    previous_count = date[1]

            if not good_dates:
                episode.downloads_history.clear()
            else:
                episode.downloads_history = {d: c for d, c in good_dates}
            logger.info("Filtered {} dates for episode {}".format(len(dates) - len(good_dates), episode))

    def get_mongo_key(self):
        if not self.url:
            raise ValueError("Missing URL, unable to build key")
        return self.url

    def sync_mongo(self, mongo_object, data_date):
        data_changed = False

        # sync release dates
        for episode_str, release_date_str in mongo_object.get("release_dates", {}).iteritems():
            episode = int(episode_str)
            release_date = datetime.datetime.strptime(release_date_str, MONGO_TIME)
            self.episodes[episode].update_release_date(release_date)
        mongo_object["release_dates"] = {str(ep): self.episodes[ep].release_date.strftime(MONGO_TIME) for ep in self.episodes.iterkeys() if self.episodes[ep].release_date}

        # sync download history
        for episode_str, download_history in mongo_object.get("download_history", {}).iteritems():
            episode = int(episode_str)
            self.episodes[episode].downloads_history = dict()
            for date_str, downloads_str in download_history.iteritems():
                self.episodes[episode].downloads_history[datetime.datetime.strptime(date_str, MONGO_TIME)] = int(downloads_str)

        if data_date:
            for episode in self.episodes.itervalues():
                episode.downloads_history[data_date] = episode.downloads_current
                data_changed = True

        mongo_object["download_history"] = {str(k): v.downloads_history_to_mongo() for k, v in self.episodes.iteritems()}

        return data_changed

    def add_torrent(self, torrent, parsed_torrent):
        self.spelling_counts[parsed_torrent.series] += torrent.downloads
        self.num_downloads += torrent.downloads

        self.sub_group_counts[parsed_torrent.sub_group] += torrent.downloads

        self.episodes[parsed_torrent.episode].downloads_current += torrent.downloads

    def add_release_date_torrent(self, torrent, parsed_torrent):
        logger = logging.getLogger(__name__)
        if not torrent.release_date:
            return

        if parsed_torrent.episode in self.episodes:
            if torrent.downloads > 1000 and (not self.episodes[parsed_torrent.episode].release_date or torrent.release_date < self.episodes[parsed_torrent.episode].release_date):
                logger.info("Updating release date of %s, episode %d from %s tp %s", self.get_name(), parsed_torrent.episode, self.episodes[parsed_torrent.episode].release_date, torrent.release_date)
                self.episodes[parsed_torrent.episode].release_date = torrent.release_date
        else:
            self.episodes[parsed_torrent.episode].release_date = torrent.release_date

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
        if self.episodes:
            self.num_downloads /= len(self.episodes)

    def __repr__(self):
        return "{} {:,} DL".format(self.get_name(), self.num_downloads)

    def get_episode_counts(self):
        episodes = sorted(self.episodes.iterkeys())
        return [(ep, max(self.episodes[ep].downloads_history.itervalues())) for ep in episodes]

    def get_sub_groups(self):
        return [name for name, _ in self.sub_group_counts.most_common()]

    def merge(self, other):
        logger = logging.getLogger(__name__)
        logger.info("Merging by URL %s and %s", self, other)
        self.num_downloads += other.num_downloads
        self.spelling_counts.update(other.spelling_counts)
        self.sub_group_counts.update(other.sub_group_counts)

        for episode in other.episodes.iterkeys():
            self.episodes[episode].update(other.episodes[episode])

    def get_seasons(self):
        max_episode = max(self.episodes.iterkeys())

        computed_dates = dict()
        for episode in xrange(1, max_episode + 1):
            if episode in self.episodes:
                computed_dates[episode] = self.episodes[episode].get_release_date()
            else:
                estimated_dates = [val.get_release_date() + datetime.timedelta((episode - e) * 7) for e, val in self.episodes.iteritems() if val.get_release_date()]
                computed_dates[episode] = median(estimated_dates)

        seasons = []
        for season in (date_to_season(d) for d in computed_dates.itervalues() if d):
            if season not in seasons:
                seasons.append(season)
        if len(seasons) > 3:
            return [4]
        else:
            return seasons

    def estimate_downloads_old(self):
        return {episode: max(val.downloads_history.itervalues()) for episode, val in self.episodes.iteritems()}

    def estimate_downloads(self, days):
        """
        Compute the predicted number of downloads per episode at a given length out.
        :param days: Number of days to extrapolate
        """
        logger = logging.getLogger(__name__)
        predictions = dict()

        for episode in self.episodes.iterkeys():
            release_date = self.episodes[episode].get_release_date()
            if not release_date:
                continue

            # build the dataset
            datapoints = [((scan_date-release_date).total_seconds()/SEC_IN_DAY, download_count) for scan_date, download_count in self.episodes[episode].downloads_history.iteritems()]
            datapoints.append((0, 0))

            default_prediction = self.get_default_prediction(datapoints, days)
            if default_prediction.confidence > 90:
                predictions[episode] = default_prediction
            elif len(datapoints) >= 3:
                x_data = numpy.array([val[0] for val in datapoints])
                y_data = numpy.array([val[1] for val in datapoints])

                try:
                    opt_params, opt_covariance = scipy.optimize.curve_fit(download_function, x_data, y_data)
                    predictions[episode] = PredictedValue(download_function(7, *opt_params), get_accuracy(len(datapoints)))
                except RuntimeError:
                    logger.warning("Failed to predict {} episode {} with {} points".format(self.url, episode, len(datapoints)))
                    predictions[episode] = default_prediction

        return predictions

    def get_score(self):
        if self.score:
            return self.score

        estimated_downloads = self.estimate_downloads(7)

        try:
            self.score = int(PredictedValue.weighted_average(estimated_downloads.values()))
        except ZeroDivisionError:
            self.score = self.num_downloads
        return self.score

    def compute_retention(self):
        estimated_downloads = self.estimate_downloads(7)

        retentions = dict()
        for episode in estimated_downloads.iterkeys():
            if episode - 1 not in estimated_downloads:
                continue

            if estimated_downloads[episode].confidence < 95:
                continue

            if estimated_downloads[episode-1].confidence < 95:
                continue

            retentions[episode] = 100. * estimated_downloads[episode].prediction / estimated_downloads[episode-1].prediction

        return retentions

    @staticmethod
    def get_default_prediction(datapoints, days):
        logger = logging.getLogger(__name__)
        transformed_points = sorted((p[0] - days, p[1]) for p in datapoints)

        index = first_positive(transformed_points, lambda pair: pair[0])
        closest_point = min(transformed_points, key=lambda pair: math.fabs(pair[0]))

        if index > 0:
            before = transformed_points[index-1]
            after = transformed_points[index]

            estimate = before[1] + (after[1] - before[1]) * -before[0] / (after[0] - before[0])

            if after[1] > 50000:
                logger.info("Found before and after points: {}, {}".format(before, after))
                percent_diff = 100. * math.fabs(closest_point[1] - estimate) / closest_point[1]
                logger.info("Old estimate: {:.0f}, New estimate: {:.0f} ({:.1f}% diff)".format(closest_point[1], estimate, percent_diff))


            if before[0] > -1 or after[0] < 1:
                accuracy = 98.
            else:
                accuracy = 80.

            return PredictedValue(estimate, accuracy)

        accuracy = 60.
        return PredictedValue(closest_point[1], accuracy)

class PredictedValue(object):
    def __init__(self, prediction, confidence):
        self.prediction = prediction
        self.confidence = confidence

    @staticmethod
    def weighted_average(predictions):
        predictions = [p for p in predictions if not math.isnan(p.prediction)]
        return sum(x.prediction * x.confidence for x in predictions) / float(sum(x.confidence for x in predictions))

    def __str__(self):
        return "{} ({})".format(self.prediction, self.confidence)

    def __repr__(self):
        return str(self)

def download_function(x, a, b, c):
    return b * numpy.power(numpy.log(x + a + 0.1), c)

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

    # deduplicate based on the torrent url (sometimes kimono returns duplicates)
    url_set = set()
    unique_torrents = []
    for torrent in torrents:
        if not torrent.url or torrent.url not in url_set:
            unique_torrents.append(torrent)
            url_set.add(torrent.url)
    logger.info("Removed %d duplicate torrent listings", len(torrents) - len(unique_torrents))

    return parse_timestamp(data["thisversionrun"]), unique_torrents


def make_table_body(series, html_templates):
    top_series = sorted(series, key=lambda s: s.get_score(), reverse=True)
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


def sync_mongo(mongo_db, animes, data_date):
    logger = logging.getLogger(__name__)

    collection = mongo_db["animes"]
    for anime in animes:
        try:
            mongo_entry = collection.find_one({"key": anime.get_mongo_key()})
        except ValueError:
            logger.info("Skipping {}; no mongo key available".format(anime))
            continue
        if mongo_entry:
            if anime.sync_mongo(mongo_entry, data_date):
                logger.info("Updating {}".format(mongo_entry))
                collection.save(mongo_entry)
            else:
                logger.info("Not updating {}, no change".format(mongo_entry))
        else:
            mongo_entry = {"key": anime.get_mongo_key()}
            if anime.sync_mongo(mongo_entry, data_date):
                logger.info("Inserting {}".format(mongo_entry))
                collection.insert(mongo_entry)


def inject_version(endpoint, api_version):
    if api_version >= 0:
        return endpoint.replace("/api/", "/api/{}/".format(api_version))
    else:
        return endpoint


def is_stale(data_date):
    logger = logging.getLogger(__name__)
    now = datetime.datetime.now()
    diff = now - data_date
    logger.info("%s days between %s and %s", diff.days, now, data_date)
    if diff.days > 1:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--template_dir", default="templates", help="Dir of templates")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Verbose logging")
    parser.add_argument("--style-file", default="res/over9000.css", help="CSS style")
    parser.add_argument("--favicon-file", default="res/favicon.ico", help="Favicon to use")
    parser.add_argument("--api-version", default=-1, type=int, help="Optional version of the main Kimono endpoint to load")
    parser.add_argument("config", help="Config file")
    parser.add_argument("output", help="Output filename or 'bitballoon' to upload to bitballoon")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)

    # load config
    config = ConfigParser.RawConfigParser()
    config.read([args.config])

    # load templates
    templates = Templates(args.template_dir)

    # load torrent list
    data_date, torrents = load(inject_version(config.get("kimono", "endpoint"), args.api_version))
    if args.api_version < 0 and is_stale(data_date):
        logger.error("Data is stale, not rebuilding")
        return -1

    # load release dates
    try:
        release_data_date, release_date_torrents = load(config.get("kimono", "release_date_endpoint"))
    except requests.exceptions.HTTPError:
        release_data_date = None
        release_date_torrents = []
        logger.exception("Failed to load release dates, skipping")

    # mongodb, google search
    mongo_client = pymongo.MongoClient(config.get("mongo", "uri"))
    mongo_db = mongo_client.get_default_database()
    search_engine = SearchEngine(config.get("google", "api_key"),
                                 config.get("google", "cx"),
                                 mongo_db)

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

    sync_mongo(mongo_db, animes, data_date)

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
            for filename in [args.style_file, args.favicon_file] + ["res/" + f for f in SEASON_IMAGES]:
                shutil.copy(filename, os.path.join(dest_dir, os.path.basename(filename)))


if __name__ == "__main__":
    sys.exit(main())
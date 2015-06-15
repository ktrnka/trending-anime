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
import math
import curve_fitting

from google_search import SearchEngine
from kimono import inject_version, is_stale, parse_timestamp
import numpy
import scipy.optimize
import bitballoon
# import download_graph
import os
import re
import requests
import pymongo

MIN_DOWNLOADS_WARNINGS = 5000

NORMALIZATION_MAP = {ord(c): None for c in "/:;._ -'\"!,~()"}
SEASONS = ["Winter season", "Spring season", "Summer season", "Fall season", "Long running show"]
SEASON_COLOR_STYLES = ["light-blue-text", "light-green-text text-accent-2", "green-text", "amber-text text-darken-1"]
SEASON_DEFAULT_COLOR_STYLE = "grey-text text-lighten-3"
MONGO_TIME = "%Y/%m/%d %H:%M"
SEC_IN_DAY = 60. * 60 * 24

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


def format_episode(html_templates, episode_no, current_downloads, episode, download_estimate, retention_rate, diagnostics=False):
    release_string = "???"
    release_date = episode.get_release_date()
    if release_date:
        release_string = release_date.strftime("%a, %b %d")

    retention_string = "???"
    if retention_rate:
        retention_string = "{:.1f}".format(retention_rate)

    download_estimate_string = "???"
    if download_estimate:
        download_estimate_string = "{:,} +/- {:,}".format(int(download_estimate.prediction), int(
            download_estimate.prediction * (1. - download_estimate.confidence / 100) / 2))

    extras = ""
    if diagnostics:
        downloads = sorted(episode.downloads_history.iteritems())
        extras = "<br>".join("{}: {:,}".format(scrape_date.strftime(MONGO_TIME), download_count) for scrape_date, download_count in downloads)

    return html_templates.sub("episode",
                              episode_number=episode_no,
                              release_date=release_string,
                              downloads="{:,}".format(current_downloads),
                              downloads_at_7=download_estimate_string,
                              retention_percent=retention_string,
                              extras=extras)


def format_episode_diagnostic(episode_no, current_downloads, episode, download_estimate):
    release_string = "???"
    release_date = episode.get_release_date()
    if release_date:
        release_string = release_date.strftime("%a, %b %d")

    download_estimate_string = "???"
    if download_estimate:
        download_estimate_string = "{:,} +/- {:,}".format(int(download_estimate.prediction), int(
            download_estimate.prediction * (1. - download_estimate.confidence / 100) / 2))

    return "{}: currently {:,}, {} at 7, released {}".format(episode_no, current_downloads, download_estimate_string, release_string)


def format_season_info(series):
    seasons = series.get_seasons()
    season_classes = ["icon-snowflake", "icon-tulip", "icon-sun", "icon-tree"]

    season_html = ""
    for i, season_color in enumerate(SEASON_COLOR_STYLES):
        if i == 2:
            season_html += '<br class="hide-on-med-and-up"/>'
        if i in seasons or 4 in seasons:
            season_class = season_classes[i]
        else:
            season_class = season_classes[i] + "-grey"
        season_html += '<i class="icon {}" style="font-size: 1.5rem;" title="{}"></i>'.format(season_class, SEASONS[i])

    return season_html


def format_list(strings, default_label):
    if strings:
        return ", ".join(strings)
    else:
        return '<span class="grey-text">{}</span>'.format(default_label)


def format_series(index, series, top_series, html_templates, diagnostics=False, image_dir=None):
    alternate_names = format_list(series.get_alternate_names(), "None")
    sub_groups = format_list(series.get_sub_groups(), "Unknown")

    episode_counts = series.get_episode_counts()
    retention_rates = series.compute_retention()
    download_estimates = series.estimate_downloads(7)
    episode_cells = [
        format_episode(html_templates, episode, count, series.episodes[episode], download_estimates.get(episode),
                       retention_rates.get(episode), diagnostics=diagnostics) for episode, count in episode_counts[-3:]]
    episode_html = "\n".join(episode_cells)

    if diagnostics:
        episode_html += "\n<!-- Diagnostics\n" + "\n".join(format_episode_diagnostic(episode_number, count, series.episodes[episode_number], download_estimates.get(episode_number)) for episode_number, count in episode_counts) + "\n-->\n"

    season_images = format_season_info(series)

    return html_templates.sub("row",
                              id="row_{}".format(index),
                              season_images=season_images,
                              alternate_titles=alternate_names,
                              sub_groups=sub_groups,
                              series_name=series.get_linked_name(),
                              episodes=episode_html,
                              value="{:,}".format(series.get_score()),
                              bar_width=int(100 * series.get_score() / top_series.get_score()))


def get_title_key(title):
    return title.lower().translate(NORMALIZATION_MAP)


class ParsedTorrent(object):
    """A torrent filename that's been parsed into components"""
    _SPLIT_PATTERN = re.compile(r"[ _,-]")
    _BRACKET_PATTERN = re.compile(r"(\[([^]]+)\])")
    _PAREN_PATTERN = re.compile(r"(\(([^)]+)\))")
    _MD5_PATTERN = re.compile(r"\s*\[[0-9A-F]{6,}\]\s*", re.IGNORECASE)
    _RESOLUTION_PATTERN = re.compile(r"^(?:\d+x)?(\d+)p?$", re.IGNORECASE)
    _FILENAME_PATTERN = re.compile(r"\[([^]]+)\] (.+) (?:-|Episode) (\d+)(?:v\d)?\s*(?:\.(\w+))?$")
    _TAIL_JUNK = re.compile(r"(?<=\.\w{3}).+")
    _FILENAME_PATTERN_ALT = re.compile(r"\[([^]]+)\] (.+) (\d\d+) +(?:\.(\w+))?$")
    _TAGS = "AAC BD BDrip FLAC 10bit 10Bit Dual v2 Dual-Audio".split()
    _unparsed_tags = collections.Counter()

    def __init__(self, series, episode, sub_group, resolution, file_type):
        self.series = series
        self.episode = int(episode)
        self.sub_group = sub_group
        self.resolution = resolution
        self.file_type = file_type

        self.logger = logging.getLogger(__name__)

    @staticmethod
    def log_unparsed(num_tags=10):
        logger = logging.getLogger(__name__)
        logger.debug("Top %d unparsed tags:", num_tags)
        for content, count in ParsedTorrent._unparsed_tags.most_common(num_tags):
            logger.debug("Unparsed %s: %d", content, count)

    @staticmethod
    def _extract_resolution_from_tags(contents):
        """Extract from lists like 720p Hi10 AAC"""
        parts = ParsedTorrent._SPLIT_PATTERN.split(contents)
        if len(parts) == 1:
            return None

        for part in parts:
            m = ParsedTorrent._RESOLUTION_PATTERN.match(part)
            if m:
                return m.group(1)

        return None

    @staticmethod
    def _consume_tags(title):
        """Parse and strip tags like [720p]"""
        resolution = None
        title_cleaned = title

        for match, contents in ParsedTorrent._BRACKET_PATTERN.findall(title):
            # don't delete the sub group
            if title.find(match) == 0:
                continue
            elif ParsedTorrent._MD5_PATTERN.match(match):
                pass
            elif contents in ParsedTorrent._TAGS:
                pass
            elif ParsedTorrent._RESOLUTION_PATTERN.match(contents):
                m = ParsedTorrent._RESOLUTION_PATTERN.match(contents)
                resolution = m.group(1)
            else:
                extracted_res = ParsedTorrent._extract_resolution_from_tags(contents)
                if not extracted_res:
                    ParsedTorrent._unparsed_tags[contents] += 1
                elif not resolution:
                    resolution = extracted_res

            # remove the bracketed expression
            title_cleaned = title_cleaned.replace(match, "")

        for match, contents in ParsedTorrent._PAREN_PATTERN.findall(title):
            # use a more restrictive match to avoid matching years
            if not resolution and contents.endswith("p") and ParsedTorrent._RESOLUTION_PATTERN.match(contents):
                m = ParsedTorrent._RESOLUTION_PATTERN.match(contents)
                resolution = m.group(1)
            elif contents in ParsedTorrent._TAGS:
                pass
            else:
                extracted_res = ParsedTorrent._extract_resolution_from_tags(contents)

                # by default don't strip unknown tags for parens cause they could be a part of the filename
                if not extracted_res:
                    ParsedTorrent._unparsed_tags[contents] += 1
                    continue
                elif not resolution:
                    resolution = extracted_res

            title_cleaned = title_cleaned.replace(match, "")

        return title_cleaned, resolution

    @staticmethod
    def from_name(title):
        """Parse a filename/title of a torrent"""

        title_cleaned, resolution = ParsedTorrent._consume_tags(title)

        # special case for some that have the translated title afterwards
        title_cleaned = ParsedTorrent._TAIL_JUNK.sub("", title_cleaned)

        title_cleaned = title_cleaned.translate({0x2012: "-"})
        title_cleaned = ParsedTorrent.normalize_spacing(title_cleaned)

        m = ParsedTorrent._FILENAME_PATTERN.match(title_cleaned)
        if m:
            p = ParsedTorrent(m.group(2), m.group(3), m.group(1), resolution, m.group(4))
            p.logger.debug("{} -> {}".format(title, p))
            return p

        m = ParsedTorrent._FILENAME_PATTERN_ALT.match(title_cleaned)
        if m:
            p = ParsedTorrent(m.group(2), m.group(3), m.group(1), resolution, m.group(4))
            p.logger.debug("{} -> {}".format(title, p))
            return p
        return None

    def __repr__(self):
        return "[{}] {} - {} [{}].{}".format(self.sub_group, self.series, self.episode, self.resolution, self.file_type)

    @staticmethod
    def normalize_spacing(title_cleaned):
        """Normalize torrents that have filenames with all underscores no spaces"""
        if " " not in title_cleaned:
            return title_cleaned.replace("_", " ")
        return title_cleaned


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


def first_positive(datapoints, key):
    for i, point in enumerate(datapoints):
        if key(point) > 0:
            return i
    return -1


class Episode(object):
    """An episode of a series with all the related info"""
    def __init__(self):
        self.downloads_current = 0
        self.downloads_history = dict()
        self.downloads_estimate = 0
        self.release_date = None

    def update(self, other):
        self.downloads_current += other.downloads_current
        self.downloads_history.update(other.downloads_history)
        self.downloads_estimate += other.downloads_estimate

        self.update_release_date(other.release_date)

    def get_release_date(self):
        if not self.downloads_history:
            return self.release_date

        earliest_date = min(self.downloads_history.iterkeys())
        if not self.release_date or earliest_date < self.release_date:
            return earliest_date - datetime.timedelta(0.5)

        return self.release_date

    def update_release_date(self, release_date, num_downloads=None):
        """Update the release date to the earlier date and filter unreliable data"""
        if not self.release_date:
            self.release_date = release_date

        if num_downloads and num_downloads < 1000:
            return

        if release_date and self.release_date > release_date:
            self.release_date = release_date

    def downloads_history_to_mongo(self):
        return {history_date.strftime(MONGO_TIME): str(count) for history_date, count in
                self.downloads_history.iteritems()}

    def clean_data(self):
        samples = sorted(self.downloads_history.iteritems(), key=lambda p: p[0])
        good_samples = []
        previous_count = None

        filter_reasons = collections.Counter()

        for sample in samples:
            if not previous_count or previous_count < sample[1]:
                good_samples.append(sample)
                previous_count = sample[1]
            elif previous_count == sample[1]:
                filter_reasons["same value"] += 1
            elif sample[1] == 0:
                filter_reasons["value is zero"] += 1
            elif previous_count > sample[1]:
                filter_reasons["lower value"] += 1

        if not good_samples:
            self.downloads_history.clear()
        else:
            self.downloads_history = {d: c for d, c in good_samples}

        return filter_reasons

    def transform_downloads_history(self):
        """Convert download history dates to deltas from the release date, sort them, add (0, 0)"""
        release_date = self.get_release_date()

        datapoints = [((scan_date - release_date).total_seconds() / SEC_IN_DAY, download_count) for
                          scan_date, download_count in self.downloads_history.iteritems()]
        datapoints.append((0, 0))
        datapoints = sorted(datapoints)

        return datapoints

    def get_max_history_downloads(self):
        return max(self.downloads_history.itervalues())


class Series(object):
    """An anime series"""
    def __init__(self):
        self.num_downloads = 0
        self.spelling_counts = collections.Counter()
        self.url = None
        self.sub_group_counts = collections.Counter()
        self.score = None

        self.episodes = collections.defaultdict(Episode)

        self.logger = logging.getLogger(__name__)

    def clean_data(self):
        for ep_num, episode in self.episodes.iteritems():
            filtering_reasons = episode.clean_data()
            if filtering_reasons:
                self.logger.debug("Filtered {} dates for {} episode {} for reasons: {}".format(sum(filtering_reasons.values()), self.get_name(), ep_num, ", ".join("{}: {}".format(k, v) for k, v in filtering_reasons.most_common())))

    def get_last_release_date(self):
        return max(self.get_release_dates())

    def get_mongo_key(self):
        if not self.url:
            raise ValueError("Missing URL, unable to build key")
        return self.url

    def sync_mongo(self, mongo_object, data_date):
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

        mongo_object["download_history"] = {str(k): v.downloads_history_to_mongo() for k, v in self.episodes.iteritems()}

    def add_torrent(self, torrent, parsed_torrent):
        self.spelling_counts[parsed_torrent.series] += torrent.downloads
        self.num_downloads += torrent.downloads

        self.sub_group_counts[parsed_torrent.sub_group] += torrent.downloads

        self.episodes[parsed_torrent.episode].downloads_current += torrent.downloads

    def add_release_date_torrent(self, torrent, parsed_torrent):
        if not torrent.release_date:
            return

        self.episodes[parsed_torrent.episode].update_release_date(torrent.release_date, torrent.downloads)

    def get_name(self):
        """Get the best name for this anime"""
        assert len(self.spelling_counts) > 0
        return self.spelling_counts.most_common(1)[0][0]

    def get_release_dates(self):
        dates = [episode.get_release_date() for episode in self.episodes.itervalues()]
        dates = [d for d in dates if d]
        return dates

    def get_alternate_names(self):
        assert len(self.spelling_counts) > 0

        normalized_names = set()
        names = []
        for name, _ in self.spelling_counts.most_common():
            normalized_name = get_title_key(name)
            if normalized_name not in normalized_names:
                normalized_names.add(normalized_name)
                names.append(name)

        return names[1:]

    def get_linked_name(self):
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
        return [(ep, max([0] + self.episodes[ep].downloads_history.values())) for ep in episodes]

    def get_sub_groups(self):
        return [name for name, _ in self.sub_group_counts.most_common()]

    def merge(self, other):
        self.logger.debug("Merging by URL: %s and %s", self, other)
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
                estimated_dates = [val.get_release_date() + datetime.timedelta((episode - e) * 7) for e, val in
                                   self.episodes.iteritems() if val.get_release_date()]
                if estimated_dates:
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
        predictions = dict()

        for episode in self.episodes.iterkeys():
            release_date = self.episodes[episode].get_release_date()
            if not release_date:
                continue

            # build the dataset
            datapoints = self.episodes[episode].transform_downloads_history()

            default_prediction = self.get_default_prediction(datapoints, days)
            if default_prediction.confidence > 90:
                predictions[episode] = default_prediction
            elif len(datapoints) >= 2:
                curve = curve_fitting.get_best_curve()
                curve.fit(datapoints)

                prediction = curve.predict(7)
                if prediction > 0:
                    predictions[episode] = PredictedValue(prediction, curve.get_accuracy(datapoints))
                else:
                    self.logger.warning("Failed to predict {} episode {} with {} points".format(self.url, episode, len(datapoints)))
                    for point in sorted(datapoints, key=lambda p: p[0]):
                        self.logger.warning("{:.1f}, {:,}".format(point[0], point[1]))
                    self.logger.warning("Setting to {}".format(default_prediction))
                    predictions[episode] = default_prediction

        return predictions

    def get_score(self):
        if self.score:
            return self.score

        estimated_downloads = self.estimate_downloads(7)

        if len(estimated_downloads) > 10:
            old_score = int(PredictedValue.weighted_average(estimated_downloads.values()))

            estimated_downloads = {k: v for k, v in sorted(estimated_downloads.items())[-10:]}
            new_score = int(PredictedValue.weighted_average(estimated_downloads.values()))

            if (new_score - old_score) / float(old_score) > 0.2:
                self.logger.info("[>20% change] Limiting {} to highest-numbered 10 episodes, changes score from {:,} to {:,}".format(self.get_name(), old_score, new_score))
            else:
                self.logger.debug("Limiting {} to highest-numbered 10 episodes, changes score from {:,} to {:,}".format(self.get_name(), old_score, new_score))


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

            if estimated_downloads[episode - 1].confidence < 95:
                continue

            if not estimated_downloads[episode - 1].prediction:
                continue

            retentions[episode] = 100. * estimated_downloads[episode].prediction / estimated_downloads[episode - 1].prediction

        return retentions

    @staticmethod
    def get_default_prediction(datapoints, days):
        transformed_points = sorted((p[0] - days, p[1]) for p in datapoints)

        index = first_positive(transformed_points, lambda pair: pair[0])
        closest_point = min(transformed_points, key=lambda pair: math.fabs(pair[0]))

        if index > 0:
            before = transformed_points[index - 1]
            after = transformed_points[index]

            estimate = before[1] + (after[1] - before[1]) * -before[0] / (after[0] - before[0])
            accuracy = 100. - 100. * (estimate - before[1]) / (1 + (after[1] + before[1]) / 2)

            # if before[0] > -1 or after[0] < 1:
            #     accuracy = 98.
            # else:
            #     accuracy = 80.

            return PredictedValue(estimate, accuracy)

        accuracy = 60.
        return PredictedValue(closest_point[1], accuracy)

    def get_max_history_downloads(self):
        return max(ep.get_max_history_downloads() for ep in self.episodes.itervalues())


class PredictedValue(object):
    def __init__(self, prediction, confidence):
        if math.isnan(prediction) or math.isnan(confidence):
            raise ValueError()
        self.prediction = prediction
        self.confidence = confidence

    @staticmethod
    def weighted_average(predictions):
        predictions = [p for p in predictions]
        return sum(x.prediction * x.confidence for x in predictions) / float(sum(x.confidence for x in predictions))

    def __str__(self):
        return "{} ({:.1f}%)".format(self.prediction, self.confidence)

    def __repr__(self):
        return str(self)


def download_function(x, a, b, c):
    return b * numpy.power(numpy.log(x + a + 0.1), c)


def deduplicate_torrents(torrents):
    """Remove duplicate torrents based on the torrent URL because Kimono may scrape while the pages are updating"""
    logger = logging.getLogger(__name__)

    url_set = set()
    unique_torrents = []
    for torrent in torrents:
        if not torrent.url or torrent.url not in url_set:
            unique_torrents.append(torrent)
            url_set.add(torrent.url)
    logger.info("Removed %d duplicate torrent listings", len(torrents) - len(unique_torrents))
    return unique_torrents


def load(endpoint, timestamp_optional=False):
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

    torrents = deduplicate_torrents(torrents)

    timestamp = None
    if not timestamp_optional or "thisversionrun" in data:
        timestamp = parse_timestamp(data["thisversionrun"])

    return timestamp, torrents


def make_table_body(series, html_templates, diagnostics=False, image_dir=None):
    top_series = sorted(series, key=lambda s: s.get_score(), reverse=True)
    data_entries = [format_series(i, anime, top_series[0], html_templates, diagnostics=diagnostics, image_dir=image_dir) for i, anime in enumerate(top_series)]
    return '<div class="divider"></div>\n'.join(data_entries)


def torrents_to_series(torrents, release_date_torrents):
    """

    :param torrents:
    :return: Mapping of normalized series names to Series objects
    """
    logger = logging.getLogger(__name__)

    success_counts = collections.Counter()
    parse_fail = collections.Counter()

    animes = collections.defaultdict(Series)

    for torrent in torrents:
        episode = ParsedTorrent.from_name(torrent.title)
        if episode:
            episode_key = get_title_key(episode.series)
            animes[episode_key].add_torrent(torrent, episode)

            success_counts[True] += torrent.downloads
        elif torrent.size > 1000 or "OVA" in torrent.title:
            pass
        else:
            parse_fail[torrent.title] += torrent.downloads
            success_counts[False] += torrent.downloads

            if torrent.downloads > MIN_DOWNLOADS_WARNINGS:
                logger.warning("Failed to parse %s with %d downloads", torrent.title, torrent.downloads)
    ParsedTorrent.log_unparsed()
    logger.info("Parsed {:.1f}% of downloads".format(100. * success_counts[True] / (success_counts[True] + success_counts[False])))

    for filename, count in parse_fail.most_common(40):
        logger.debug("Failed to parse %s, %d", filename, count)

    # add release dates when possible
    for torrent in release_date_torrents:
        episode = ParsedTorrent.from_name(torrent.title)
        if episode:
            episode_key = get_title_key(episode.series)
            if episode_key in animes:
                animes[episode_key].add_release_date_torrent(torrent, episode)

    return animes


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


def sync_mongo(mongo_db, animes, data_date, disable_update=False):
    logger = logging.getLogger(__name__)

    collection = mongo_db["animes"]
    num_updated = 0
    num_added = 0
    for anime in animes:
        try:
            mongo_entry = collection.find_one({"key": anime.get_mongo_key()})
        except ValueError:
            logger.info("Skipping {}; no mongo key available".format(anime))
            continue

        if mongo_entry:
            anime.sync_mongo(mongo_entry, data_date)

            if not disable_update:
                collection.save(mongo_entry)
                num_updated += 1
        else:
            mongo_entry = {"key": anime.get_mongo_key()}
            anime.sync_mongo(mongo_entry, data_date)

            if not disable_update:
                collection.insert(mongo_entry)
                num_added += 1

    logger.info("%d mongo records updated", num_updated)
    logger.info("%d mongo records added", num_added)


def update_anime_links(animes, config, mongo_db):
    search_engine = SearchEngine(config.get("google", "api_key"),
                                 config.get("google", "cx"),
                                 mongo_db)
    for anime in animes.itervalues():
        anime.url = search_engine.get_link(anime.get_name())

    search_engine.log_summary()


def filter_old_series(animes):
    logger = logging.getLogger(__name__)
    today = datetime.datetime.now()

    for series in animes:
        release_dates = series.get_release_dates()
        if not release_dates:
            logger.warn("No release dates for series {}".format(series))
            yield series
            continue

        most_recent = max(release_dates)
        age = today - most_recent
        if age.days <= 30:
            yield series
        else:
            logger.info("Removing old series {}, age {}".format(series, age))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--template_dir", default="templates", help="Dir of templates")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Verbose logging")
    parser.add_argument("--style-file", default="res/over9000.css", help="CSS style")
    parser.add_argument("--favicon-file", default="res/favicon.ico", help="Favicon to use")
    parser.add_argument("--api-version", default=-1, type=int,
                        help="Optional version of the main Kimono endpoint to load")
    parser.add_argument("--diagnostic", default=False, action="store_true", help="Include detailed diagnostics in the output")
    parser.add_argument("--one-way-sync", default=False, action="store_true", help="Only sync from mongo but not back to mongo")
    parser.add_argument("config", help="Config file")
    parser.add_argument("output", help="Output filename or 'bitballoon' to upload to bitballoon")
    parser.add_argument("additional_files", nargs="*", help="Additional files or directories to copy to site release")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
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
        _, release_date_torrents = load(config.get("kimono", "release_date_endpoint"), True)
    except requests.exceptions.HTTPError:
        release_date_torrents = []
        logger.exception("Failed to load release dates, skipping")

    mongo_client = pymongo.MongoClient(config.get("mongo", "uri"))
    mongo_db = mongo_client.get_default_database()

    animes = torrents_to_series(torrents, release_date_torrents)

    for anime in animes.itervalues():
        anime.normalize_counts()

    animes = collections.Counter({k: v for k, v in animes.iteritems() if v.num_downloads > 1000})

    update_anime_links(animes, config, mongo_db)

    animes = merge_by_link(animes.values())

    sync_mongo(mongo_db, animes, data_date, disable_update=args.one_way_sync)

    for anime in animes:
        anime.clean_data()

    animes = list(filter_old_series(animes))

    navbar = templates.sub("navbar", current_class="active", winter2015_class="", about_class="")
    table_data = make_table_body(animes, templates, diagnostics=args.diagnostic, image_dir=os.path.dirname(args.output))
    html_data = templates.sub("main",
                              refreshed_timestamp=data_date.strftime("%A, %B %d"),
                              table_body=table_data,
                              navbar=navbar)

    if args.output == "bitballoon":
        bb = bitballoon.BitBalloon(config.get("bitballoon", "access_key"),
                                   config.get("bitballoon", "site_id"),
                                   config.get("bitballoon", "email"))

        bb.update_file_data(html_data.encode("UTF-8"), "index.html", deploy=True)
    else:
        with io.open(args.output, "w", encoding="UTF-8") as html_out:
            html_out.write(html_data)

        dest_dir = os.path.dirname(args.output)
        if dest_dir:
            for filename in [args.style_file, args.favicon_file]:
                shutil.copy(filename, os.path.join(dest_dir, os.path.basename(filename)))
            for filename in args.additional_files:
                dest_path = os.path.join(dest_dir, os.path.basename(filename))
                if os.path.isdir(filename):
                    shutil.rmtree(dest_path, True)
                    shutil.copytree(filename, dest_path)
                else:
                    shutil.copy(filename, dest_path)


if __name__ == "__main__":
    sys.exit(main())
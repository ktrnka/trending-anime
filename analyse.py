from __future__ import unicode_literals

import argparse
import collections
import io
import json
import logging
import sys
import re

import requests


BASIC_PATTERN = re.compile(r"\[([^]]+)\] (.+) - (\d+)\s*(?:\[([\d]+p)\])?.(\w+)")
MD5_PATTERN = re.compile(r"\s*\[[0-9A-F]+\]{6,}\s*", re.IGNORECASE)
NORMALIZATION_MAP = {ord(c): None for c in "/:;._ -'\"!,~()"}

webpage_begin = """
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta charset="UTF-8">
<link href='http://fonts.googleapis.com/css?family=Roboto:500' rel='stylesheet' type='text/css'>
<title>Over 9000: Anime Trends</title>
<style type="text/css">
<!--
a:link {
    text-decoration: none;
}

a:visited {
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

a:active {
    text-decoration: underline;
}
td.value {
    padding:0;
    border-bottom: none;
}
td.number {
    text-align: right;
}
td {
    padding: 4px 6px;
    height: 2em;
}
td.name {
    width: 15em;
}
body {
    font-family: Roboto, sans-serif;
    background: lightgrey;
}
th {
    text-align: left;
    vertical-align:top;
}
caption {
    font-size:90%;
    font-style:italic;
}
div.rectangle {
    height: 16px;
    background-color: darkgreen;
    width: 500px;
}
#middle {
    width: 800px;
    margin: 0 auto;
    background: #ffffff;
    padding: 5px 20px;
}
h1 {
    text-align: center;
    font-style: italic;
    color: grey;
}
-->
</style>
</head>
<body>
<div id="middle">
    <h1>Over 9000: Anime Trends</h1>
    <table cellspacing="0" cellpadding="0" summary="...">
"""
webpage_end = """
    </table>
</div>
</body>
</html>
"""

row_html = """
      <tr>
        <td class="name">{series}</td>
        <td class="number">{value}</td>
        <td class="value"><div class="rectangle" style="width: {bar_width}px;"></div></td>
      </tr>
"""


class SearchEngine(object):
    def __init__(self):
        self.api_key = "***REMOVED***"
        self.cx = "***REMOVED***"
        self.url = "https://www.googleapis.com/customsearch/v1"

        self.max_requests = 10
        self.requests = 0
        self.logger = logging.getLogger(__name__)

    def get_link(self, series_name):
        if self.requests > self.max_requests:
            return None

        r = requests.get(self.url, params={"key": self.api_key, "cx": self.cx, "q": series_name})
        self.logger.info("Request URL: {}".format(r.url))
        r.raise_for_status()
        self.requests += 1
        data = r.json()
        first_result = data["items"][0]["link"]
        self.logger.info("Link for {}: {}".format(series_name, first_result))
        return first_result

    def linkify(self, series_name):
        url = self.get_link(series_name)
        if url:
            return '<a href="{}">{}</a>'.format(url, series_name)
        return series_name


def format_row(series_names, downloads, max_downloads, search_engine):
    best_name = series_names.most_common(1)[0][0]
    return row_html.format(series=search_engine.linkify(best_name),
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


def make_table(series_downloads, spelling_map):
    search_engine = SearchEngine()
    top_series = series_downloads.most_common()
    data_entries = [format_row(spelling_map[series], downloads, top_series[0][1], search_engine) for series, downloads in top_series]
    return "\n".join(data_entries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint", help="Json file or web location to fetch data")
    parser.add_argument("output", help="Output webpage")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    logger = logging.getLogger(__name__)

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
        html_out.write(webpage_begin + make_table(series_downloads, spelling_map) + webpage_end)


if __name__ == "__main__":
    sys.exit(main())
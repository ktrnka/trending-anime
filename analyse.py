from __future__ import unicode_literals

import argparse
import codecs
import collections
import io
import json

import sys
import re

BASIC_PATTERN = re.compile(r"\[(\w+)\] (.+) - (\d+)\s*(?:\[([\d]+p)\])?.(\w+)")
MD5_PATTERN = re.compile(r"\s*\[[0-9A-F]+\]\s*", re.IGNORECASE)

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
        m = BASIC_PATTERN.match(title_cleaned)
        if m:
            return Episode(m.group(2), m.group(3), m.group(1), m.group(4), m.group(5))
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
        return torrent


def load(endpoint):
    torrents = []
    with io.open(endpoint, "rb") as json_in:
        data = json.load(json_in)

        for torrent in data["results"]["collection1"]:
            torrents.append(Torrent.from_json(torrent))

    return torrents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint", help="Json file or web location to fetch data")
    args = parser.parse_args()

    torrents = load(args.endpoint)

    c = collections.Counter()
    series_downloads = collections.Counter()
    resolutions = collections.Counter()
    for torrent in torrents:
        episode = Episode.from_name(torrent.title)
        if episode:
            series_downloads[episode.series] += torrent.downloads
            resolutions[episode.resolution] += torrent.downloads
            c["success"] += 1
            c["success_by_downloads"] += torrent.downloads
        else:
            c["failure"] += 1
            c["failure_by_downloads"] += torrent.downloads

    #print "Success rate by torrents: {:.1f}%".format(100. * c["success"] / (c["success"] + c["failure"]))
    #print "Success rate by downloads: {:.1f}%".format(100. * c["success_by_downloads"] / (c["success_by_downloads"] + c["failure_by_downloads"]))

    #print resolutions.most_common()

    for series, downloads in series_downloads.most_common():
        print "{}: {:,} downloads".format(series, downloads)


if __name__ == "__main__":
    sys.exit(main())
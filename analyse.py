from __future__ import unicode_literals

import argparse
import collections
import io
import json

import sys
import re

BASIC_PATTERN = re.compile(r"\[(\w+)\] (.+) - (\d+)\s*(?:\[([\d]+p)\])?.(\w+)")
MD5_PATTERN = re.compile(r"\s*\[[0-9A-F]+\]\s*", re.IGNORECASE)
NORMALIZATION_MAP = {ord(c): None for c in "/:;._ -'\"!,~()"}


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

    resolutions = collections.Counter()
    spelling_map = collections.defaultdict(collections.Counter)

    norm_downloads = collections.Counter()
    norm_episodes = collections.defaultdict(set)

    who_subs = collections.defaultdict(collections.Counter)

    for torrent in torrents:
        episode = Episode.from_name(torrent.title)
        if episode:
            episode_key = get_title_key(episode.series)

            resolutions[episode.resolution] += torrent.downloads

            spelling_map[episode_key][episode.series] += torrent.downloads

            norm_downloads[episode_key] += torrent.downloads
            norm_episodes[episode_key].add(episode.episode)
            who_subs[episode_key][episode.sub_group] += torrent.downloads

    for series in norm_downloads.iterkeys():
        if norm_episodes[series]:
            norm_downloads[series] /= len(norm_episodes[series])

    for series, downloads in norm_downloads.most_common():
        best_title, _ = spelling_map[series].most_common(1)[0]
        print "{}: {:,} downloads per episode".format(best_title, downloads)
        print "\tEpisodes {}".format(", ".join(unicode(i) for i in sorted(norm_episodes[series])))

        if len(spelling_map[series]) > 1:
            print "\tSpellings: {}".format(", ".join(t for t, v in spelling_map[series].most_common()))

        print "\tSub groups: {}".format(", ".join(g for g, _ in who_subs[series].most_common()))

if __name__ == "__main__":
    sys.exit(main())
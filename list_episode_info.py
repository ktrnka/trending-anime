import ConfigParser
import argparse
import collections
import io
import json
import logging
from pprint import pprint
import sys
import datetime
import pymongo

__author__ = 'keith'

import make_site


def mean(values):
    if not values:
        return -1
    return sum(values) / float(len(values))

def weighted_average(values, weights):
    if not values:
        return -1

    return sum((v * w for v, w in zip(values, weights))) / sum(weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    config = ConfigParser.RawConfigParser()
    config.read([args.config])

    mongo_client = pymongo.MongoClient(config.get("mongo", "uri"))
    collection = mongo_client.get_default_database()["animes"]

    scores_old = dict()
    scores_new = dict()

    for anime in collection.find():
        if "parade" not in anime["key"].lower():
            continue

        series = make_site.Series()
        series.url = anime["key"]
        series.sync_mongo(anime, None)

        print "Anime", series.url

        series.clean_download_history()

        episodes = sorted(series.episodes.iterkeys())

        downloads_old = series.estimate_downloads_old()
        downloads_new = series.estimate_downloads(7)

        scores_old[series.url] = mean(downloads_old.values())
        print "Old value: {}".format(mean(downloads_old.values()))

        try:
            scores_new[series.url] = make_site.PredictedValue.weighted_average(downloads_new.values())
        except ZeroDivisionError:
            scores_new[series.url] = -1
        print "New value: {:.1f}".format(scores_new[series.url])

        for episode in episodes:
            release_date = series.episodes[episode].get_release_date()
            print "Episode {}: {}".format(episode, release_date.strftime("%Y-%m-%d"))
            print "\tDownloads (old): {}".format(downloads_old.get(episode, -1))
            print "\tDownloads (new): {}".format(downloads_new.get(episode, -1))

            for date, downloads in sorted(series.episodes[episode].downloads_history.iteritems(), key=lambda p: p[0]):
                print "\t{}: {:,}".format(date.strftime("%Y-%m-%d %H:%M"), downloads)

    for scores in [scores_old, scores_new]:
        print "Scores:"
        sorted_pairs = sorted(scores.iteritems(), key=lambda p: p[1], reverse=True)
        for pair in sorted_pairs[:20]:
            print "\t{}: {:,}".format(pair[0], int(pair[1]))



if __name__ == "__main__":
    sys.exit(main())
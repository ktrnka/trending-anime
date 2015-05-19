import ConfigParser
import argparse
import logging
import sys

import pymongo
import download_graph


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
        if "seraph" not in anime["key"].lower():
            continue

        series = make_site.Series()
        series.url = anime["key"]
        series.spelling_counts[series.url] += 1
        series.sync_mongo(anime, None)

        print "Anime", series.url

        series.clean_data()

        episodes = sorted(series.episodes.iterkeys())

        predictions = series.estimate_downloads_debug(7, 4)

        for episode in episodes:
            release_date = series.episodes[episode].get_release_date()
            print "Episode {}: {}".format(episode, release_date.strftime("%Y-%m-%d"))

            reference_data, prediction_data = predictions[episode]
            reference_x, reference_y = reference_data
            pred_x, pred_y = prediction_data
            download_graph.make_downloads_graph(zip(reference_x, reference_y), "temp/{}.png".format(episode), prediction_data=zip(pred_x, pred_y))

            for date, downloads in sorted(series.episodes[episode].downloads_history.iteritems(), key=lambda p: p[0]):
                print "\t{}: {:,}".format(date.strftime("%Y-%m-%d %H:%M"), downloads)




if __name__ == "__main__":
    sys.exit(main())
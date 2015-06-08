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
    parser.add_argument("title_query", help="Keyword to find in the title")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = ConfigParser.RawConfigParser()
    config.read([args.config])

    mongo_client = pymongo.MongoClient(config.get("mongo", "uri"))

    db = mongo_client.get_default_database()
    anime_collection = db["animes"]
    link_collection = db["links"]

    link_collection.ensure_index([("title", pymongo.TEXT)], background=False)

    for match in link_collection.find({"$text": { "$search": args.title_query }}):
        anime = anime_collection.find_one({"key": match["url"]})

        series = make_site.Series()
        series.url = anime["key"]
        series.spelling_counts[match["title"]] += 1
        series.sync_mongo(anime, None)

        print "Anime", series.get_name()

        series.clean_data()

        episodes = sorted(series.episodes.iterkeys())

        for episode in episodes:
            release_date = series.episodes[episode].get_release_date()
            print "Episode {}: {}".format(episode, release_date.strftime("%Y-%m-%d"))

            for date, downloads in sorted(series.episodes[episode].downloads_history.iteritems(), key=lambda p: p[0]):
                print "\t{}: {:,}".format(date.strftime("%Y-%m-%d %H:%M"), downloads)

if __name__ == "__main__":
    sys.exit(main())
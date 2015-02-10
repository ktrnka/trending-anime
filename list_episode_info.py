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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    config = ConfigParser.RawConfigParser()
    config.read([args.config])

    mongo_client = pymongo.MongoClient(config.get("mongo", "uri"))
    collection = mongo_client.get_default_database()["animes"]

    all_errors = collections.defaultdict(list)

    for anime in collection.find():
        # if "kise" not in anime["key"].lower():
        #     continue

        series = make_site.Series()
        series.url = anime["key"]
        series.sync_mongo(anime, None)

        series.clean_download_history()

        episodes = sorted(series.download_history.iteritems(), key=lambda p: p[0])

        prediction_errors = series.estimate_downloads(7)
        if prediction_errors:
            for num_points, error in prediction_errors.iteritems():
                all_errors[num_points].append(error)

        for episode, download_counts in episodes:
            if episode not in series.episode_dates:
                continue

            release_date = series.episode_dates[episode]
            print "Episode {}: {}".format(episode, release_date.strftime("%Y-%m-%d"))

            for date, downloads in sorted(download_counts.iteritems(), key=lambda p: p[0]):
                print "\t{}: {:,}".format(date.strftime("%Y-%m-%d"), downloads)

    with io.open("prediction_accuracy.json", "wb") as json_out:
        json.dump({k: 100.-sum(errors)/len(errors) for k, errors in all_errors.iteritems() if k <= 10}, json_out)




if __name__ == "__main__":
    sys.exit(main())
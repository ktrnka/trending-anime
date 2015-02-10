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

    for anime in collection.find():
        # if "kise" not in anime["key"].lower():
        #     continue

        series = make_site.Series()
        series.url = anime["key"]
        series.sync_mongo(anime, None)

        series.clean_download_history()

        episodes = sorted(series.download_history.iteritems(), key=lambda p: p[0])

        downloads_old = series.estimate_downloads_old()
        downloads_new = series.estimate_downloads(7)

        for episode, download_counts in episodes:
            if episode not in series.episode_dates:
                continue

            release_date = series.episode_dates[episode]
            print "Episode {}: {}".format(episode, release_date.strftime("%Y-%m-%d"))
            print "\tDownloads (old): {}".format(downloads_old.get(episode, -1))
            print "\tDownloads (new): {}".format(downloads_new.get(episode, -1))


            # for date, downloads in sorted(download_counts.iteritems(), key=lambda p: p[0]):
            #     print "\t{}: {:,}".format(date.strftime("%Y-%m-%d"), downloads)




if __name__ == "__main__":
    sys.exit(main())
import ConfigParser
import argparse
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
        print anime["key"]

        episodes = [(int(ep), v) for ep, v in anime.get("download_history", {}).iteritems()]
        episodes = sorted(episodes, key=lambda p: p[0])

        for episode, download_counts in episodes:
            print "Episode {}".format(episode)
            download_counts = {datetime.datetime.strptime(k, make_site.MONGO_TIME):int(v) for k, v in download_counts.iteritems()}
            for date, downloads in sorted(download_counts.iteritems(), key=lambda p: p[0]):
                print "\t{}: {:,}".format(date.strftime("%Y-%m-%d"), downloads)



if __name__ == "__main__":
    sys.exit(main())
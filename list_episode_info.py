import ConfigParser
import argparse
import logging
import sys
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
    collection = mongo_client["anime-trends"]



if __name__ == "__main__":
    sys.exit(main())
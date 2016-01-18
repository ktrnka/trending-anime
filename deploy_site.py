from __future__ import unicode_literals
import ConfigParser
import logging
import sys
import argparse
import itertools
import bitballoon
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Verbose logging")
    parser.add_argument("config", help="Config file with API keys for Bitballoon")
    parser.add_argument("site_dir", help="Staging directory with all files to be uploaded")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # load config
    config = ConfigParser.RawConfigParser()
    config.read([args.config])

    bb = bitballoon.BitBalloon(config.get("bitballoon", "access_key"),
                               config.get("bitballoon", "site_id"),
                               config.get("bitballoon", "email"))

    # deploy all files under the directory tree
    all_files = [[os.path.join(dirpath, f) for f in filenames] for dirpath, dirnames, filenames in os.walk(args.site_dir)]
    all_files = list(itertools.chain(*all_files))

    if not any(filename.endswith("index.html") for filename in all_files):
        logger.error("No index.html in the deploy, aborting.")
        return

    logger.debug("File list: %s", all_files)
    bb.deploy_files(all_files)


if __name__ == "__main__":
    sys.exit(main())
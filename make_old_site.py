from __future__ import unicode_literals
import ConfigParser
import logging
import shutil
import sys
import argparse
import collections
import datetime
import io
import bitballoon
from make_site import Templates
import make_site
import os
import pymongo
import pymongo.errors
import pymongo.cursor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--template_dir", default="templates", help="Dir of templates")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Verbose logging")
    parser.add_argument("--style-file", default="res/over9000.css", help="CSS style")
    parser.add_argument("--favicon-file", default="res/favicon.ico", help="Favicon to use")
    parser.add_argument("--diagnostic", default=False, action="store_true", help="Include detailed diagnostics in the output")
    parser.add_argument("--weeks", default=2, type=int, help="Number of weeks back to look")
    parser.add_argument("config", help="Config file")
    parser.add_argument("date", help="End date to check in year/month/day format")
    parser.add_argument("output", help="Output filename or 'bitballoon' to upload to bitballoon")
    parser.add_argument("additional_files", nargs="*", help="Additional files or directories to copy to site release")
    return parser.parse_args()


def get_date_range(args):
    end_date = datetime.datetime.strptime(args.date, "%Y/%m/%d")
    start_date = end_date - datetime.timedelta(7 * args.weeks)

    return start_date, end_date

def load_all(collection, retries=3):
    results = []
    try:
        for result in collection.find():
            results.append(result)
        return results
    except pymongo.errors.AutoReconnect as e:
        if retries > 0:
            return load_all(collection, retries - 1)
        else:
            raise e


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

    # load templates
    templates = Templates(args.template_dir)

    mongo_client = pymongo.MongoClient(config.get("mongo", "uri"))
    mongo_db = mongo_client.get_default_database()

    min_date, max_date = get_date_range(args)

    # load all series, one-way sync from mongo
    link_collection = mongo_db["links"]
    logger.info("%d objects in link collection", link_collection.count())

    titles = collections.defaultdict(collections.Counter)
    for match in load_all(link_collection):
        titles[match["url"]][match["title"]] += 1


    animes = []
    anime_collection = mongo_db["animes"]
    logger.info("%d objects in anime collection", anime_collection.count())
    for anime_object in load_all(anime_collection):
        series = make_site.Series()
        series.url = anime_object["key"]

        if series.url in titles:
            series.spelling_counts = titles[series.url]
        else:
            series.spelling_counts[series.url] += 1
            logger.warning("No title found for anime %s", series.url)

        series.sync_mongo(anime_object, None)
        series.clean_data()

        animes.append(series)

    logger.info("Processed %d shows", len(animes))

    animes = [anime for anime in animes if min_date < anime.get_last_release_date() < max_date]

    navbar = templates.sub("navbar", current_class="", winter2015_class="active", about_class="")

    table_data = make_site.make_table_body(animes, templates, diagnostics=args.diagnostic, image_dir=os.path.dirname(args.output))
    html_data = templates.sub("main",
                              refreshed_timestamp=datetime.datetime.now().strftime("%A, %B %d"),
                              table_body=table_data,
                              navbar=navbar,
                              season_name=make_site.make_page_title(max_date - datetime.timedelta(2)),
                              inline_style=make_site.load_styles([args.style_file] + [f for f in args.additional_files if f.endswith(".css")]))

    if args.output.startswith("bitballoon"):
        bb = bitballoon.BitBalloon(config.get("bitballoon", "access_key"),
                                   config.get("bitballoon", "site_id"),
                                   config.get("bitballoon", "email"))
        _, filename = os.path.split(args.output)

        bb.update_file_data(html_data.encode("UTF-8"), filename, deploy=True)
    else:
        with io.open(args.output, "w", encoding="UTF-8") as html_out:
            html_out.write(html_data)

        dest_dir = os.path.dirname(args.output)
        if dest_dir:
            for filename in [args.style_file, args.favicon_file]:
                shutil.copy(filename, os.path.join(dest_dir, os.path.basename(filename)))
            for filename in args.additional_files:
                dest_path = os.path.join(dest_dir, os.path.basename(filename))
                if os.path.isdir(filename):
                    shutil.rmtree(dest_path, True)
                    shutil.copytree(filename, dest_path)
                else:
                    shutil.copy(filename, dest_path)


if __name__ == "__main__":
    sys.exit(main())
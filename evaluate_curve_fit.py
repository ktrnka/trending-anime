import ConfigParser
import argparse
import logging
import string
import sys
import pprint
import collections
import datetime
import urllib
import numpy
import pandas

import pymongo
import download_graph
import matplotlib.pyplot as plt


import make_site
import curve_fitting


FILENAME_CHARS = set("-_.() " + string.ascii_letters + string.digits)

def mean(values):
    if not values:
        return -1
    return sum(values) / float(len(values))


def clean_filename(s):
    return "".join(c for c in s if c in FILENAME_CHARS)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    config = ConfigParser.RawConfigParser()
    config.read([args.config])

    mongo_client = pymongo.MongoClient(config.get("mongo", "uri"))
    collection = mongo_client.get_default_database()["animes"]

    backoff_zero = curve_fitting.ConstantCurve(lambda x: 0, "Zero", "0")

    simplest_log = curve_fitting.SimpleLogCurve()

    log_curve = curve_fitting.Curve(lambda x, a, b, c: b * numpy.power(numpy.log(x + a + 1), c), "log3", "{1} * (log(x + {0} + 1) ^ {2}",
                      backoff_curve=backoff_zero)
    log_curve_simple = curve_fitting.Curve(lambda x, b: b * numpy.power(numpy.log(x + 1), 0.5), "log1", "{0} * log(x + 1) ^ 0.5",
                             backoff_curve=simplest_log)

    log_curve_plus = curve_fitting.Curve(lambda x, a, b, c: b * numpy.power(numpy.log(x + a + 1), c), "log3+1", "{1} * (log(x + {0} + 1) ^ {2}",
                      backoff_curve=log_curve_simple, min_points=4)

    inverse_curve = curve_fitting.Curve(lambda x, a, b: x / (x + a ** 2) * b, "inv", "x / (x + {0}^2) * {1}", backoff_curve=log_curve_simple, min_points=3)

    evaluation_suite = curve_fitting.EvaluationSuite([curve_fitting.Evaluation(2), curve_fitting.Evaluation(3), curve_fitting.Evaluation(4), curve_fitting.Evaluation(5)], [log_curve, log_curve_simple, log_curve_plus, inverse_curve])

    scores = collections.defaultdict(lambda: collections.defaultdict(list))

    for anime in collection.find():
        series = make_site.Series()
        series.url = anime["key"]
        series.spelling_counts[series.url] += 1
        series.sync_mongo(anime, None)
        if series.get_max_history_downloads() < 5000:
            continue

        print "Anime", series.url

        series.clean_data()

        episodes = sorted(series.episodes.iterkeys())

        for episode in episodes:
            release_date = series.episodes[episode].get_release_date()

            # 3/1-5/1
            if release_date < datetime.datetime(2015, 3, 1) or release_date >= datetime.datetime(2015, 5, 1):
                continue

            print "Episode {}: {}".format(episode, release_date.strftime("%Y-%m-%d") if release_date else "???")

            for date, downloads in sorted(series.episodes[episode].downloads_history.iteritems(), key=lambda p: p[0]):
                print "\t{}: {:,}".format(date.strftime("%Y-%m-%d %H:%M"), downloads)

            if not release_date:
                continue

            datapoints = series.episodes[episode].transform_downloads_history()
            if len(datapoints) < 2:
                continue

            try:
                all_scores = []
                for evaluation, model, score in evaluation_suite.evaluate(datapoints):
                    scores[evaluation.x_max][model.name].append(score)
                    all_scores.append(score)

                    if score > 0.5 and score < 1.0:
                        print "Hard data set [{:.3f}] for {}, {}:".format(score, evaluation, model)
                        print "\t{}".format(datapoints)
                        print "\tPredict @ 7: {}".format(model.predict(7.))

                if all(s > 0.4 for s in all_scores):
                    series_part = clean_filename(series.url.split("/")[-1])
                    download_graph.make_downloads_graph(datapoints, "unusual_downloads/{}_{}.png".format(series_part, episode))

                if all(s < 0.25 for s in all_scores):
                    series_part = clean_filename(series.url.split("/")[-1])
                    download_graph.make_downloads_graph(datapoints, "normal_downloads/{}_{}.png".format(series_part, episode))

            except curve_fitting.InsufficientDataError:
                pass

    for evaluation, scored_models in scores.iteritems():
        print "Summary at evaluation xmax {}".format(evaluation)

        scored_models = {model: pandas.Series(scores) for model, scores in scored_models.iteritems()}
        for model_name, scores in scored_models.iteritems():
            print "\t{}: {:.3f} +/- {:.3f} in {:,} tests".format(model_name, scores.mean(), scores.std(), scores.shape[0])

            nondefault = scores[scores < 0.99]
            print "\t\tnondefault: {:.3f} +/- {:.3f} in {:,} tests".format(nondefault.mean(), nondefault.std(), nondefault.shape[0])

            plt.clf()
            scores.hist(bins=25)
            plt.savefig("model_scores_{}_{}.png".format(evaluation, model_name))
            plt.close()







if __name__ == "__main__":
    sys.exit(main())
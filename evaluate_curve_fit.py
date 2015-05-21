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


def learning_curve(data, filename):
    plt.figure()
    plt.title("Prediction accuracy vs number of days training data")
    plt.xlabel("Number of days of training data available")
    plt.ylabel("Fraction difference from actual values")

    plt.ylim((0., 1.))

    plt.grid()

    for data_name in data.iterkeys():
        x_values = sorted(data[data_name].iterkeys())
        y_values = [data[data_name][x].mean() for x in x_values]

        plt.plot(x_values, y_values, "o-", label=data_name)

    plt.legend(loc="best")

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


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

    # simplest_log = curve_fitting.SimpleLogCurve()

    log_curve_3 = curve_fitting.Curve(lambda x, a, b, c: b * numpy.power(numpy.log(x + a + 1), c), "log 3p",
                                      "{1} * (log(x + {0} + 1) ^ {2}",
                                      backoff_curve=backoff_zero)

    log_curve_2exp = curve_fitting.Curve(lambda x, a, b: b * numpy.power(numpy.log(x + 1), a), "log 2p_exp",
                                      "{1} * (log(x + 1) ^ {0}",
                                      backoff_curve=backoff_zero)

    log_curve_2shift = curve_fitting.Curve(lambda x, a, b: b * numpy.power(numpy.log(x + a + 1), 0.5), "log 2p_shift",
                                      "{1} * (log(x + {0} + 1) ^ 0.5",
                                      backoff_curve=backoff_zero)

    log_curve_1 = curve_fitting.Curve(lambda x, b: b * numpy.power(numpy.log(x + 1), 0.5), "log 1p",
                                      "{0} * log(x + 1) ^ 0.5",
                                      backoff_curve=backoff_zero)

    # log_curve_plus = curve_fitting.Curve(lambda x, a, b, c: b * numpy.power(numpy.log(x + a + 1), c), "log3+1", "{1} * (log(x + {0} + 1) ^ {2}",
    # backoff_curve=log_curve_simple, min_points=4)

    inverse_curve = curve_fitting.Curve(lambda x, a, b: x / (x + a ** 2) * b, "inv 2p", "x / (x + {0}^2) * {1}",
                                        backoff_curve=backoff_zero, min_points=3)

    evaluation_suite = curve_fitting.EvaluationSuite(
        [curve_fitting.Evaluation(1), curve_fitting.Evaluation(2), curve_fitting.Evaluation(3),
         curve_fitting.Evaluation(4), curve_fitting.Evaluation(5), curve_fitting.Evaluation(6),
         curve_fitting.Evaluation(7), curve_fitting.Evaluation(8)], [log_curve_1, log_curve_2exp, log_curve_2shift, log_curve_3, inverse_curve])

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
                    scores[model.name][evaluation.x_max].append(score)
                    all_scores.append(score)

                    if score > 0.5 and score < 1.0:
                        print "Hard data set [{:.3f}] for {}, {}:".format(score, evaluation, model)
                        print "\t{}".format(datapoints)
                        print "\tPredict @ 7: {}".format(model.predict(7.))

                if all(s > 0.4 for s in all_scores):
                    series_part = clean_filename(series.url.split("/")[-1])
                    download_graph.make_downloads_graph(datapoints,
                                                        "unusual_downloads/{}_{}.png".format(series_part, episode))

                if all(s < 0.25 for s in all_scores):
                    series_part = clean_filename(series.url.split("/")[-1])
                    download_graph.make_downloads_graph(datapoints,
                                                        "normal_downloads/{}_{}.png".format(series_part, episode))

            except curve_fitting.InsufficientDataError:
                pass

    for model_name, scored_models in scores.items():
        scores[model_name] = {x: pandas.Series(scores) for x, scores in scored_models.iteritems()}

    learning_curve(scores, "learning_curve.png")

    for model_name, scored_models in scores.iteritems():
        print "Model {}".format(model_name)

        for evaluation, scores in scored_models.iteritems():
            print "\t{}: {:.3f} +/- {:.3f} in {:,} tests".format(evaluation, scores.mean(), scores.std(),
                                                                 scores.shape[0])

            nondefault = scores[scores < 0.99]
            print "\t\tnondefault: {:.3f} +/- {:.3f} in {:,} tests".format(nondefault.mean(), nondefault.std(),
                                                                           nondefault.shape[0])

            plt.clf()
            scores.hist(bins=25)
            plt.savefig("model_scores_{}_{}.png".format(evaluation, model_name))
            plt.close()


if __name__ == "__main__":
    sys.exit(main())
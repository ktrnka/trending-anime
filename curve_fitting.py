from __future__ import unicode_literals
import logging
import sys
import argparse
import collections

import numpy
import scipy.optimize


class InsufficientDataError(ValueError):
    pass


class InsufficientTestingDataError(InsufficientDataError):
    pass


class Evaluation(object):
    def __init__(self, x_max, testing_x_min=6.5, testing_x_max=7.5):
        self.logger = logging.getLogger(__name__ + ".Evaluation")
        self.x_max = x_max
        self.testing_x_range = (testing_x_min, testing_x_max)

        if self.x_max >= self.testing_x_range[0]:
            self.logger.warning(
                "Testing x range overlaps training range: {} vs {}".format(self.x_max, self.testing_x_range))

    def evaluate(self, model, datapoints):
        training_data = [p for p in datapoints if p[0] < self.x_max]
        testing_data = [p for p in datapoints if self.testing_x_range[0] <= p[0] <= self.testing_x_range[1]]

        # filter anything that has no points other than (0, 0)
        if len([p for p in training_data if p[0] > 0]) == 0:
            raise InsufficientDataError()

        if not testing_data:
            raise InsufficientTestingDataError()

        model.fit(training_data)

        predictions = [model.predict(p[0]) for p in testing_data]
        return self.score([p[1] for p in testing_data], predictions), len(training_data)

    def score(self, reference_y, predicted_y):
        errors = []
        for ref, pred in zip(reference_y, predicted_y):
            errors.append(abs(pred - ref) / float(ref))

        return sum(errors) / len(errors)

    def __str__(self):
        return "Evaluation(x <= {}, {} <= testing_x <= {})".format(self.x_max, self.testing_x_range[0],
                                                                   self.testing_x_range[1])


class EvaluationSuite(object):
    def __init__(self, evaluations, models):
        self.evaluations = evaluations
        self.models = models

        self.scores_by_xmax = collections.defaultdict(lambda: collections.defaultdict(list))
        self.scores_by_training_size = collections.defaultdict(lambda: collections.defaultdict(list))

    def evaluate(self, datapoints):
        for evaluation in self.evaluations:
            assert isinstance(evaluation, Evaluation)

            score_pairs = [evaluation.evaluate(model, datapoints) for model in self.models]
            for model, score_pair in zip(self.models, score_pairs):
                score, effective_training_size = score_pair

                self.scores_by_xmax[evaluation.x_max][model.name].append(score)
                self.scores_by_training_size[effective_training_size][model.name].append(score)
                yield evaluation, model, score, effective_training_size

    def describe(self):
        for model in self.models:
            print model.name

            for evaluation in sorted(self.evaluations, key=lambda e: e.x_max):
                scores = numpy.array(self.scores_by_xmax[evaluation.x_max][model.name])
                num_nan = numpy.isnan(scores).sum()
                num_default = (scores >= 1.0).sum()

                print "\t{} days: {:.3f} +/- {:.3f} ({} nan values, {} failed to predict)".format(evaluation.x_max, scores.mean(), scores.std(),
                                                                       num_nan, num_default)

            for num_points in sorted(self.scores_by_training_size.iterkeys()):
                scores = numpy.array(self.scores_by_training_size[num_points][model.name])
                num_nan = numpy.isnan(scores).sum()
                num_default = (scores >= 1.0).sum()

                print "\t{} points: {:.3f} +/- {:.3f} ({} nan values, {} failed to predict)".format(num_points, scores.mean(), scores.std(),
                                                                       num_nan, num_default)



class Curve(object):
    def __init__(self, function, name, format_string, min_points=None, backoff_curve=None, default_prediction=0):
        self.logger = logging.getLogger(__name__ + ".Curve")
        self.function = function
        self.name = name
        self.format_string = format_string

        self.default_prediction = 0
        self.min_points = min_points
        self.backoff_curve = backoff_curve

        self.params = None
        self.y_max = None
        self.use_backoff = False

        self.accuracy_table = None

    def fit(self, datapoints):
        x, y = zip(*datapoints)
        x = numpy.array(x)
        y = numpy.array(y)
        uncertainties = 1 / (x + 0.5)

        self.y_max = y.max()

        # reset values to prevent carrying over
        self.params = None
        self.use_backoff = False

        if not self.min_points or len(datapoints) >= self.min_points:
            try:
                self.params, opt_covariance = scipy.optimize.curve_fit(self.function, x, y, sigma=uncertainties)
            except (TypeError, RuntimeError):
                pass
                # self.logger.exception("Fitting curve failed, backing off")
        else:
            pass
            # self.logger.info("Too few points, backing off")

        if self.params is None and self.backoff_curve:
            self.use_backoff = True
            self.backoff_curve.fit(datapoints)

    def predict(self, x):
        if self.use_backoff:
            return self.backoff_curve.predict(x)
        elif self.params is not None:
            return min(self.function(x, *self.params), self.y_max * 2)
        else:
            return self.default_prediction

    def set_accuracy_table(self, accuracy_table, transform=float):
        self.accuracy_table = {int(k): transform(v) for k, v in accuracy_table.iteritems()}

    def get_accuracy(self, datapoints):
        if not self.accuracy_table:
            return -1

        num_datapoints = len(datapoints)

        if num_datapoints in self.accuracy_table:
            return self.accuracy_table[num_datapoints]

        min_param = min(self.accuracy_table.iterkeys())
        if num_datapoints < min_param:
            return self.accuracy_table[min_param]

        max_param = max(self.accuracy_table.iterkeys())
        if num_datapoints > max_param:
            return self.accuracy_table[max_param]

        return -1


    def __str__(self):
        if self.use_backoff:
            return "{}-Backoff({})".format(self.name, self.backoff_curve)
        elif self.params is not None:
            return self.name + "(" + self.format_string.format(*self.params) + ")"
        else:
            return "Default({})".format(self.default_prediction)


class SimpleLogCurve(Curve):
    def __init__(self):
        super(SimpleLogCurve, self).__init__(lambda x, a: a * numpy.pow(numpy.log(x + 1), 0.5), "simple log",
                                             "{0} * log(x + 1) ^ 0.5")

    def fit(self, datapoints):
        max_point = max(datapoints, key=lambda p: p[0])
        self.params = [max_point[1] / self.function(max_point[0], 1.)]


class LinearMetaCurve(Curve):
    def __init__(self, curves, name="LinearMetaCurve"):
        super(LinearMetaCurve, self).__init__(None, name, None)
        self.curves = list(curves)

    def fit(self, datapoints):
        for curve in self.curves:
            curve.fit(datapoints)

    def predict(self, x):
        predictions = [curve.predict(x) for curve in self.curves]

        # filter any that fail and fall to backoff
        predictions = [p for p in predictions if p > 0]

        if predictions:
            return numpy.mean(predictions)
        else:
            return self.default_prediction

    def __str__(self):
        return "{}: {}".format(self.name, " | ".join(str(c) for c in self.curves))


def get_best_curve():
    log_curve_1 = Curve(lambda x, b: b * numpy.power(numpy.log(x + 1), 0.5), "log 1p", "{0} * log(x + 1) ^ 0.5")

    log_curve_3 = Curve(lambda x, a, b, c: b * numpy.power(numpy.log(x + a + 1), c), "log 3p backoff", "{1} * (log(x + {0} + 1) ^ {2}", backoff_curve=log_curve_1, min_points=6)

    asymptote_curve = Curve(lambda x, a, b: x / (x + a ** 2) * b, "asymptote 2p backoff", "x / (x + {0}^2) * {1}", backoff_curve=log_curve_1, min_points=3)

    combined_curve = LinearMetaCurve([log_curve_3, asymptote_curve], name="average")
    combined_curve.set_accuracy_table({2: 0.340, 3: 0.114, 4: 0.068, 5: 0.061, 6: 0.037, 7: 0.028, 8: 0.019}, transform=lambda v: 100 * (1 - v))

    return combined_curve


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()

    data = [(0, 0), (2.2069444444444444, 66844), (3.1944444444444446, 71377), (4.186111111111111, 77021),
            (5.218055555555556, 83822), (5.288888888888889, 84046), (6.292361111111111, 86508),
            (7.298611111111111, 89478), (8.36875, 92649), (9.373611111111112, 94531), (10.377777777777778, 95869),
            (11.384027777777778, 96871), (12.3875, 97685), (13.390972222222222, 98331), (14.394444444444444, 99435),
            (15.397916666666667, 100797), (16.40138888888889, 101751), (17.404166666666665, 102434),
            (18.40763888888889, 103014), (19.41111111111111, 103555), (20.413888888888888, 104097),
            (21.417361111111113, 104915), (22.42013888888889, 105829), (23.42361111111111, 106560),
            (24.426388888888887, 107230), (25.429166666666667, 107795)]

    log_curve = Curve(lambda x, a, b, c: b * numpy.power(numpy.log(x + a + 1), c), "log", "{1} * (log(x + {0} + 1) ^ {2}")
    log_curve_simple = Curve(lambda x, b: b * numpy.power(numpy.log(x + 1), 0.5), "log root", "{0} * log(x + 1) ^ 0.5")
    inverse_curve = Curve(lambda x, a, b: x / (x + a ** 2) * b, "inv", "x / (x + {0}^2) * {1}")

    evaluation_suite = EvaluationSuite([Evaluation(3), Evaluation(4), Evaluation(5)],
                                       [log_curve, log_curve_simple, inverse_curve])
    for evaluation, model, score, effective_training_size in evaluation_suite.evaluate(data):
        print "{}, [{:.3f}], {}".format(evaluation, score, model)


if __name__ == "__main__":
    sys.exit(main())
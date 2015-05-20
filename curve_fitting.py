from __future__ import unicode_literals
import logging
import sys
import argparse
import download_graph

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

    def evaluate(self, model, datapoints, graph_file=None):
        training_data = [p for p in datapoints if p[0] < self.x_max]
        testing_data = [p for p in datapoints if self.testing_x_range[0] <= p[0] <= self.testing_x_range[1]]

        if not training_data:
            raise InsufficientDataError()

        if not testing_data:
            raise InsufficientTestingDataError()

        model.fit(training_data)

        if graph_file:
            pred = [(p[0], model.predict(p[0])) for p in datapoints]

            download_graph.make_downloads_graph(datapoints, graph_file, prediction_data=pred)

        predictions = [model.predict(p[0]) for p in testing_data]
        return self.score([p[1] for p in testing_data], predictions)

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

    def evaluate(self, datapoints):
        for evaluation in self.evaluations:
            assert isinstance(evaluation, Evaluation)

            scores = [evaluation.evaluate(model, datapoints) for model in self.models]
            for model, score in zip(self.models, scores):
                yield evaluation, model, score


class Curve(object):
    def __init__(self, function, name, format_string, backoff_curve=None):
        self.logger = logging.getLogger(__name__ + ".Curve")
        self.function = function
        self.backoff_curve = backoff_curve
        self.name = name
        self.format_string = format_string
        self.params = None

        self.y_max = None

    def fit(self, datapoints):
        x, y = zip(*datapoints)
        x = numpy.array(x)
        y = numpy.array(y)

        self.y_max = y.max()

        try:
            self.params, opt_covariance = scipy.optimize.curve_fit(self.function, x, y)
        except (TypeError, RuntimeError):
            self.logger.info("Fitting curve failed, backing off")
            if self.backoff_curve:
                self.backoff_curve.fit(datapoints)

    def predict(self, x):
        if self.params is not None:
            return min(self.function(x, *self.params), self.y_max * 2)
        elif self.backoff_curve:
            return self.backoff_curve.predict(x)

    def __str__(self):
        if self.params is not None:
            return self.name + "(" + self.format_string.format(*self.params) + ")"
        elif self.backoff_curve:
            return "{}-Backoff({})".format(self.name, self.backoff_curve)
        else:
            return "Unfit curve"


class ConstantCurve(Curve):
    def fit(self, datapoints):
        pass

    def predict(self, x):
        return self.function(x)

    def __str__(self):
        return self.format_string


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

    backoff_zero = ConstantCurve(lambda x: 0, "Zero", "0")

    log_curve = Curve(lambda x, a, b, c: b * numpy.power(numpy.log(x + a + 1), c), "log", "{1} * (log(x + {0} + 1) ^ {2}",
                      backoff_curve=backoff_zero)
    log_curve_simple = Curve(lambda x, b: b * numpy.power(numpy.log(x + 1), 0.5), "log root", "{0} * log(x + 1) ^ 0.5",
                             backoff_curve=backoff_zero)
    inverse_curve = Curve(lambda x, a, b: x / (x + a ** 2) * b, "inv", "x / (x + {0}^2) * {1}", backoff_curve=backoff_zero)

    evaluation_suite = EvaluationSuite([Evaluation(3), Evaluation(4), Evaluation(5)],
                                       [log_curve, log_curve_simple, inverse_curve])
    for evaluation, model, score in evaluation_suite.evaluate(data):
        print "{}, [{:.3f}], {}".format(evaluation, score, model)


if __name__ == "__main__":
    sys.exit(main())
from __future__ import unicode_literals
import logging
import sys
import argparse
import math
import numpy
import scipy


class InsufficientDataError(ValueError):
    pass

class InsufficientTestingDataError(InsufficientDataError):
    pass


class Evaluation(object):
    def __init__(self, x_max):
        self.logger = logging.getLogger(__name__ + ".Evaluation")
        self.x_max = x_max
        self.testing_x_range = (7, 8)

        if self.x_max >= self.testing_x_range[0]:
            self.logger.warning(
                "Testing x range overlaps training range: {} vs {}".format(self.x_max, self.testing_x_range))

    def evaluate(self, model, datapoints):
        training_data = [p for p in datapoints if p[0] < self.x_max]
        testing_data = [p for p in datapoints if p[0] >= self.testing_x_range[0] and p[0] <= self.testing_x_range[1]]

        if not training_data:
            raise InsufficientDataError()

        if not testing_data:
            raise InsufficientTestingDataError()

        model.fit(training_data)

        predictions = [model.predict(p[0]) for p in testing_data]
        return self.score([p[1] for p in testing_data], predictions)

    def score(self, reference_y, predicted_y):
        errors = []
        for ref, pred in zip(reference_y, predicted_y):
            errors.append(abs(pred - ref) / float(ref))

        return sum(errors) / len(errors)


class Curve(object):
    def __init__(self, function, format_string):
        self.logger = logging.getLogger(__name__ + ".Curve")
        self.function = function
        self.format_string = format_string
        self.params = None

    def fit(self, datapoints):
        x, y = zip(*datapoints)
        x = numpy.array(x)
        y = numpy.array(y)

        self.params, opt_covariance = scipy.optimize.curve_fit(self.function, x, y)
        self.logger.info("Fitting curve to {}".format(datapoints))
        self.logger.info("Fit params {} to data".format(self.params))

    def predict(self, x):
        return self.function(*self.params)

    def __str__(self):
        return self.format_string.format(*self.params)


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()

    data = [(0, 0), (0.21041666666666667, 28418), (1.2006944444444445, 57567), (3.027083333333333, 84158), (4.211805555555555, 90606), (5.4215277777777775, 94122), (9.20138888888889, 104490), (10.188888888888888, 105996), (11.180555555555555, 107288), (12.2125, 108501), (12.283333333333333, 108573), (13.286805555555556, 109588), (14.293055555555556, 111004), (15.363194444444444, 112638), (16.368055555555557, 113675), (17.372222222222224, 114421), (18.37847222222222, 115140), (19.381944444444443, 115780), (20.385416666666668, 116267), (21.38888888888889, 117132), (22.39236111111111, 118233), (23.395833333333332, 118992), (24.398611111111112, 119553)]
    predictor = lambda x, a, b, c: b * numpy.power(numpy.log(x + a + 0.1), c)
    curve = Curve(predictor, "{1} * (log(x + {0} + 0.1) ^ {2}")
    evaluation = Evaluation(4)
    print evaluation.evaluate(curve, data)


if __name__ == "__main__":
    sys.exit(main())
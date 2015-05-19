from __future__ import unicode_literals
import logging
import sys
import argparse
import math


class InsufficientDataError(ValueError):
    pass

class Evaluation(object):
    def __init__(self, x_max):
        self.logger = logging.getLogger(__name__)
        self.x_max = x_max
        self.testing_x_range = (7, 8)

        if self.x_max >= self.testing_x_range[0]:
            self.logger.warning("Testing x range overlaps training range: {} vs {}".format(self.x_max, self.testing_x_range))

    def evaluate(self, model, datapoints):
        training_data = [p for p in datapoints if p[0] < self.x_max]
        testing_data = [p for p in datapoints if p[0] >= self.testing_x_range[0] and p[0] <= self.testing_x_range[1]]

        if not training_data:
            raise InsufficientDataError()
        if not testing_data:
            raise InsufficientDataError()

        model.fit(training_data)

        predictions = [model.predict(p[0]) for p in testing_data]
        return self.score([p[1] for p in testing_data], predictions)

    def score(self, reference_y, predicted_y):
        errors = []
        for ref, pred in zip(reference_y, predicted_y):
            errors.append(abs(pred - ref) / float(ref))

        return sum(errors) / len(errors)


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    sys.exit(main())
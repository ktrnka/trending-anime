from __future__ import unicode_literals
import sys
import argparse
import math


class InsufficientDataError(ValueError):
    pass

class Evaluation(object):
    def __init__(self):
        self.max_days_training = 4
        self.predict_days_range = (7, 8)

    def evaluate(self, model, datapoints):
        training_data = [p for p in datapoints if p[0] < self.max_days_training]
        testing_data = [p for p in datapoints if p[0] >= self.predict_days_range[0] and p[0] <= self.predict_days_range[1]]

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
            errors.append(math.abs(pred - ref) / ref)

        return sum(errors) / float(len(errors))


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    sys.exit(main())
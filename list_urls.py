from __future__ import unicode_literals
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()

    for i in xrange(1, 25):
        print "http://www.nyaa.se/?cats=1_37&offset={}".format(i)


if __name__ == "__main__":
    sys.exit(main())
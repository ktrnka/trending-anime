
"""
Make graphs of downloads by day, heavily inspired by
http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
"""
import sys
import matplotlib.pyplot as plt

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)


def choose_ticks(y_min, y_max):
    interval_priorities = [20000, 10000, 5000, 1000, 100]
    for interval in interval_priorities:
        ticks = range(y_min + interval, y_max, interval)
        if len(ticks) > 1:
            return ticks

    raise ValueError("No appropriate axis ticks found")


def make_downloads_graph(data, filename, prediction_data=None):
    x_data, y_data = zip(*data)

    x_min = 0
    x_max = max(x_data) * 1.02
    y_min = 0
    y_max = max(y_data) * 1.02

    plt.figure(figsize=(12, 9))
    plt.plot(x_data, y_data, lw=2, color=tableau20[0])
    if prediction_data:
        x_pred, y_pred = zip(*prediction_data)

        x_max = max(x_max, max(x_pred) * 1.02)
        y_max = max(y_max, max(y_pred) * 1.02)

        plt.plot(x_pred, y_pred, "--", lw=2, color=tableau20[1])

    plt.xlabel("Days since release", fontsize=16)
    plt.ylabel("Downloads", fontsize=16)
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_ticks = choose_ticks(y_min, int(y_max))
    plt.yticks(y_ticks, ["{:,}".format(y) for y in y_ticks], fontsize=14)
    plt.xticks(fontsize=14)
    for y in y_ticks:
        x_points = range(int(x_min), int(x_max + 1))
        plt.plot(x_points, [y] * len(x_points), "--", lw=0.5, color="black", alpha=0.3)

    plt.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    plt.axis([x_min, x_max, y_min, y_max])
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def main():
    data = [(0.0, 2963), (1.0, 17758), (2.1, 20749), (3.1, 22455), (4.1, 23447), (4.9, 24263), (5.0, 24326)]
    make_downloads_graph(data, "test_graph.png")


if __name__ == "__main__":
    sys.exit(main())
from typing import NoReturn
import matplotlib.pyplot as plt


N_ROWS = 4
N_COLS = 2


def plot_metric(name: str, result: list, ax: plt.Axes) -> NoReturn:
    """
    Create plot for metric
    :param name: name of metric
    :param result: metric values in successive epochs
    :param ax: axis
    :return: NoReturn
    """
    ax.set_title(name)
    ax.plot(range(len(result)), result)
    ax.set_ylim(0, 1.01)


def visualize_loss(metrics: dict) -> NoReturn:
    """
    Plots all metric results in successive epochs
    :param metrics: dictionary with metrics (w. metrics.MetricWriter)
    :return: NoReturn
    """
    fig, ax = plt.subplots(N_ROWS, N_COLS, figsize=(10, 10))
    dict_iterator = iter(metrics)
    for i in range(N_ROWS):
        for j in range(N_COLS):
            name = next(dict_iterator)
            result = metrics[name]
            plot_metric(name, result, ax[i, j] if N_COLS > 1 else ax[i])
    fig.tight_layout()
    fig.savefig('metrics.png')
    plt.show()

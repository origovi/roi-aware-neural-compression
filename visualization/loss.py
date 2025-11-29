import matplotlib.pyplot as plt
import numpy as np

def plot_loss(losses, descriptions, x_label, y_label, title, save_file_route=None):
    """
    Plots Loss values for multiple test cases.
    """
    show_legend = len(losses) >= 2 and len(losses) == len(descriptions)

    # Plotting
    for i in range(len(losses)):
        # Plot the curve
        if show_legend: plt.plot(np.arange(len(losses[i])), losses[i], marker='o', linestyle='-', label=descriptions[i])
        else: plt.plot(np.arange(len(losses[i])), losses[i], marker='o', linestyle='-', label=descriptions[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if show_legend: plt.legend()
    plt.grid(True)
    if save_file_route is None:
        plt.show()
    else:
        plt.savefig(save_file_route, pad_inches=0)
        plt.close()
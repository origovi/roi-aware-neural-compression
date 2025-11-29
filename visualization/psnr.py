import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_psnr(mses, bpps, descriptions, title, save_file_route=None):
    """
    Plots PSNR values for multiple test cases.
    """
    max_pixel = 1.0  # Assuming 8-bit images
    show_legend = len(mses) >= 2 and len(mses) == len(descriptions)


    # Plotting
    for i in range(len(mses)):
        if len(mses[i]) != len(bpps[i]):
            raise ValueError(f"Mismatch in length of MSEs and bpps for curve {i}.")

        # Compute PSNR values for this curve
        psnr_values = [10 * np.log10((max_pixel ** 2) / mse) if mse > 0 else float('inf') for mse in mses[i]]

        # Sort values by bpp for a smooth curve
        # sorted_indices = np.argsort(bpps[i])
        # sorted_bpps = np.array(bpps[i])[sorted_indices]
        # sorted_psnr = np.array(psnr_values)[sorted_indices]

        # Plot the curve
        if show_legend: plt.plot(bpps[i], psnr_values, marker='o', linestyle='-', label=descriptions[i])
        else: plt.plot(bpps[i], psnr_values, marker='o', linestyle='-')

        # Annotate points
        if i < 2:
            for j in range(len(mses[i])):
                plt.annotate(f'({chr(100 + j)})', (bpps[i][j], psnr_values[j]), textcoords="offset points", xytext=(-5,5), ha='center')
        else:
            for j in range(len(mses[i])):
                plt.annotate(f'({chr(97 + j)})', (bpps[i][j], psnr_values[j]), textcoords="offset points", xytext=(-5,5), ha='center')

    plt.xlabel("Bits per Pixel (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.title(title)
    if show_legend: plt.legend()
    plt.grid(True)
    if save_file_route is None:
        plt.show()
    else:
        # plt.savefig(save_file_route, bbox_inches='tight')
        plt.savefig(save_file_route)
        plt.close()
import numpy as np
import pylab
import matplotlib.pyplot as plt
from pprint import pprint


# Plotting Parameters
color_cfd = "k"
color_exp = "r"
linewidth_global = 0.6


def import_data(path: str, exp=True) -> np.ndarray:

    n_skip_head = 6
    n_skip_foot = 1

    return np.genfromtxt(path, skip_header=n_skip_head, skip_footer=n_skip_foot)[:, (0, -1)]




def format_cfd_data(data: np.ndarray) -> np.ndarray:
    formatted_data = np.zeros(shape=data.shape)
    formatted_data[:, 0] = data[:, 0]
    formatted_data[:, 1] = data[:,1]

    return formatted_data


def plot_residual(res1=[], res2=[], res3=[]) -> None:
    plt.plot(res1[:, 0], res1[:, 1], linewidth=1.5, color="g", label="SSG 6 AoA fine mesh")
    plt.plot(res2[:, 0], res2[:, 1], linewidth=1.5, color="b", label="k-omega 6 AoA fine mesh")
    plt.plot(res3[:, 0], res3[:, 1], linewidth=1.5, color="orange", label="first order upwind 6 AoA fine mesh")

    plt.yscale('log')
    plt.legend(fontsize=18)
    plt.xlabel("Iterations", size=20)
    plt.ylabel("RMS U-Mom",size=20)
    plt.grid()
    plt.show()

    return None

#read residual data

res_6_c = format_cfd_data(import_data("residuals_6_c.csv"))
res_6_f = format_cfd_data(import_data("residuals_6_f.csv"))
res_13_f = format_cfd_data(import_data("residual_131_f.csv"))
res_6_SSG = format_cfd_data(import_data("residual_6_SSG.csv"))
res_6_upwind = format_cfd_data(import_data("residuals_6_upwind.csv"))
res_13_SSG = format_cfd_data(import_data("residual_13_SSG.csv"))


plot_residual(res_6_SSG, res_6_f, res_6_upwind)


#print(import_data("Cp_flap_6_c.csv",False))
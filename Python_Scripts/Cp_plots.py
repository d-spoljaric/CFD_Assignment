import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


# Plotting Parameters
color_cfd = "k"
color_exp = "r"
linewidth_global = 0.6


def import_data(path: str, exp=True) -> np.ndarray:
    if exp:
        n_skip_head = 2
        n_skip_foot = 1

        return np.genfromtxt(path, skip_header=n_skip_head, skip_footer=n_skip_foot)
    else:
        n_skip_head = 6
        n_skip_foot = 1

        return np.genfromtxt(path, skip_header=n_skip_head, skip_footer=n_skip_foot)[:, (0, -1)]




def format_cfd_data(data: np.ndarray) -> np.ndarray:
    formatted_data = np.zeros(shape=data.shape)
    formatted_data[:, 0] = ((data[:, 0] - np.min(data[:,0]))/ np.max(data[:, 0]))
    formatted_data[:, 1] = data[:,1]

    return formatted_data


def plot_pressure_coeff(exp_data: np.ndarray, cfd_data=[], cfd=False) -> None:
    plt.plot(exp_data[:, 0], exp_data[:, 1], linewidth=linewidth_global, color=color_exp, label="Experimental")

    if cfd:
        plt.scatter(cfd_data[:, 0], cfd_data[:, 1], color=color_cfd, marker=".", s=0.6, label="CFD")

    plt.legend()
    plt.gca().invert_yaxis()
    plt.xlabel("x/c")
    plt.ylabel("$c_{p}$")
    plt.show()

    return None

#read experimental data
wing_exp_6 = import_data("ex_cp_wing_6.dat", True)
wing_exp_13 = import_data("ex_cp_wing_13.dat", True)
flap_exp_6 = import_data("ex_cp_flap_6.dat", True)
flap_exp_13 = import_data("ex_cp_flap_13.dat", True)

#read cfd flap data
flap_cfd_6_c = format_cfd_data(import_data("Cp_flap_6_c.csv", False))
flap_cfd_6_f = format_cfd_data(import_data("Cp_flap_6_f.csv", False))
flap_cfd_13_f = format_cfd_data(import_data("Cp_flap_131_f.csv", False))
flap_cfd_6_upwind = format_cfd_data(import_data("Cp_flap_6_upwind.csv", False))
flap_cfd_6_SSG = format_cfd_data(import_data("Cp_flap_6_SSG.csv", False))

#read cfd wing data
wing_cfd_6_c = format_cfd_data(import_data("Cp_wing_6_c.csv", False))
wing_cfd_6_f = format_cfd_data(import_data("Cp_wing_6_f.csv", False))
wing_cfd_13_f = format_cfd_data(import_data("Cp_wing_131_f.csv", False))
wing_cfd_6_upwind = format_cfd_data(import_data("Cp_wing_6_upwind.csv", False))
wing_cfd_6_SSG = format_cfd_data(import_data("Cp_wing_6_SSG.csv", False))


plot_pressure_coeff(wing_exp_13, wing_cfd_13_f, True)

print(import_data("Cp_flap_6_c.csv",False))
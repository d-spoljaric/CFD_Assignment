import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

# Simulation Variables
v_ref = 63.476
p_ref = 101325
rho_ref = p_ref/(287.1*293)
c_ref = 0.5948

# Plotting Parameters
color_cfd = "k"
color_exp = "r"
linewidth_global = 0.6


def import_data(path: str, exp = True) -> np.ndarray:
    if exp:
        n_skip_head = 2
        n_skip_foot = 1
    
        return np.genfromtxt(path, skip_header=n_skip_head, skip_footer=n_skip_foot)
    else:
        n_skip_head = 6
        n_skip_foot = 1
        
        return np.genfromtxt(path, skip_header=n_skip_head, skip_footer=n_skip_foot)[:, (0, -1)]
        
def rel_to_absolute(value: np.ndarray, ref_val: float) -> np.ndarray:
    return ref_val + value

def calc_pressure_coeff(pressure: np.ndarray, pressure_inf: float, rho_inf: float, v_inf: float) -> np.ndarray:
    return (pressure - pressure_inf)/(0.5 * rho_inf* (v_inf**2))

def format_cfd_data(data: np.ndarray) -> np.ndarray:
    formatted_data = np.zeros(shape = data.shape)
    formatted_data[:, 0] = data[:, 0]/np.max(data[:, 0])
    formatted_data[:, 1] = calc_pressure_coeff(rel_to_absolute(data[:, 1], p_ref), p_ref, rho_ref, v_ref)
    
    return formatted_data

def plot_pressure_coeff(exp_data: np.ndarray, cfd_data = [], cfd = False) -> None:
    plt.plot(exp_data[:, 0], exp_data[:, 1], linewidth = linewidth_global, color = color_exp, label = "Experimental")
    
    if cfd:
        plt.scatter(cfd_data[:, 0], cfd_data[:, 1], color = color_cfd, marker = ".", s = 0.6, label = "CFD")
        
    plt.legend()
    plt.gca().invert_yaxis()
    plt.xlabel("x/c")
    plt.ylabel("$c_{p}$")
    plt.show()
    
    return None



wing_exp_6_data = import_data("ex_cp_wing_6.dat", True)
wing_exp_13_data = import_data("ex_cp_wing_13.dat", True)
flap_exp_6_data = import_data("ex_cp_flap_6.dat", True)
flap_exp_13_data = import_data("ex_cp_flap_13.dat", True)

wing_cfd_6_data = format_cfd_data(import_data("wing_pressure.csv", False))

plot_pressure_coeff(wing_exp_6_data, wing_cfd_6_data, True)
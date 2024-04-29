# %% to produce contour plots from a design analysys
## Dependencies
from libs.timing_diagram import *
from libs.design_functions import *
import numpy as np
import matplotlib.pyplot as plt
from libs.spherical_earth_geometry_radar import *
from libs.radartools.farField import UniformAperture
from libs.ambiguity_functions import *
from unambiguous_imaging_design_analysis import *
import matplotlib as mpl
import json
import pickle

# %% opening files

# Load the results_list
with open('analysis_results/results_list.pk', 'rb') as f:
    results_list = pickle.load(f)
# Load the x_coordinate
with open('analysis_results/x_coordinate.pk', 'rb') as f:
    x_coordinate = pickle.load(f)
# Load the y_coordinate
with open('analysis_results/y_coordinate.pk', 'rb') as f:
    y_coordinate = pickle.load(f)
# Load the undersampling_list
with open('analysis_results/undersampling_list.pk', 'rb') as f:
    undersampling_list = pickle.load(f)
# Load the rasr_free_swath_list
with open('analysis_results/rasr_free_swath_list.pk', 'rb') as f:
    rasr_free_swath_list = pickle.load(f)
# Load the core_snr_list
with open('analysis_results/core_snr_list.pk', 'rb') as f:
    core_snr_list = pickle.load(f)

with open("analysis_results/settings.json", "r") as read_file:
    data = json.load(read_file)

La = data["antenna_l"]
Wa = data["antenna_w"]
wavelength = data["wavelength"]
vs = data["orbital_speed"]
h = data["altitude"]
dutycycle = data["dutycycle"]
nadir_duration = data["nadir_duration"]

# %% Defining the thresholds for the contours
minimum_ground_swath = np.array([20e3, 30e3, 40e3])  # m
minimum_azimuth_resolution = np.array([5])  # m

# %% Filtering the designs
filtered_designs, bounds = designs_filter(minimum_ground_swath, minimum_azimuth_resolution, results_list)

# %% Plotting an empty timing diagram
## canonical constraints line plots
# 1 ground range axis in m
ground_range = np.linspace(0, 2000, 500) * 1000

prf_opt = optimal_prf_line(ground_range, Wa, wavelength, h)

fig, (ax, ax1) = plt.subplots(2, dpi=150, gridspec_kw={'height_ratios': [30, 1]})

# minimum lines
prf_min = minimum_prf_line(ground_range, vs, La, h)
PRI = 1 / 7050
prf = np.linspace(1 / PRI - 1500, 1 / PRI + 10000, 100)

time_diagram_plotter(ax, prf, dutycycle, h, nadir=False, integrationtime=False, color='dimgray')
nadir_return_plotter(ax, prf, dutycycle, nadir_duration, h, color='silver')
ax.set_xlabel('PRF [Hz]')
ax.set_ylabel(' Ground range [km]')
ax.set_xlim(1 / PRI - 1000, 1 / PRI + 1000)
ax.set_ylim(100, 300)

## canonical constraints line plots
ax.plot(prf_opt, ground_range / 1000, 'k')
ax.plot(prf_min, ground_range / 1000, '--k')
ax.set_title('Winners visualization')
fig.tight_layout()

# adjust the plot scale
ax.set_xlim(np.array(x_coordinate).min() - 3e2, np.array(x_coordinate).max() + 3e2)
ax.set_ylim(np.array(y_coordinate).min() / 1e3 - 30, np.array(y_coordinate).max() / 1e3 + 30);

# %% Plot the contour lines
styles = ['-r', '-.r', ':r'] # for the 3 contours
for ii in range(len(bounds)):
    design_set_boundary(ax, results_list, filtered_designs[ii], linestyle = styles[ii], labl=str(bounds[ii]))

ax.legend()


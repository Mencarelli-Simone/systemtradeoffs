#########################################################################################
#      Simone  Mencarelli                                                               #
#       23/2/23                                                                         #
#                                                                                       #
# This file takes the analysys of  the notebook unambiguous-imaging-design-analysys     #
# and makes it into a function.                                                         #
# An interactive main script is also provided to analyze dewsign points manually.       #
#########################################################################################


# %% Dependencies
import sys
from libs.timing_diagram import *
from libs.design_functions import *
import numpy as np
import matplotlib.pyplot as plt
from libs.spherical_earth_geometry_radar import *
from libs.radartools.farField import Aperture, UniformAperture
from libs.ambiguity_functions import *
import matplotlib
matplotlib.use('qt5Agg')
from matplotlib.widgets import Button

# screen dpi,
dpii = 100


# %%

def optimal_prf_line(ground_range, Wa, wavelength, h=500e3, c=299792458):
    """
    minimum prf condition for ideal aperture of width Wa
    :param ground_range: ground range axis of timing diagram
    :param Wa: Antenna width in elevation
    :param wavelength: radar wavelength
    :param h: radar altitude
    :param c: speed of light
    :return: prf points
    """
    # %% (In5)
    # 1 ground range axis in m
    # 2 incidence angle
    slant_range = range_ground_to_slant(ground_range, h)
    # loopy but functional
    ground_range, eta = range_slant_to_ground(slant_range, h)  # the incidence angle is the second parameter returned
    # 3 incidence range
    eta1 = eta - wavelength / (2 * Wa)
    eta2 = eta + wavelength / (2 * Wa)
    slant_range_1, ground_range_1 = range_from_theta(eta1 * 180 / np.pi, h)
    slant_range_2, ground_range_2 = range_from_theta(eta2 * 180 / np.pi, h)
    # 4 slant range delta
    slant_delta_range = slant_range_2 - slant_range_1
    # 5 prf from slant range
    pri = 2 * slant_delta_range / c
    prf_opt = 1 / pri
    return prf_opt


def minimum_prf_line(ground_range, vs, La, h=500e3):
    """
    minimum PRF To correctly sample the doppler bandwidth (LFM approx)
    :param ground_range: ground range axis
    :param vs: satellite speed
    :param La: antenna length (ideal aperture)
    :param h: satellite altitude
    :return: minimum PRF
    """
    prf_min = 2 * ground_speed(ground_range, vs, h) / La
    return prf_min


def closest_valid_timing_selection(h, rg_coordinate,
                                   prf_coordinate, dutycycle,
                                   c=299792458):
    """
    :param h:
    :param rg_coordinate:
    :param prf_coordinate:
    :param dutycycle:
    :param c:
    :return: dict:{
        'ground-swath': array of 2 points: ground range between transmit events (considering pulse duration),
        'prf': array of 2 points: pulse repetition frequency,
        'compressed-ground-swath': array of 2 points: ground swath after pulse compression (integration time included),
        'compressed-slant-swath': array of 2 points: slant range swath after pulse compression,
        'compressed-incidence-swath': array of 2 points: incidence angles (spherical earth) at the extremes of
        the compressed swath
    }
    """

    # Find the swath given a point in the time diagram
    # i.e.
    # find the closest end of transmission
    # find the closest start of transmission

    # (In 12)
    swath_prf = np.zeros(2)
    swath_rg = np.zeros_like(swath_prf)
    swath_rg[0], swath_prf[0] = last_end_of_transmission(rg_coordinate, prf_coordinate, dutycycle, h)
    swath_rg[1], swath_prf[1] = next_start_of_transmission(rg_coordinate, prf_coordinate, dutycycle, h)

    # total swath considering compression time (In 13)
    # 1 swath from pulse to pulse
    swath_rs = range_ground_to_slant(swath_rg, h)
    # 2 swath without 1 pulse period (half on each side)
    swath_rs_compressed = swath_rs + np.array(
        [(1 / swath_prf[0] * dutycycle * c / 4), - 1 / swath_prf[1] * dutycycle * c / 4])
    # 3 convert back to ground range
    swath_rg_compressed, swath_eta_compressed = range_slant_to_ground(swath_rs_compressed, h)

    return_dict = {
        'ground-swath': np.nan_to_num(swath_rg),
        'prf': swath_prf,
        'compressed-ground-swath': np.nan_to_num(swath_rg_compressed),
        'compressed-slant-swath': swath_rs_compressed,
        'compressed-incidence-swath': swath_eta_compressed
    }

    return return_dict


def umambiguous_mode_analysis(radar_geo: RadarGeometry,
                              uniap: UniformAperture, rg_coordinate,
                              prf_coordinate, AASR_max, RASR_max,
                              vs, wavelength, dutycycle,
                              aasr_points=5, rasr_points=40,
                              c=299792458):
    """
    optimize the design given an initial prf and ground range point for the desired maximum AASR and RASR.
    note, both the undersampling ratio and the rasr free swath are approximated using only a few points chosen
    accordingly to rasr_points and aasr_points. #todo complete this manual
    :param radar_geo:
    :param uniap:
    :param rg_coordinate:
    :param prf_coordinate:
    :param AASR_max:
    :param RASR_max:
    :param vs:
    :param wavelength:
    :param dutycycle:
    :param aasr_points:
    :param rasr_points:
    :param c:
    :return:
    """

    # platform height
    h = radar_geo.S_0[2]
    # Find the swath given a point in the time diagram
    # i.e.
    # find the closest end of transmission
    # find the closest start of transmission
    td_vals = closest_valid_timing_selection(h, rg_coordinate, prf_coordinate, dutycycle, c)

    # extracting the values from the return dictionary
    swath_rg = td_vals['ground-swath']
    swath_prf = td_vals['prf']
    swath_rs_compressed = td_vals['compressed-slant-swath']
    swath_eta_compressed = td_vals['compressed-incidence-swath']
    swath_rg_compressed = td_vals['compressed-ground-swath']

    # 1. Define a RadarGeometry centered at the center of the swath
    # 2. adjust the looking angle to maximize NESZ
    # 3. compute RASR and AASR
    # 4. Plot RASR and  AASR over doppler bandwidth / processed bandwidth
    # 5. Print NESZ at the swath center

    # 1 (In 15)
    # swath center incidence angle
    rg_center, eta_center = range_slant_to_ground(np.average(swath_rs_compressed), h)
    looking_angle = incidence_angle_to_looking_angle(eta_center, h)
    radar_geo.set_rotation(looking_angle, 0, 0)

    # 2 (In 16)
    # Optimization error function (core SNR in spherical earth at the edges of the swath)
    error_func = lambda b: snr_error(b, swath_eta_compressed[0], swath_eta_compressed[1], radar_geo, uniap)
    # b is the looking angle of the radar
    # Optimization using python native optimization methods
    looking_angle_opt = fsolve(error_func, looking_angle, maxfev=100)
    print('optimized looking angle:', looking_angle_opt[0] * 180 / np.pi)
    # set the new looking angle
    radar_geo.set_rotation(float(looking_angle_opt), 0, 0)

    # 3 (In 17)
    # find the RASR over the observable range
    # support axis
    incidence_axis = np.linspace(swath_eta_compressed[0], swath_eta_compressed[1], rasr_points)
    slant_range_axis, ground_range_axis = range_from_theta(incidence_axis * 180 / np.pi, h)
    # nominal Doppler bandwidth
    Bd = 2 * ground_speed(rg_center, vs, h) / uniap.L * 0.5
    rasr = RASR(radar_geo, uniap, incidence_axis,
                1 / prf_coordinate, Bd, wavelength, vs, h, pbaroff=False)
    # todo return this

    # AASR (In 19)
    doppler_undersampling_ratio = np.linspace(.01, 1, aasr_points)
    aasr = np.zeros_like(doppler_undersampling_ratio)
    for ii in tqdm(range(len(aasr))):
        aasr[ii] = AASR(radar_geo, uniap, eta_center * 180 / np.pi, prf_coordinate,
                        Bd * doppler_undersampling_ratio[ii],
                        wavelength, pbaroff=True)

    # Core SNR (In 21)
    core_snr, azres = core_snr_spherical(radar_geo, uniap, eta_center, wavelength, vs, h, 1)

    # RASR threshold, numerical on the computed vector, no recalculation. (In 22)
    # find the swath with rasr smaller than RASR min
    indexes = np.argwhere(rasr < 10 ** (RASR_max / 10))
    ranges = ground_range_axis[indexes]
    # differential integration
    swath_rasr = 0
    for ii in range(len(ranges) - 1):
        swath_rasr += ranges[-ii - 1] - ranges[-ii - 2]

    # AASR threshold linear interpolation between computed samples (In 24)
    indexes = np.argwhere(aasr < 10 ** (AASR_max / 10))
    if len(indexes) != 0:
        undersampling = doppler_undersampling_ratio[max(indexes)]
        # print(undersampling)
        if max(indexes) < len(aasr) - 1:
            # linear interpolation
            x = (AASR_max)
            x_1 = 10 * np.log10(aasr[max(indexes)])
            x_0 = 10 * np.log10(aasr[max(indexes) + 1])
            y_1 = doppler_undersampling_ratio[max(indexes)]
            y_0 = doppler_undersampling_ratio[max(indexes) + 1]
            undersampling = (y_0 * (x_1 - x) + y_1 * (x - x_0)) / (x_1 - x_0)
            if undersampling > 1:
                undersampling = 1
        # print(undersampling)

    else:  # it means that no undersampling ratio was found to stay below  AASR
        undersampling = 0

    # Core SNR with corrected undersampling
    core_snr_underprocessed, azres_underprocessed = core_snr_spherical(radar_geo, uniap, eta_center, wavelength, vs, h, undersampling)

    return_dict = {
        'ground-swath': swath_rg,
        'prf': swath_prf,
        'compressed-ground-swath': swath_rg_compressed,
        'compressed-slant-swath': swath_rs_compressed,
        'compressed-incidence-swath': swath_eta_compressed,
        'rasr-ground-range-axis': ground_range_axis,
        'rasr-slant-range-axis': slant_range_axis,
        'rasr-incidence-axis': incidence_axis,
        'rasr-linear': rasr,
        'rasr-doppler-bandwidth': Bd,
        'aasr-undersampling-axis': doppler_undersampling_ratio,
        'aasr-linear': aasr,
        'core-snr-linear': core_snr,
        'corrected-azimuth-resolution': azres,
        'minimum-undersampling': undersampling,
        'usable-rasr-swath': swath_rasr,
        'usable-swath-ranges': ranges,
        'optimized-looking-angle': looking_angle_opt[0],
        'core-snr-linear-underprocessed': core_snr_underprocessed,
        'corrected-azimuth-resolution-underprocessed': azres_underprocessed
    }
    return return_dict

    # ifmain here

    # 1 plot the tddiagram and 2 empty figures for aasr rasr
    # 2 add ideal line
    # 2 infinite loop
    # 3 handele td click events by computing the swath
    # 4 add a gui button to initiate rasr and aasr computing and update the plots


# visualizer
def design_point_visualizer(axis, dictionary, scalar_element, colormap, scalar_min, scalar_max,
                            labeling=True,
                            cmapaxis=0,
                            scale=1,
                            unit=' ',
                            logscale=False, h=500e3,
                            re=6371e3):
    """

    :param axis: matplotlib axis for the plot
    :param dictionary: dictionary from analysis
    :param scalar_element: a string containing the dictionary key pointing to a scalar value
    :param colormap: matplotlib colormap key
    :param scalar_min: minimum scalar value for the colorbar, linear
    :param scalar_max: maximum scalar value for the colorbar, linear
    :param labeling: default True, adds a label to every line in the plot
    :param cmapaxis: axis for the colormap, if not set, no colormap is produced
    :param scale: multiplier for the scalar value
    :param unit: string to display on the label after the number
    :param logscale: if set to true, the scalar value is converted to decibel
    :return: stocazzo
    """
    #1 find holes in the usable swath ranges, note the swath ranges are linearly spaced over the incidence angle axis
    ranges = dictionary['usable-swath-ranges']
    slant_ranges = range_ground_to_slant(ranges, h, re)
    ranges1, incidences = range_slant_to_ground(slant_ranges, h, re)
    deltas = incidences[1:-1] - incidences[0:-2]  # so far so good
    if deltas.size != 0:
        delta = min(deltas)
    else:
        delta = 0
    #print(delta)
    holes = np.argwhere(deltas > 1.01 * delta)  #contains the indexes of the ranges element preceding a hole
    #2 color code for scalar value
    cmap = matplotlib.colormaps[colormap]  # todo change to colormap
    scalar = np.average(dictionary[scalar_element]) * scale
    if logscale:
        scalar = 10 * np.log10(scalar)
        scalar_min = 10 * np.log10(scalar_min)
        scalar_max = 10 * np.log10(scalar_max)
    mapped_color = cmap((scalar - scalar_min) / (scalar_max - scalar_min))
    #3 draw a line for every contiguous section of range
    if holes.size == 0:
        holes = np.array([-1])
    else:
        holes += 1
        holes = np.append(holes, -1)
    holes = np.insert(holes, 0, 0)
    #print(deltas)
    #print(holes)
    for ii in range(holes.size - 1):
        yy = ranges[holes[ii]: holes[ii + 1]]
        xx = np.ones_like(yy) * np.average(dictionary['prf'])
        label = str(round(scalar, 2)) + unit
        axis.plot(xx, yy / 1000, color=mapped_color, label=label, linewidth=2.5)
        if labeling:
            axis.annotate(label, (np.average(xx) + 2, np.average(yy) / 1000))

    # plot a colored dot if the range is zero
    if ranges.size == 0:
        label = str(round(scalar, 2)) + unit
        axis.plot(np.average(dictionary['prf']), np.average(dictionary['compressed-ground-swath']) / 1000, '.',
                  color=mapped_color, label=label, linewidth=2.5)
        if labeling:
            axis.annotate(label, (np.average(dictionary['prf']) + 2,
                                  np.average(dictionary['compressed-ground-swath']) / 1000))

    if cmapaxis != 0:
        # colormap
        #cmap = plt.cm.get_cmap(colormap) #deprecated
        cmap = matplotlib.colormaps[colormap]
        norm = matplotlib.colors.Normalize(vmin=scalar_min, vmax=scalar_max)
        cb1 = matplotlib.colorbar.ColorbarBase(cmapaxis, cmap=cmap,
                                        norm=norm,
                                        orientation='horizontal')
        cb1.set_label(unit)

# Design filter
def designs_filter(min_swath, min_az_resolution, results):
    """

    :param min_swath: numpy array of minimum swathes to select
    :param min_az_resolution: numpy array of minimum azimuth resolutions to select
    :param results:
    :return:    filtered_designs: list of numpy arrays containing the indexes of the selected designs for a min_swat - min_az_resolution pair
                boundaries: list of minimum swath and azimuth resolution corresponding to the filtered_designs list elements.
    """
    # MESHGRIDS i.e. all combinations
    GS, AR = np.meshgrid(min_swath, min_az_resolution)
    # flatten them all
    GS = GS.flatten()
    AR = AR.flatten()
    # unpack the corrected azimuth resolution and swath widths
    azimuth_resolutions = []
    swathes = []
    for dictionary in results:
        ar = float(dictionary["corrected-azimuth-resolution-underprocessed"])
        sw = float(dictionary["usable-rasr-swath"])
        azimuth_resolutions.append(ar)
        swathes.append(sw)
    # convert dictionaries to numpy arrays
    azimuth_resolutions = np.array(azimuth_resolutions)
    swathes = np.array(swathes)
    # thresholded lists
    filtered_designs = []  # list of index vectors
    # reference parameters
    boundaries = []
    for ii in range(len(AR)):
        indexes1 = np.argwhere(azimuth_resolutions <= AR[ii])
        indexes2 = np.argwhere(swathes >= GS[ii])
        intersection_set = np.intersect1d(indexes1, indexes2)
        filtered_designs.append(intersection_set)
        boundaries.append({'minmum-azimuth-resolution': AR[ii], 'minimum-ground-swath': GS[ii]})

    return filtered_designs, boundaries
# %% callbacks
# Filtered level curve
def design_set_boundary(axis, results, filtered_indexes, box=30, linestyle='k', labl=''):
    """

    :param axis: axis to plot on
    :param results: resuts list unfiltered
    :param filtered_indexes: indexes of the winning set
    :param box: boxcar filter length in sample default 30
    :param linestyle: arg variable for plot configuration
    :return:
    """
    # retrieve the coordinates
    x_coordinate = []
    y_coordinate = []
    y_top = []
    y_bottom = []
    for ii in range(len(results)):
        dictionary = results[ii]
        x_coordinate.append(np.average(dictionary['prf']))
        y_coordinate.append(
            np.average(dictionary['compressed-ground-swath']))
        if len(dictionary['usable-swath-ranges']) > 0:
            y_top.append(dictionary['usable-swath-ranges'].max())
            y_bottom.append(dictionary['usable-swath-ranges'].min())
        else:
            y_top.append(np.average(dictionary['compressed-ground-swath']))
            y_bottom.append(np.average(dictionary['compressed-ground-swath']))

    # compute the lines
    max_indexes = []
    min_indexes = []
    for ii in range(len(filtered_indexes) + box):
        indexes = 0
        if ii >= len(filtered_indexes):
            indexes = filtered_indexes[ii - box: len(filtered_indexes)]
        elif ii - box >= 0:
            indexes = filtered_indexes[ii - box: ii]
        else:
            indexes = filtered_indexes[0: ii]
        subset_range = np.array([y_coordinate[jj] for jj in indexes])
        if subset_range.size != 0:
            maxi = np.argmax(subset_range)
            mini = np.argmin(subset_range)
            if indexes[maxi] not in max_indexes:  # if not in list already
                max_indexes.append(indexes[maxi])
            if indexes[mini] not in min_indexes:
                min_indexes.append(indexes[mini])

    # plot the lines
    x_coordinate = np.array(x_coordinate)
    y_top = np.array(y_top)
    y_bottom = np.array(y_bottom)
    axis.plot(x_coordinate[min_indexes], y_bottom[min_indexes] * 1e-3, linestyle)
    axis.plot(x_coordinate[max_indexes], y_top[max_indexes] * 1e-3, linestyle)
    # close the lines
    axis.plot([x_coordinate[max_indexes[0]], x_coordinate[max_indexes[0]]],
              [y_top[max_indexes[0]] * 1e-3, y_bottom[min_indexes[0]] * 1e-3], linestyle)
    axis.plot([x_coordinate[min_indexes[-1]], x_coordinate[min_indexes[-1]]],
          [y_bottom[min_indexes[-1]] * 1e-3, y_top[max_indexes[-1]] * 1e-3], linestyle, label=labl)







def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    global update
    global rg_coordinate
    global prf_coordinate
    if event.dblclick:
        rg_coordinate1 = event.ydata
        prf_coordinate1 = event.xdata
        if ylim[0] < rg_coordinate1 < ylim[1] and xlim[0] < prf_coordinate1 < xlim[1]:
            rg_coordinate = rg_coordinate1
            prf_coordinate = prf_coordinate1
            update = True


def asrupdate_onclick(event):
    global asrupdate
    asrupdate = True
    print("updating AASR plot")


def on_press(event):
    print('press', event.key)
    sys.stdout.flush()


# %%
# global variables
update = False
rg_coordinate = 210e3  # m
prf_coordinate = 6800  # Hz
asrupdate = False

if __name__ == "__main__":
    # %% (In 3)
    # constants
    c = 299792458

    # radar parameters
    dutycycle = 0.25  # duty cycle
    h = 500e3  # height
    wavelength = c / 10e9  # wavelength

    # satellite speed
    vs = orbital_speed(h)

    # antenna size
    Wa = 0.3  # antenna width in meters
    La = 2  # antenna length in meters

    # nadir duration in fractions of PRI for visualization in timing diagram
    # note, it makes sense to use this fractional quantity as the nadir duration, if unfocused (e.g. saturated receiver), will be proportional to the impulse on time.
    nadir_duration = 2 * dutycycle

    # ASR limits
    AASR_max, RASR_max = -20, -20

    # %% (In 4)
    # PRF axis
    PRI = 1 / 7050
    prf = np.linspace(1 / PRI - 4000, 1 / PRI + 4000, 100)
    # %%
    # timing diagram plot
    fig, ax = plt.subplots(1, dpi=dpii)
    time_diagram_plotter(ax, prf, dutycycle, h, nadir=False, integrationtime=False)
    nadir_return_plotter(ax, prf, dutycycle, nadir_duration, h)
    ax.set_xlabel('PRF [Hz]')
    ax.set_ylabel(' Ground range [km]')
    ax.set_xlim(1 / PRI - 1000, 1 / PRI + 1000)
    ylim = 10, 300
    ax.set_ylim(ylim)
    # make space for the buttons
    fig.subplots_adjust(bottom=0.2)

    # %% (In5)
    # 1 ground range axis in m
    ground_range = np.linspace(0, 2000, 500) * 1000
    # 2 incidence angle
    slant_range = range_ground_to_slant(ground_range, h)
    # loopy but functional
    ground_range, eta = range_slant_to_ground(slant_range, h)  # the incidence angle is the second parameter returned
    # 3 incidence range
    eta1 = eta - wavelength / (2 * Wa)
    eta2 = eta + wavelength / (2 * Wa)
    slant_range_1, ground_range_1 = range_from_theta(eta1 * 180 / np.pi, h)
    slant_range_2, ground_range_2 = range_from_theta(eta2 * 180 / np.pi, h)
    # 4 slant range delta
    slant_delta_range = slant_range_2 - slant_range_1
    # 5 prf from slant range
    pri = 2 * slant_delta_range / c
    prf_opt = 1 / pri
    # %% (In 6)
    # Add to the plot
    ax.plot(prf_opt, ground_range / 1000, 'red')
    # %% (In 7)
    sr = range_ground_to_slant(200e3, h)
    # loopy but functional
    gr, eta = range_slant_to_ground(sr, h)  # the incidence angle is the second parameter returned
    prf_min = 2 * ground_speed(ground_range, vs, h) / La
    # %% (In 9)
    # Add to the plot
    ax.plot(prf_min, ground_range / 1000, 'green')
    # make a second axis for updatable stuff
    ax2 = ax.twinx()
    ax2.get_yaxis().set_visible(False)
    ax2.set_ylim(ylim)

    # %%
    dictionary = closest_valid_timing_selection(h, rg_coordinate,
                                                prf_coordinate, dutycycle,
                                                )
    swath_prf = dictionary['prf']
    swath_rg = dictionary['ground-swath']
    swath_rg_compressed = dictionary['compressed-ground-swath']
    ax2.plot(swath_prf, swath_rg / 1000, 'k')

    print('thinking')
    plt.pause(1)

    # time.sleep(5)
    # %% init second event handler
    plt.pause(1)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    # also connect keyboard events
    fig.canvas.mpl_connect('key_press_event', on_press)
    plt.pause(1)
    plt.show(block=False)
    # add a start looping button
    axnext = fig.add_axes([0.70, 0.05, 0.2, 0.076])  # button axes
    axnext.get_yaxis().set_visible(False)
    axnext.get_xaxis().set_visible(False)
    asrup = Button(axnext, "ASR update")
    asrup.on_clicked(asrupdate_onclick)

    # %% init ASR plots
    fig_a, ax_a = plt.subplots(1, dpi=dpii)
    fig_a.show(False)
    fig_r, ax_r = plt.subplots(1, dpi=dpii)
    fig_r.show(False)

    # %% init ASR objects
    # radar geometry class initialization (In 15)
    radar_geo = RadarGeometry()
    radar_geo.set_rotation(30 * np.pi / 180, 0, 0)
    radar_geo.set_initial_position(0, 0, h)
    radar_geo.set_speed(vs)
    # swath center incidence angle
    swath_rs_compressed = dictionary['compressed-slant-swath']
    rg_center, eta_center = range_slant_to_ground(np.average(swath_rs_compressed), h)
    looking_angle = incidence_angle_to_looking_angle(eta_center, h)
    radar_geo.set_rotation(looking_angle, 0, 0)

    print('initial looking angle:', looking_angle * 180 / np.pi)
    # 2 (In 16)
    # Uniform aperture antenna initialization
    uniap = UniformAperture(La, Wa, c / wavelength)

    # %% infinite loop
    print('looping')
    while (True):
        if update:
            ax2.cla()
            ax2.get_yaxis().set_visible(False)
            ax2.set_ylim(ax.get_ylim())
            # %% update ax2
            dictionary = closest_valid_timing_selection(h, float(rg_coordinate * 1000),
                                                        float(prf_coordinate), dutycycle)
            swath_prf = dictionary['prf']
            swath_rg = dictionary['ground-swath']
            swath_rg_compressed = dictionary['compressed-ground-swath']
            ax2.plot(swath_prf, swath_rg / 1000, 'k')
            ax2.plot(swath_prf, swath_rg_compressed / 1000, 'P', color='k')
            print(swath_prf, swath_rg)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.draw()
            plt.show(block=False)
            update = False
            # %%
        if asrupdate:
            dictionary = umambiguous_mode_analysis(radar_geo,
                                                   uniap, rg_coordinate * 1000,
                                                   prf_coordinate, AASR_max, RASR_max,
                                                   vs, wavelength, dutycycle,
                                                   aasr_points=6, rasr_points=27)
            # plotting
            ax_r.cla()
            ax_a.cla()
            plt.pause(.2)
            # rasr
            rasr = dictionary['rasr-linear']
            ground_range_axis = dictionary['rasr-ground-range-axis']
            ax_r.plot(ground_range_axis / 1000, 10 * np.log10(rasr))
            ax_r.set_xlabel('ground range [km]')
            ax_r.set_ylabel('RASR [dB]')
            # aasr
            aasr = dictionary['aasr-linear']
            doppler_undersampling_ratio = dictionary['aasr-undersampling-axis']
            ax_a.plot(doppler_undersampling_ratio, 10 * np.log10(aasr))
            ax_a.set_xlabel('Doppler undersampling ratio')
            ax_a.set_ylabel('AASR [dB]')
            # in 23
            # add to plot the RASR threshold
            swath_rasr = dictionary['usable-rasr-swath']
            ranges = dictionary['usable-swath-ranges']
            if swath_rasr != 0:
                ax_r.scatter(ranges / 1000, np.ones_like(ranges) * RASR_max, marker='.', color='r',
                             label=str('rg=' + str(np.round(swath_rasr[0] / 1000, 2)) + '[km]'))
                ax_r.legend()
            # AASR undersampling point (In 25)
            undersampling = dictionary['minimum-undersampling']
            ax_a.scatter(undersampling, AASR_max, marker='o', color='r',
                         label='undersampling ratio =' + str(np.round(undersampling, 2)))
            ax_a.legend()
            fig_a.show()
            fig_r.show()
            fig_a.canvas.draw()
            fig_a.canvas.flush_events()
            fig_r.canvas.draw()
            fig_r.canvas.flush_events()
            plt.draw()
            plt.show()
            asrupdate = False
        # print(update)
        plt.pause(2)
        print('ping')

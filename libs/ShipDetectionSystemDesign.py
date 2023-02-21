## imported from the project "design-baseline"
## 4 October 22
# This files contains the functions needed to perform the analisys as in the shipdetection notebook
# the idea is to produce a scripted sweep

import numpy as np
from design_functions import *
import matplotlib.pyplot as plt
import matplotlib
from timing_diagram import time_diagram_plotter
from radartools.farField import UniformAperture
from spherical_earth_geometry_radar import *
from scipy.optimize import fsolve, bisect


# probability of detection function
def pd(radGeo, uniap, incidence, wavel, losses, bandwidth, p_avg, Pfa, A_ship, expected, variance, c=299792458.0):
    h = radGeo.S_0[2]
    v_s = radGeo.abs_v
    # core snr
    c_snr, daz = core_snr_spherical(radGeo, uniap, incidence, wavel, v_s, h)
    # parametrized NESZ
    Ta = 300  # k
    nesz = 10 ** (losses / 10) * Ta * bandwidth / (c_snr * p_avg)
    # probability of detection
    # resolution
    A_cell = daz * c / (2 * sin(incidence) * bandwidth)
    # probability of detection
    Pd = pd_from_nesz_res(nesz, A_cell, Pfa, A_ship, expected, variance)
    return Pd


def shipDetectionSystemDesigner(theta, la, wa, ares, h, freq, p_peak, dutycycle, losses, Pfa, A_ship, expected,
                                variance, pd_min, c=299792458.0, re=6371e3, reducedSwath=False, swathnominalfraction=1):
    """
    1.	Set the initial looking angle
    2.	Find the signal bandwidth assuming a target resolution area e.g. Ares = 2m^2
    3.	Compute the NESZ over range assuming a spherical earth model, an ideal aperture antenna of given size, the bandwidth of point 2, and a set of fixed system parameters (losses, noise figure, power etc.)
    4.	Compute the Pd curve over range from the NESZ of point 3
    5.	Threshold the Pd curve to a minimum level e.g. Pd = 0.5, and find the swath width.
    6.	If the swath width is smaller than the 3-dB elevation beamwidth, consider the 3-dB beamwidth as the swath for the following steps. Else use the swath width of point 5
    7.	Using the Swath Width of point 6. Find an initial PRF guess assuming a certain duty cycle e.g. 25 %. (quite large to allow for low peak power)
    8.	Align the PRF of 7 to the closest timing diagram valid point to avoid nadir returns
    9.	Re-Evaluate the Probability of detection at the swath edges given by the new PRF and duty cycle of point 8 and slightly change the looking angle to center the Probability of detection “beam”
    10.	Save the results in a dictionary and return

    :param: theta_i: starting incidence angl
    :return:
    """
    # average power as function of peak power and duty cycle
    pavg = p_peak * dutycycle

    # dict_in to store imput parameters for future reference
    dict_in = {'incidence_angle': theta,
               'antenna_l': la,
               'antenna_w': wa,
               'resolution_area': ares,
               'orbital_height': h,
               'frequency': freq,
               'peak power': p_peak,
               'duty_cycle': dutycycle,
               'losses': losses,
               'Pfa_min': Pfa,
               'A_ship': A_ship,
               'average_i': expected,
               'variance_i': variance,
               'Pd_min': pd_min,
               'c': c,
               're': re}
    # dictionary to be filled for output
    dict_out = {}

    # 2 find bandwidth

    B = find_bandwidth(la, theta, ares)
    ## out
    dict_out['bandwidth'] = B

    # 3 # 4 # 5

    # geometry
    radGeo = RadarGeometry()
    # looking angle
    looking_angle = incidence_angle_to_looking_angle(theta * np.pi / 180, h)
    radGeo.set_rotation(looking_angle, 0, 0)
    # altitude
    radGeo.set_initial_position(0, 0, h)
    # orbital speed
    v_s = radGeo.orbital_speed()
    radGeo.set_speed(v_s)
    ## out
    dict_out['initial_looking_angle'] = looking_angle
    ## out
    dict_out['v_s'] = v_s
    # ideal aperture antenna
    uniap = UniformAperture(la, wa, freq)

    # 5 Threshold the Pd curve to a minimum level e.g. Pd = 0.5, and find the swath width.

    # wavelength
    wavel = c / freq
    # antenna beam zero position in elevation
    el_width = wavel / wa
    # function to solve for swath width
    func = lambda inc: pd(radGeo, uniap, inc, wavel, losses, B, pavg, Pfa, A_ship, expected, variance) - pd_min
    # left problem
    s_inc_right = theta * np.pi / 180 - 0.001
    s_inc_left = theta * np.pi / 180 - el_width
    # bisecant method
    inc_left = bisect(func, s_inc_left, s_inc_right, full_output=True)
    # right problem
    s_inc_right = theta * np.pi / 180 + el_width
    s_inc_left = theta * np.pi / 180 + 0.0001
    # bisecant method
    inc_right = bisect(func, s_inc_left, s_inc_right, full_output=True)
    inc_pd = np.array([inc_left[0], inc_right[0]])
    ## out initial incidence angle limits for probability of detection > pd_min
    dict_out['initial_inc_pd'] = inc_pd
    # translation to swath
    swath_s, swath_g = range_from_theta(inc_pd * 180 / np.pi, h)

    # 6 If the swath width is smaller than the 3-dB elevation beamwidth,
    # consider the 3-dB beamwidth as the swath for the following steps. Else use the swath width of point 5

    # 3 dB beamwidth approx
    inc_3dB = np.array((theta * np.pi / 180 - el_width * 1 / 2, theta * np.pi / 180 + el_width * 1 / 2))
    # swath choice
    inc_left = min(inc_pd[0], inc_3dB[0])
    inc_right = max(inc_pd[1], inc_3dB[1])
    inc_swat_0 = np.array((inc_left, inc_right))

    if reducedSwath:
        inc_swat_0 = np.array((theta * np.pi / 180 - el_width * swathnominalfraction / 2,
                               theta * np.pi / 180 + el_width * swathnominalfraction / 2))

    ## out initial incidence angle swath used for prf selection
    dict_out['initial_inc_swath'] = inc_swat_0
    if (inc_right - inc_left) > (inc_3dB[1] - inc_3dB[0]):
        ## out "probability of detection swath, larger than nominal swath")
        dict_out['larger_than_nominal'] = True
    else:
        dict_out['larger_than_nominal'] = False

    # 7 Using the Swath Width of point 6. Find an initial PRF guess assuming a certain duty cycle e.g. 25 %.
    # (quite large to allow for low peak power)

    # slant range from angles
    swath_s, swath_g = range_from_theta(inc_swat_0 * 180 / np.pi, h)
    swath_0 = swath_s[1] - swath_s[0]
    # PRI considering duty cycle
    PRI_0 = 2 * swath_0 / c * 1 / (1 - 2 * dutycycle)
    ## out very initial PRI
    dict_out['PRI_0'] = PRI_0
    # alligning the PRI to fast time axis
    # # step 1 impulse order determination
    # average slant range
    rs = np.average(swath_s)
    order = np.floor(2 * rs / (PRI_0 * c))
    # # step 2 adjusted PRI
    PRI_1 = 2 * rs / (c * (.5 + order))
    ## out first refined PRI selected (with nadir ambiguity present)
    dict_out['PRI_1'] = PRI_1
    # # step 3 adjusted near end and far end range and swath
    r2s1 = order * c * PRI_1 / 2
    r1s1 = (order + 1) * c * PRI_1 / 2
    # # step 4 adjusted ground ranges
    r1g1, theta1 = range_slant_to_ground(r1s1)
    r2g1, theta2 = range_slant_to_ground(r2s1)
    rg1 = r1g1 - r2g1
    ## updated swath
    swath_s = np.array([r2s1, r1s1])
    swath_g0 = np.array([r2g1, r1g1])
    inc_swat = np.array([theta2, theta1])

    # 8 Align the PRF of 7 to the closest timing diagram valid point to avoid nadir returns

    # optimized looking angle
    theta_1, PRI_2, swath_g = closest_nadir_null(PRI_1, swath_s[0], h, end='near')
    ## out final optimized PRI
    dict_out['PRI_2'] = PRI_2
    ## out final optimized PRF
    dict_out['PRF'] = 1 / PRI_2
    # new slant range
    swath_s = range_ground_to_slant(swath_g, h)
    # new incidence range
    swath_g, inc_swat = range_slant_to_ground(swath_s, h)
    # dutycycle limits
    usable_swath_s = swath_s + np.array([dutycycle * PRI_2 * c / 2, - dutycycle * PRI_2 * c / 2])
    usable_swath_g, usable_inc = range_slant_to_ground(usable_swath_s, h)
    # set the incidence angle in the middle of the usable swath
    theta_1 = np.average(usable_inc * 180 / np.pi)
    # adjust radar looking angle
    looking_angle = incidence_angle_to_looking_angle(theta_1 / 180 * np.pi, h)
    radGeo.set_rotation(looking_angle, 0, 0)
    ## out optimized incidence angle swath
    dict_out['inc_swath'] = inc_swat
    ## out optimized ground swath
    dict_out['ground_swath'] = swath_g
    ## out optimized slant range swath
    dict_out['slant_swath'] = swath_s
    ## out usable incidence angle swath
    dict_out['usable_inc_swath'] = usable_inc
    ## out usable ground range swath
    dict_out['usable_ground_swath'] = usable_swath_g
    ## out usable slant range swath
    dict_out['usable_slant_swath'] = usable_swath_s

    # 9 Re-Evaluate the Probability of detection at the swath edges given by the new PRF and duty cycle of point 8
    #   and slightly change the looking angle to center the Probability of detection “beam”
    # antenna temperature
    # redefine the bandwidth
    B = find_bandwidth(la, theta_1, ares)
    ## out bandwidth updated
    dict_out['bandwidth'] = B
    Ta = 300  # kelvin
    func = lambda b: pd_error(b, usable_inc[0], usable_inc[1], radGeo, uniap, 10 ** (losses / 10) * Ta * B / pavg, B,
                              Pfa, A_ship, expected,
                              variance)
    looking_angle_opt = fsolve(func, looking_angle)
    # print(looking_angle_opt * 180 / np.pi)
    # recenter the radar
    radGeo.set_rotation(looking_angle_opt, 0, 0)
    looking_angle = looking_angle_opt
    ## out final looking angle
    dict_out['looking_angle'] = looking_angle
    ## out radar geometry object
    dict_out['radarGeo'] = radGeo
    ## out uniform aperture object
    dict_out['uniAp'] = uniap

    # 10 find the minumum probability of detection swath after optimization

    #  solve the function for left and right with respect of broadside
    func = lambda inc: pd(radGeo, uniap, inc, wavel, losses, B, pavg, Pfa, A_ship, expected, variance) - pd_min
    # left problem
    if func(usable_inc[0]) < 0:  # if the usable swath left edge does not exceed pdmin
        # then the actual pd min swath will be smaller
        s_inc_right = np.average(usable_inc) - 0.001
        s_inc_left = usable_inc[0]
        # bisecant method
        inc_left = bisect(func, s_inc_left, s_inc_right, full_output=False)
    else:  # if the pd at the swath edge is higher than the minimum probability of detection
        inc_left = usable_inc[0]  # then that's the limit in any case

    pdmin = pd_min
    # right problem
    if func(usable_inc[0]) < 0:  # if the usable swath right edge does not exceed pdmin
        # then the actual pd min swath will be smaller
        s_inc_right = usable_inc[1]
        s_inc_left = np.average(usable_inc) + 0.0001
        # bisecant method
        inc_right = bisect(func, s_inc_left, s_inc_right, full_output=False)
    else:  # if the pd at the swath edge is higher than the minimum probability of detection
        inc_right = usable_inc[1]  # then that's the limit in any case
        pdmin = func(usable_inc[1]) + pd_min

    pd_inc = np.array([inc_left, inc_right])
    ## out
    dict_out['pd_inc_swath'] = pd_inc
    # translate to ground and slant swath
    pd_swath_s, pd_swath_g = range_from_theta(pd_inc * 180 / np.pi, h)
    ## out
    dict_out['pd_ground_swath'] = pd_swath_g
    ## out
    dict_out['pd_slant_swath'] = pd_swath_s
    ## out
    dict_out['pd_min'] = np.ones_like(pd_inc) * pdmin
    # Broadside incidence angle
    # using the theorem of sines
    bs_inc = np.arcsin((re + h) / re * np.sin(looking_angle))
    dict_out['broadside_incidence'] = bs_inc

    return dict_out, dict_in

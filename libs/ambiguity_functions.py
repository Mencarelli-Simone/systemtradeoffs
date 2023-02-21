import numpy as np
from numpy import sin, cos, arccos, arcsin
from design_functions import *
from spherical_earth_geometry_radar import *
from radartools.farField import UniformAperture
from tqdm import tqdm


# Gain_from_doppler
def gain_from_doppler(doppler, incidence, radGeo: RadarGeometry, uniap: UniformAperture, lambda_c, v_s, h=500e3, aasr=False):
    """
    returns the antennna pattern peak approximately close to the doppler point of choice
    :param doppler: doppler meshgrid
    :param incidence: incidence angle of scene center meshgrid
    :param radGeo: radar Geometry object
    :param uniap: uniform aperture object
    :param lambda_c: wavelength
    :param v_s: orbital speed
    :param h: platform height
    :param aasr: turn this to true if computing aasr
    :return: gain pattern, maximum gain broadside
    """
    # doppler incidence angle
    D, I = doppler, incidence
    # Incidence azimuth position stationary slow time
    I, A, Tk = mesh_doppler_to_azimuth(I, D, lambda_c, v_s, h)
    # gcs coordinate
    X, Y, Z = mesh_incidence_azimuth_to_gcs(I, A, lambda_c, v_s, h)
    # lcs coordinate
    Xl, Yl, Zl = mesh_gcs_to_lcs(X, Y, Z, radGeo.Bc2s, radGeo.S_0)
    # lcs spherical coordinates
    R, T, P = meshCart2sph(Xl, Yl, Zl)

    if aasr:
        # phi allingnment step
        # allign Phi coordinate to closest 0 , 180 value
        P = np.round(P / np.pi) * np.pi
        P = np.where(P == 0, 2 * np.pi, P)

    G = uniap.mesh_gain_pattern_theor(T, P)
    G = np.where(np.isnan(G), uniap.max_gain(), G)
    maxg = uniap.max_gain()
    return G, maxg


def AASR(radGeo: RadarGeometry, uniap: UniformAperture, incidence, prf, Bd, lambda_c, re=6371e3, pbaroff=False):
    """
    returns the maximum expected Azimuth Ambiguity to signal ratio at the given incidence angle
    100 % processed doppler bandwidth is assumed
    :param radGeo: radar Geometry object
    :param uniap: uniform aperture object
    :param incidence: incidence angle in degrees
    :param prf: pulse repetition frequency
    :param Bd: Processed Doppler Bandwidth
    :param lambda_c: wavelength
    :return:
    """
    h = radGeo.S_0[2]
    v_s = radGeo.abs_v
    # print(v_s)
    dop_ax = np.linspace(-Bd / 2, Bd / 2, 101)  # to obtain an higher order dop just  add n*prf
    incidence_angle = incidence

    # 0 max order

    looking_angle = incidence_angle_to_looking_angle(incidence_angle * np.pi / 180, h)

    # azimuth circle radius
    r0, rg = range_from_theta(incidence_angle, h)
    costheta_e = ((re + h) ** 2 + re ** 2 - r0 ** 2) / (2 * (re + h) * re)
    raz = re * costheta_e
    d = r0 * cos(looking_angle)
    # projected maximum slant range
    l = np.sqrt((d + raz) ** 2 - raz ** 2)

    # maximum azimuthal angle
    costheta_a = (raz ** 2 + (raz + d) ** 2 - l ** 2) / (2 * raz * (raz + d))
    theta_a = arccos(costheta_a)

    cos_phimax = (-raz ** 2 + (raz + d) ** 2 + l ** 2) / (2 * l * (raz + d))
    phimax = arccos(cos_phimax)

    # then the max doppler shift is:
    A = (re + h) ** 2 + re ** 2
    B = 2 * (re + h) * re * costheta_e
    v = v_s / (re + h)

    doppler_max = (B * v * sin(theta_a) / (lambda_c * np.sqrt(A - B * cos(theta_a))))

    # this maximum doppler corresponds to a sum order of
    n = int(np.floor(doppler_max / prf))

    # 1 denominator
    D, I = np.meshgrid(dop_ax, incidence_angle * np.pi / 180)
    den_g, maxg = gain_from_doppler(D, I, radGeo, uniap, lambda_c, v_s, h, aasr=True)
    den_g /= maxg
    # integrated denominator
    den = integrate.simps(den_g ** 2, D, axis=1)
    num = np.zeros_like(den)
    # print(den)
    # 2 numerator sum
    for nn in tqdm(range(1, int(n + 1)), disable=pbaroff):
        #D, I = np.meshgrid(dop_ax, incidence_angle * np.pi / 180)
        # positive n
        num_g, maxg = gain_from_doppler(D + nn * prf, I, radGeo, uniap, lambda_c, v_s, h, aasr=True)
        num_g /= maxg
        num += integrate.simps(num_g ** 2, D, axis=1)
        # negative n
        num_g, maxg = gain_from_doppler(D - nn * prf, I, radGeo, uniap, lambda_c, v_s, h, aasr=True)
        num_g /= maxg
        num += integrate.simps(num_g ** 2, D, axis=1)
    return num / den


# %%
def RASR(radGeo: RadarGeometry, uniap: UniformAperture, incidence, PRI, Bd, lambda_c, v_s, h=500e3, c=299792458.0, pbaroff=False):
    """
    Range ambiguity to signal ratio as per 6.5.20 in curlander
    :param radGeo: radar Geometry object
    :param uniap: uniform aperture object
    :param incidence: incidence angle vector
    :param PRI: Pulse Repetition Interval = 1 / Bd
    :param Bd: Doppler Bandwidth
    :param lambda_c: Wavelength
    :param v_s: Satellite speed
    :param h: satellite Height
    :param c: optional, default Speed of light
    :return:
    """
    # step 1 find orders
    incidence_angle = np.average(incidence)  # scene average incidence angle
    r0, rg = range_from_theta(incidence_angle * 180 / np.pi, h)  # todo find incidence from rax or rax from incidence
    order = int(r0 * 2 / (c * PRI))
    # horizon
    r_hor, rghor = range_from_theta(90, h)
    nH = int(r_hor * 2 / (c * PRI)) - order
    # nadir
    nN = order - int(h * 2 / (c * PRI))
    # print('order', order, '\nnH', nH, '\nnN', nN)

    # Step 2 Denominator
    # Range Axis
    rax, rgax = range_from_theta(incidence * 180 / np.pi, h)
    # Doppler Axis
    dax = np.linspace(-Bd / 2, Bd / 2, 111)
    # gain
    D, I = np.meshgrid(dax, incidence) # todo fix gain from doppler usage
    G, maxg = gain_from_doppler(D, I, radGeo, uniap, lambda_c, v_s, h)
    G /= maxg
    # square and integrate over doppler
    Gint = integrate.simps(G ** 2, D, axis=1)
    Denom = Gint / (rax ** 3 * sin(incidence))

    # Step 3 Numerator
    # for every possible replica
    j = np.arange(-nN, nH)
    Numer = np.zeros_like(Denom)
    raxjj = np.zeros_like(Numer).astype('float64')
    for jj in tqdm(j, disable=pbaroff):
        # 1 range axis
        raxjj = float(jj) * c * PRI / 2 + rax
        raxj = np.where(raxjj <= h, h, raxjj)
        # doppler axis is still the same
        # 2 incidence angle axis
        rgax, thetaj = range_slant_to_ground(raxj, h)
        D, Ij = np.meshgrid(dax, thetaj)
        # 3 returns from after nadir (in the looking direction of the radar)
        if jj != 0:  # j = 0 is the signal, not the ambiguity
            G, max = gain_from_doppler(D, Ij, radGeo, uniap, lambda_c, v_s, h)
            G /= maxg
            Gint = integrate.simps(G ** 2, D, axis=1)
            Gint = np.where(raxjj <= h, 0, Gint)
            Numer += np.where(sin(thetaj) != 0, Gint / (raxj ** 3 * sin(thetaj)), 0)
        # 4 returns from behind the nadir (in the other direction)
        G, maxg = gain_from_doppler(D, -Ij, radGeo, uniap, lambda_c, v_s, h)
        G /= maxg
        Gint = integrate.simps(G ** 2, D, axis=1)
        Gint = np.where(raxjj <= h, 0, Gint)
        Numer += np.where(sin(thetaj) != 0, Gint / (raxj ** 3 * sin(thetaj)), 0)

    return Numer / Denom  # todo test

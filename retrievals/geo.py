from math import sin, cos, sqrt, asin, radians
import numpy as np
import pyproj


def geospace_zonal(lon, lat, d_west, d_east, num):
    """
    Create a longitude grid with a certain extent and number of elements
    for a given location.
    :param lon: Longitude of location
    :param lat: Latitude of location
    :param d_west: Extent to the west (m)
    :param d_east: Extent to the east (m)
    :param num: Number of elements
    :return: Array of longitudes.
    """
    proj = pyproj.Proj(proj='eqc', lon_0=lon, lat_0=lat, lat_ts=lat)
    xs = np.linspace(-d_west, d_east, num)
    ys = np.zeros_like(xs)
    lons, lats = proj(xs, ys, inverse=True)
    return lons


def geospace_meridional(lon, lat, d_south, d_north, num):
    """
    Create a latitude grid with a certain extent and number of elements
    for a given location.
    :param lon: Longitude of location
    :param lat: Latitude of location
    :param d_south: Extent to the south (m)
    :param d_north: Extent to the north (m)
    :param num: Number of elements
    :return: Array of latitudes.
    """
    proj = pyproj.Proj(proj='eqc', lon_0=lon, lat_0=lat, lat_ts=lat)
    ys = np.linspace(-d_south, d_north, num)
    xs = np.zeros_like(ys)
    lons, lats = proj(xs, ys, inverse=True)
    return lats


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in meters between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371e3 * c
    return km

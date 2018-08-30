import pytest
from retrievals import geo


@pytest.mark.parametrize('lon, lat', [(55, -21), (7.5, 47), (16, 69)])
def test_geospace_zonal(lon, lat):
    lons = geo.geospace_zonal(lon, lat, 200e3, 300e3, 5)

    tol = 2e3  # km
    assert abs(geo.haversine(lon, lat, lons[0], lat) - 200e3) < tol
    assert abs(geo.haversine(lon, lat, lons[-1], lat) - 300e3) < tol
    assert abs(geo.haversine(lons[0], lat, lons[-1], lat) - 500e3) < tol


@pytest.mark.parametrize('lon, lat', [(55, -21), (7.5, 47), (16, 69)])
def test_geospace_meridional(lon, lat):
    lats = geo.geospace_meridional(lon, lat, 200e3, 300e3, 5)

    tol = 2e3  # km
    assert abs(geo.haversine(lon, lat, lon, lats[0]) - 200e3) < tol
    assert abs(geo.haversine(lon, lat, lon, lats[-1]) - 300e3) < tol
    assert abs(geo.haversine(lon, lats[0], lon, lats[-1]) - 500e3) < tol

from dotenv import load_dotenv
load_dotenv(dotenv_path='./.env')

import os
import numpy as np
import matplotlib.pyplot as plt

from typhon.arts.workspace import Workspace

from retrievals import utils
from retrievals import instrument


class WiracSimulator:

    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.ws = None
        self.num_meas = 0
        self.tags = None
        self.frequency = None

        # Sensor
        self.fc = 142.17504e9
        self.num_channels = 2 ** 14 * 120 / 200
        self.f_min = 142.0960424414062e9
        self.f_max = 142.2340253515625e9

        # Init
        self._init_workspace()
        self._copy_agendas()
        self._setup_atmosphere()
        self._setup_sensor(self.f_min, self.f_max, self.fc, self.num_channels)

    def _init_workspace(self):
        # Start Arts
        ws = Workspace()
        ws.verbosityInit(0)
        ws.verbositySet(0, 0, 0, 0)
        ws.execute_controlfile("general/general.arts")
        ws.verbositySet(0, 0, 0, 0)
        ws.execute_controlfile("general/agendas.arts")
        ws.execute_controlfile("general/continua.arts")
        ws.execute_controlfile("general/planet_earth.arts")
        self.ws = ws

    def _copy_agendas(self):
        ws = self.ws
        ws.Copy(ws.abs_xsec_agenda, ws.abs_xsec_agenda__noCIA)
        ws.Copy(ws.ppath_agenda, ws.ppath_agenda__FollowSensorLosPath)
        ws.Copy(ws.ppath_step_agenda, ws.ppath_step_agenda__GeometricPath)
        ws.Copy(ws.iy_space_agenda, ws.iy_space_agenda__CosmicBackground)
        ws.Copy(ws.iy_surface_agenda, ws.iy_surface_agenda__UseSurfaceRtprop)
        ws.Copy(ws.iy_main_agenda, ws.iy_main_agenda__Emission)
        ws.Copy(ws.propmat_clearsky_agenda, ws.propmat_clearsky_agenda__OnTheFly)

    def _setup_atmosphere(self):
        ws = self.ws

        # General Settings
        # For the wind retrievals, the forward model calculations are performed on a 3D atmosphere grid.
        # Radiation is assumed to be unpolarized.
        ws.atmosphere_dim = 3
        ws.stokes_dim = 1
        ws.iy_unit = 'RJBT'

        # Absorption
        # We only consider absorption from ozone in this example.
        ws.abs_speciesSet(["O3", "H2O-PWR98"])
        ws.abs_lineshapeDefine("Voigt_Kuntz6", "VVH", 750e9)
        ws.ReadXML(ws.abs_lines, os.path.join(self.data_dir, 'Perrin_O3_142.xml'))
        ws.abs_lines_per_speciesCreateFromLines()

        # Atmosphere (A Priori)
        z_toa = 100e3
        z_surf = 1e3
        z_grid = np.arange(z_surf - 1e3, z_toa, 2e3)
        ws.PFromZSimple(ws.p_grid, z_grid)
        ws.lat_grid = np.array([-30, -22.5, -22, -21.5, -14])
        ws.lon_grid = np.array([40, 54, 55, 56, 70])
        ws.z_surface = z_surf * np.ones((np.asarray(ws.lat_grid).size,
                                         np.asarray(ws.lon_grid).size))

        # For the a priori state we read data from the Fascod climatology that is part of the ARTS xml data.
        ws.AtmRawRead(basename="planets/Earth/Fascod/tropical/tropical")
        ws.AtmFieldsCalcExpand1D()

    def _setup_sensor(self, f_min, f_max, fc, num_channels):
        ws = self.ws
        resolution = (f_max - f_min) / num_channels

        # Frequency Grid and Sensor
        self.f_grid = utils.exp_space(f_min - 10e6, f_max + 10e6, 300, fc, 2)
        ws.f_grid = self.f_grid

        # Backend
        self.f_backend = np.linspace(f_min, f_max, num_channels)

        ws.AntennaOff()
        ws.f_backend = self.f_backend
        ws.sensor_norm = 1
        ws.sensor_time = np.zeros(1)
        ws.sensor_responseInit()

        ws.backend_channel_response = instrument.ffts_channel_response(resolution)
        ws.sensor_responseBackend()

        self.frequency = self.f_backend

    def sensor_on(self):
        self._setup_sensor(self.f_min, self.f_max, self.fc, self.num_channels)

    def sensor_off(self):
        ws = self.ws
        ws.sensorOff()
        self.frequency = self.f_grid

    def _check_setup(self):
        ws = self.ws

        # Reference Measurement
        # Before we can calculate `y`, our setup needs to pass the following tests:
        ws.abs_f_interp_order = 3
        ws.propmat_clearsky_agenda_checkedCalc()
        ws.sensor_checkedCalc()
        ws.atmgeom_checkedCalc()
        ws.atmfields_checkedCalc()
        ws.abs_xsec_agenda_checkedCalc()
        ws.jacobianOff()
        ws.cloudboxOff()
        ws.cloudbox_checkedCalc()

    def set_pos_los(self, elevations, azimuths, tags=None):
        ws = self.ws
        # Sensor Position and Viewing Geometry
        # In ARTS the measurement directions are given by a two-column matrix, where the
        # first column contains the zenith angle and the second column the azimuth angle.
        elevations = np.array(elevations)
        azimuths = np.array(azimuths)
        self.num_meas = len(elevations)

        ws.sensor_los = np.vstack([90-elevations, azimuths]).transpose()
        ws.sensor_pos = np.array([[2000, -22, 55]] * self.num_meas)

        if tags is None:
            self.tags = list(map(lambda i: 'M'+str(i), range(2)))
        else:
            self.tags = tags

    def set_wind_simple(self, u_wind=0, v_wind=0):
        ws = self.ws

        # Adding Wind
        ws.wind_u_field = u_wind * np.ones((ws.p_grid.value.size,
                                            ws.lat_grid.value.size,
                                            ws.lon_grid.value.size))
        ws.wind_v_field = v_wind * np.ones((ws.p_grid.value.size,
                                            ws.lat_grid.value.size,
                                            ws.lon_grid.value.size))
        ws.wind_w_field = np.zeros((0, 0, 0))

    def measure(self):
        self._check_setup()

        ws = self.ws

        ws.yCalc()
        y = np.copy(ws.y.value)
        num_ch = len(y) // self.num_meas

        ys = []
        for i in range(self.num_meas):
            i_start = i * num_ch
            i_end = (i+1) * num_ch
            ys.append(y[i_start:i_end])
        return ys


if __name__ == '__main__':
    data_dir = os.environ['DATA_DIR']
    sim = WiracSimulator(data_dir)

    sim.set_wind_simple(u_wind=60, v_wind=-40)
    sim.set_pos_los([22, 22], [90, -90], ['East', 'West'])
    sim.sensor_off()
    y_east, y_west = sim.measure()
    freq = sim.frequency
    freq_diff = (freq - sim.fc)/1e6

    plt.plot(freq_diff, y_east)
    plt.plot(freq_diff, y_west)
    plt.show()
    print('len:', len(y_west))

from dotenv import load_dotenv
load_dotenv(dotenv_path='./.env')

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from typhon.arts.workspace import Workspace, arts_agenda
from typhon.arts.griddedfield import GriddedField1

from retrievals import utils
from retrievals import instrument


# Config
data_dir = os.environ['DATA_DIR']
fc = 142.17504e9

# Sensor
num_channels = 2**14 * 120 / 200
f_min = 142.0950424414062e9 + 10e6
f_max = 142.2350253515625e9 - 10e6
resolution = (f_max - f_min) / num_channels

# Location
latitude = -22
longitude = 55
altitude = 2000

# Geometry
elevation = 22  # degree


# Start Arts
ws = Workspace()

ws.verbosityInit(0)
ws.verbositySet(0, 0, 0, 0)

ws.execute_controlfile("general/general.arts")
ws.execute_controlfile("general/agendas.arts")
ws.execute_controlfile("general/continua.arts")
ws.execute_controlfile("general/planet_earth.arts")

ws.Copy(ws.abs_xsec_agenda,   ws.abs_xsec_agenda__noCIA)
ws.Copy(ws.ppath_agenda,      ws.ppath_agenda__FollowSensorLosPath)
ws.Copy(ws.ppath_step_agenda, ws.ppath_step_agenda__GeometricPath)
ws.Copy(ws.iy_space_agenda,   ws.iy_space_agenda__CosmicBackground)
ws.Copy(ws.iy_surface_agenda, ws.iy_surface_agenda__UseSurfaceRtprop)
ws.Copy(ws.iy_main_agenda,    ws.iy_main_agenda__Emission)
ws.Copy(ws.propmat_clearsky_agenda, ws.propmat_clearsky_agenda__OnTheFly)

# General Settings
# For the wind retrievals, the forward model calculations are performed on a 3D atmosphere grid.
# Radiation is assumed to be unpolarized.
ws.atmosphere_dim = 3
ws.stokes_dim     = 1
ws.iy_unit = "RJBT"

# Absorption
# We only consider absorption from ozone in this example.
ws.abs_speciesSet(["O3", "H2O-PWR98"])
ws.abs_lineshapeDefine("Voigt_Kuntz6", "VVH", 750e9)
ws.ReadXML(ws.abs_lines, os.path.join(data_dir, 'Perrin_O3_142.xml'))
ws.abs_lines_per_speciesCreateFromLines()

# Atmosphere (A Priori)
# We create a pressure grid using the `PFromZSimple` function to create a grid of approximate pressure levels
# corresponding to altitudes in the range
# z = 0.0, 2000.0, ..., 94000.0
z_toa  = 100e3
z_surf = 1e3
z_grid = np.arange(z_surf - 1e3, z_toa, 2e3)
ws.PFromZSimple(ws.p_grid, z_grid)
ws.lat_grid = np.array([-30, -22.5, -22, -21.5, -14])
ws.lon_grid = np.array([40, 54, 55, 56, 70])
ws.z_surface = z_surf * np.ones((np.asarray(ws.lat_grid).size,
                                 np.asarray(ws.lon_grid).size))

# For the a priori state we read data from the Fascod climatology that is part of the ARTS xml data.
ws.AtmRawRead(basename = "planets/Earth/Fascod/tropical/tropical")
ws.AtmFieldsCalcExpand1D()

# Adding Wind
# Wind in ARTS is represented by the `wind_u_field` and `wind_v_field` WSVs, which hold the horizontal components
# of the wind at each grid point of the atmosphere model. For this example, a constant wind is assumed.
u_wind = 60.0
v_wind = -40.0
ws.wind_u_field = u_wind * np.ones((ws.p_grid.value.size,
                                    ws.lat_grid.value.size,
                                    ws.lon_grid.value.size))
ws.wind_v_field = v_wind * np.ones((ws.p_grid.value.size,
                                    ws.lat_grid.value.size,
                                    ws.lon_grid.value.size))
ws.wind_w_field = np.zeros((0,0,0))

# Frequency Grid and Sensor
ws.f_grid = utils.exp_space(f_min - 10e6, f_max + 10e6, 300, fc, 2)

plt.plot(ws.f_grid.value * 1e-9)
plt.xlabel("Channel Number")
plt.ylabel("Frequency [GHz]")
plt.show()


# Backend
f_grid = utils.exp_space(f_min - 10e6, f_max + 10e6, 300, fc)
f_backend = np.linspace(f_min, f_max, num_channels)

ws.AntennaOff()
ws.f_backend = f_backend
ws.sensor_norm = 1
ws.sensor_time = np.zeros(1)
ws.sensor_responseInit()

ws.backend_channel_response = instrument.ffts_channel_response(resolution)
ws.sensor_responseBackend()

# Sensor Position and Viewing Geometry
# In ARTS the measurement directions are given by a two-column matrix, where the
# first column contains the zenith angle and the second column the azimuth angle.
ws.sensor_los = np.array([[90 - elevation, 90],
                          [90 - elevation, -90]])
ws.sensor_pos = np.array([[altitude, latitude, longitude]] * 2)


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

# Now, we can run the simulation using the `yCalc` WSM and display the results.
# # The results for all 5 measurements are stacked in `ws.y`.
ws.yCalc()
y = np.copy(ws.y.value)

print('len(y) // 2 =', len(y)//2)

labels = ["E", "W"]
f, axs = plt.subplots(1, 2, sharey=True, sharex=True)
n_meas = ws.y.value.size // 2

freqs = ws.f_backend.value
z = ws.z_field.value[:, 0, 0].ravel()

for i in range(2):
    ax = axs.ravel()[i]
    if i < 5:
        i_start = n_meas * i
        i_end = n_meas * (i + 1)
        y = ws.y.value[i_start:i_end]
        ax.plot((freqs-fc) / 1e6, y)
        ax.set_title(labels[i])
        ax.set_xlabel("f [GHz]")
        ax.set_ylabel("$T_B$ [K]")
    else:
        ax.axis("off")
plt.tight_layout()
plt.show()


y = np.copy(ws.y.value)
fig, ax = plt.subplots()
ax.plot((freqs-fc) / 1e6, y[:n_meas])
ax.plot((freqs-fc) / 1e6, y[n_meas:])
ax.twinx().plot((freqs-fc) / 1e6, y[:n_meas]-y[n_meas:])
fig.show()


# Setting up the Retrieval
# In this example, we retrieve ozone and the horizontal and vertical components of the wind velocities.
# The state space covariance matrix in ARTS is represented by the **covmat_sa** WSV.
# It belongs to the CovarianceMatrix group, which is used to represent block diagonal matrices.
# For each retrieval quantity that is added to the retrieval, a corresponding block must be added to **covmat_sa**.
# This is usually done by the corresponding **retrievalAdd...** call, which looks for this block
# in the **covmat_block** WSV.
# In short the general workflow for adding a retrieval quantity is as follows:
#  - Create the covariance matrix for the retrieval quantity either calling one of the **covmat...** WSV or
#    by loading your own matrix
#  - Write the matrix block into **covmat_block**
#  - Call the **retrievalAdd...** method to add the retrieval quantity and the covariance matrix block
#    to **covmat_sa**
lat_ret_grid = np.array([np.mean(ws.lat_grid)])
lon_ret_grid = np.array([np.mean(ws.lon_grid)])
n_p = ws.p_grid.value.size

ws.retrievalDefInit()
ws.covmat1D(ws.covmat_block,
            grid_1 = z_grid,
            sigma_1 = 0.1  * np.ones(n_p), # Relative uncertainty
            cls_1   = 10e3 * np.ones(n_p), # 10km correlation length
            fname   = "lin")
ws.retrievalAddAbsSpecies(species = "O3",
                          unit    = "rel",
                          g1      = ws.p_grid,
                          g2      = lat_ret_grid,
                          g3      = lon_ret_grid)
# Wind u-component
ws.covmat1D(ws.covmat_block,
            grid_1 = z_grid[::2],
           sigma_1 = 100.0  * np.ones(n_p // 2), # Relative uncertainty
           cls_1   = 10e3 * np.ones(n_p // 2), # 10km correlation length
           fname   = "lin")
ws.retrievalAddWind(g1 = ws.p_grid.value[::2],
                    g2 = np.array([np.mean(ws.lat_grid)]),
                    g3 = np.array([np.mean(ws.lon_grid)]),
                    component = "u")
# Wind v-component
ws.covmat1D(ws.covmat_block,
            grid_1 = z_grid[::2],
           sigma_1 = 100.0  * np.ones(n_p // 2), # Relative uncertainty
           cls_1   = 10e3 * np.ones(n_p // 2), # 10km correlation length
           fname   = "lin")
ws.retrievalAddWind(g1 = ws.p_grid.value[::2],
                    g2 = np.array([np.mean(ws.lat_grid)]),
                    g3 = np.array([np.mean(ws.lon_grid)]),
                    component = "v")
ws.retrievalDefClose()
ws.covmatDiagonal(ws.covmat_block, ws.covmat_inv_block, vars = 0.0001 * np.ones(ws.y.value.shape))
ws.covmat_seSet(ws.covmat_block)

@arts_agenda
def inversion_iterate_agenda(ws):
    ws.x2artsStandard()
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.yCalc()
    ws.Print(ws.y)
    ws.Print(ws.jacobian)
    ws.VectorAddVector( ws.yf, ws.y, ws.y_baseline )
    ws.IndexAdd(ws.inversion_iteration_counter, ws.inversion_iteration_counter, 1)

ws.Copy(ws.inversion_iterate_agenda, inversion_iterate_agenda)


# Jacobians
# We plot the Jacobians w.r.t. to the different retrieval quantities to check that we are sensitive to them.
ws.jacobian_do = 1
ws.yCalc()
y = np.copy(ws.y.value)

# Ozone
from matplotlib.colors import Normalize

f, axs = plt.subplots(1,2)
n_meas = ws.y.value.size // 2

freqs = f_backend
z     = ws.z_field.value[:, 0, 0].ravel()
norm  = Normalize(vmin = 0.0, vmax = 1.5)

for i in range(2):
    ax = axs.ravel()[i]
    if i < 5:
        i_start = n_meas * i
        i_end   = n_meas * (i + 1)
        j = ws.jacobian.value[i_start:i_end, :n_p]

        im = axs.ravel()[i].pcolormesh(freqs / 1e9, z / 1e3, j.T, norm = norm)
        ax.set_title(labels[i])
        ax.set_xlabel("f [GHz]")
        ax.set_ylabel("Altitude [km]")
    else:
        ax.axis("off")
cax = f.add_axes([1.04, 0.1, 0.05, 0.8])
cbax = f.colorbar(im, cax = cax,
                  label = "Weighting Function Value", norm = norm)
plt.tight_layout()
plt.show()

# Zonal Winds
# For zonal winds we plot the logarithm of the absolute value to visualize the different sensitivity scales throughout
# the model atmosphere.
from matplotlib.colors import LogNorm

f, axs = plt.subplots(1, 2)
n_meas = ws.y.value.size // 2
norm = LogNorm(vmin = 1e-10, vmax = 1e-4)

for i in range(2):
    ax = axs.ravel()[i]
    if i < 5:
        i_start = n_meas * i
        i_end   = n_meas * (i + 1)
        j = ws.jacobian.value[i_start:i_end, n_p: 3 * n_p // 2]

        im = axs.ravel()[i].pcolormesh(freqs / 1e9, z[::2 ] / 1e3, np.abs(j.T), norm = norm)
        ax.set_title(labels[i])
        ax.set_xlabel("f [GHz]")
        ax.set_ylabel("Altitude [km]")
    else:
        ax.axis("off")
cax = f.add_axes([1.04, 0.1, 0.05, 0.8])
cbax = f.colorbar(im, cax = cax,
                  label = "Absolute Value of Weighting Function Value", norm = norm)
plt.tight_layout()
plt.show()

# Meridional Winds
f, axs = plt.subplots(1, 2)
norm   = LogNorm(vmin = 1e-10, vmax = 1e-4)

for i in range(2):
    ax = axs.ravel()[i]
    if i < 5:
        i_start = n_meas * i
        i_end   = n_meas * (i + 1)
        j = ws.jacobian.value[i_start:i_end, 3 * (n_p // 2): 5 * (n_p // 2)]

        im = axs.ravel()[i].pcolormesh(freqs / 1e9, z[::2 ] / 1e3, np.abs(j.T), norm = norm)
        ax.set_title(labels[i])
        ax.set_xlabel("f [GHz]")
        ax.set_ylabel("Altitude [km]")
    else:
        ax.axis("off")
cax = f.add_axes([1.04, 0.1, 0.05, 0.8])
cbax = f.colorbar(im, cax = cax,
                  label = "Absolute Value of Weighting Function Value", norm = norm)
plt.tight_layout()
plt.show()

exit(0)
print('Retrieval.....')

# A Priori State
# For the a priori state we assume zero wind in any direction. The a priori vector for the OEM is created by
# the `xaStandard` WSM, which computes $x_a$ from the current atmospheric state.
ws.wind_u_field.value[:] = 0.0
ws.wind_v_field.value[:] = 0.0
ws.xaStandard()

# The OEM Calculation
ws.x        = np.zeros(0)
ws.jacobian = np.zeros((0,0))
ws.y.value[:] = y
ws.OEM(method="lm", max_iter=20, display_progress=1,
       lm_ga_settings=np.array([100.0,2.0,2.0,10.0,1.0,1.0]))
ws.x2artsStandard()

# Retrieval Results
f, axs = plt.subplots(1, 2)

ax = axs[0]
ax.axvline(u_wind, color = "grey", ls = "--", label = "Truth")
ax.plot(ws.wind_u_field.value[:, 0, 0], z // 1e3, label = "Retrieved")
ax.set_xlabel("$v_u$ [m/s]")
ax.set_ylabel("Altitude [km]")
ax.legend()

ax = axs[1]
ax.axvline(v_wind, color = "grey", ls = "--", label = "Truth")
ax.plot(ws.wind_v_field.value[:, 0, 0], z // 1e3, label = "Retrieved")
ax.set_title("$v_v$")
ax.set_xlabel("$v_v$ [m/s]")
ax.set_ylabel("Altitude [km]")
ax.legend()
plt.show()
import numpy as np
from typhon.arts.griddedfield import GriddedField1


def ffts_channel_response(resolution, num_channels=10):
    """
    Channel response for an FFT Spectrometer with sinc^2 response.
    :param resolution: The frequency resolution of the FFTS in Hz
    :param num_channels: Number of channels with nonzero response, default: 10
    :return: ArrayOfGriddedField1 for use as Arts WSV backend_channel_response.
    """
    grid = np.linspace(-num_channels / 2, num_channels / 2, 20 * num_channels)
    response = np.sinc(grid) ** 2
    bcr = GriddedField1(name='Backend channel response function for FFTS',
                        gridnames=['Frequency'], dataname='Data',
                        grids=[resolution * grid],
                        data=response)
    return [bcr, ]

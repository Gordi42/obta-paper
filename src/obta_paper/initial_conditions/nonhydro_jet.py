"""The unstable jet initial condition for the Nonhydrostatic model."""
from __future__ import annotations

import fridom.nonhydro as nh


class NonhydroJet(nh.State):

    r"""
    Two opposing unstable jets.

    Description
    -----------
    An instable jet setup with a small pressure perturbation
    on top of it. The jet is given by:

    .. math::
        u(x,y,z) = \left(u_0(x,y) - u_1(x,y)\right) \cos(2 \pi / L_z z)

    with the zonal jets given by:

    .. math::
        u_i(x,y) = \exp\left(-\left(\frac{y - p_{i} L_y}{\sigma \pi}\right)^2\right)

    where :math:`L_y` is the domain length in the y-direction,
    :math:`p_i` are the relative positions of the jets in the y-direction,
    and :math:`\\sigma` is the width of the jet. The perturbation is given by:

    .. math::
        v = A \sin \left( \frac{2 \pi}{L_x} k_p x \right)

    where :math:`A` is the amplitude of the perturbation and :math:`k_p` is the
    wavenumber of the perturbation.

    Parameters
    ----------
    mset : ModelSettings
        The model settings.
    wavenum : int
        The relative wavenumber of the perturbation.
    waveamp : float
        The amplitude of the perturbation.
    pos : tuple(float)
        The relative positions of the jet in the y-direction
    width : float
        The width of the jet.

    """

    def __init__(self,
                 mset: nh.ModelSettings,
                 wavenum: int = 3,
                 waveamp: float = 0.05,
                 pos: tuple[float] = (0.25, 0.75),
                 width: float = 0.04) -> None:
        super().__init__(mset)
        # Shortcuts
        ncp = nh.config.ncp
        lx, ly, lz = self.grid.L

        # Construct the zonal jets
        state = nh.State(mset)
        x, y, z = state.u.get_mesh()
        state.u.arr = (+ ncp.exp(- ((y - pos[1] * ly)/(width * ncp.pi))**2)
                       - ncp.exp(- ((y - pos[0] * ly)/(width * ncp.pi))**2) )
        kz = 2 * ncp.pi / lz
        state.u.arr *= ncp.cos(kz * z)

        # Construct the perturbation
        kx_p = 2 * ncp.pi / lx * wavenum
        x, y, z = state.v.get_mesh()
        state.v.arr = waveamp * ncp.sin(kx_p * x)

        self.fields = state.fields

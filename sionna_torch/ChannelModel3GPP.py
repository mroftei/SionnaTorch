#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import torch
from .ChannelCoefficients import ChannelCoefficientsGenerator

from .LSPGenerator import LSPGenerator
from .RaysGenerator import RaysGenerator

class Topology:
    # pylint: disable=line-too-long
    r"""
    Class for conveniently storing the network topology information required
    for sampling channel impulse responses

    Parameters
    -----------

    velocities : [batch size, number of UTs], float
        UT velocities

    moving_end : str
        Indicated which end of the channel (TX or RX) is moving. Either "tx" or
        "rx".

    los_aoa : [batch size, number of BSs, number of UTs], float
        Azimuth angle of arrival of LoS path [radian]

    los_aod : [batch size, number of BSs, number of UTs], float
        Azimuth angle of departure of LoS path [radian]

    los_zoa : [batch size, number of BSs, number of UTs], float
        Zenith angle of arrival for of path [radian]

    los_zod : [batch size, number of BSs, number of UTs], float
        Zenith angle of departure for of path [radian]

    los : [batch size, number of BSs, number of UTs], bool
        Indicate for each BS-UT link if it is in LoS

    distance_3d : [batch size, number of UTs, number of UTs], float
        Distance between the UTs in X-Y-Z space (not only X-Y plan).

    tx_orientations : [batch size, number of TXs, 3], float
        Orientations of the transmitters, which are either BSs or UTs depending
        on the link direction [radian].

    rx_orientations : [batch size, number of RXs, 3], float
        Orientations of the receivers, which are either BSs or UTs depending on
        the link direction [radian].
    """

    def __init__(self,  velocities,
                        moving_end,
                        los_aoa,
                        los_aod,
                        los_zoa,
                        los_zod,
                        los,
                        distance_3d,
                        tx_orientations,
                        rx_orientations):
        self.velocities = velocities
        self.moving_end = moving_end
        self.los_aoa = los_aoa
        self.los_aod = los_aod
        self.los_zoa = los_zoa
        self.los_zod = los_zod
        self.los = los
        self.tx_orientations = tx_orientations
        self.rx_orientations = rx_orientations
        self.distance_3d = distance_3d


class ChannelModel3GPP():
    # pylint: disable=line-too-long
    r"""
    Baseclass for implementing 3GPP system level channel models, such as UMi,
    UMa, and RMa.

    Parameters
    -----------
    scenario : SystemLevelScenario
        Scenario for the channel simulation

    always_generate_lsp : bool
        If `True`, new large scale parameters (LSPs) are generated for every
        new generation of channel impulse responses. Otherwise, always reuse
        the same LSPs, except if the topology is changed. Defaults to
        `False`.

    Input
    -----

    num_time_samples : int
        Number of time samples

    sampling_frequency : float
        Sampling frequency [Hz]

    Output
    -------
        a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_samples], complex
            Path coefficients

        tau : [batch size, num_rx, num_tx, num_paths], float
            Path delays [s]
    """

    def __init__(self, scenario, rng: torch.Generator, always_generate_lsp=False):

        self._scenario = scenario
        self._lsp_sampler = LSPGenerator(scenario, rng)
        self._ray_sampler = RaysGenerator(scenario, rng)
        self._set_topology_called = False

        self._cir_sampler = ChannelCoefficientsGenerator(
                                            scenario.carrier_frequency,
                                            subclustering=True,
                                            rng = rng)

        # Are new LSPs needed
        self._always_generate_lsp = always_generate_lsp

    def set_topology(self, ut_loc=None, bs_loc=None, ut_orientations=None,
        bs_orientations=None, ut_velocities=None, los=None):
        r"""
        Set the network topology.

        It is possible to set up a different network topology for each batch
        example. The batch size used when setting up the network topology
        is used for the link simulations.

        When calling this function, not specifying a parameter leads to the
        reuse of the previously given value. Not specifying a value that was not
        set at a former call rises an error.

        Input
        ------
            ut_loc : [batch size,num_ut, 3], float
                Locations of the UTs

            bs_loc : [batch size,num_bs, 3], float
                Locations of BSs

            ut_orientations : [batch size,num_ut, 3], float
                Orientations of the UTs arrays [radian]

            bs_orientations : [batch size,num_bs, 3], float
                Orientations of the BSs arrays [radian]

            ut_velocities : [batch size,num_ut, 3], float
                Velocity vectors of UTs

            los : bool or `None`
                If not `None` (default value), all UTs located outdoor are
                forced to be in LoS if ``los`` is set to `True`, or in NLoS
                if it is set to `False`. If set to `None`, the LoS/NLoS states
                of UTs is set following 3GPP specification [TR38901]_.
        """

        # Update the scenario topology
        need_for_update = self._scenario.set_topology(  ut_loc,
                                                        bs_loc,
                                                        ut_orientations,
                                                        bs_orientations,
                                                        ut_velocities,
                                                        los)

        if need_for_update:
            # Update the LSP sampler
            self._lsp_sampler.topology_updated_callback()

            # Update the ray sampler
            self._ray_sampler.topology_updated_callback()

            # Sample LSPs if no need to generate them everytime
            if not self._always_generate_lsp:
                self._lsp = self._lsp_sampler()

        if not self._set_topology_called:
            self._set_topology_called = True

    def cir_to_time_channel(self, bandwidth, a, tau, l_min, l_max, normalize=False):
        # pylint: disable=line-too-long
        r"""
        Compute the channel taps forming the discrete complex-baseband
        representation of the channel from the channel impulse response
        (``a``, ``tau``).

        This function assumes that a sinc filter is used for pulse shaping and receive
        filtering. Therefore, given a channel impulse response
        :math:`(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1`, the channel taps
        are computed as follows:

        .. math::
            \bar{h}_{b, \ell}
            = \sum_{m=0}^{M-1} a_{m}\left(\frac{b}{W}\right)
                \text{sinc}\left( \ell - W\tau_{m} \right)

        for :math:`\ell` ranging from ``l_min`` to ``l_max``, and where :math:`W` is
        the ``bandwidth``.

        Input
        ------
        bandwidth : float
            Bandwidth [Hz]

        a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], complex
            Path coefficients

        tau : [batch size, num_rx, num_tx, num_paths] or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], float
            Path delays [s]

        l_min : int
            Smallest time-lag for the discrete complex baseband channel (:math:`L_{\text{min}}`)

        l_max : int
            Largest time-lag for the discrete complex baseband channel (:math:`L_{\text{max}}`)

        normalize : bool
            If set to `True`, the channel is normalized over the block size
            to ensure unit average energy per time step. Defaults to `False`.

        Output
        -------
        hm :  [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1], complex
            Channel taps coefficients
        """
        if len(tau.shape) == 4:
            # Expand dims to broadcast with h. Add the following dimensions:
            #  - number of rx antennas (2)
            #  - number of tx antennas (4)
            tau = torch.unsqueeze(torch.unsqueeze(tau, axis=2), axis=4)
            # Broadcast is not supported by TF for such high rank tensors.
            # We therefore do part of it manually
            tau = torch.tile(tau, [1, 1, 1, 1, a.shape[4], 1])

        # Add a time samples dimension for broadcasting
        tau = torch.unsqueeze(tau, axis=6)

        # Time lags for which to compute the channel taps
        l = torch.arange(l_min, l_max+1, dtype=torch.float32)

        # Bring tau and l to broadcastable shapes
        tau = torch.unsqueeze(tau, axis=-1)
        # l = np.expand_dims(l, (0,1,2,3,4,5,6))

        # sinc pulse shaping
        g = torch.sinc(l-tau*bandwidth) # L dims should be at end
        g = g + 0.0j
        a = torch.unsqueeze(a, axis=-1)

        # For every tap, sum the sinc-weighted coefficients
        hm = torch.sum(a*g, axis=-3)

        # if normalize:
        #     # Normalization is performed such that for each batch example and
        #     # link the energy per block is one.
        #     # The total energy of a channel response is the sum of the squared
        #     # norm over the channel taps.
        #     # Average over block size, RX antennas, and TX antennas
        #     c = np.mean(np.sum(np.square(np.abs(hm)),
        #                                     axis=6, keepdims=True),
        #                     axis=(2,4,5), keepdims=True)
        #     c = np.sqrt(c) + 0.0j
        #     hm = math.divide_no_nan(hm, c)

        return hm

    def __call__(self, num_time_samples, sampling_frequency, l_min, l_max):
        # Sample LSPs if required
        if self._always_generate_lsp:
            lsp = self._lsp_sampler()
        else:
            lsp = self._lsp

        # Sample rays
        rays = self._ray_sampler(lsp)

        # Sample channel responses
        # First we need to create a topology
        # Indicates which end of the channel is moving: TX or RX
        if self._scenario.direction == 'downlink':
            moving_end = 'rx'
            tx_orientations = self._scenario.bs_orientations
            rx_orientations = self._scenario.ut_orientations
        elif self._scenario.direction == 'uplink':
            moving_end = 'tx'
            tx_orientations = self._scenario.ut_orientations
            rx_orientations = self._scenario.bs_orientations
        topology = Topology(    velocities=self._scenario.ut_velocities,
                                moving_end=moving_end,
                                los_aoa=torch.deg2rad(self._scenario.los_aoa),
                                los_aod=torch.deg2rad(self._scenario.los_aod),
                                los_zoa=torch.deg2rad(self._scenario.los_zoa),
                                los_zod=torch.deg2rad(self._scenario.los_zod),
                                los=self._scenario.los,
                                distance_3d=self._scenario.distance_3d,
                                tx_orientations=tx_orientations,
                                rx_orientations=rx_orientations)

        # The channel coefficient needs the cluster delay spread parameter in ns
        c_ds = self._scenario.get_param("cDS")*1e-9

        # According to the link direction, we need to specify which from BS
        # and UT is uplink, and which is downlink.
        # Default is downlink, so we need to do some tranpose to switch tx and
        # rx and to switch angle of arrivals and departure if direction is set
        # to uplink. Nothing needs to be done if direction is downlink
        if self._scenario.direction == "uplink":
            aoa = rays.aoa
            zoa = rays.zoa
            aod = rays.aod
            zod = rays.zod
            rays.aod = torch.permute(aoa, [0, 2, 1, 3, 4])
            rays.zod = torch.permute(zoa, [0, 2, 1, 3, 4])
            rays.aoa = torch.permute(aod, [0, 2, 1, 3, 4])
            rays.zoa = torch.permute(zod, [0, 2, 1, 3, 4])
            rays.powers = torch.permute(rays.powers, [0, 2, 1, 3])
            rays.delays = torch.permute(rays.delays, [0, 2, 1, 3])
            rays.xpr = torch.permute(rays.xpr, [0, 2, 1, 3, 4])
            los_aod = topology.los_aod
            los_aoa = topology.los_aoa
            los_zod = topology.los_zod
            los_zoa = topology.los_zoa
            topology.los_aoa = torch.permute(los_aod, [0, 2, 1])
            topology.los_aod = torch.permute(los_aoa, [0, 2, 1])
            topology.los_zoa = torch.permute(los_zod, [0, 2, 1])
            topology.los_zod = torch.permute(los_zoa, [0, 2, 1])
            topology.los = torch.permute(topology.los, [0, 2, 1])
            c_ds = torch.permute(c_ds, [0, 2, 1])
            topology.distance_3d = torch.permute(topology.distance_3d, [0, 2, 1])
            # Concerning LSPs, only these two are used.
            # We do not transpose the others to reduce complexity
            k_factor = torch.permute(lsp.k_factor, [0, 2, 1])
            sf = torch.permute(lsp.sf, [0, 2, 1])
        else:
            k_factor = lsp.k_factor
            sf = lsp.sf

        # pylint: disable=unbalanced-tuple-unpacking
        h, delays = self._cir_sampler(num_time_samples, sampling_frequency,
                                      k_factor, rays, topology, c_ds)

        # Step 12 (path loss and shadow fading)
        if self._scenario.pathloss_enabled:
            pl_db = self._lsp_sampler.sample_pathloss()
            if self._scenario.direction == 'uplink':
                pl_db = torch.permute(pl_db, [0,2,1])
        else:
            pl_db = 0.0
        if not self._scenario.shadow_fading_enabled:
            sf = torch.ones_like(sf)
        gain = torch.pow(10.0, -(pl_db)/20.)*torch.sqrt(sf)
        gain = gain[(...,)+(None,)*(len(h.shape)-len(gain.shape))]
        h *= gain + 0.0j

        # Reshaping to match the expected output
        h = torch.permute(h, [0, 2, 4, 1, 5, 3, 6])
        delays = torch.permute(delays, [0, 2, 1, 3])

        h_T = self.cir_to_time_channel(sampling_frequency, h, delays, l_min, l_max)

        return h_T, h, delays

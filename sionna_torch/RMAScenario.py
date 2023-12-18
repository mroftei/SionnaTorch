#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""3GPP TR39.801 rural macrocell (RMa) channel scenario"""
import torch
import numpy as np
import scipy.constants
from .SystemLevelScenario import SystemLevelScenario

class RMaScenario(SystemLevelScenario):
    r"""
    3GPP TR 38.901 rural macrocell (RMa) channel model scenario.

    Parameters
    -----------

    carrier_frequency : float
        Carrier frequency [Hz]

    rx_array : PanelArray
        Panel array used by the receivers. All receivers share the same
        antenna array configuration.

    tx_array : PanelArray
        Panel array used by the transmitters. All transmitters share the
        same antenna array configuration.

    direction : str
        Link direction. Either "uplink" or "downlink".

    enable_pathloss : bool
        If `True`, apply pathloss. Otherwise doesn't. Defaults to `True`.

    enable_shadow_fading : bool
        If `True`, apply shadow fading. Otherwise doesn't.
        Defaults to `True`.

    average_street_width : float
        Average street width [m]. Defaults to 5m.

    average_street_width : float
        Average building height [m]. Defaults to 20m.

    always_generate_lsp : bool
        If `True`, new large scale parameters (LSPs) are generated for every
        new generation of channel impulse responses. Otherwise, always reuse
        the same LSPs, except if the topology is changed. Defaults to
        `False`.

    dtype : Complex DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `complex64`.
    """

    def __init__(self, carrier_frequency,
        direction, rng: torch.Generator, enable_pathloss=True, enable_shadow_fading=True,
        average_street_width=20.0, average_building_height=5.0, dtype=torch.complex64):

        # Only the low-loss O2I model if available for RMa.
        super().__init__(carrier_frequency, False,
            direction, rng, enable_pathloss, enable_shadow_fading, dtype)

        # Average street width [m]
        self.average_street_width = average_street_width

        # Average building height [m]
        self.average_building_height = average_building_height

        self.is_urban = False

    #########################################
    # Public methods and properties
    #########################################

    def clip_carrier_frequency_lsp(self, fc):
        r"""Clip the carrier frequency ``fc`` in GHz for LSP calculation

        Input
        -----
        fc : float
            Carrier frequency [GHz]

        Output
        -------
        : float
            Clipped carrier frequency, that should be used for LSp computation
        """
        return fc

    @property
    def los_probability(self):
        r"""Probability of each UT to be LoS. Used to randomly generate LoS
        status of outdoor UTs.

        Computed following section 7.4.2 of TR 38.901.

        [batch size, num_ut]"""
        distance_2d = self.distance_2d
        los_probability = torch.exp(-(distance_2d-10.0)/1000.0)
        los_probability = torch.where(torch.less(distance_2d, 10.0), 1.0, los_probability)
        return los_probability

    @property
    def rays_per_cluster(self):
        r"""Number of rays per cluster"""
        return 20

    #########################
    # Utility methods
    #########################

    def _compute_lsp_log_mean_std(self):
        r"""Computes the mean and standard deviations of LSPs in log-domain"""

        batch_size = self.batch_size
        num_bs = self.num_bs
        num_ut = self.num_ut
        distance_2d = self.distance_2d
        h_bs = self.h_bs
        h_bs = np.expand_dims(h_bs, axis=2) # For broadcasting
        h_ut = self.h_ut
        h_ut = np.expand_dims(h_ut, axis=1) # For broadcasting

        ## Mean
        # DS
        log_mean_ds = self.get_param("muDS")
        # ASD
        log_mean_asd = self.get_param("muASD")
        # ASA
        log_mean_asa = self.get_param("muASA")
        # SF.  Has zero-mean.
        log_mean_sf = torch.zeros((batch_size, num_bs, num_ut), dtype=self._dtype_real).numpy()
        # K.  Given in dB in the 3GPP tables, hence the division by 10
        log_mean_k = self.get_param("muK")/10.0
        # ZSA
        log_mean_zsa = self.get_param("muZSA")
        # ZSD mean is of the form max(-1, A*d2D/1000 - 0.01*(hUT-1.5) + B)
        log_mean_zsd = (self.get_param("muZSDa")*(distance_2d/1000.)
            - 0.01*(h_ut-1.5) + self.get_param("muZSDb"))
        log_mean_zsd = np.maximum(-1.0, log_mean_zsd)

        lsp_log_mean = np.stack([log_mean_ds,
                                log_mean_asd,
                                log_mean_asa,
                                log_mean_sf,
                                log_mean_k,
                                log_mean_zsa,
                                log_mean_zsd], axis=3)

        ## STD
        # DS
        log_std_ds = self.get_param("sigmaDS")
        # ASD
        log_std_asd = self.get_param("sigmaASD")
        # ASA
        log_std_asa = self.get_param("sigmaASA")
        # SF. Given in dB in the 3GPP tables, hence the division by 10
        # O2I and NLoS cases just require the use of a predefined value
        log_std_sf_o2i_nlos = self.get_param("sigmaSF")/10.0
        # For LoS, two possible scenarion depending on the 2D location of the
        # user
        distance_breakpoint = (2.*np.pi*h_bs*h_ut*self.carrier_frequency/scipy.constants.c)
        log_std_sf_los=np.where(np.less(distance_2d, distance_breakpoint),
            self.get_param("sigmaSF1")/10.0, self.get_param("sigmaSF2")/10.0)
        # Use the correct SF STD according to the user scenario: NLoS/O2I, or
        # LoS
        log_std_sf = np.where(self.los, log_std_sf_los, log_std_sf_o2i_nlos)
        # K. Given in dB in the 3GPP tables, hence the division by 10.
        log_std_k = self.get_param("sigmaK")/10.0
        # ZSA
        log_std_zsa = self.get_param("sigmaZSA")
        # ZSD
        log_std_zsd = self.get_param("sigmaZSD")

        lsp_log_std = np.stack([log_std_ds,
                               log_std_asd,
                               log_std_asa,
                               log_std_sf,
                               log_std_k,
                               log_std_zsa,
                               log_std_zsd], axis=3)

        self.lsp_log_mean = lsp_log_mean
        self.lsp_log_std = lsp_log_std

        # ZOD offset
        zod_offset = (np.arctan((35.-3.5)/distance_2d)
          - np.arctan((35.-1.5)/distance_2d))
        zod_offset = np.where(self.los, 0.0, zod_offset)
        self.zod_offset = zod_offset

    def _compute_pathloss_basic(self):
        r"""Computes the basic component of the pathloss [dB]"""

        distance_2d = self.distance_2d
        distance_3d = self.distance_3d
        fc = self.carrier_frequency/1e9 # Carrier frequency (GHz)
        h_bs = self.h_bs
        h_bs = torch.unsqueeze(h_bs, axis=2) # For broadcasting
        h_ut = self.h_ut
        h_ut = torch.unsqueeze(h_ut, axis=1) # For broadcasting
        average_building_height = self.average_building_height

        # Beak point distance
        # For this computation, the carrifer frequency needs to be in Hz
        distance_breakpoint = (2.*torch.pi*h_bs*h_ut*self.carrier_frequency/scipy.constants.c)

        ## Basic path loss for LoS
        pl_1 = (20.0*torch.log10(40.0*torch.pi*distance_3d*fc/3.)
            + np.minimum(0.03*np.power(average_building_height,1.72),
                10.0)*torch.log10(distance_3d)
            - np.minimum(0.044*np.power(average_building_height,1.72),
                14.77)
            + 0.002*np.log10(average_building_height)*distance_3d)
        pl_2 = (20.0*torch.log10(40.0*torch.pi*distance_breakpoint*fc/3.)
            + np.minimum(0.03*np.power(average_building_height,1.72),
                10.0)*torch.log10(distance_breakpoint)
            - np.minimum(0.044*np.power(average_building_height,1.72),
                14.77)
            + 0.002*np.log10(average_building_height)*distance_breakpoint
            + 40.0*torch.log10(distance_3d/distance_breakpoint))
        pl_los = torch.where(torch.less(distance_2d, distance_breakpoint),
            pl_1, pl_2)

        ## Basic pathloss for NLoS and O2I

        pl_3 = (161.04 - 7.1*np.log10(self.average_street_width)
                + 7.5*np.log10(average_building_height)
                - (24.37 - 3.7*np.square(average_building_height/h_bs))
                *np.log10(h_bs)
                + (43.42 - 3.1*np.log10(h_bs))*(np.log10(distance_3d)-3.0)
                + 20.0*np.log10(fc) - (3.2*np.square(np.log10(11.75*h_ut))
                - 4.97))
        pl_nlos = np.maximum(pl_los, pl_3)

        ## Set the basic pathloss according to UT state

        # LoS
        pl_b = np.where(self.los, pl_los, pl_nlos)

        self.basic_pathloss = pl_b

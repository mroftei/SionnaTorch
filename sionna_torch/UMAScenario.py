#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""3GPP TR39.801 urban macrocell (UMa) channel model"""
import torch
import numpy as np
import scipy.constants
from .SystemLevelScenario import SystemLevelScenario


class UMaScenario(SystemLevelScenario):
    r"""
    3GPP TR 38.901 urban macrocell (UMa) channel model scenario.

    Parameters
    -----------
    carrier_frequency : float
        Carrier frequency [Hz]

    ut_array : PanelArray
        Panel array configuration used by UTs

    bs_array : PanelArray
        Panel array configuration used by BSs

    direction : str
        Link direction. Either "uplink" or "downlink"

    enable_pathloss : bool
        If set to `True`, apply pathloss. Otherwise, does not. Defaults to True.

    enable_shadow_fading : bool
        If set to `True`, apply shadow fading. Otherwise, does not.
        Defaults to True.

    dtype : DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `complex64`.
    """

    def __init__(self, carrier_frequency,
        direction, rng: torch.Generator, enable_pathloss=True, enable_shadow_fading=True, dtype=torch.complex64):

        # Only the low-loss O2I model if available for RMa.
        super().__init__(carrier_frequency, True,
            direction, rng, enable_pathloss, enable_shadow_fading, dtype)

        self.is_urban = True

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
        if fc < 6.:
            fc = 6.0
        return fc

    @property
    def los_probability(self):
        r"""Probability of each UT to be LoS. Used to randomly generate LoS
        status of outdoor UTs.

        Computed following section 7.4.2 of TR 38.901.

        [batch size, num_ut]"""

        h_ut = self.h_ut
        c = torch.pow(torch.abs(h_ut-13.)/10., 1.5)
        c = torch.where(torch.less(h_ut, 13.0), 0.0, c)
        c = torch.unsqueeze(c, axis=1)

        distance_2d = self.distance_2d
        los_probability = ((18.0/distance_2d
            + torch.exp(-distance_2d/63.0)*(1.-18./distance_2d))
            *(1.+c*5./4.*torch.pow(distance_2d/100., 3)
                *torch.exp(-distance_2d/150.0)))

        los_probability = torch.where(torch.less(distance_2d, 18.0), 1.0, los_probability)
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
        log_mean_sf = torch.zeros([batch_size, num_bs, num_ut], dtype=self._dtype_real).numpy()
        # K.  Given in dB in the 3GPP tables, hence the division by 10
        log_mean_k = self.get_param("muK")/10.0
        # ZSA
        log_mean_zsa = self.get_param("muZSA")
        # ZSD
        log_mean_zsd_los = np.clip(-2.1*(distance_2d/1000.0) - 0.01*np.abs(h_ut-1.5)+0.75, -0.5, None)
        log_mean_zsd_nlos = np.clip(-2.1*(distance_2d/1000.0) - 0.01*np.abs(h_ut-1.5)+0.9, -0.5, None)
        log_mean_zsd = np.where(self.los, log_mean_zsd_los, log_mean_zsd_nlos)

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
        log_std_sf = self.get_param("sigmaSF")/10.0
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
        fc = self.carrier_frequency/1e9
        if fc < 6.:
            fc = 6.0
        a = 0.208*np.log10(fc)-0.782
        b = 25.0
        c = -0.13*np.log10(fc)+2.03
        e = 7.66*np.log10(fc)-5.96
        zod_offset =(e-np.power(10.0, a*np.log10(np.clip(distance_2d, b, None)) + c - 0.07*(h_ut-1.5)))
        zod_offset = np.where(self.los, 0.0, zod_offset)
        self.zod_offset = zod_offset

    def _compute_pathloss_basic(self):
        r"""Computes the basic component of the pathloss [dB]"""

        batch_size = self.batch_size
        num_bs = self.num_bs
        num_ut = self.num_ut
        distance_2d = self.distance_2d
        distance_3d = self.distance_3d
        fc = self.carrier_frequency # Carrier frequency (Hz)
        h_bs = self.h_bs
        h_bs = np.expand_dims(h_bs, axis=2) # For broadcasting
        h_ut = self.h_ut
        h_ut = np.expand_dims(h_ut, axis=1) # For broadcasting

        # Beak point distance
        g = ((5./4.)*np.power(distance_2d/100., 3.)
            *np.exp(-distance_2d/150.0))
        g = np.where(np.clip(distance_2d, None, 18.), 0.0, g)
        c = g*np.power((h_ut-13.)/10., 1.5)
        c = np.where(np.clip(h_ut, None, 13.), 0.0, c)
        p = 1./(1.+c)
        r = torch.rand([batch_size, num_bs, num_ut], generator=self.rng, dtype=self._dtype_real).numpy()
        r = np.where(np.less(r, p), 1.0, 0.0)

        max_value = h_ut- 1.5
        # Random uniform integer generation is not supported when maxval and
        # are not scalar. The following commented would therefore not work.
        # Therefore, for now, we just sample from a continuous
        # distribution.
        s = torch.rand([batch_size, num_bs, num_ut], generator=self.rng, dtype=self._dtype_real).numpy()
        s = (12.0 - max_value) * s + max_value
        # Itc could happen that h_ut = 13m, and therefore max_value < 13m
        s = np.where(np.less(s, 12.0), 12.0, s)

        h_e = r + (1.-r)*s
        h_bs_prime = h_bs - h_e
        h_ut_prime = h_ut - h_e
        distance_breakpoint = 4*h_bs_prime*h_ut_prime*fc/scipy.constants.c

        ## Basic path loss for LoS

        pl_1 = 28.0 + 22.0*np.log10(distance_3d) + 20.0*np.log10(fc/1e9)
        pl_2 = (28.0 + 40.0*np.log10(distance_3d) + 20.0*np.log10(fc/1e9)
         - 9.0*np.log10(np.square(distance_breakpoint)+np.square(h_bs-h_ut)))
        pl_los = np.where(np.less(distance_2d, distance_breakpoint),
            pl_1, pl_2)

        ## Basic pathloss for NLoS and O2I

        pl_3 = (13.54 + 39.08*np.log10(distance_3d) + 20.0*np.log10(fc/1e9)
            - 0.6*(h_ut-1.5))
        pl_nlos = np.maximum(pl_los, pl_3)

        ## Set the basic pathloss according to UT state

        # Expand to allow broadcasting with the BS dimension
        # LoS
        pl_b = np.where(self.los, pl_los, pl_nlos)

        self.basic_pathloss = pl_b

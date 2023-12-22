#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any
import torch
import numpy as np
import scipy.constants

from .ChannelCoefficients import ChannelCoefficientsGenerator
from .LSPGenerator import LSPGenerator
from .RaysGenerator import RaysGenerator
from .ApplyTimeChannel import ApplyTimeChannel
from .Parameters import PARAMS_LOS_RURAL, PARAMS_LOS_URBAN, PARAMS_NLOS_RURAL, PARAMS_NLOS_URBAN


class SionnaScenario:
    def __init__(self, 
                 ut_xy: np.ndarray, #[batch size,num_ut, 3]
                 bs_xy: np.ndarray, #[batch size,num_bs, 3]
                 urban_state: np.ndarray, #[batch size,num_bs]
                 ut_velocities: np.ndarray = None, #[batch size,num_ut, 3]
                 los_requested: np.ndarray = None,
                 direction: str = "uplink", #uplink/downlink
                 n_time_samples: int = 1024,
                 f_c: float = .92e9,
                 bw: float = 30e3,
                 noise_power = 1e-9,
                 seed: int = 42,
                 dtype=torch.complex64
            ) -> None:
        
        self.f_c = f_c
        self.lambda_0 = scipy.constants.c/f_c # wavelength
        self.bw = bw
        self.noise_power = noise_power
        self.rng = rng = torch.Generator().manual_seed(seed)
        self.average_street_width = 20.0
        self.average_building_height = 5.0
        self.rays_per_cluster = 20
        self.direction = direction #uplink/downlink
        self.n_samples = n_time_samples
        self.los_requested = los_requested

        # data type
        assert dtype.is_complex, "'dtype' must be complex type"
        self._dtype = dtype
        self._dtype_real = dtype.to_real()

        self.l_min, self.l_max = -6, int(np.ceil(3e-6*self.bw)) + 6
        l_tot = self.l_max-self.l_min+1

        self._cir_sampler = ChannelCoefficientsGenerator(f_c, subclustering=True, rng = rng)
        self._lsp_sampler = LSPGenerator(self, rng)
        self._ray_sampler = RaysGenerator(self, rng)
        self._apply_channel = ApplyTimeChannel(n_time_samples, l_tot=l_tot, rng=rng, add_awgn=True)

        self.update_topology(ut_xy, bs_xy, urban_state, ut_velocities, los_requested)
        

    def update_topology(self,
                        ut_xy: np.ndarray, #[batch size,num_ut, 3],
                        bs_xy: np.ndarray, #[batch size,num_bs, 3]
                        urban_state: np.ndarray, #[batch size,num_bs]
                        ut_velocities: np.ndarray = None, #[batch size,num_ut, 3]
                        los_requested: np.ndarray = None):
        # set_topology
        self.ut_xy = torch.from_numpy(ut_xy)
        self.h_ut = self.ut_xy[:,:,2]
        self.bs_xy = torch.from_numpy(bs_xy)
        self.is_urban = torch.from_numpy(urban_state)
        self.los_requested = los_requested
        self.batch_size = self.ut_xy.shape[0]
        self.num_ut = self.ut_xy.shape[1]
        self.num_bs = self.bs_xy.shape[1]
        self.h_bs = self.bs_xy[:,:,2]
        if ut_velocities is None:
            self.ut_velocities = torch.zeros_like(self.ut_xy)
        else:
            self.ut_velocities = ut_velocities

        # Update topology-related quantities
        self._compute_distance_2d_3d_and_angles()
        self.is_los = self._compute_los()

        # Compute the LSPs means and stds
        self.lsp_log_mean, self.lsp_log_std, self.zod_offset = self._compute_lsp_log_mean_std()

        # Compute the basic path-loss
        self.basic_pathloss = self._compute_pathloss_basic()

        # Update the LSP sampler
        self._lsp_sampler.topology_updated_callback()

        # Update the ray sampler
        self.num_clusters_max = int(self.get_param("numClusters").max())
        self._ray_sampler.topology_updated_callback()
        
        
    def __call__(self, x: torch.Tensor) -> Any:
        assert x.shape[-1] == self.n_samples, "Input frame size mismatch"

        # Sample LSPs 
        lsp = self._lsp_sampler()
        # Sample rays
        rays = self._ray_sampler(lsp)

        # Sample channel responses

        # The channel coefficient needs the cluster delay spread parameter in ns
        c_ds = self.get_param("cDS")*1e-9

        # According to the link direction, we need to specify which from BS
        # and UT is uplink, and which is downlink.
        # Default is downlink, so we need to do some tranpose to switch tx and
        # rx and to switch angle of arrivals and departure if direction is set
        # to uplink. Nothing needs to be done if direction is downlink
        if self.direction == "uplink":
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
            c_ds = torch.permute(c_ds, [0, 2, 1])
            # Concerning LSPs, only these two are used.
            # We do not transpose the others to reduce complexity
            lsp.k_factor = torch.permute(lsp.k_factor, [0, 2, 1])
            lsp.sf = torch.permute(lsp.sf, [0, 2, 1])

        num_time_samples_lag = self.n_samples+(self.l_max-self.l_min)
        h, tau = self._cir_sampler(num_time_samples_lag, self.f_c,
                                      lsp.k_factor, rays, self, c_ds)

        # Step 12 (path loss and shadow fading)
        gain = torch.pow(10.0, -(self.basic_pathloss)/20.)*torch.sqrt(lsp.sf)
        gain = gain[(...,)+(None,)*(len(h.shape)-len(gain.shape))]
        h *= gain + 0.0j

        ## cir_to_time_channel 
        # Reshaping to match the expected output
        h = torch.permute(h, [0, 2, 4, 1, 5, 3, 6])[...,None]
        tau = torch.permute(tau, [0, 2, 1, 3])[:,:,None,:,None,:,None,None]
        # Expand dims to broadcast with h.
        tau = torch.tile(tau, [1, 1, 1, 1, h.shape[4], 1,1])

        # Time lags for which to compute the channel taps
        l = torch.arange(self.l_min, self.l_max+1, dtype=torch.float32)

        # sinc pulse shaping
        g = torch.sinc(l-tau*self.bw) + 0.0j # L dims should be at end

        # For every tap, sum the sinc-weighted coefficients
        h_T = torch.sum(h*g, axis=-3)

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
        y_torch, snr = self._apply_channel(x, h_T, self.noise_power)

        return y_torch, snr

    def get_param(self, parameter_name):
        r"""
        Given a ``parameter_name`` used in the configuration, returns a
        tensor with shape [batch size, number of BSs, number of UTs] of the
        parameter value according to each BS-UT link state (LoS or NLoS).

        Input
        ------
        parameter_name : str
            Name of the parameter used in the configuration file

        Output
        -------
        : [batch size, number of BSs, number of UTs], float
            Parameter value for each BS-UT link
        """

        fc = self.f_c/1e9
        if fc < 6.:
            fc = np.where(self.is_urban, 6.0, fc)

        # Parameter value
        if parameter_name in ('muDS', 'sigmaDS', 'muASD', 'sigmaASD', 'muASA',
                                'sigmaASA', 'muZSA', 'sigmaZSA'):
            pa_los = PARAMS_LOS_URBAN[parameter_name + 'a']
            pb_los = PARAMS_LOS_URBAN[parameter_name + 'b']
            pc_los = PARAMS_LOS_URBAN[parameter_name + 'c']
            parameter_value_los_urban = pa_los*np.log10(pb_los+fc) + pc_los

            pa_nlos = PARAMS_NLOS_URBAN[parameter_name + 'a']
            pb_nlos = PARAMS_NLOS_URBAN[parameter_name + 'b']
            pc_nlos = PARAMS_NLOS_URBAN[parameter_name + 'c']
            parameter_value_nlos_urban = pa_nlos*np.log10(pb_nlos+fc) + pc_nlos

            pa_los = PARAMS_LOS_RURAL[parameter_name + 'a']
            pb_los = PARAMS_LOS_RURAL[parameter_name + 'b']
            pc_los = PARAMS_LOS_RURAL[parameter_name + 'c']
            parameter_value_los_rural = pa_los*np.log10(pb_los+fc) + pc_los

            pa_nlos = PARAMS_NLOS_RURAL[parameter_name + 'a']
            pb_nlos = PARAMS_NLOS_RURAL[parameter_name + 'b']
            pc_nlos = PARAMS_NLOS_RURAL[parameter_name + 'c']
            parameter_value_nlos_rural = pa_nlos*np.log10(pb_nlos+fc) + pc_nlos

        elif parameter_name == "cDS":
            pa_los = PARAMS_LOS_URBAN[parameter_name + 'a']
            pb_los = PARAMS_LOS_URBAN[parameter_name + 'b']
            pc_los = PARAMS_LOS_URBAN[parameter_name + 'c']
            parameter_value_los_urban = np.maximum(pa_los, pb_los - pc_los*np.log10(fc))

            pa_nlos = PARAMS_NLOS_URBAN[parameter_name + 'a']
            pb_nlos = PARAMS_NLOS_URBAN[parameter_name + 'b']
            pc_nlos = PARAMS_NLOS_URBAN[parameter_name + 'c']
            parameter_value_nlos_urban = np.maximum(pa_nlos, pb_nlos - pc_nlos*np.log10(fc))

            pa_los = PARAMS_LOS_RURAL[parameter_name + 'a']
            pb_los = PARAMS_LOS_RURAL[parameter_name + 'b']
            pc_los = PARAMS_LOS_RURAL[parameter_name + 'c']
            parameter_value_los_rural = np.maximum(pa_los, pb_los - pc_los*np.log10(fc))

            pa_nlos = PARAMS_NLOS_RURAL[parameter_name + 'a']
            pb_nlos = PARAMS_NLOS_RURAL[parameter_name + 'b']
            pc_nlos = PARAMS_NLOS_RURAL[parameter_name + 'c']
            parameter_value_nlos_rural = np.maximum(pa_nlos, pb_nlos - pc_nlos*np.log10(fc))

        else:
            parameter_value_los_urban = PARAMS_LOS_URBAN[parameter_name]
            parameter_value_nlos_urban = PARAMS_NLOS_URBAN[parameter_name]
            parameter_value_los_rural = PARAMS_LOS_RURAL[parameter_name]
            parameter_value_nlos_rural = PARAMS_NLOS_RURAL[parameter_name]

        parameter_value_los = np.where(self.is_urban, parameter_value_los_urban, parameter_value_los_rural)
        parameter_value_nlos = np.where(self.is_urban, parameter_value_nlos_urban, parameter_value_nlos_rural)
        parameter_tensor = np.where(self.is_los, parameter_value_los, parameter_value_nlos)

        return torch.from_numpy(parameter_tensor).type(self._dtype_real)

    def _compute_distance_2d_3d_and_angles(self):
        r"""
        Computes the following internal values:
        * 2D distances for all BS-UT pairs in the X-Y plane
        * 3D distances for all BS-UT pairs
        * 2D distances for all pairs of UTs in the X-Y plane
        * LoS AoA, AoD, ZoA, ZoD for all BS-UT pairs

        This function is called at every update of the topology.
        """

        ut_loc = self.ut_xy
        ut_loc = torch.unsqueeze(ut_loc, axis=1)

        bs_loc = self.bs_xy
        bs_loc = torch.unsqueeze(bs_loc, axis=2)

        delta_loc_xy = ut_loc[:,:,:,:2] - bs_loc[:,:,:,:2]
        delta_loc = ut_loc - bs_loc

        # 2D distances for all BS-UT pairs in the (x-y) plane
        distance_2d = torch.sqrt(torch.sum(torch.square(delta_loc_xy), axis=3))
        self.distance_2d = distance_2d

        # 3D distances for all BS-UT pairs
        distance_3d = torch.sqrt(torch.sum(torch.square(delta_loc), axis=3))
        self.distance_3d = distance_3d

        # LoS AoA, AoD, ZoA, ZoD
        los_aod = torch.arctan2(delta_loc[:,:,:,1], delta_loc[:,:,:,0])
        los_aoa = los_aod + torch.pi
        los_zod = torch.arctan2(distance_2d, delta_loc[:,:,:,2])
        los_zoa = los_zod - torch.pi
        # Angles are wrapped to (0,360)
        self.los_aod_rad = torch.remainder(los_aod, 2*torch.pi)
        self.los_aoa_rad = torch.remainder(los_aoa, 2*torch.pi)
        self.los_zod_rad = torch.remainder(los_zod, 2*torch.pi)
        self.los_zoa_rad = torch.remainder(los_zoa, 2*torch.pi)

        # 2D distances for all pairs of UTs in the (x-y) plane
        ut_loc_xy = self.ut_xy[:,:,:2]

        ut_loc_xy_expanded_1 = torch.unsqueeze(ut_loc_xy, axis=1)
        ut_loc_xy_expanded_2 = torch.unsqueeze(ut_loc_xy, axis=2)

        delta_loc_xy = ut_loc_xy_expanded_1 - ut_loc_xy_expanded_2

        matrix_ut_distance_2d = torch.sqrt(torch.sum(torch.square(delta_loc_xy),
                                                       axis=3))
        self.matrix_ut_distance_2d = matrix_ut_distance_2d

    def _compute_los(self):
        # urban
        c = torch.pow(torch.abs(self.h_ut-13.)/10., 1.5)
        c = torch.where(torch.less(self.h_ut, 13.0), 0.0, c)
        c = torch.unsqueeze(c, axis=1)
        urban_los_probability = ((18.0/self.distance_2d
            + torch.exp(-self.distance_2d/63.0)*(1.-18./self.distance_2d))
            *(1.+c*5./4.*torch.pow(self.distance_2d/100., 3)
                *torch.exp(-self.distance_2d/150.0)))
        urban_los_probability = torch.where(torch.less(self.distance_2d, 18.0), 1.0, urban_los_probability)

        # rural
        rural_los_probability = torch.exp(-(self.distance_2d-10.0)/1000.0)
        rural_los_probability = torch.where(torch.less(self.distance_2d, 10.0), 1.0, rural_los_probability)

        self.los_probability = torch.where(self.is_urban, urban_los_probability, rural_los_probability)
        los = torch.rand([self.batch_size, self.num_bs, self.num_ut], generator=self.rng, dtype=self._dtype_real)

        if self.los_requested is not None:
            return torch.ones_like(self.distance_2d, dtype=bool) * self.los_requested
        return torch.less(los, self.los_probability)
    
    def _compute_lsp_log_mean_std(self):
        distance_2d = self.distance_2d
        h_bs = torch.unsqueeze(self.h_bs, axis=2) # For broadcasting
        h_ut = torch.unsqueeze(self.h_ut, axis=1) # For broadcasting

        ## Mean
        # DS
        log_mean_ds = self.get_param("muDS")
        # ASD
        log_mean_asd = self.get_param("muASD")
        # ASA
        log_mean_asa = self.get_param("muASA")
        # SF.  Has zero-mean.
        log_mean_sf = torch.zeros([self.batch_size, self.num_bs, self.num_ut], dtype=self._dtype_real)
        # K.  Given in dB in the 3GPP tables, hence the division by 10
        log_mean_k = self.get_param("muK")/10.0
        # ZSA
        log_mean_zsa = self.get_param("muZSA")
        # ZSD
        # urban
        urban_log_mean_zsd_los = torch.clip(-2.1*(distance_2d/1000.0) - 0.01*torch.abs(h_ut-1.5)+0.75, -0.5, None)
        urban_log_mean_zsd_nlos = torch.clip(-2.1*(distance_2d/1000.0) - 0.01*torch.abs(h_ut-1.5)+0.9, -0.5, None)
        urban_log_mean_zsd = torch.where(self.is_los, urban_log_mean_zsd_los, urban_log_mean_zsd_nlos)
        #rural
        rural_log_mean_zsd = (self.get_param("muZSDa")*(distance_2d/1000.)
            - 0.01*(h_ut-1.5) + self.get_param("muZSDb"))
        rural_log_mean_zsd = torch.clip(rural_log_mean_zsd, -1.0)

        log_mean_zsd = torch.where(self.is_urban, urban_log_mean_zsd, rural_log_mean_zsd)


        lsp_log_mean = torch.stack([log_mean_ds,
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
        # Rural LOS has more work
        # For LoS, two possible scenarion depending on the 2D location of the user
        distance_breakpoint = (2.*torch.pi*h_bs*h_ut*self.f_c/scipy.constants.c)
        log_std_sf_los=torch.where(torch.less(distance_2d, distance_breakpoint),
            self.get_param("sigmaSF1")/10.0, self.get_param("sigmaSF2")/10.0)
        # Use the correct SF STD according to the user scenario: NLoS/O2I, LoS
        log_std_sf = torch.where(torch.logical_and(self.is_los, torch.logical_not(self.is_urban)), log_std_sf_los, log_std_sf)
        # K. Given in dB in the 3GPP tables, hence the division by 10.
        log_std_k = self.get_param("sigmaK")/10.0
        # ZSA
        log_std_zsa = self.get_param("sigmaZSA")
        # ZSD
        log_std_zsd = self.get_param("sigmaZSD")

        lsp_log_std = torch.stack([log_std_ds,
                               log_std_asd,
                               log_std_asa,
                               log_std_sf,
                               log_std_k,
                               log_std_zsa,
                               log_std_zsd], axis=3)

        # ZOD offset
        fc = self.f_c/1e9
        if fc < 6.:
            fc = 6.0
        a = 0.208*np.log10(fc)-0.782
        b = 25.0
        c = -0.13*np.log10(fc)+2.03
        e = 7.66*np.log10(fc)-5.96
        urban_zod_offset =(e-torch.pow(10.0, a*torch.log10(torch.clip(distance_2d, b, None)) + c - 0.07*(h_ut-1.5)))
        rural_zod_offset = (torch.arctan((35.-3.5)/distance_2d) - torch.arctan((35.-1.5)/distance_2d))
        zod_offset = torch.where(self.is_urban, urban_zod_offset, rural_zod_offset)
        zod_offset = torch.where(self.is_los, 0.0, zod_offset)

        return lsp_log_mean, lsp_log_std, zod_offset
    
    def _compute_pathloss_basic(self):
        r"""Computes the basic component of the pathloss [dB]"""
        h_bs = torch.unsqueeze(self.h_bs, dim=2) # For broadcasting
        h_ut = torch.unsqueeze(self.h_ut, dim=1) # For broadcasting

        # Beak point distance
        g = ((5./4.)*torch.pow(self.distance_2d/100., 3.)
            *torch.exp(-self.distance_2d/150.0))
        g = torch.where(torch.less_equal(self.distance_2d, 18.0), 0.0, g)
        c = g*torch.pow((h_ut-13.)/10., 1.5)
        c = torch.where(torch.less(h_ut, 13.), 0.0, c)
        p = 1./(1.+c)
        r = torch.rand([self.batch_size, self.num_bs, self.num_ut], generator=self.rng, dtype=self._dtype_real)
        r = torch.where(torch.less(r, p), 1.0, 0.0)

        max_value = h_ut- 1.5
        # Random uniform integer generation is not supported when maxval and
        # are not scalar. The following commented would therefore not work.
        # Therefore, for now, we just sample from a continuous
        # distribution.
        s = torch.rand([self.batch_size, self.num_bs, self.num_ut], generator=self.rng, dtype=self._dtype_real)
        s = (12.0 - max_value) * s + max_value
        # Itc could happen that h_ut = 13m, and therefore max_value < 13m
        s = torch.where(torch.less(s, 12.0), 12.0, s)

        h_e = r + (1.-r)*s
        h_bs_prime = h_bs - h_e
        h_ut_prime = h_ut - h_e
        urban_distance_breakpoint = 4*h_bs_prime*h_ut_prime*self.f_c/scipy.constants.c
        rural_distance_breakpoint = (2.*torch.pi*h_bs*h_ut*self.f_c/scipy.constants.c)

        ## Basic path loss for LoS
        # Urban
        pl_1 = 28.0 + 22.0*torch.log10(self.distance_3d) + 20.0*np.log10(self.f_c/1e9)
        pl_2 = (28.0 + 40.0*torch.log10(self.distance_3d) + 20.0*np.log10(self.f_c/1e9)
         - 9.0*torch.log10(np.square(urban_distance_breakpoint)+torch.square(h_bs-h_ut)))
        urban_pl_los = torch.where(torch.less(self.distance_2d, urban_distance_breakpoint),
            pl_1, pl_2)
        # Rural
        pl_1 = (20.0*torch.log10(40.0*torch.pi*self.distance_3d*self.f_c/3e9)
            + np.minimum(0.03*np.power(self.average_building_height,1.72),
                10.0)*torch.log10(self.distance_3d)
            - np.minimum(0.044*np.power(self.average_building_height,1.72),
                14.77)
            + 0.002*np.log10(self.average_building_height)*self.distance_3d)
        pl_2 = (20.0*torch.log10(40.0*torch.pi*rural_distance_breakpoint*self.f_c/3e9)
            + np.minimum(0.03*np.power(self.average_building_height,1.72),
                10.0)*torch.log10(rural_distance_breakpoint)
            - np.minimum(0.044*np.power(self.average_building_height,1.72),
                14.77)
            + 0.002*np.log10(self.average_building_height)*rural_distance_breakpoint
            + 40.0*torch.log10(self.distance_3d/rural_distance_breakpoint))
        rural_pl_los = torch.where(torch.less(self.distance_2d, rural_distance_breakpoint),
            pl_1, pl_2)

        ## Basic pathloss for NLoS and O2I
        # Urban
        pl_3 = (13.54 + 39.08*torch.log10(self.distance_3d) + 20.0*np.log10(self.f_c/1e9)
            - 0.6*(h_ut-1.5))
        urban_pl_nlos = torch.maximum(urban_pl_los, pl_3)
        # Rural
        pl_3 = (161.04 - 7.1*np.log10(self.average_street_width)
                + 7.5*np.log10(self.average_building_height)
                - (24.37 - 3.7*torch.square(self.average_building_height/h_bs))
                *torch.log10(h_bs)
                + (43.42 - 3.1*torch.log10(h_bs))*(torch.log10(self.distance_3d)-3.0)
                + 20.0*np.log10(self.f_c/1e9) - (3.2*torch.square(torch.log10(11.75*h_ut))
                - 4.97))
        rural_pl_nlos = torch.maximum(rural_pl_los, pl_3)

        ## Set the basic pathloss according to UT state
        # Expand to allow broadcasting with the BS dimension
        # LoS
        urban_pl_b = torch.where(self.is_los, urban_pl_los, urban_pl_nlos)
        rural_pl_b = torch.where(self.is_los, rural_pl_los, rural_pl_nlos)
        pl_b = torch.where(self.is_urban, urban_pl_b, rural_pl_b)

        if self.direction == 'uplink':
            pl_b = torch.permute(pl_b, [0,2,1])

        return pl_b
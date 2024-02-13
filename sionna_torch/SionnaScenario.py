#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Optional, Tuple
import torch
import numpy as np
import scipy.constants

from .ChannelCoefficients import ChannelCoefficientsGenerator
from .LSPGenerator import LSPGenerator
from .RaysGenerator import RaysGenerator
from .ApplyTimeChannel import ApplyTimeChannel
from .Parameters import PARAMS_IDX, PARAMS_LOS_RURAL, PARAMS_LOS_URBAN, PARAMS_NLOS_RURAL, PARAMS_NLOS_URBAN


class SionnaScenario:
    def __init__(self,
                 n_bs=1,
                 n_ut=1,
                 batch_size=1,
                 n_time_samples: int = 1024,
                 f_c: float = .92e9,
                 bw: float = 30e3,
                 noise_power_dB = None,
                 enable_sf = True,
                 enable_fading = True,
                 seed: int = 42,
                 dtype=torch.complex64,
                 device: Optional[torch.device] = None,
    ) -> None:
        self.batch_size = batch_size
        self.num_bs = n_bs
        self.num_ut = n_ut
        self.f_c = f_c
        self.lambda_0 = scipy.constants.c/f_c # wavelength
        self.bw = bw
        self.rng = rng = torch.Generator(device=device).manual_seed(seed)
        self.average_street_width = 20.0
        self.average_building_height = 5.0
        self.rays_per_cluster = 20
        self.n_samples = n_time_samples
        self.enable_sf = enable_sf
        self.enable_fading = enable_fading

        # data type
        assert dtype.is_complex, "'dtype' must be complex type"
        self._dtype = dtype
        self._dtype_real = dtype.to_real()
        self.device = device

        self.l_min, self.l_max = -6, int(np.ceil(3e-6*self.bw)) + 6
        l_tot = self.l_max-self.l_min+1

        if noise_power_dB is not None:
            self.noise_power_lin = 10**(noise_power_dB/10)
        else:
            self.noise_power_lin = 10**((-173.8 + 10 * np.log10(bw))/10)

        self._cir_sampler = ChannelCoefficientsGenerator(f_c, subclustering=True, rng = rng, dtype=dtype, device=device)
        self._lsp_sampler = LSPGenerator(self, rng)
        self._ray_sampler = RaysGenerator(self, rng)
        self._apply_channel = ApplyTimeChannel(n_time_samples, l_tot=l_tot, rng=rng, add_awgn=True, device=device)
        
        self.load_params() # Load all parameters in to device tensors
        
    def update_topology(self,
                        ut_xy: torch.Tensor, #[batch size,num_ut, 3] in map pixels
                        bs_xy: torch.Tensor, #[batch size,num_bs, 3] in map pixels
                        map: torch.Tensor, #[batch size,num_bs] terrain class at each pixel
                        map_resolution: float = 1.0, # map pixels to meters conversion factor
                        ut_velocities: torch.Tensor = None, #[batch size,num_ut, 3]
                        los_requested: torch.Tensor = None,
                        direction: str = "uplink" #uplink/downlink
    ) -> None:
        # set_topology
        self.ut_xy = ut_xy.clone().to(self.device)
        self.h_ut = self.ut_xy[:,:,2]
        self.ut_xy[:,:,:2] = self.ut_xy[:,:,:2] * map_resolution
        self.bs_xy = bs_xy.clone().to(self.device)
        self.h_bs = self.bs_xy[:,:,2]
        self.bs_xy[:,:,:2] = self.bs_xy[:,:,:2] * map_resolution
        assert self.num_ut == self.ut_xy.shape[1]
        assert self.num_bs == self.bs_xy.shape[1]
        assert self.batch_size == self.bs_xy.shape[0]
        self.map = map.to(self.device)
        self.map_resolution = map_resolution
        self.los_requested = los_requested
        self.direction = direction #uplink/downlink
        self.batch_size = self.ut_xy.shape[0]
        if ut_velocities is None:
            self.ut_velocities = torch.zeros_like(self.ut_xy, device=self.device)
        else:
            self.ut_velocities = ut_velocities.to(self.device) * map_resolution

        # Update topology-related quantities
        self._compute_distance_2d_3d_and_angles()
        self.is_los = self._compute_los()
        self.param_list = self.init_param_list()

        # Compute the LSPs means and stds
        self.lsp_log_mean, self.lsp_log_std, self.zod_offset = self._compute_lsp_log_mean_std()

        # Compute the basic path-loss
        self.basic_pathloss = self._compute_pathloss_basic()

        # Update the LSP sampler
        self._lsp_sampler.topology_updated_callback()

        # Update the ray sampler
        self.num_clusters_max = int(self.get_param("numClusters").max())
        self._ray_sampler.topology_updated_callback()
        
        
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[-1] == self.n_samples, "Input frame size mismatch"
        assert self.ut_xy is not None, "Call update_topology before applying channel"

        if self.enable_fading:
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
        else:
            h = torch.ones((1,1,1,1,1,1,num_time_samples_lag), dtype=self._dtype, device=self.device)
            tau = torch.zeros((1,1,1,1), dtype=self._dtype_real, device=self.device)

        # Step 12 (path loss and shadow fading)
        sf = lsp.sf if self.enable_sf else torch.ones_like(lsp.sf)
        gain = torch.pow(10.0, -(self.basic_pathloss)/20.)*torch.sqrt(sf)
        gain = gain[(...,)+(None,)*(len(h.shape)-len(gain.shape))]
        h *= gain + 0.0j

        ## cir_to_time_channel 
        # Reshaping to match the expected output
        h = torch.permute(h, [0, 2, 4, 1, 5, 3, 6])[...,None]
        tau = torch.permute(tau, [0, 2, 1, 3])[:,:,None,:,None,:,None,None]
        # Expand dims to broadcast with h.
        tau = torch.tile(tau, [1, 1, 1, 1, h.shape[4], 1,1])

        # Time lags for which to compute the channel taps
        l = torch.arange(self.l_min, self.l_max+1, dtype=torch.float32, device=self.device)

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
        y_torch, snr = self._apply_channel(x, h_T, self.noise_power_lin)

        return y_torch, snr
    
    def load_params(self):
        fc = self.f_c/1e9
        fc_urban = 6.0 if fc/1e9 < 6 else fc

        self.param_list_los_urban = torch.empty((len(PARAMS_IDX)), dtype=self._dtype_real, device=self.device)
        self.param_list_nlos_urban = torch.empty((len(PARAMS_IDX)), dtype=self._dtype_real, device=self.device)
        self.param_list_los_rural = torch.empty((len(PARAMS_IDX)), dtype=self._dtype_real, device=self.device)
        self.param_list_nlos_rural = torch.empty((len(PARAMS_IDX)), dtype=self._dtype_real, device=self.device)

        for k, list_idx in PARAMS_IDX.items():
            if k in ('muDS', 'sigmaDS', 'muASD', 'sigmaASD', 'muASA',
                                'sigmaASA', 'muZSA', 'sigmaZSA'):
                pa_los = PARAMS_LOS_URBAN[k + 'a']
                pb_los = PARAMS_LOS_URBAN[k + 'b']
                pc_los = PARAMS_LOS_URBAN[k + 'c']
                self.param_list_los_urban[list_idx] = pa_los*np.log10(pb_los+fc_urban) + pc_los

                pa_nlos = PARAMS_NLOS_URBAN[k + 'a']
                pb_nlos = PARAMS_NLOS_URBAN[k + 'b']
                pc_nlos = PARAMS_NLOS_URBAN[k + 'c']
                self.param_list_nlos_urban[list_idx] = pa_nlos*np.log10(pb_nlos+fc_urban) + pc_nlos

                pa_los = PARAMS_LOS_RURAL[k + 'a']
                pb_los = PARAMS_LOS_RURAL[k + 'b']
                pc_los = PARAMS_LOS_RURAL[k + 'c']
                self.param_list_los_rural[list_idx] = pa_los*np.log10(pb_los+fc) + pc_los

                pa_nlos = PARAMS_NLOS_RURAL[k + 'a']
                pb_nlos = PARAMS_NLOS_RURAL[k + 'b']
                pc_nlos = PARAMS_NLOS_RURAL[k + 'c']
                self.param_list_nlos_rural[list_idx] = pa_nlos*np.log10(pb_nlos+fc) + pc_nlos
            elif k == "cDS":
                pa_los = PARAMS_LOS_URBAN['cDSa']
                pb_los = PARAMS_LOS_URBAN['cDSb']
                pc_los = PARAMS_LOS_URBAN['cDSc']
                self.param_list_los_urban[list_idx] = np.maximum(pa_los, pb_los - pc_los*np.log10(fc_urban))

                pa_nlos = PARAMS_NLOS_URBAN['cDSa']
                pb_nlos = PARAMS_NLOS_URBAN['cDSb']
                pc_nlos = PARAMS_NLOS_URBAN['cDSc']
                self.param_list_nlos_urban[list_idx] = np.maximum(pa_nlos, pb_nlos - pc_nlos*np.log10(fc_urban))

                pa_los = PARAMS_LOS_RURAL['cDSa']
                pb_los = PARAMS_LOS_RURAL['cDSb']
                pc_los = PARAMS_LOS_RURAL['cDSc']
                self.param_list_los_rural[list_idx] = np.maximum(pa_los, pb_los - pc_los*np.log10(self.f_c/1e9))

                pa_nlos = PARAMS_NLOS_RURAL['cDSa']
                pb_nlos = PARAMS_NLOS_RURAL['cDSb']
                pc_nlos = PARAMS_NLOS_RURAL['cDSc']
                self.param_list_nlos_rural[list_idx] = np.maximum(pa_nlos, pb_nlos - pc_nlos*np.log10(self.f_c/1e9))
            else:
                self.param_list_los_urban[list_idx] = PARAMS_LOS_URBAN[k]
                self.param_list_nlos_urban[list_idx] = PARAMS_NLOS_URBAN[k]
                self.param_list_los_rural[list_idx] = PARAMS_LOS_RURAL[k]
                self.param_list_nlos_rural[list_idx] = PARAMS_NLOS_RURAL[k]
    
    def init_param_list(self):
        parameter_value_los = torch.where(self.is_urban[...,None], self.param_list_los_urban, self.param_list_los_rural)
        parameter_value_nlos = torch.where(self.is_urban[...,None], self.param_list_nlos_urban, self.param_list_nlos_rural)
        return torch.where(self.is_los[...,None], parameter_value_los, parameter_value_nlos)

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
        return self.param_list[...,PARAMS_IDX[parameter_name]]

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

        ## Terrain distance calculations
        max_dist = max(self.map.shape) * self.map_resolution
        steps = torch.arange(max_dist, dtype=torch.float32, device=self.device) / (max_dist - 1)

        steps = steps[None,None,None,:,None]

        delta_loc_xy = self.ut_xy[:,None,:,None,:2] - self.bs_xy[:,:,None,None,:2]
        dist_vector = steps*delta_loc_xy
        xy_vectors = dist_vector + self.bs_xy[:,:,None,None,:2]
        xy_vectors = (xy_vectors/self.map_resolution).int()
        dist_vector = torch.sqrt(torch.sum(torch.square(dist_vector), axis=-1))[...,1:] # Convert to 2d distance

        zi = self.map[xy_vectors[...,0], xy_vectors[...,1]]

        # Filter out small changes in terrain
        n = 7 # filter size
        trans_filt = zi.flatten(0,-2)[:,None].float()
        trans_filt = torch.nn.functional.pad(trans_filt, (0,n-1), 'replicate')
        kernel = torch.ones((1,1,n), dtype=torch.float, device=self.device)/n
        trans_filt = torch.nn.functional.conv1d(trans_filt, kernel)
        trans_filt = trans_filt.reshape_as(zi).round().int()
        terrain_transitions = (zi[...,1:] - zi[...,:-1]) != 0
        # terrain_transitions[...,0] = True
        terrain_transitions[...,-1] = True

        terrain_transitions_mask = torch.any(terrain_transitions.flatten(0,-2), 0)

        self.terrain_is_urban = zi[...,1:][...,terrain_transitions_mask].bool()
        self.is_urban = self.terrain_is_urban[...,-1]
        self.distances = dist_vector[...,terrain_transitions_mask]


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
        los = torch.rand([self.batch_size, self.num_bs, self.num_ut], generator=self.rng, dtype=self._dtype_real, device=self.device)

        if self.los_requested is not None:
            return torch.ones_like(self.distance_2d, dtype=bool, device=self.device) * self.los_requested
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
        log_mean_sf = torch.zeros([self.batch_size, self.num_bs, self.num_ut], dtype=self._dtype_real, device=self.device)
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
        h_ut = self.h_ut[:,None,:,None] # For broadcasting
        h_bs = self.h_bs[:,:,None,None] # For broadcasting

        # Beak point distance
        g = ((5./4.)*torch.pow(self.distances/100., 3.)
            *torch.exp(-self.distances/150.0))
        g = torch.where(torch.less_equal(self.distances, 18.0), 0.0, g)
        c = g*torch.pow((h_ut-13.)/10., 1.5)
        c = torch.where(torch.less(h_ut, 13.), 0.0, c)
        p = 1./(1.+c)
        r = torch.rand([self.batch_size, self.num_bs, self.num_ut, 1], generator=self.rng, dtype=self._dtype_real, device=self.device)
        r = torch.where(torch.less(r, p), 1.0, 0.0)

        max_value = h_ut- 1.5
        # Random uniform integer generation is not supported when maxval and
        # are not scalar. The following commented would therefore not work.
        # Therefore, for now, we just sample from a continuous
        # distribution.
        s = torch.rand([self.batch_size, self.num_bs, self.num_ut, 1], generator=self.rng, dtype=self._dtype_real, device=self.device)
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
        pl_1 = 28.0 + 22.0*torch.log10(self.distances) + 20.0*np.log10(self.f_c/1e9)
        pl_2 = (28.0 + 40.0*torch.log10(self.distances) + 20.0*np.log10(self.f_c/1e9)
         - 9.0*torch.log10(torch.square(urban_distance_breakpoint)+torch.square(h_bs-h_ut)))
        urban_pl_los = torch.where(torch.less(self.distances, urban_distance_breakpoint),
            pl_1, pl_2)
        # Rural
        pl_1 = (20.0*torch.log10(40.0*torch.pi*self.distances*self.f_c/3e9)
            + np.minimum(0.03*np.power(self.average_building_height,1.72),
                10.0)*torch.log10(self.distances)
            - np.minimum(0.044*np.power(self.average_building_height,1.72),
                14.77)
            + 0.002*np.log10(self.average_building_height)*self.distances)
        pl_2 = (20.0*torch.log10(40.0*torch.pi*rural_distance_breakpoint*self.f_c/3e9)
            + np.minimum(0.03*np.power(self.average_building_height,1.72),
                10.0)*torch.log10(rural_distance_breakpoint)
            - np.minimum(0.044*np.power(self.average_building_height,1.72),
                14.77)
            + 0.002*np.log10(self.average_building_height)*rural_distance_breakpoint
            + 40.0*torch.log10(self.distances/rural_distance_breakpoint))
        rural_pl_los = torch.where(torch.less(self.distances, rural_distance_breakpoint),
            pl_1, pl_2)

        ## Basic pathloss for NLoS and O2I
        # Urban
        pl_3 = (13.54 + 39.08*torch.log10(self.distances) + 20.0*np.log10(self.f_c/1e9)
            - 0.6*(h_ut-1.5))
        urban_pl_nlos = torch.maximum(urban_pl_los, pl_3)
        # Rural
        pl_3 = (161.04 - 7.1*np.log10(self.average_street_width)
                + 7.5*np.log10(self.average_building_height)
                - (24.37 - 3.7*torch.square(self.average_building_height/h_bs))
                *torch.log10(h_bs)
                + (43.42 - 3.1*torch.log10(h_bs))*(torch.log10(self.distances)-3.0)
                + 20.0*np.log10(self.f_c/1e9) - (3.2*torch.square(torch.log10(11.75*h_ut))
                - 4.97))
        rural_pl_nlos = torch.maximum(rural_pl_los, pl_3)

        ## Set the basic pathloss according to UT state
        # LoS
        urban_pl_b = torch.where(self.is_los[...,None], urban_pl_los, urban_pl_nlos)
        rural_pl_b = torch.where(self.is_los[...,None], rural_pl_los, rural_pl_nlos)

        # Terrain change contributions
        urban_pl_lin = 10**(urban_pl_b/10)
        rural_pl_lin = 10**(rural_pl_b/10)
        rural_pl_lin = torch.nn.functional.pad(rural_pl_lin, (1,0))
        rural_pl_diff = torch.diff(rural_pl_lin)
        # rural_pl_diff[...,0] += rural_pl_lin[...,0]
        urban_pl_lin = torch.nn.functional.pad(urban_pl_lin, (1,0))
        urban_pl_diff = torch.diff(urban_pl_lin)
        # urban_pl_diff[...,0] += urban_pl_lin[...,0]
        pl_b = torch.where(self.terrain_is_urban, urban_pl_diff, rural_pl_diff)
        pl_b = torch.sum(pl_b, -1)
        pl_b = 10*torch.log10(pl_b)

        if self.direction == 'uplink':
            pl_b = torch.permute(pl_b, [0,2,1])

        return pl_b
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class used to define a system level 3GPP channel simulation scenario"""

import json
import os
import numpy as np
import torch
import scipy.constants


class SystemLevelScenario():
    r"""
    This class is used to set up the scenario for system level 3GPP channel
    simulation.

    Scenarios for system level channel simulation, such as UMi, UMa, or RMa,
    are defined by implementing this base class.

    Input
    ------
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

    def __init__(self, carrier_frequency, urban,
        direction, rng: torch.Generator, enable_pathloss=True, enable_shadow_fading=True,
        dtype=torch.complex64):

        self.rng = rng

        # Carrier frequency (Hz)
        self.carrier_frequency = carrier_frequency

        # Wavelength (m)
        self.lambda_0 = scipy.constants.c/carrier_frequency

        # data type
        assert dtype.is_complex, "'dtype' must be complex type"
        self._dtype = dtype
        self._dtype_real = dtype.to_real()

        # Direction
        assert direction in ("uplink", "downlink"), \
            "'direction' must be 'uplink' or 'downlink'"
        self.direction = direction

        # Pathloss and shadow fading
        self.pathloss_enabled = enable_pathloss
        self.shadow_fading_enabled = enable_shadow_fading

        # Scenario
        self.ut_loc = None
        self.bs_loc = None
        self.ut_orientations = None
        self.bs_orientations = None
        self.ut_velocities = None
        self.requested_los = None

        # Load parameters for this scenario
        self._load_params()
        if urban:
            self._params_nlos = self._params_nlos_urban
            self._params_los = self._params_los_urban
        else:
            self._params_nlos = self._params_nlos_rural
            self._params_los = self._params_los_rural

    @property
    def batch_size(self):
        """Batch size"""
        return self.ut_loc.shape[0]

    @property
    def num_ut(self):
        """Number of UTs."""
        return self.ut_loc.shape[1]

    @property
    def num_bs(self):
        """Number of BSs."""
        return self.bs_loc.shape[1]

    @property
    def matrix_ut_distance_2d(self):
        r"""Distance between all pairs for UTs in the X-Y plan [m].
        [batch size, number of UTs, number of UTs]"""
        return self._matrix_ut_distance_2d

    @property
    def num_clusters_los(self):
        r"""Number of clusters for LoS scenario"""
        return self._params_los["numClusters"]

    @property
    def num_clusters_nlos(self):
        r"""Number of clusters for NLoS scenario"""
        return self._params_nlos["numClusters"]

    @property
    def num_clusters_max(self):
        r"""Maximum number of clusters over LoS and NLoS scenarios"""
        # Different models have different number of clusters
        num_clusters_los = self._params_los["numClusters"]
        num_clusters_nlos = self._params_nlos["numClusters"]
        num_clusters_max = np.max([num_clusters_los, num_clusters_nlos])
        return num_clusters_max

    def set_topology(self, ut_loc=None, bs_loc=None, ut_orientations=None,
        bs_orientations=None, ut_velocities=None, los=None):
        r"""
        Set the network topology.

        It is possible to set up a different network topology for each batch
        example.

        When calling this function, not specifying a parameter leads to the
        reuse of the previously given value. Not specifying a value that was not
        set at a former call rises an error.

        Input
        ------
            ut_loc : [batch size, number of UTs, 3], float
                Locations of the UTs [m]

            bs_loc : [batch size, number of BSs, 3], float
                Locations of BSs [m]

            ut_orientations : [batch size, number of UTs, 3], float
                Orientations of the UTs arrays [radian]

            bs_orientations : [batch size, number of BSs, 3], float
                Orientations of the BSs arrays [radian]

            ut_velocities : [batch size, number of UTs, 3], float
                Velocity vectors of UTs [m/s]

            los : bool or `None`
                If not `None` (default value), all UTs located outdoor are
                forced to be in LoS if ``los`` is set to `True`, or in NLoS
                if it is set to `False`. If set to `None`, the LoS/NLoS states
                of UTs is set following 3GPP specification
                (Section 7.4.2 of TR 38.901).
        """

        assert (ut_loc is not None) or (self.ut_loc is not None),\
            "`ut_loc` is None and was not previously set"

        assert (bs_loc is not None) or (self.bs_loc is not None),\
            "`bs_loc` is None and was not previously set"

        assert (ut_orientations is not None)\
            or (self.ut_orientations is not None),\
            "`ut_orientations` is None and was not previously set"

        assert (bs_orientations is not None)\
            or (self.bs_orientations is not None),\
            "`bs_orientations` is None and was not previously set"

        assert (ut_velocities is not None)\
            or (self.ut_velocities is not None),\
            "`ut_velocities` is None and was not previously set"

        # Boolean used to keep track of whether or not we need to (re-)compute
        # the distances between users, correlation matrices...
        # This is required if the UT locations, BS locations,
        # state of UTs, or LoS/NLoS states of outdoor UTs are updated.
        need_for_update = False

        if ut_loc is not None:
            self.ut_loc = torch.from_numpy(ut_loc)
            self.h_ut = self.ut_loc[:,:,2]
            need_for_update = True

        if bs_loc is not None:
            self.bs_loc = torch.from_numpy(bs_loc)
            self.h_bs = self.bs_loc[:,:,2]
            need_for_update = True

        if bs_orientations is not None:
            self.bs_orientations = torch.from_numpy(bs_orientations)

        if ut_orientations is not None:
            self.ut_orientations = torch.from_numpy(ut_orientations)

        if ut_velocities is not None:
            self.ut_velocities = torch.from_numpy(ut_velocities)

        if los is not None:
            self.requested_los = los
            need_for_update = True

        if need_for_update:
            # Update topology-related quantities
            self._compute_distance_2d_3d_and_angles()
            self._sample_los()

            # Compute the LSPs means and stds
            self._compute_lsp_log_mean_std()

            # Compute the basic path-loss
            self._compute_pathloss_basic()

        return need_for_update

    def spatial_correlation_matrix(self, correlation_distance):
        r"""Computes and returns a 2D spatial exponential correlation matrix
        :math:`C` over the UTs, such that :math:`C`has shape
        (number of UTs)x(number of UTs), and

        .. math::
            C_{n,m} = \exp{-\frac{d_{n,m}}{D}}

        where :math:`d_{n,m}` is the distance between UT :math:`n` and UT
        :math:`m` in the X-Y plan, and :math:`D` the correlation distance.

        Input
        ------
        correlation_distance : float
            Correlation distance, i.e., distance such that the correlation
            is :math:`e^{-1} \approx 0.37`

        Output
        --------
        : [batch size, number of UTs, number of UTs], float
            Spatial correlation :math:`C`
        """
        spatial_correlation_matrix = np.exp(-self.matrix_ut_distance_2d/
                                                 correlation_distance)
        return spatial_correlation_matrix

    def get_param(self, parameter_name):
        r"""
        Given a ``parameter_name`` used in the configuration file, returns a
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

        fc = self.carrier_frequency/1e9
        fc = self.clip_carrier_frequency_lsp(fc)

        parameter_tensor = np.zeros(shape=[self.batch_size,
                                            self.num_bs,
                                            self.num_ut])

        # Parameter value
        if parameter_name in ('muDS', 'sigmaDS', 'muASD', 'sigmaASD', 'muASA',
                             'sigmaASA', 'muZSA', 'sigmaZSA'):

            pa_los = self._params_los_urban[parameter_name + 'a']
            pb_los = self._params_los_urban[parameter_name + 'b']
            pc_los = self._params_los_urban[parameter_name + 'c']

            pa_nlos = self._params_nlos_urban[parameter_name + 'a']
            pb_nlos = self._params_nlos_urban[parameter_name + 'b']
            pc_nlos = self._params_nlos_urban[parameter_name + 'c']

            parameter_value_los_urban = pa_los*np.log10(pb_los+fc) + pc_los
            parameter_value_nlos_urban = pa_nlos*np.log10(pb_nlos+fc) + pc_nlos

            pa_los = self._params_los_rural[parameter_name + 'a']
            pb_los = self._params_los_rural[parameter_name + 'b']
            pc_los = self._params_los_rural[parameter_name + 'c']

            pa_nlos = self._params_nlos_rural[parameter_name + 'a']
            pb_nlos = self._params_nlos_rural[parameter_name + 'b']
            pc_nlos = self._params_nlos_rural[parameter_name + 'c']

            parameter_value_los_rural = pa_los*np.log10(pb_los+fc) + pc_los
            parameter_value_nlos_rural = pa_nlos*np.log10(pb_nlos+fc) + pc_nlos
        elif parameter_name == "cDS":

            pa_los = self._params_los_urban[parameter_name + 'a']
            pb_los = self._params_los_urban[parameter_name + 'b']
            pc_los = self._params_los_urban[parameter_name + 'c']

            pa_nlos = self._params_nlos_urban[parameter_name + 'a']
            pb_nlos = self._params_nlos_urban[parameter_name + 'b']
            pc_nlos = self._params_nlos_urban[parameter_name + 'c']

            parameter_value_los_urban = np.maximum(pa_los,
                pb_los - pc_los*np.log10(fc))
            parameter_value_nlos_urban = np.maximum(pa_nlos,
                pb_nlos - pc_nlos*np.log10(fc))
            
            pa_los = self._params_los_rural[parameter_name + 'a']
            pb_los = self._params_los_rural[parameter_name + 'b']
            pc_los = self._params_los_rural[parameter_name + 'c']

            pa_nlos = self._params_nlos_rural[parameter_name + 'a']
            pb_nlos = self._params_nlos_rural[parameter_name + 'b']
            pc_nlos = self._params_nlos_rural[parameter_name + 'c']

            parameter_value_los_rural = np.maximum(pa_los,
                pb_los - pc_los*np.log10(fc))
            parameter_value_nlos_rural = np.maximum(pa_nlos,
                pb_nlos - pc_nlos*np.log10(fc))
        else:
            parameter_value_los_urban = self._params_los_urban[parameter_name]
            parameter_value_nlos_urban = self._params_nlos_urban[parameter_name]
            parameter_value_los_rural = self._params_los_rural[parameter_name]
            parameter_value_nlos_rural = self._params_nlos_rural[parameter_name]

        # LoS
        parameter_value_los = np.where(self.is_urban, parameter_value_los_urban, parameter_value_los_rural)
        parameter_tensor = np.where(self.los, parameter_value_los, parameter_tensor)
        # NLoS
        parameter_value_nlos = np.where(self.is_urban, parameter_value_nlos_urban, parameter_value_nlos_rural)
        parameter_tensor = np.where(np.logical_not(self.los), parameter_value_nlos, parameter_tensor)

        return torch.from_numpy(parameter_tensor).type(self._dtype_real)

    #####################################################
    # Internal utility methods
    #####################################################

    def _compute_distance_2d_3d_and_angles(self):
        r"""
        Computes the following internal values:
        * 2D distances for all BS-UT pairs in the X-Y plane
        * 3D distances for all BS-UT pairs
        * 2D distances for all pairs of UTs in the X-Y plane
        * LoS AoA, AoD, ZoA, ZoD for all BS-UT pairs

        This function is called at every update of the topology.
        """

        ut_loc = self.ut_loc
        ut_loc = torch.unsqueeze(ut_loc, axis=1)

        bs_loc = self.bs_loc
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
        # Angles are converted to degrees and wrapped to (0,360)
        self.los_aod = torch.remainder(torch.rad2deg(los_aod), 360.0)
        self.los_aoa = torch.remainder(torch.rad2deg(los_aoa), 360.0)
        self.los_zod = torch.remainder(torch.rad2deg(los_zod), 360.0)
        self.los_zoa = torch.remainder(torch.rad2deg(los_zoa), 360.0)

        # 2D distances for all pairs of UTs in the (x-y) plane
        ut_loc_xy = self.ut_loc[:,:,:2]

        ut_loc_xy_expanded_1 = torch.unsqueeze(ut_loc_xy, axis=1)
        ut_loc_xy_expanded_2 = torch.unsqueeze(ut_loc_xy, axis=2)

        delta_loc_xy = ut_loc_xy_expanded_1 - ut_loc_xy_expanded_2

        matrix_ut_distance_2d = torch.sqrt(torch.sum(torch.square(delta_loc_xy),
                                                       axis=3))
        self._matrix_ut_distance_2d = matrix_ut_distance_2d

    def _sample_los(self):
        r"""Set the LoS state of each UT randomly, following the procedure
        described in section 7.4.2 of TR 38.901.
        LoS state of each UT is randomly assigned according to a Bernoulli
        distribution, which probability depends on the channel model.
        """
        if self.requested_los is None:
            los_probability = self.los_probability
            los = torch.rand([self.batch_size, self.num_bs, self.num_ut], generator=self.rng, dtype=self._dtype_real)
            los = torch.less(los, los_probability)
        else:
            los = torch.full([self.batch_size, self.num_bs, self.num_ut],
                            self.requested_los)            

        self.los = los

    def _load_params(self):
        r"""Load the configuration files corresponding to the 3 possible states
        of UTs: LoS, NLoS, and O2I"""

        self._params_los_rural = {
            "muDSa" : 0.0,
            "muDSb" : 0.0,
            "muDSc" : -7.49,

            "sigmaDSa" : 0.0,
            "sigmaDSb" : 0.0,
            "sigmaDSc" : 0.55,

            "muASDa" : 0.0,
            "muASDb" : 0.0,
            "muASDc" : 0.90,

            "sigmaASDa" : 0.0,
            "sigmaASDb" : 0.0,
            "sigmaASDc" : 0.38,

            "muASAa" : 0.0,
            "muASAb" : 0.0,
            "muASAc" : 1.52,

            "sigmaASAa" : 0.0,
            "sigmaASAb" : 0.0,
            "sigmaASAc" : 0.24,

            "muZSAa" : 0.0,
            "muZSAb" : 0.0,
            "muZSAc" : 0.47,

            "sigmaZSAa" : 0.0,
            "sigmaZSAb" : 0.0,
            "sigmaZSAc" : 0.40,

            "muK" : 7.0,
            "sigmaK" : 4.0,

            "muXPR" : 12.0,
            "sigmaXPR" : 4.0,

            "sigmaSF" : 8.0,
            "sigmaSF1" : 4.0,
            "sigmaSF2" : 6.0,

            "muZSDa" : -0.17,
            "muZSDb" : 0.22,
            "sigmaZSD" : 0.34,

            "numClusters" : 11,

            "rTau" : 3.8,

            "zeta" : 3.0,

            "cDSa" : 3.91,
            "cDSb" : 0.0,
            "cDSc" : 0.0,
            "cASD" : 2.0,
            "cASA" : 3.0,
            "cZSA" : 3.0,

            "CPhiNLoS" : 1.123,
            "CThetaNLoS" : 1.031,

            "corrDistDS" : 50,
            "corrDistASD" : 25,
            "corrDistASA" : 35,
            "corrDistSF" : 37,
            "corrDistK" : 40,
            "corrDistZSA" : 15,
            "corrDistZSD" : 15,

            "corrASDvsDS" : 0,
            "corrASAvsDS" : 0,
            "corrASAvsSF" : 0,
            "corrASDvsSF" : 0,
            "corrDSvsSF" : -0.5,
            "corrASDvsASA" : 0,
            "corrASDvsK" : 0,
            "corrASAvsK" : 0,
            "corrDSvsK" : 0,
            "corrSFvsK" : 0,
            "corrZSDvsSF" : 0.01,
            "corrZSAvsSF" : -0.17,
            "corrZSDvsK" : 0,
            "corrZSAvsK" : -0.02,
            "corrZSDvsDS" : -0.05,
            "corrZSAvsDS" : 0.27,
            "corrZSDvsASD" : 0.73,
            "corrZSAvsASD" : -0.14,
            "corrZSDvsASA" : -0.20,
            "corrZSAvsASA" : 0.24,
            "corrZSDvsZSA": -0.07
        }

        self._params_nlos_rural = {
            "muDSa" : 0.0,
            "muDSb" : 0.0,
            "muDSc" : -7.43,

            "sigmaDSa" : 0.0,
            "sigmaDSb" : 0.0,
            "sigmaDSc" : 0.48,

            "muASDa" : 0.0,
            "muASDb" : 0.0,
            "muASDc" : 0.95,

            "sigmaASDa" : 0.0,
            "sigmaASDb" : 0.0,
            "sigmaASDc" : 0.45,

            "muASAa" : 0.0,
            "muASAb" : 0.0,
            "muASAc" : 1.52,

            "sigmaASAa" : 0.0,
            "sigmaASAb" : 0.0,
            "sigmaASAc" : 0.13,

            "muZSAa" : 0.0,
            "muZSAb" : 0.0,
            "muZSAc" : 0.58,

            "sigmaZSAa" : 0.0,
            "sigmaZSAb" : 0.0,
            "sigmaZSAc" : 0.37,

            "muK" : 0.0,
            "sigmaK" : 0.0,

            "muXPR" : 7.0,
            "sigmaXPR" : 3.0,

            "sigmaSF" : 8.0,
            "sigmaSF1" : 4.0,
            "sigmaSF2" : 6.0,

            "muZSDa" : -0.19,
            "muZSDb" : 0.28,
            "sigmaZSD" : 0.30,

            "numClusters" : 10,

            "rTau" : 1.7,

            "zeta" : 3.0,

            "cDSa" : 3.91,
            "cDSb" : 0.0,
            "cDSc" : 0.0,
            "cASD" : 2.0,
            "cASA" : 3.0,
            "cZSA" : 3.0,

            "CPhiNLoS" : 1.090,
            "CThetaNLoS" : 0.957,

            "corrDistDS" : 36,
            "corrDistASD" : 30,
            "corrDistASA" : 40,
            "corrDistSF" : 120,
            "corrDistK" : 1,
            "corrDistZSA" : 50,
            "corrDistZSD" : 50,

            "corrASDvsDS" : -0.4,
            "corrASAvsDS" : 0,
            "corrASAvsSF" : 0,
            "corrASDvsSF" : 0.6,
            "corrDSvsSF" : -0.5,
            "corrASDvsASA" : 0,
            "corrASDvsK" : 0,
            "corrASAvsK" : 0,
            "corrDSvsK" : 0,
            "corrSFvsK" : 0,
            "corrZSDvsSF" : -0.04,
            "corrZSAvsSF" : -0.25,
            "corrZSDvsK" : 0,
            "corrZSAvsK" : 0,
            "corrZSDvsDS" : -0.10,
            "corrZSAvsDS" : -0.40,
            "corrZSDvsASD" : 0.42,
            "corrZSAvsASD" : -0.27,
            "corrZSDvsASA" : -0.18,
            "corrZSAvsASA" : 0.26,
            "corrZSDvsZSA": -0.27
        }

        self._params_los_urban = {
            "muDSa" : -0.0963,
            "muDSb" : 0.0,
            "muDSc" : -6.955,

            "sigmaDSa" : 0.0,
            "sigmaDSb" : 0.0,
            "sigmaDSc" : 0.66,

            "muASDa" : 0.1114,
            "muASDb" : 0.0,
            "muASDc" : 1.06,

            "sigmaASDa" : 0.0,
            "sigmaASDb" : 0.0,
            "sigmaASDc" : 0.28,

            "muASAa" : 0.0,
            "muASAb" : 0.0,
            "muASAc" : 1.81,

            "sigmaASAa" : 0.0,
            "sigmaASAb" : 0.0,
            "sigmaASAc" : 0.20,

            "muZSAa" : 0.0,
            "muZSAb" : 0.0,
            "muZSAc" : 0.95,

            "sigmaZSAa" : 0.0,
            "sigmaZSAb" : 0.0,
            "sigmaZSAc" : 0.16,

            "muK" : 9.0,
            "sigmaK" : 3.5,

            "muXPR" : 8.0,
            "sigmaXPR" : 4.0,

            "sigmaSF" : 4.0,
            "sigmaSF1" : 0.0,
            "sigmaSF2" : 0.0,

            "muZSDa" : 0.0,
            "muZSDb" : 0.0,
            "sigmaZSD" : 0.0,

            "sigmaZSD" : 0.40,

            "numClusters" : 12,

            "rTau" : 2.5,

            "zeta" : 3.0,

            "cDSa" : 0.25,
            "cDSb" : 6.5622,
            "cDSc" : 3.4084,
            "cASD" : 5.0,
            "cASA" : 11.0,
            "cZSA" : 7.0,

            "CPhiNLoS" : 1.146,
            "CThetaNLoS" : 1.104,

            "corrDistDS" : 30,
            "corrDistASD" : 18,
            "corrDistASA" : 15,
            "corrDistSF" : 37,
            "corrDistK" : 12,
            "corrDistZSA" : 15,
            "corrDistZSD" : 15,

            "corrASDvsDS" : 0.4,
            "corrASAvsDS" : 0.8,
            "corrASAvsSF" : -0.5,
            "corrASDvsSF" : -0.5,
            "corrDSvsSF" : -0.4,
            "corrASDvsASA" : 0.0,
            "corrASDvsK" : 0.0,
            "corrASAvsK" : -0.2,
            "corrDSvsK" : -0.4,
            "corrSFvsK" : 0.0,
            "corrZSDvsSF" : 0.0,
            "corrZSAvsSF" : -0.8,
            "corrZSDvsK" : 0.0,
            "corrZSAvsK" : 0.0,
            "corrZSDvsDS" : -0.2,
            "corrZSAvsDS" : 0.0,
            "corrZSDvsASD" : 0.5,
            "corrZSAvsASD" : 0.0,
            "corrZSDvsASA" : -0.3,
            "corrZSAvsASA" : 0.4,
            "corrZSDvsZSA": 0.0
        }

        self._params_nlos_urban = {
            "muDSa" : -0.204,
            "muDSb" : 0.0,
            "muDSc" : -6.28,

            "sigmaDSa" : 0.0,
            "sigmaDSb" : 0.0,
            "sigmaDSc" : 0.39,

            "muASDa" : -0.1144,
            "muASDb" : 0.0,
            "muASDc" : 1.5,

            "sigmaASDa" : 0.0,
            "sigmaASDb" : 0.0,
            "sigmaASDc" : 0.28,

            "muASAa" : -0.27,
            "muASAb" : 0.0,
            "muASAc" : 2.08,

            "sigmaASAa" : 0.0,
            "sigmaASAb" : 0.0,
            "sigmaASAc" : 0.11,

            "muZSAa" : -0.3236,
            "muZSAb" : 0.0,
            "muZSAc" : 1.512,

            "sigmaZSAa" : 0.0,
            "sigmaZSAb" : 0.0,
            "sigmaZSAc" : 0.16,

            "muK" : 0.0,
            "sigmaK" : 0.0,

            "muXPR" : 7.0,
            "sigmaXPR" : 3.0,

            "sigmaSF" : 6.0,
            "sigmaSF1" : 0.0,
            "sigmaSF2" : 0.0,

            "muZSDa" : 0.0,
            "muZSDb" : 0.0,
            "sigmaZSD" : 0.0,

            "sigmaZSD" : 0.49,

            "numClusters" : 20,

            "rTau" : 2.3,

            "zeta" : 3.0,

            "cDSa" : 0.25,
            "cDSb" : 6.5622,
            "cDSc" : 3.4084,
            "cASD" : 2.0,
            "cASA" : 15.0,
            "cZSA" : 7.0,

            "CPhiNLoS" : 1.289,
            "CThetaNLoS" : 1.178,

            "corrDistDS" : 40,
            "corrDistASD" : 50,
            "corrDistASA" : 50,
            "corrDistSF" : 50,
            "corrDistK" : 1,
            "corrDistZSA" : 50,
            "corrDistZSD" : 50,

            "corrASDvsDS" : 0.4,
            "corrASAvsDS" : 0.6,
            "corrASAvsSF" : 0.0,
            "corrASDvsSF" : -0.6,
            "corrDSvsSF" : -0.4,
            "corrASDvsASA" : 0.4,
            "corrASDvsK" : 0.0,
            "corrASAvsK" : 0.0,
            "corrDSvsK" : 0.0,
            "corrSFvsK" : 0.0,
            "corrZSDvsSF" : 0.0,
            "corrZSAvsSF" : -0.4,
            "corrZSDvsK" : 0.0,
            "corrZSAvsK" : 0.0,
            "corrZSDvsDS" : -0.5,
            "corrZSAvsDS" : 0.0,
            "corrZSDvsASD" : 0.5,
            "corrZSAvsASD" : -0.1,
            "corrZSDvsASA" : 0.0,
            "corrZSAvsASA" : 0.0,
            "corrZSDvsZSA": 0.0
        }

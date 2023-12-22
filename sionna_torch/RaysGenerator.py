#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Class for sampling rays following 3GPP TR38.901 specifications and giving a
channel simulation scenario and LSPs.
"""
import torch

class Rays:
    # pylint: disable=line-too-long
    r"""
    Class for conveniently storing rays

    Parameters
    -----------

    delays : [batch size, number of BSs, number of UTs, number of clusters], 
        Paths delays [s]

    powers : [batch size, number of BSs, number of UTs, number of clusters], 
        Normalized path powers

    aoa : (batch size, number of BSs, number of UTs, number of clusters, number of rays], 
        Azimuth angles of arrival [radian]

    aod : [batch size, number of BSs, number of UTs, number of clusters, number of rays], 
        Azimuth angles of departure [radian]

    zoa : [batch size, number of BSs, number of UTs, number of clusters, number of rays], 
        Zenith angles of arrival [radian]

    zod : [batch size, number of BSs, number of UTs, number of clusters, number of rays], 
        Zenith angles of departure [radian]

    xpr [batch size, number of BSs, number of UTs, number of clusters, number of rays], 
        Coss-polarization power ratios.
    """

    def __init__(self, delays, powers, aoa, aod, zoa, zod, xpr):
        self.delays = delays
        self.powers = powers
        self.aoa = aoa
        self.aod = aod
        self.zoa = zoa
        self.zod = zod
        self.xpr = xpr


class RaysGenerator:
    """
    Sample rays according to a given channel scenario and large scale
    parameters (LSP).

    This class implements steps 6 to 9 from the TR 38.901 specifications,
    (section 7.5).

    Note that a global scenario is set for the entire batches when instantiating
    this class (UMa, UMi, or RMa). However, each UT-BS link can have its
    specific state (LoS or NLoS).

    The batch size is set by the ``scenario`` given as argument when
    constructing the class.

    Parameters
    ----------
    scenario : :class:`~sionna.channel.tr38901.SystemLevelScenario``
        Scenario used to generate LSPs

    Input
    -----
    lsp : :class:`~sionna.channel.tr38901.LSP`
        LSPs samples

    Output
    ------
    rays : :class:`~sionna.channel.tr38901.Rays`
        Rays samples
    """

    def __init__(self, scenario, rng: torch.Generator):
        # Scenario
        self._scenario = scenario
        self.rng = rng

        # For AoA, AoD, ZoA, and ZoD, offset to add to cluster angles to get ray
        # angles. This is hardcoded from table 7.5-3 for 3GPP 38.901
        # specification.
        self._ray_offsets = torch.tensor([0.0447, -0.0447,
                                         0.1413, -0.1413,
                                         0.2492, -0.2492,
                                         0.3715, -0.3715,
                                         0.5129, -0.5129,
                                         0.6797, -0.6797,
                                         0.8844, -0.8844,
                                         1.1481, -0.1481,
                                         1.5195, -1.5195,
                                         2.1551, -2.1551], dtype=self._scenario._dtype_real)

    #########################################
    # Public methods and properties
    #########################################

    def __call__(self, lsp):
        # Sample cluster delays
        delays, delays_unscaled = self._cluster_delays(lsp.ds, lsp.k_factor)

        # Sample cluster powers
        powers, powers_for_angles_gen = self._cluster_powers(lsp.ds,
                                            lsp.k_factor, delays_unscaled)

        # Sample AoA
        aoa = self._azimuth_angles(lsp.asa, lsp.k_factor, powers_for_angles_gen, 'aoa')

        # Sample AoD
        aod = self._azimuth_angles(lsp.asd, lsp.k_factor, powers_for_angles_gen, 'aod')

        # Sample ZoA
        zoa = self._zenith_angles(lsp.zsa, lsp.k_factor, powers_for_angles_gen, 'zoa')

        # Sample ZoD
        zod = self._zenith_angles(lsp.zsd, lsp.k_factor, powers_for_angles_gen, 'zod')

        # XPRs
        xpr = self._cross_polarization_power_ratios()

        # Random coupling
        aoa, aod, zoa, zod = self._random_coupling(aoa, aod, zoa, zod)

        # Convert angles of arrival and departure from degree to radian
        aoa = torch.deg2rad(aoa)
        aod = torch.deg2rad(aod)
        zoa = torch.deg2rad(zoa)
        zod = torch.deg2rad(zod)

        # Storing and returning rays
        rays = Rays(delays = delays,
                    powers = powers,
                    aoa    = aoa,
                    aod    = aod,
                    zoa    = zoa,
                    zod    = zod,
                    xpr    = xpr)

        return rays

    def topology_updated_callback(self):
        """
        Updates internal quantities. Must be called at every update of the
        scenario that changes the state of UTs or their locations.

        Input
        ------
        None

        Output
        ------
        None
        """
        self._compute_clusters_mask()

    ########################################
    # Internal utility methods
    ########################################

    def _compute_clusters_mask(self):
        """
        Given a scenario (UMi, UMa, RMa), the number of clusters is different
        for different state of UT-BS links (LoS or NLoS).

        Because we use tensors with predefined dimension size (not ragged), the
        cluster dimension is always set to the maximum number of clusters the
        scenario requires. A mask is then used to discard not required tensors,
        depending on the state of each UT-BS link.

        This function computes and stores this mask of size
        [batch size, number of BSs, number of UTs, maximum number of cluster]
        where an element equals 0 if the cluster is used, 1 otherwise.
        """

        scenario = self._scenario
        num_clusters = scenario.get_param("numClusters")
        num_clusters_max = scenario.num_clusters_max

        # Initialize an empty mask
        mask = torch.arange(0, num_clusters_max, dtype=self._scenario._dtype_real)
        mask = torch.where(torch.lt(mask,num_clusters[...,None]), 0.0, 1.0)

        # Save the mask
        self._cluster_mask = mask

    def _cluster_delays(self, delay_spread, rician_k_factor):
        # pylint: disable=line-too-long
        """
        Generate cluster delays.
        See step 5 of section 7.5 from TR 38.901 specification.

        Input
        ------
        delay_spread : [batch size, num of BSs, num of UTs], 
            RMS delay spread of each BS-UT link.

        rician_k_factor : [batch size, num of BSs, num of UTs], 
            Rician K-factor of each BS-UT link. Used only for LoS links.

        Output
        -------
        delays : [batch size, num of BSs, num of UTs, maximum number of clusters], 
            Path delays [s]

        unscaled_delays [batch size, num of BSs, num of UTs, maximum number of clusters], 
            Unscaled path delays [s]
        """

        scenario = self._scenario

        batch_size = scenario.batch_size
        num_bs = scenario.num_bs
        num_ut = scenario.num_ut

        num_clusters_max = scenario.num_clusters_max

        # Getting scaling parameter according to each BS-UT link scenario
        delay_scaling_parameter = scenario.get_param("rTau")
        delay_scaling_parameter = torch.unsqueeze(delay_scaling_parameter, 3)

        # Generating random cluster delays
        # We don't start at 0 to avoid numerical errors
        delay_spread = torch.unsqueeze(delay_spread, 3)
        x = torch.rand([batch_size, num_bs, num_ut, num_clusters_max], generator=self.rng, dtype=self._scenario._dtype_real)
        x = (1e-6 - 1.0) * x + 1.0 # 1e-6:1.0

        # Moving to linear domain
        unscaled_delays = -delay_scaling_parameter*delay_spread*torch.log(x)
        # Forcing the cluster that should not exist to huge delays (1s)
        unscaled_delays = (unscaled_delays*(1.-self._cluster_mask)
            + self._cluster_mask)

        # Normalizing and sorting the delays
        unscaled_delays = unscaled_delays - torch.min(unscaled_delays, 3, keepdim=True)[0]
        unscaled_delays = torch.sort(unscaled_delays, 3)[0]

        # Additional scaling applied to LoS links
        rician_k_factor_db = 10.0*torch.log10(rician_k_factor) # to dB
        scaling_factor = (0.7705 - 0.0433*rician_k_factor_db
            + 0.0002*torch.square(rician_k_factor_db)
            + 0.000017*torch.pow(rician_k_factor_db, 3.0))
        scaling_factor = torch.unsqueeze(scaling_factor, 3)
        delays = torch.where(torch.unsqueeze(scenario.is_los, 3),
            unscaled_delays / scaling_factor, unscaled_delays)

        return delays.type(self._scenario._dtype_real), unscaled_delays.type(self._scenario._dtype_real)

    def _cluster_powers(self, delay_spread, rician_k_factor, unscaled_delays):
        # pylint: disable=line-too-long
        """
        Generate cluster powers.
        See step 6 of section 7.5 from TR 38.901 specification.

        Input
        ------
        delays : [batch size, num of BSs, num of UTs, maximum number of clusters], 
            Path delays [s]

        rician_k_factor : [batch size, num of BSs, num of UTs], 
            Rician K-factor of each BS-UT link. Used only for LoS links.

        unscaled_delays [batch size, num of BSs, num of UTs, maximum number of clusters], 
            Unscaled path delays [s]. Required to compute the path powers.

        Output
        -------
        powers : [batch size, num of BSs, num of UTs, maximum number of clusters], 
            Normalized path powers
        """

        scenario = self._scenario

        num_clusters_max = scenario.num_clusters_max

        delay_scaling_parameter = scenario.get_param("rTau")
        cluster_shadowing_std_db = scenario.get_param("zeta")
        delay_spread = torch.unsqueeze(delay_spread, 3)
        cluster_shadowing_std_db = torch.unsqueeze(cluster_shadowing_std_db,
            3)
        delay_scaling_parameter = torch.unsqueeze(delay_scaling_parameter,
            3)

        # Generate unnormalized cluster powers
        cluster_shadowing_std_db = cluster_shadowing_std_db.repeat(1,1,1,num_clusters_max)
        z = torch.normal(mean=0.0, std=cluster_shadowing_std_db, generator=self.rng).type(self._scenario._dtype_real)

        # Moving to linear domain
        powers_unnormalized = (torch.exp(-unscaled_delays*
            (delay_scaling_parameter - 1.0)/
            (delay_scaling_parameter*delay_spread))*torch.pow(10.0, -z/10.0))

        # Force the power of unused cluster to zero
        powers_unnormalized = powers_unnormalized*(1.-self._cluster_mask)

        # Normalizing cluster powers
        powers = (powers_unnormalized/
            torch.sum(powers_unnormalized, 3, keepdims=True))

        # Additional specular component for LoS
        rician_k_factor = torch.unsqueeze(rician_k_factor, 3)
        p_nlos_scaling = 1.0/(rician_k_factor + 1.0)
        p_1_los = rician_k_factor*p_nlos_scaling
        powers_1 = p_nlos_scaling*powers[:,:,:,:1] + p_1_los
        powers_n = p_nlos_scaling*powers[:,:,:,1:]
        powers_for_angles_gen = torch.where(torch.unsqueeze(scenario.is_los, 3),
            torch.concatenate([powers_1, powers_n], 3), powers)

        return powers.type(self._scenario._dtype_real), powers_for_angles_gen.type(self._scenario._dtype_real)

    def _azimuth_angles(self, azimuth_spread, rician_k_factor, cluster_powers,
                        angle_type):
        # pylint: disable=line-too-long
        """
        Generate departure or arrival azimuth angles (degrees).
        See step 7 of section 7.5 from TR 38.901 specification.

        Input
        ------
        azimuth_spread : [batch size, num of BSs, num of UTs], 
            Angle spread, (ASD or ASA) depending on ``angle_type`` [deg]

        rician_k_factor : [batch size, num of BSs, num of UTs], 
            Rician K-factor of each BS-UT link. Used only for LoS links.

        cluster_powers : [batch size, num of BSs, num of UTs, maximum number of clusters], 
            Normalized path powers

        angle_type : str
            Type of angle to compute. Must be 'aoa' or 'aod'.

        Output
        -------
        azimuth_angles : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], 
            Paths azimuth angles wrapped within (-180, 180) [degree]. Either the AoA or AoD depending on ``angle_type``.
        """

        scenario = self._scenario

        batch_size = scenario.batch_size
        num_bs = scenario.num_bs
        num_ut = scenario.num_ut

        num_clusters_max = scenario.num_clusters_max

        azimuth_spread = torch.unsqueeze(azimuth_spread, 3)

        # Loading the angle spread
        if angle_type == 'aod':
            azimuth_angles_los = torch.rad2deg(scenario.los_aod_rad)
            cluster_angle_spread = scenario.get_param('cASD')
        else:
            azimuth_angles_los =torch.rad2deg(scenario.los_aoa_rad)
            cluster_angle_spread = scenario.get_param('cASA')
        # Adding cluster dimension for broadcasting
        azimuth_angles_los = torch.unsqueeze(azimuth_angles_los, 3)
        cluster_angle_spread = torch.unsqueeze(torch.unsqueeze(
            cluster_angle_spread, 3), 4)

        # Compute C-phi constant
        rician_k_factor = torch.unsqueeze(rician_k_factor, 3)
        rician_k_factor_db = 10.0*torch.log10(rician_k_factor) # to dB
        c_phi_nlos = torch.unsqueeze(scenario.get_param("CPhiNLoS"), 3)
        c_phi_los = c_phi_nlos*(1.1035- 0.028*rician_k_factor_db
            - 0.002*torch.square(rician_k_factor_db)
            + 0.0001*torch.pow(rician_k_factor_db, 3.))
        c_phi = torch.where(torch.unsqueeze(scenario.is_los, 3),
            c_phi_los, c_phi_nlos)

        # Inverse Gaussian function
        z = cluster_powers/torch.max(cluster_powers, dim=3, keepdims=True)[0]
        z = torch.clip(z, 1e-6, 1.0)
        azimuth_angles_prime = (2.*azimuth_spread/1.4)*(torch.sqrt(-torch.log(z)
                                                                )/c_phi)

        # Introducing random variation
        random_sign = torch.randint(0, 2, [batch_size, num_bs, 1, num_clusters_max], generator=self.rng, dtype=torch.int32)
        random_sign = 2*random_sign - 1
        random_sign = random_sign.type(self._scenario._dtype_real)
        azimuth_spread = azimuth_spread.repeat(1,1,1,num_clusters_max)
        random_comp = torch.normal(mean=0.0, std=azimuth_spread/7.0, generator=self.rng).type(self._scenario._dtype_real)
        azimuth_angles = (random_sign*azimuth_angles_prime + random_comp + azimuth_angles_los)
        azimuth_angles = (azimuth_angles -
            torch.where(torch.unsqueeze(scenario.is_los, 3),
            random_sign[:,:,:,:1]*azimuth_angles_prime[:,:,:,:1]
            + random_comp[:,:,:,:1], 0.0))

        # Add offset angles to cluster angles to get the ray angles
        ray_offsets = self._ray_offsets[:scenario.rays_per_cluster]
        # Add dimensions for batch size, num bs, num ut, num clusters
        ray_offsets = torch.reshape(ray_offsets, [1,1,1,1,
                                                scenario.rays_per_cluster])
        # Rays angles
        azimuth_angles = torch.unsqueeze(azimuth_angles, 4)
        azimuth_angles = azimuth_angles + cluster_angle_spread*ray_offsets

        # Wrapping to (-180, 180)
        azimuth_angles = torch.remainder(azimuth_angles, 360.0)
        azimuth_angles = torch.where(torch.greater(azimuth_angles, 180.),
            azimuth_angles-360., azimuth_angles)

        return azimuth_angles.type(self._scenario._dtype_real)

    def _zenith_angles(self, zenith_spread, rician_k_factor, cluster_powers,
                       angle_type):
        # pylint: disable=line-too-long
        """
        Generate departure or arrival zenith angles (degrees).
        See step 7 of section 7.5 from TR 38.901 specification.

        Input
        ------
        zenith_spread : [batch size, num of BSs, num of UTs], 
            Angle spread, (ZSD or ZSA) depending on ``angle_type`` [deg]

        rician_k_factor : [batch size, num of BSs, num of UTs], 
            Rician K-factor of each BS-UT link. Used only for LoS links.

        cluster_powers : [batch size, num of BSs, num of UTs, maximum number of clusters], 
            Normalized path powers

        angle_type : str
            Type of angle to compute. Must be 'zoa' or 'zod'.

        Output
        -------
        zenith_angles : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], 
            Paths zenith angles wrapped within (0,180) [degree]. Either the ZoA or ZoD depending on ``angle_type``.
        """

        scenario = self._scenario

        batch_size = scenario.batch_size
        num_bs = scenario.num_bs
        num_ut = scenario.num_ut

        # Tensors giving UTs states
        los = scenario.is_los
        los_uts = los
        nlos_uts = torch.logical_not(los)

        num_clusters_max = scenario.num_clusters_max

        # Adding cluster dimension for broadcasting
        zenith_spread = torch.unsqueeze(zenith_spread, 3)
        rician_k_factor = torch.unsqueeze(rician_k_factor, 3)
        los_uts = torch.unsqueeze(los_uts, 3)
        nlos_uts = torch.unsqueeze(nlos_uts, 3)

        # Loading angle spread
        if angle_type == 'zod':
            zenith_angles_los = torch.rad2deg(scenario.los_zod_rad)
            cluster_angle_spread = (3./8.)*torch.pow(10.0, scenario.lsp_log_mean[:,:,:,6])
        else:
            cluster_angle_spread = scenario.get_param('cZSA')
            zenith_angles_los = torch.rad2deg(scenario.los_zoa_rad)
        zod_offset = scenario.zod_offset
        # Adding cluster dimension for broadcasting
        zod_offset = torch.unsqueeze(zod_offset, 3)
        zenith_angles_los = torch.unsqueeze(zenith_angles_los, 3)
        cluster_angle_spread = torch.unsqueeze(cluster_angle_spread, 3)

        # Compute the C_theta
        rician_k_factor_db = 10.0*torch.log10(rician_k_factor) # to dB
        c_theta_nlos = torch.unsqueeze(scenario.get_param("CThetaNLoS"),3)
        c_theta_los = c_theta_nlos*(1.3086 + 0.0339*rician_k_factor_db
            - 0.0077*torch.square(rician_k_factor_db)
            + 0.0002*torch.pow(rician_k_factor_db, 3.))
        c_theta = torch.where(los_uts, c_theta_los, c_theta_nlos)

        # Inverse Laplacian function
        z = cluster_powers/torch.max(cluster_powers, 3, keepdim=True)[0]
        z = torch.clip(z, 1e-6, 1.0)
        zenith_angles_prime = -zenith_spread*torch.log(z)/c_theta

        # Random component
        random_sign = torch.randint(0, 2, [batch_size, num_bs, 1, num_clusters_max], generator=self.rng, dtype=torch.int32)
        random_sign = 2*random_sign - 1
        random_sign = random_sign.type(self._scenario._dtype_real)
        zenith_spread = zenith_spread.repeat(1,1,1,num_clusters_max)
        random_comp = torch.normal(mean=0.0, std=zenith_spread/7.0, generator=self.rng).type(self._scenario._dtype_real)

        # The center cluster angles depend on the UT scenario
        zenith_angles = random_sign*zenith_angles_prime + random_comp
        los_additinoal_comp = -(random_sign[:,:,:,:1]*
            zenith_angles_prime[:,:,:,:1] + random_comp[:,:,:,:1]
            - zenith_angles_los)
        if angle_type == 'zod':
            additional_comp = torch.where(los_uts, los_additinoal_comp,
                zenith_angles_los + zod_offset)
        else:
            additional_comp = torch.where(los_uts, los_additinoal_comp,
                0.0)
            additional_comp = torch.where(nlos_uts, zenith_angles_los,
                additional_comp)
        zenith_angles = zenith_angles + additional_comp

        # Generating rays for every cluster
        # Add offset angles to cluster angles to get the ray angles
        ray_offsets = self._ray_offsets[:scenario.rays_per_cluster]
        # # Add dimensions for batch size, num bs, num ut, num clusters
        ray_offsets = torch.reshape(ray_offsets, [1,1,1,1,
                                                scenario.rays_per_cluster])
        # Adding ray dimension for broadcasting
        zenith_angles = torch.unsqueeze(zenith_angles, axis=4)
        cluster_angle_spread = torch.unsqueeze(cluster_angle_spread, 4)
        zenith_angles = zenith_angles + cluster_angle_spread*ray_offsets

        # Wrapping to (0, 180)
        zenith_angles = torch.remainder(zenith_angles, 360.0)
        zenith_angles = torch.where(torch.greater(zenith_angles, 180.),
            360.-zenith_angles, zenith_angles)

        return zenith_angles

    def _shuffle_angles(self, angles):
        # pylint: disable=line-too-long
        """
        Randomly shuffle a tensor carrying azimuth/zenith angles
        of arrival/departure.

        Input
        ------
        angles : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], 
            Angles to shuffle

        Output
        -------
        shuffled_angles : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], 
            Shuffled ``angles``
        """

        scenario = self._scenario

        batch_size = scenario.batch_size
        num_bs = scenario.num_bs
        num_ut = scenario.num_ut

        # Create randomly shuffled indices by arg-sorting samples from a random
        # normal distribution
        random_numbers = torch.normal(0.0, 1.0, size=[batch_size, num_bs, 1, scenario.num_clusters_max, scenario.rays_per_cluster], generator=self.rng)
        shuffled_indices = torch.argsort(random_numbers)
        shuffled_indices = torch.tile(shuffled_indices, [1, 1, num_ut, 1, 1])
        # Shuffling the angles
        # shuffled_angles = tf.gather(angles,shuffled_indices, batch_dims=4)
        shuffled_angles = torch.take_along_dim(angles, shuffled_indices, 4)
        return shuffled_angles

    def _random_coupling(self, aoa, aod, zoa, zod):
        # pylint: disable=line-too-long
        """
        Randomly couples the angles within a cluster for both azimuth and
        elevation.

        Step 8 in TR 38.901 specification.

        Input
        ------
        aoa : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], 
            Paths azimuth angles of arrival [degree] (AoA)

        aod : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], 
            Paths azimuth angles of departure (AoD) [degree]

        zoa : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], 
            Paths zenith angles of arrival [degree] (ZoA)

        zod : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], 
            Paths zenith angles of departure [degree] (ZoD)

        Output
        -------
        shuffled_aoa : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], 
            Shuffled `aoa`

        shuffled_aod : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], 
            Shuffled `aod`

        shuffled_zoa : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], 
            Shuffled `zoa`

        shuffled_zod : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], 
            Shuffled `zod`
        """
        shuffled_aoa = self._shuffle_angles(aoa)
        shuffled_aod = self._shuffle_angles(aod)
        shuffled_zoa = self._shuffle_angles(zoa)
        shuffled_zod = self._shuffle_angles(zod)

        return shuffled_aoa, shuffled_aod, shuffled_zoa, shuffled_zod

    def _cross_polarization_power_ratios(self):
        # pylint: disable=line-too-long
        """
        Generate cross-polarization power ratios.

        Step 9 in TR 38.901 specification.

        Input
        ------
        None

        Output
        -------
        cross_polarization_power_ratios : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], 
            Polarization power ratios
        """

        scenario = self._scenario

        batch_size = scenario.batch_size
        num_bs = scenario.num_bs
        num_ut = scenario.num_ut
        num_clusters = scenario.num_clusters_max
        num_rays_per_cluster = scenario.rays_per_cluster

        # Loading XPR mean and standard deviation
        mu_xpr = scenario.get_param("muXPR")
        std_xpr = scenario.get_param("sigmaXPR")
        # Expanding for broadcasting with clusters and rays dims
        mu_xpr = mu_xpr[...,None,None].expand(batch_size,num_bs,num_ut,num_clusters,num_rays_per_cluster)
        std_xpr = std_xpr[...,None,None].expand(batch_size,num_bs,num_ut,num_clusters,num_rays_per_cluster)

        # XPR are assumed to follow a log-normal distribution.
        # Generate XPR in log-domain
        x = torch.normal(mean=mu_xpr, std=std_xpr, generator=self.rng).type(self._scenario._dtype_real)
        # To linear domain
        cross_polarization_power_ratios = torch.pow(10.0, x/10.0)
        return cross_polarization_power_ratios

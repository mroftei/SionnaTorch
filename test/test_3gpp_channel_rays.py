#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import unittest
import numpy as np
from channel_test_utils import *
from scipy.stats import kstest
from sionna_torch.LSPGenerator import LSP
from sionna_torch.SionnaScenario import SionnaScenario


class TestRays(unittest.TestCase):
    r"""Test the rays generated for 3GPP system level simulations
    """

    # Batch size used to check the LSP distribution
    BATCH_SIZE = 10000
    # BATCH_SIZE = 100000

    # Carrier frequency
    CARRIER_FREQUENCY = 3.5e9 # Hz

    # Maximum allowed deviation for distance calculation (relative error)
    # MAX_ERR = 0.1
    MAX_ERR = 3e-1

    # Heigh of UTs
    H_UT = 1.5

    # Heigh of BSs
    H_BS = 35.0

    # Map square resolution
    MAP_RES = 2000

    def setUpClass():
        r"""Sample rays from all LoS and NLoS channel models for testing"""

        # Forcing the seed to make the tests deterministic
        seed = 43
        torch.manual_seed(seed)
        rng = torch.Generator().manual_seed(seed)
        np.random.seed(seed)

        batch_size = TestRays.BATCH_SIZE
        fc = TestRays.CARRIER_FREQUENCY

        # 1 UT and 1 BS
        ut_loc = generate_random_loc(batch_size, 1, (100,TestRays.MAP_RES), (100,TestRays.MAP_RES),
                                        (1.5, 1.5), share_loc=True,
                                        dtype=torch.float64)
        bs_loc = generate_random_loc(batch_size, 1, (0,100), (0,100),
                                        (35.0, 35.0), share_loc=True,
                                        dtype=torch.float64)

        # Force the LSPs
        TestRays.ds = np.power(10.0,-7.49)
        ds_ = torch.full((batch_size, 1, 1), TestRays.ds.astype(np.float64), dtype=torch.float64)
        TestRays.asd = np.power(10.0, 0.90)
        asd_ = torch.full((batch_size, 1, 1), TestRays.asd.astype(np.float64), dtype=torch.float64)
        TestRays.asa = np.power(10.0, 1.52)
        asa_ = torch.full((batch_size, 1, 1), TestRays.asa.astype(np.float64), dtype=torch.float64)
        TestRays.zsa = np.power(10.0, 0.47)
        zsa_ = torch.full((batch_size, 1, 1), TestRays.zsa.astype(np.float64), dtype=torch.float64)
        TestRays.zsd = np.power(10.0, -0.29)
        zsd_ = torch.full((batch_size, 1, 1), TestRays.zsd.astype(np.float64), dtype=torch.float64)
        TestRays.k = np.power(10.0, 7./10.)
        k_ = torch.full((batch_size, 1, 1), TestRays.k.astype(np.float64), dtype=torch.float64)
        sf_ = torch.zeros((batch_size, 1, 1), dtype=torch.float64)
        lsp = LSP(ds_, asd_, asa_, sf_, k_, zsa_, zsd_)

        # Store the sampled rays
        TestRays.delays = {}
        TestRays.powers = {}
        TestRays.aoa = {}
        TestRays.aod = {}
        TestRays.zoa = {}
        TestRays.zod = {}
        TestRays.xpr = {}
        TestRays.num_clusters = {}
        TestRays.los_aoa = {}
        TestRays.los_aod = {}
        TestRays.los_zoa = {}
        TestRays.los_zod = {}
        TestRays.mu_log_zsd = {}

        #################### RMa
        TestRays.delays['rma'] = {}
        TestRays.powers['rma'] = {}
        TestRays.aoa['rma'] = {}
        TestRays.aod['rma'] = {}
        TestRays.zoa['rma'] = {}
        TestRays.zod['rma'] = {}
        TestRays.xpr['rma'] = {}
        TestRays.num_clusters['rma'] = {}
        TestRays.los_aoa['rma'] = {}
        TestRays.los_aod['rma'] = {}
        TestRays.los_zoa['rma'] = {}
        TestRays.los_zod['rma'] = {}
        TestRays.mu_log_zsd['rma'] = {}
        scen_map = np.zeros([TestRays.MAP_RES, TestRays.MAP_RES], dtype=int)
        # scenario = RMaScenario(fc, "downlink", rng=rng, dtype=torch.complex128)
        # ray_sampler = RaysGenerator(scenario, rng=rng)

        #### LoS
        # scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
        #                                     ut_velocities, los=True)
        # ray_sampler.topology_updated_callback()
        # rays = ray_sampler(lsp)
        scenario = SionnaScenario(ut_loc, bs_loc, scen_map, los_requested=True, f_c=fc, seed=seed, dtype=torch.complex128)
        rays = scenario._ray_sampler(lsp)
        TestRays.delays['rma']['los'] = torch.squeeze(rays.delays).numpy()
        TestRays.powers['rma']['los'] = torch.squeeze(rays.powers).numpy()
        TestRays.aoa['rma']['los'] = torch.squeeze(rays.aoa).numpy()
        TestRays.aod['rma']['los'] = torch.squeeze(rays.aod).numpy()
        TestRays.zoa['rma']['los'] = torch.squeeze(rays.zoa).numpy()
        TestRays.zod['rma']['los'] = torch.squeeze(rays.zod).numpy()
        TestRays.xpr['rma']['los'] = torch.squeeze(rays.xpr).numpy()
        TestRays.num_clusters['rma']['los'] = 11
        TestRays.los_aoa['rma']['los'] = torch.rad2deg(scenario.los_aoa_rad).numpy()
        TestRays.los_aod['rma']['los'] = torch.rad2deg(scenario.los_aod_rad).numpy()
        TestRays.los_zoa['rma']['los'] = torch.rad2deg(scenario.los_zoa_rad).numpy()
        TestRays.los_zod['rma']['los'] = torch.rad2deg(scenario.los_zod_rad).numpy()
        TestRays.mu_log_zsd['rma']['los'] = scenario.lsp_log_mean[:,0,0,6]

        #### NLoS
        # scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
        #                                     ut_velocities, los=False)
        # ray_sampler.topology_updated_callback()
        # rays = ray_sampler(lsp)
        scenario = SionnaScenario(ut_loc, bs_loc, scen_map, los_requested=False, f_c=fc, seed=seed, dtype=torch.complex128)
        rays = scenario._ray_sampler(lsp)
        TestRays.delays['rma']['nlos'] = torch.squeeze(rays.delays).numpy()
        TestRays.powers['rma']['nlos'] = torch.squeeze(rays.powers).numpy()
        TestRays.aoa['rma']['nlos'] = torch.squeeze(rays.aoa).numpy()
        TestRays.aod['rma']['nlos'] = torch.squeeze(rays.aod).numpy()
        TestRays.zoa['rma']['nlos'] = torch.squeeze(rays.zoa).numpy()
        TestRays.zod['rma']['nlos'] = torch.squeeze(rays.zod).numpy()
        TestRays.xpr['rma']['nlos'] = torch.squeeze(rays.xpr).numpy()
        TestRays.num_clusters['rma']['nlos'] = 10
        TestRays.los_aoa['rma']['nlos'] = torch.rad2deg(scenario.los_aoa_rad).numpy()
        TestRays.los_aod['rma']['nlos'] = torch.rad2deg(scenario.los_aod_rad).numpy()
        TestRays.los_zoa['rma']['nlos'] = torch.rad2deg(scenario.los_zoa_rad).numpy()
        TestRays.los_zod['rma']['nlos'] = torch.rad2deg(scenario.los_zod_rad).numpy()
        TestRays.mu_log_zsd['rma']['nlos'] = scenario.lsp_log_mean[:,0,0,6]

        #################### UMa
        TestRays.delays['uma'] = {}
        TestRays.powers['uma'] = {}
        TestRays.aoa['uma'] = {}
        TestRays.aod['uma'] = {}
        TestRays.zoa['uma'] = {}
        TestRays.zod['uma'] = {}
        TestRays.xpr['uma'] = {}
        TestRays.num_clusters['uma'] = {}
        TestRays.los_aoa['uma'] = {}
        TestRays.los_aod['uma'] = {}
        TestRays.los_zoa['uma'] = {}
        TestRays.los_zod['uma'] = {}
        TestRays.mu_log_zsd['uma'] = {}
        scen_map = np.ones([TestRays.MAP_RES, TestRays.MAP_RES], dtype=int)
        # scenario = UMaScenario(  fc, "downlink", rng=rng, dtype=torch.complex128)
        # ray_sampler = RaysGenerator(scenario, rng=rng)

        #### LoS
        # scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
        #                                     ut_velocities, los=True)
        # ray_sampler.topology_updated_callback()
        # rays = ray_sampler(lsp)
        scenario = SionnaScenario(ut_loc, bs_loc, scen_map, los_requested=True, f_c=fc, seed=seed, dtype=torch.complex128)
        rays = scenario._ray_sampler(lsp)
        TestRays.delays['uma']['los'] = torch.squeeze(rays.delays).numpy()
        TestRays.powers['uma']['los'] = torch.squeeze(rays.powers).numpy()
        TestRays.aoa['uma']['los'] = torch.squeeze(rays.aoa).numpy()
        TestRays.aod['uma']['los'] = torch.squeeze(rays.aod).numpy()
        TestRays.zoa['uma']['los'] = torch.squeeze(rays.zoa).numpy()
        TestRays.zod['uma']['los'] = torch.squeeze(rays.zod).numpy()
        TestRays.xpr['uma']['los'] = torch.squeeze(rays.xpr).numpy()
        TestRays.num_clusters['uma']['los'] = 12
        TestRays.los_aoa['uma']['los'] = torch.rad2deg(scenario.los_aoa_rad).numpy()
        TestRays.los_aod['uma']['los'] = torch.rad2deg(scenario.los_aod_rad).numpy()
        TestRays.los_zoa['uma']['los'] = torch.rad2deg(scenario.los_zoa_rad).numpy()
        TestRays.los_zod['uma']['los'] = torch.rad2deg(scenario.los_zod_rad).numpy()
        TestRays.mu_log_zsd['uma']['los'] = scenario.lsp_log_mean[:,0,0,6]

        #### NLoS
        # scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
        #                                     ut_velocities, los=False)
        # ray_sampler.topology_updated_callback()
        # rays = ray_sampler(lsp)
        scenario = SionnaScenario(ut_loc, bs_loc, scen_map, los_requested=False, f_c=fc, seed=seed, dtype=torch.complex128)
        rays = scenario._ray_sampler(lsp)
        TestRays.delays['uma']['nlos'] = torch.squeeze(rays.delays).numpy()
        TestRays.powers['uma']['nlos'] = torch.squeeze(rays.powers).numpy()
        TestRays.aoa['uma']['nlos'] = torch.squeeze(rays.aoa).numpy()
        TestRays.aod['uma']['nlos'] = torch.squeeze(rays.aod).numpy()
        TestRays.zoa['uma']['nlos'] = torch.squeeze(rays.zoa).numpy()
        TestRays.zod['uma']['nlos'] = torch.squeeze(rays.zod).numpy()
        TestRays.xpr['uma']['nlos'] = torch.squeeze(rays.xpr).numpy()
        TestRays.num_clusters['uma']['nlos'] = 20
        TestRays.los_aoa['uma']['nlos'] = torch.rad2deg(scenario.los_aoa_rad).numpy()
        TestRays.los_aod['uma']['nlos'] = torch.rad2deg(scenario.los_aod_rad).numpy()
        TestRays.los_zoa['uma']['nlos'] = torch.rad2deg(scenario.los_zoa_rad).numpy()
        TestRays.los_zod['uma']['nlos'] = torch.rad2deg(scenario.los_zod_rad).numpy()
        TestRays.mu_log_zsd['uma']['nlos'] = scenario.lsp_log_mean[:,0,0,6]

        ###### General
        TestRays.d_2d = scenario.distance_2d[0,0,0].numpy()

    @channel_test_on_models(('rma', 'uma'), ('los', 'nlos'))
    def test_delays(self, model, submodel):
        """Test ray generation: Delays"""
        num_clusters = TestRays.num_clusters[model][submodel]
        batch_size = TestRays.BATCH_SIZE
        tau = TestRays.delays[model][submodel][:,:num_clusters].flatten()
        _, ref_tau = delays(model, submodel, batch_size, num_clusters,
                                    TestRays.ds, TestRays.k)
        ref_tau = ref_tau[:,:num_clusters].flatten()
        D,_ = kstest(tau,ref_tau)
        self.assertLessEqual(D, TestRays.MAX_ERR, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'uma'), ('los', 'nlos'))
    def test_powers(self, model, submodel):
        """Test ray generation: Powers"""
        num_clusters = TestRays.num_clusters[model][submodel]
        batch_size = TestRays.BATCH_SIZE
        p = self.powers[model][submodel][:,:num_clusters].flatten()
        unscaled_tau, _ = delays(model, submodel, batch_size, num_clusters,
                                    TestRays.ds, TestRays.k)
        ref_p,_ = powers(model, submodel, batch_size, num_clusters,
                unscaled_tau, self.ds, self.k)
        ref_p = ref_p[:,:num_clusters].flatten()
        D,_ = kstest(ref_p,p)
        self.assertLessEqual(D, TestRays.MAX_ERR, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'uma'), ('los', 'nlos'))
    def test_aoa(self, model, submodel):
        """Test ray generation: AoA"""
        num_clusters = TestRays.num_clusters[model][submodel]
        batch_size = TestRays.BATCH_SIZE
        k = None
        if submodel == 'los':
            k = TestRays.k
        unscaled_tau, _ = delays(model, submodel, batch_size, num_clusters,
                                    TestRays.ds, TestRays.k)
        _, ref_p_angles = powers(model, submodel, batch_size, num_clusters,
            unscaled_tau, TestRays.ds, TestRays.k)
        ref_p_angles = ref_p_angles[:,:num_clusters]
        ref_samples = aoa(model, submodel, batch_size, num_clusters,
            TestRays.asa, ref_p_angles, TestRays.los_aoa[model][submodel], k)
        ref_samples = ref_samples[:,:num_clusters].flatten()
        samples = TestRays.aoa[model][submodel][:,:num_clusters].flatten()
        D,_ = kstest(ref_samples, samples)
        self.assertLessEqual(D, TestRays.MAX_ERR, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'uma'), ('los', 'nlos'))
    def test_aod(self, model, submodel):
        """Test ray generation: AoD"""
        num_clusters = TestRays.num_clusters[model][submodel]
        batch_size = TestRays.BATCH_SIZE
        k = None
        if submodel == 'los':
            k = TestRays.k
        unscaled_tau, _ = delays(model, submodel, batch_size, num_clusters,
                                    TestRays.ds, TestRays.k)
        _, ref_p_angles = powers(model, submodel, batch_size, num_clusters,
                unscaled_tau, TestRays.ds, TestRays.k)
        ref_p_angles = ref_p_angles[:,:num_clusters]
        ref_samples = aod(model, submodel, batch_size, num_clusters,
            TestRays.asd, ref_p_angles, TestRays.los_aod[model][submodel], k)
        ref_samples = ref_samples[:,:num_clusters].flatten()
        samples = TestRays.aod[model][submodel][:,:num_clusters].flatten()
        D,_ = kstest(ref_samples, samples)
        self.assertLessEqual(D, TestRays.MAX_ERR, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'uma'), ('los', 'nlos'))
    def test_zoa(self, model, submodel):
        """Test ray generation: ZoA"""
        num_clusters = TestRays.num_clusters[model][submodel]
        batch_size = TestRays.BATCH_SIZE
        k = None
        if submodel == 'los':
            k = TestRays.k
        unscaled_tau, _ = delays(model, submodel, batch_size, num_clusters,
                                    TestRays.ds, TestRays.k)
        _, ref_p_angles = powers(model, submodel, batch_size, num_clusters,
                    unscaled_tau, TestRays.ds, TestRays.k)
        ref_p_angles = ref_p_angles[:,:num_clusters]
        ref_samples = zoa(model, submodel, batch_size, num_clusters,
            TestRays.zsa, ref_p_angles, TestRays.los_zoa[model][submodel], k)
        ref_samples = ref_samples[:,:num_clusters].flatten()
        samples = TestRays.zoa[model][submodel][:,:num_clusters].flatten()
        D,_ = kstest(ref_samples, samples)
        self.assertLessEqual(D, TestRays.MAX_ERR, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'uma'), ('los', 'nlos'))
    def test_zod(self, model, submodel):
        """Test ray generation: ZoD"""
        num_clusters = TestRays.num_clusters[model][submodel]
        batch_size = TestRays.BATCH_SIZE
        fc = TestRays.CARRIER_FREQUENCY
        d_2d = TestRays.d_2d
        h_ut = TestRays.H_UT
        mu_log_zod = TestRays.mu_log_zsd[model][submodel]
        k = None
        if submodel == 'los':
            k = TestRays.k
        unscaled_tau, _ = delays(model, submodel, batch_size, num_clusters,
                                    TestRays.ds, TestRays.k)
        _, ref_p_angles = powers(model, submodel, batch_size, num_clusters,
                    unscaled_tau, TestRays.ds, TestRays.k)
        ref_p_angles = ref_p_angles[:,:num_clusters]
        offset = zod_offset(model, submodel, fc, d_2d, h_ut)
        ref_samples = zod(model, submodel, batch_size, num_clusters,
            TestRays.zsd, ref_p_angles, TestRays.los_zod[model][submodel],
            offset, mu_log_zod, k)
        ref_samples = ref_samples[:,:num_clusters].flatten()
        samples = TestRays.zod[model][submodel][:,:num_clusters].flatten()
        D,_ = kstest(ref_samples, samples)
        self.assertLessEqual(D, TestRays.MAX_ERR, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'uma'), ('los', 'nlos'))
    def test_xpr(self, model, submodel):
        """Test ray generation: XPR"""
        num_clusters = TestRays.num_clusters[model][submodel]
        batch_size = TestRays.BATCH_SIZE
        samples = TestRays.xpr[model][submodel][:,:num_clusters].flatten()
        ref_samples = xpr(model, submodel, batch_size, num_clusters)
        ref_samples = ref_samples[:,:num_clusters].flatten()
        D,_ = kstest(ref_samples, samples)
        self.assertLessEqual(D, TestRays.MAX_ERR)

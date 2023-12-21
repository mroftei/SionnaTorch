#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import unittest
import numpy as np
from channel_test_utils import *
from scipy.stats import kstest, norm
from sionna_torch.SionnaScenario import SionnaScenario


class TestLSP(unittest.TestCase):
    r"""Test the distribution, cross-correlation, and spatial correlation of
    3GPP channel models' LSPs
    """

    # Carrier frequency
    CARRIER_FREQUENCY = 3.5e9  # Hz

    # Heigh of UTs
    H_UT = 1.5

    # Heigh of BSs
    H_BS = 35.0

    # Batch size for generating samples of LSPs and pathlosses
    BATCH_SIZE = 50000
    # BATCH_SIZE = 500000

    # More than one UT is required for testing the spatial and cross-correlation
    # of LSPs
    NB_UT = 5

    # The LSPs follow either a Gaussian or a truncated Gaussian
    # distribution. A Kolmogorov-Smirnov (KS) test is used to check that the
    # LSP follow the appropriate distribution. This is the threshold below
    # which the KS statistic `D` should be for passing the test.
    # MAX_ERR_KS = 0.1
    MAX_ERR_KS = 1e-2

    # # Maximum allowed deviation for cross-correlation of LSP parameters
    MAX_ERR_CROSS_CORR = 5e-2
    # MAX_ERR_CROSS_CORR = 3e-2

    # # Maximum allowed deviation for spatial correlation of LSP parameters
    # MAX_ERR_SPAT_CORR = 0.1
    MAX_ERR_SPAT_CORR = 3e-2

    # LoS probability
    MAX_ERR_LOS_PROB = 1e-2

    # ZOD Offset maximum relative error
    MAX_ERR_ZOD_OFFSET = 1e-2

    # Maximum allowed deviation for pathloss
    MAX_ERR_PATHLOSS_MEAN = 1.0
    MAX_ERR_PATHLOSS_STD = 1e-1

    def limited_normal(self, batch_size, minval, maxval, mu, std):
        r"""
        Return a limited normal distribution. This is different from a truncated
        normal distribution, as the samples exceed ``minval`` and ``maxval`` are
        clipped.

        More precisely, ``x`` is generated as follows:
        1. Sample ``y`` of shape [``batch_size``] from a Gaussian distribution N(mu,std)
        2. x = max(min(x, maxval), minval)
        """
        x = np.random.normal(size=[batch_size])
        x = np.maximum(x, minval)
        x = np.minimum(x, maxval)
        x = std * x + mu
        return x

    def setUpClass():
        r"""Sample LSPs and pathlosses from all channel models for testing"""

        # Forcing the seed to make the tests deterministic
        seed = 43
        torch.manual_seed(seed)
        rng = torch.Generator().manual_seed(seed)
        np.random.seed(seed)

        nb_bs = 1
        fc = TestLSP.CARRIER_FREQUENCY
        h_ut = TestLSP.H_UT
        h_bs = TestLSP.H_BS
        batch_size = TestLSP.BATCH_SIZE
        nb_ut = TestLSP.NB_UT

        # The following quantities have no impact on LSP
        # However,these are needed to instantiate the model
        ut_orientations = torch.zeros([batch_size, nb_ut], dtype=torch.float64).numpy()
        bs_orientations = torch.zeros([batch_size, nb_ut], dtype=torch.float64).numpy()
        ut_velocities = torch.zeros([batch_size, nb_ut], dtype=torch.float64).numpy()

        # LSPs, ZoD offset, pathlosses
        TestLSP.lsp_samples = {}
        TestLSP.zod_offset = {}
        TestLSP.pathlosses = {}
        TestLSP.los_prob = {}

        ut_loc = generate_random_loc(
            batch_size,
            nb_ut,
            (100, 2000),
            (100, 2000),
            (h_ut, h_ut),
            share_loc=True,
            dtype=torch.float64,
        )
        bs_loc = generate_random_loc(
            batch_size,
            nb_bs,
            (0, 100),
            (0, 100),
            (h_bs, h_bs),
            share_loc=True,
            dtype=torch.float64,
        )

        ####### RMa
        TestLSP.lsp_samples["rma"] = {}
        TestLSP.zod_offset["rma"] = {}
        TestLSP.pathlosses["rma"] = {}
        is_urban = np.zeros([batch_size, nb_bs, nb_ut], dtype=bool)

        # LoS
        scenario = SionnaScenario(ut_loc, bs_loc, is_urban, los_requested=True, f_c=fc, seed=seed, dtype=torch.complex128)
        TestLSP.lsp_samples["rma"]["los"] = scenario._lsp_sampler()
        TestLSP.zod_offset["rma"]["los"] = scenario.zod_offset
        TestLSP.pathlosses["rma"]["los"] = scenario.basic_pathloss.numpy().reshape((batch_size,-1))

        # NLoS
        scenario = SionnaScenario(ut_loc, bs_loc, is_urban, los_requested=False, f_c=fc, seed=seed, dtype=torch.complex128)
        TestLSP.lsp_samples["rma"]["nlos"] = scenario._lsp_sampler()
        TestLSP.zod_offset["rma"]["nlos"] = scenario.zod_offset
        TestLSP.pathlosses["rma"]["nlos"] = scenario.basic_pathloss.numpy().reshape((batch_size,-1))

        TestLSP.los_prob["rma"] = scenario.los_probability.numpy()
        TestLSP.rma_w = scenario.average_street_width
        TestLSP.rma_h = scenario.average_building_height

        ####### UMa
        TestLSP.lsp_samples["uma"] = {}
        TestLSP.zod_offset["uma"] = {}
        TestLSP.pathlosses["uma"] = {}
        is_urban = np.ones([batch_size, nb_bs, nb_ut], dtype=bool)

        # LoS
        scenario = SionnaScenario(ut_loc, bs_loc, is_urban, los_requested=True, f_c=fc, seed=seed, dtype=torch.complex128)
        TestLSP.lsp_samples["uma"]["los"] = scenario._lsp_sampler()
        TestLSP.zod_offset["uma"]["los"] = scenario.zod_offset
        TestLSP.pathlosses["uma"]["los"] = scenario.basic_pathloss.numpy().reshape((batch_size,-1))

        # NLoS
        scenario = SionnaScenario(ut_loc, bs_loc, is_urban, los_requested=False, f_c=fc, seed=seed, dtype=torch.complex128)
        TestLSP.lsp_samples["uma"]["nlos"] = scenario._lsp_sampler()
        TestLSP.zod_offset["uma"]["nlos"] = scenario.zod_offset
        TestLSP.pathlosses["uma"]["nlos"] = scenario.basic_pathloss.numpy().reshape((batch_size,-1))

        TestLSP.los_prob["uma"] = scenario.los_probability.numpy()

        # The following values do not depend on the scenario
        TestLSP.d_2d = scenario.distance_2d.numpy()
        TestLSP.d_2d_ut = scenario.matrix_ut_distance_2d.numpy()
        TestLSP.d_3d = scenario.distance_3d[0, 0, :].numpy()

    @channel_test_on_models(("rma", "uma"), ("los", "nlos"))
    def test_ds_dist(self, model, submodel):
        """Test the distribution of LSP DS"""
        samples = TestLSP.lsp_samples[model][submodel].ds.numpy().flatten()
        samples = np.log10(samples)
        mu, std = log10DS(model, submodel, TestLSP.CARRIER_FREQUENCY)
        D, _ = kstest(samples, norm.cdf, args=(mu, std))
        self.assertLessEqual(D, TestLSP.MAX_ERR_KS, f"{model}:{submodel}")

    @channel_test_on_models(("rma", "uma"), ("los", "nlos"))
    def test_asa_dist(self, model, submodel):
        """Test the distribution of LSP ASA"""
        samples = TestLSP.lsp_samples[model][submodel].asa.numpy().flatten()
        samples = np.log10(samples)
        mu, std = log10ASA(model, submodel, TestLSP.CARRIER_FREQUENCY)
        a = -np.inf
        b = (np.log10(104) - mu) / std
        samples_ref = self.limited_normal(TestLSP.BATCH_SIZE, a, b, mu, std)
        # KS-test does not work great with discontinuties.
        # Therefore, we test only the continuous part of the CDF, and also test
        # that the maximum value allowed is not exceeded
        maxval = np.max(samples)
        samples = samples[samples < np.log10(104)]
        samples_ref = samples_ref[samples_ref < np.log10(104)]
        D, _ = kstest(samples, samples_ref)
        self.assertLessEqual(maxval, np.log10(104), f"{model}:{submodel}")
        self.assertLessEqual(D, TestLSP.MAX_ERR_KS, f"{model}:{submodel}")

    @channel_test_on_models(("rma", "uma"), ("los", "nlos"))
    def test_asd_dist(self, model, submodel):
        """Test the distribution of LSP ASD"""
        samples = TestLSP.lsp_samples[model][submodel].asd.numpy()
        samples = np.log10(samples)
        mu, std = log10ASD(model, submodel, TestLSP.CARRIER_FREQUENCY)
        a = -np.inf
        b = (np.log10(104) - mu) / std
        samples_ref = self.limited_normal(TestLSP.BATCH_SIZE, a, b, mu, std)
        # KS-test does not work great with discontinuties.
        # Therefore, we test only the continuous part of the CDF, and also test
        # that the maximum value allowed is not exceeded
        maxval = np.max(samples)
        samples = samples[samples < np.log10(104)]
        samples_ref = samples_ref[samples_ref < np.log10(104)]
        D, _ = kstest(samples, samples_ref)
        self.assertLessEqual(maxval, np.log10(104), f"{model}:{submodel}")
        self.assertLessEqual(D, TestLSP.MAX_ERR_KS, f"{model}:{submodel}")

    @channel_test_on_models(("rma", "uma"), ("los", "nlos"))
    def test_zsa_dist(self, model, submodel):
        """Test the distribution of LSP ZSA"""
        samples = TestLSP.lsp_samples[model][submodel].zsa.numpy().flatten()
        samples = np.log10(samples)
        mu, std = log10ZSA(model, submodel, TestLSP.CARRIER_FREQUENCY)
        a = -np.inf
        b = (np.log10(52) - mu) / std
        samples_ref = self.limited_normal(TestLSP.BATCH_SIZE, a, b, mu, std)
        # KS-test does not work great with discontinuties.
        # Therefore, we test only the continuous part of the CDF, and also test
        # that the maximum value allowed is not exceeded
        maxval = np.max(samples)
        samples = samples[samples < np.log10(52)]
        samples_ref = samples_ref[samples_ref < np.log10(52)]
        D, _ = kstest(samples, samples_ref)
        self.assertLessEqual(maxval, np.log10(52), f"{model}:{submodel}")
        self.assertLessEqual(D, TestLSP.MAX_ERR_KS, f"{model}:{submodel}")

    @channel_test_on_models(("rma", "uma"), ("los", "nlos"))
    def test_zsd_dist(self, model, submodel):
        """Test the distribution of LSP ZSD"""
        d_2d = TestLSP.d_2d[0, 0, 0]
        samples = TestLSP.lsp_samples[model][submodel].zsd[:,0,0].numpy().flatten()
        samples = np.log10(samples)
        mu, std = log10ZSD(
            model, submodel, d_2d, TestLSP.CARRIER_FREQUENCY, TestLSP.H_BS, TestLSP.H_UT
        )
        a = -np.inf
        b = (np.log10(52) - mu) / std
        samples_ref = self.limited_normal(TestLSP.BATCH_SIZE, a, b, mu, std)
        # KS-test does not work great with discontinuties.
        # Therefore, we test only the continuous part of the CDF, and also test
        # that the maximum value allowed is not exceeded
        maxval = np.max(samples)
        samples = samples[samples < np.log10(52)]
        samples_ref = samples_ref[samples_ref < np.log10(52)]
        D, _ = kstest(samples, samples_ref)
        self.assertLessEqual(maxval, np.log10(52), f"{model}:{submodel}")
        self.assertLessEqual(D, TestLSP.MAX_ERR_KS, f"{model}:{submodel}")

    @channel_test_on_models(("rma", "uma"), ("los", "nlos"))
    def test_sf_dist(self, model, submodel):
        """Test the distribution of LSP SF"""
        d_2d = TestLSP.d_2d[0, 0, 0]
        samples = TestLSP.lsp_samples[model][submodel].sf.numpy().flatten()
        samples = 10.0 * np.log10(samples)
        mu, std = log10SF_dB(
            model, submodel, d_2d, TestLSP.CARRIER_FREQUENCY, TestLSP.H_BS, TestLSP.H_UT
        )
        D, _ = kstest(samples, norm.cdf, args=(mu, std))
        self.assertLessEqual(D, TestLSP.MAX_ERR_KS, f"{model}:{submodel}")

    @channel_test_on_models(("rma", "uma"), ("los",))
    def test_k_dist(self, model, submodel):
        """Test the distribution of LSP K"""
        samples = TestLSP.lsp_samples[model][submodel].k_factor.numpy().flatten()
        samples = 10.0 * np.log10(samples)
        mu, std = log10K_dB(model, submodel)
        D, _ = kstest(samples, norm.cdf, args=(mu, std))
        self.assertLessEqual(D, TestLSP.MAX_ERR_KS, f"{model}:{submodel}")

    @channel_test_on_models(("rma", "uma"), ("los", "nlos"))
    def test_cross_correlation(self, model, submodel):
        """Test the LSP cross correlation"""
        lsp_list = []
        ds_samples = TestLSP.lsp_samples[model][submodel].ds.numpy().flatten()
        ds_samples = np.log10(ds_samples)
        lsp_list.append(ds_samples)
        asd_samples = TestLSP.lsp_samples[model][submodel].asd.numpy().flatten()
        asd_samples = np.log10(asd_samples)
        lsp_list.append(asd_samples)
        asa_samples = TestLSP.lsp_samples[model][submodel].asa.numpy().flatten()
        asa_samples = np.log10(asa_samples)
        lsp_list.append(asa_samples)
        sf_samples = TestLSP.lsp_samples[model][submodel].sf.numpy().flatten()
        sf_samples = np.log10(sf_samples)
        lsp_list.append(sf_samples)
        if submodel == "los":
            k_samples = TestLSP.lsp_samples[model][submodel].k_factor.numpy().flatten()
            k_samples = np.log10(k_samples)
            lsp_list.append(k_samples)
        zsa_samples = TestLSP.lsp_samples[model][submodel].zsa.numpy().flatten()
        zsa_samples = np.log10(zsa_samples)
        lsp_list.append(zsa_samples)
        zsd_samples = TestLSP.lsp_samples[model][submodel].zsd.numpy().flatten()
        zsd_samples = np.log10(zsd_samples)
        lsp_list.append(zsd_samples)
        lsp_list = np.stack(lsp_list, axis=-1)
        cross_corr_measured = np.corrcoef(lsp_list.T)
        abs_err = np.abs(cross_corr(model, submodel) - cross_corr_measured)
        max_err = np.max(abs_err)
        self.assertLessEqual(max_err, TestLSP.MAX_ERR_CROSS_CORR, f"{model}:{submodel}")

    @channel_test_on_models(("rma", "uma"), ("los", "nlos"))
    def test_spatial_correlation(self, model, submodel):
        """Test the spatial correlation of LSPs"""
        d_2d_ut = TestLSP.d_2d_ut[0, 0]
        #### LoS
        ds_samples = TestLSP.lsp_samples[model][submodel].ds.numpy()[:, 0, :]
        ds_samples = np.log10(ds_samples)
        asd_samples = TestLSP.lsp_samples[model][submodel].asd.numpy()[:, 0, :]
        asd_samples = np.log10(asd_samples)
        asa_samples = TestLSP.lsp_samples[model][submodel].asa.numpy()[:, 0, :]
        asa_samples = np.log10(asa_samples)
        sf_samples = TestLSP.lsp_samples[model][submodel].sf.numpy()[:, 0, :]
        sf_samples = np.log10(sf_samples)
        if submodel == "los":
            k_samples = TestLSP.lsp_samples[model][submodel].k_factor.numpy()[:, 0, :]
            k_samples = np.log10(k_samples)
        zsa_samples = TestLSP.lsp_samples[model][submodel].zsa.numpy()[:, 0, :]
        zsa_samples = np.log10(zsa_samples)
        zsd_samples = TestLSP.lsp_samples[model][submodel].zsd.numpy()[:, 0, :]
        zsd_samples = np.log10(zsd_samples)
        #
        C_ds_measured = np.corrcoef(ds_samples.T)[0]
        C_asd_measured = np.corrcoef(asd_samples.T)[0]
        C_asa_measured = np.corrcoef(asa_samples.T)[0]
        C_sf_measured = np.corrcoef(sf_samples.T)[0]
        if submodel == "los":
            C_k_measured = np.corrcoef(k_samples.T)[0]
        C_zsa_measured = np.corrcoef(zsa_samples.T)[0]
        C_zsd_measured = np.corrcoef(zsd_samples.T)[0]
        #
        C_ds = np.exp(-d_2d_ut / corr_dist_ds(model, submodel))
        C_asd = np.exp(-d_2d_ut / corr_dist_asd(model, submodel))
        C_asa = np.exp(-d_2d_ut / corr_dist_asa(model, submodel))
        C_sf = np.exp(-d_2d_ut / corr_dist_sf(model, submodel))
        if submodel == "los":
            C_k = np.exp(-d_2d_ut / corr_dist_k(model, submodel))
        C_zsa = np.exp(-d_2d_ut / corr_dist_zsa(model, submodel))
        C_zsd = np.exp(-d_2d_ut / corr_dist_zsd(model, submodel))
        #
        ds_max_err = np.max(np.abs(C_ds_measured - C_ds))
        self.assertLessEqual(
            ds_max_err, TestLSP.MAX_ERR_SPAT_CORR, f"{model}:{submodel}"
        )
        asd_max_err = np.max(np.abs(C_asd_measured - C_asd))
        self.assertLessEqual(
            asd_max_err, TestLSP.MAX_ERR_SPAT_CORR, f"{model}:{submodel}"
        )
        asa_max_err = np.max(np.abs(C_asa_measured - C_asa))
        self.assertLessEqual(
            asa_max_err, TestLSP.MAX_ERR_SPAT_CORR, f"{model}:{submodel}"
        )
        sf_max_err = np.max(np.abs(C_sf_measured - C_sf))
        self.assertLessEqual(
            sf_max_err, TestLSP.MAX_ERR_SPAT_CORR, f"{model}:{submodel}"
        )
        if submodel == "los":
            k_max_err = np.max(np.abs(C_k_measured - C_k))
            self.assertLessEqual(
                k_max_err, TestLSP.MAX_ERR_SPAT_CORR, f"{model}:{submodel}"
            )
        zsa_max_err = np.max(np.abs(C_zsa_measured - C_zsa))
        self.assertLessEqual(
            zsa_max_err, TestLSP.MAX_ERR_SPAT_CORR, f"{model}:{submodel}"
        )
        zsd_max_err = np.max(np.abs(C_zsd_measured - C_zsd))
        self.assertLessEqual(
            zsd_max_err, TestLSP.MAX_ERR_SPAT_CORR, f"{model}:{submodel}"
        )

    # Submodel is not needed for LoS probability
    @channel_test_on_models(("rma", "uma"), ("foo",))
    def test_los_probability(self, model, submodel):
        """Test LoS probability"""
        d_2d = TestLSP.d_2d
        h_ut = TestLSP.H_UT
        #
        los_prob_ref = los_probability(model, d_2d, h_ut)
        los_prob = TestLSP.los_prob[model]
        #
        max_err = np.max(np.abs(los_prob_ref - los_prob))
        self.assertLessEqual(max_err, TestLSP.MAX_ERR_LOS_PROB, f"{model}:{submodel}")

    @channel_test_on_models(("rma", "uma"), ("los", "nlos"))
    def test_zod_offset(self, model, submodel):
        """Test ZOD offset"""
        d_2d = self.d_2d
        fc = TestLSP.CARRIER_FREQUENCY
        h_ut = TestLSP.H_UT
        samples = self.zod_offset[model][submodel].numpy()
        samples_ref = zod_offset(model, submodel, fc, d_2d, h_ut)
        max_err = np.max(np.abs(samples - samples_ref))
        self.assertLessEqual(max_err, TestLSP.MAX_ERR_ZOD_OFFSET, f"{model}:{submodel}")

    @channel_test_on_models(("rma", "uma"), ("los", "nlos"))
    def test_pathloss(self, model, submodel):
        """Test the pathloss"""
        fc = TestLSP.CARRIER_FREQUENCY
        h_ut = TestLSP.H_UT
        h_bs = TestLSP.H_BS
        if model == "rma":
            samples = TestLSP.pathlosses[model][submodel]
            mean_samples = np.mean(samples, axis=0)
            std_samples = np.std(samples, axis=0)
            #
            d_2ds = TestLSP.d_2d[0, 0]
            d_3ds = TestLSP.d_3d
            w = TestLSP.rma_w
            h = TestLSP.rma_h
            samples_ref = np.array(
                [
                    pathloss(model, submodel, d_2d, d_3d, fc, h_bs, h_ut, h, w)
                    for d_2d, d_3d in zip(d_2ds, d_3ds)
                ]
            )
            #
            max_err = np.max(np.abs(mean_samples - samples_ref))
            self.assertLessEqual(
                max_err, TestLSP.MAX_ERR_PATHLOSS_MEAN, f"{model}:{submodel}"
            )
            max_err = np.max(np.abs(std_samples - pathloss_std(model, submodel)))
            self.assertLessEqual(
                max_err, TestLSP.MAX_ERR_PATHLOSS_STD, f"{model}:{submodel}"
            )
        elif model == "uma":
            samples = TestLSP.pathlosses[model][submodel]
            mean_samples = np.mean(samples, axis=0)
            std_samples = np.std(samples, axis=0)
            #
            d_2ds = TestLSP.d_2d[0, 0]
            d_3ds = TestLSP.d_3d
            samples_ref = np.array(
                [
                    pathloss(model, submodel, d_2d, d_3d, fc, h_bs, h_ut)
                    for d_2d, d_3d in zip(d_2ds, d_3ds)
                ]
            )
            #
            max_err = np.max(np.abs(mean_samples - samples_ref))
            self.assertLessEqual(
                max_err, TestLSP.MAX_ERR_PATHLOSS_MEAN, f"{model}:{submodel}"
            )
            max_err = np.max(np.abs(std_samples - pathloss_std(model, submodel)))
            self.assertLessEqual(
                max_err, TestLSP.MAX_ERR_PATHLOSS_STD, f"{model}:{submodel}"
            )

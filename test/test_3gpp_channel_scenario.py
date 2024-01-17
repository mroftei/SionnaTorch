#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import torch
from channel_test_utils import generate_random_loc
from sionna_torch.SionnaScenario import SionnaScenario

class TestScenario(unittest.TestCase):
    r"""Test the distance calculations and function that get the parameters
    according to the scenario.
    """

    # Batch size used to check the LSP distribution
    BATCH_SIZE = 100

    # Carrier frequency
    CARRIER_FREQUENCY = 3.5e9 # Hz

    # Maximum allowed deviation for distance calculation (relative error)
    MAX_ERR = 1e-2

    # Heigh of UTs
    H_UT = 1.5

    # Heigh of BSs
    H_BS = 10.0

    # Number of BS
    NB_BS = 3

    # Number of UT
    NB_UT = 10

    # Map square resolution
    MAP_RES = 2000

    def setUpClass():

        # Forcing the seed to make the tests deterministic
        seed = 43
        torch.manual_seed(seed)
        rng = torch.Generator().manual_seed(seed)
        np.random.seed(seed)

        batch_size = TestScenario.BATCH_SIZE
        nb_bs = TestScenario.NB_BS
        nb_ut = TestScenario.NB_UT
        fc = TestScenario.CARRIER_FREQUENCY
        h_ut = TestScenario.H_UT
        h_bs = TestScenario.H_BS

        # ut_velocities = np.zeros([batch_size, nb_ut])
        scen_map = np.ones([TestScenario.MAP_RES, TestScenario.MAP_RES], dtype=int)

        ut_loc = generate_random_loc(batch_size, nb_ut, (100,TestScenario.MAP_RES),
                                     (100,TestScenario.MAP_RES), (h_ut, h_ut))
        bs_loc = generate_random_loc(batch_size, nb_bs, (0,100),
                                            (0,100), (h_bs, h_bs))

        TestScenario.scenario = SionnaScenario(n_bs=nb_bs, n_ut=nb_ut, batch_size=batch_size, f_c=fc, seed=seed)
        TestScenario.scenario.update_topology(ut_loc, bs_loc, scen_map, map_resolution=20.0)

    def test_dist(self):
        """Test calculation of distances (total, in, and out)"""
        d_3d = self.scenario.distance_3d.numpy()
        d_2d = self.scenario.distance_2d.numpy()
        d_vect = self.scenario.distances.numpy()[...,-1]
        # Checking total 3D distances
        ut_loc = self.scenario.ut_xy.numpy()
        bs_loc = self.scenario.bs_xy.numpy()
        bs_loc = np.expand_dims(bs_loc, axis=2)
        ut_loc = np.expand_dims(ut_loc, axis=1)
        d_3d_ref = np.sqrt(np.sum(np.square(ut_loc-bs_loc), axis=3))
        max_err = np.max(np.abs(d_3d - d_3d_ref)/d_3d_ref)
        self.assertLessEqual(max_err, TestScenario.MAX_ERR)
        # Checking total 2D distances
        ut_loc = self.scenario.ut_xy.numpy()
        bs_loc = self.scenario.bs_xy.numpy()
        bs_loc = np.expand_dims(bs_loc, axis=2)
        ut_loc = np.expand_dims(ut_loc, axis=1)
        d_2d_ref = np.sqrt(np.sum(np.square(ut_loc[:,:,:,:2]-bs_loc[:,:,:,:2]), axis=3))
        max_err = np.max(np.abs(d_2d - d_2d_ref)/d_2d_ref)
        self.assertLessEqual(max_err, TestScenario.MAX_ERR)
        # Check distance vector calculation
        max_err = np.max(np.abs(d_vect - d_2d_ref)/d_2d_ref)
        self.assertLessEqual(max_err, TestScenario.MAX_ERR)

    def test_get_param(self):
        """Test the get_param() function"""
        # Test if muDSc is correctly extracted from the file (RMa)
        param_tensor_ref = np.zeros([TestScenario.BATCH_SIZE,
                                        TestScenario.NB_BS, TestScenario.NB_UT])
        los_index = np.where(self.scenario.is_los.numpy())
        nlos_index = np.where(np.logical_not(self.scenario.is_los.numpy()))
        param_tensor_ref[los_index] = 2.5
        param_tensor_ref[nlos_index] = 2.3
        #
        param_tensor = TestScenario.scenario.get_param('rTau').numpy()
        max_err = np.max(np.abs(param_tensor-param_tensor_ref))
        self.assertLessEqual(max_err, 1e-6)

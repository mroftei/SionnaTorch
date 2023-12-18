#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
from channel_test_utils import *
from sionna_torch.RMAScenario import RMaScenario
from sionna_torch.UMAScenario import UMaScenario

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

        # The following quantities have no impact on LSP
        # However,these are needed to instantiate the model
        ut_orientations = np.zeros([batch_size, nb_ut])
        bs_orientations = np.zeros([batch_size, nb_ut])
        ut_velocities = np.zeros([batch_size, nb_ut])

        TestScenario.scenario = RMaScenario(fc, "uplink", rng=rng)

        ut_loc = generate_random_loc(batch_size, nb_ut, (100,2000),
                                     (100,2000), (h_ut, h_ut))
        bs_loc = generate_random_loc(batch_size, nb_bs, (0,100),
                                            (0,100), (h_bs, h_bs))

        TestScenario.scenario.set_topology(ut_loc, bs_loc, ut_orientations,
                                bs_orientations, ut_velocities)

    def test_dist(self):
        """Test calculation of distances (total, in, and out)"""
        d_3d = self.scenario.distance_3d.numpy()
        d_2d = self.scenario.distance_2d.numpy()
        # Checking total 3D distances
        ut_loc = self.scenario.ut_loc.numpy()
        bs_loc = self.scenario.bs_loc.numpy()
        bs_loc = np.expand_dims(bs_loc, axis=2)
        ut_loc = np.expand_dims(ut_loc, axis=1)
        d_3d_ref = np.sqrt(np.sum(np.square(ut_loc-bs_loc), axis=3))
        max_err = np.max(np.abs(d_3d - d_3d_ref)/d_3d_ref)
        self.assertLessEqual(max_err, TestScenario.MAX_ERR)
        # Checking total 2D distances
        ut_loc = self.scenario.ut_loc.numpy()
        bs_loc = self.scenario.bs_loc.numpy()
        bs_loc = np.expand_dims(bs_loc, axis=2)
        ut_loc = np.expand_dims(ut_loc, axis=1)
        d_2d_ref = np.sqrt(np.sum(np.square(ut_loc[:,:,:,:2]-bs_loc[:,:,:,:2]), axis=3))
        max_err = np.max(np.abs(d_2d - d_2d_ref)/d_2d_ref)
        self.assertLessEqual(max_err, TestScenario.MAX_ERR)

    def test_get_param(self):
        """Test the get_param() function"""
        # Test if muDSc is correctly extracted from the file (RMa)
        param_tensor_ref = np.zeros([TestScenario.BATCH_SIZE,
                                        TestScenario.NB_BS, TestScenario.NB_UT])
        los_index = np.where(self.scenario.los.numpy())
        nlos_index = np.where(np.logical_not(self.scenario.los.numpy()))
        param_tensor_ref[los_index] = -7.49
        param_tensor_ref[nlos_index] = -7.43
        #
        param_tensor = self.scenario.get_param('muDSc').numpy()
        max_err = np.max(np.abs(param_tensor-param_tensor_ref))
        self.assertLessEqual(max_err, 1e-6)

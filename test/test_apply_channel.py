#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import unittest
import numpy as np
import torch
from sionna_torch.ApplyTimeChannel import ApplyTimeChannel

# TODO: snr test

class TestApplyTimeChannel(unittest.TestCase):

    def test_apply_time_channel(self):
        # Forcing the seed to make the tests deterministic
        seed = 43
        torch.manual_seed(seed)
        rng = torch.Generator().manual_seed(seed)
        np.random.seed(seed)

        batch_size = 16
        num_rx = 4
        num_rx_ant = 4
        num_tx = 2
        num_tx_ant = 2
        NUM_TIME_SAMPLES = [1, 5, 32, 128]
        L_TOT = [1, 3, 8, 16]
        for num_time_samples in NUM_TIME_SAMPLES:
            for l_tot in L_TOT:
                apply = ApplyTimeChannel(num_time_samples, l_tot, rng=rng, add_awgn=False)
                x = torch.randn([batch_size,
                                      num_tx,
                                      num_tx_ant,
                                      num_time_samples])
                h_time = torch.randn([batch_size,
                                           num_rx,
                                           num_rx_ant,
                                           num_tx,
                                           num_tx_ant,
                                           num_time_samples+l_tot-1,
                                           l_tot])
                y, snr = apply(x, h_time)
                self.assertEqual(y.shape, (batch_size,
                                           num_rx,
                                           num_rx_ant,
                                           num_time_samples+l_tot-1))
                y_ref = np.zeros([batch_size,
                                  num_rx,
                                  num_rx_ant,
                                  num_time_samples+l_tot-1], dtype=np.complex64)
                h_time = h_time.numpy()
                x = x.numpy()
                for b in np.arange(batch_size):
                    for rx in np.arange(num_rx):
                        for ra in np.arange(num_rx_ant):
                            for t in np.arange(num_time_samples+l_tot-1):
                                h_ = h_time[b,rx,ra,:,:,t,:]
                                x_ = x[b]
                                for l in np.arange(l_tot):
                                    if t-l < 0:
                                        break
                                    if t-l > num_time_samples-1:
                                        continue
                                    y_ref[b,rx,ra,t] += np.sum(x_[:,:,t-l]*h_[:,:,l])
                self.assertTrue(np.allclose(y_ref, y, atol=1e-5))

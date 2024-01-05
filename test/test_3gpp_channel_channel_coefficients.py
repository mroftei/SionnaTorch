#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
from channel_test_utils import *
from sionna_torch.LSPGenerator import LSPGenerator
from sionna_torch.RaysGenerator import RaysGenerator
from sionna_torch.ChannelCoefficients import ChannelCoefficientsGenerator
from sionna_torch.SionnaScenario import SionnaScenario


class TestChannelCoefficientsGenerator(unittest.TestCase):
    r"""Test the computation of channel coefficients"""

    # Batch size used to check the LSP distribution
    BATCH_SIZE = 32

    # Carrier frequency
    CARRIER_FREQUENCY = 3.5e9 # Hz

    # Maximum allowed deviation for calculation (relative error)
    MAX_ERR = 1e-2

    # # Heigh of UTs
    H_UT = 1.5

    # # Heigh of BSs
    H_BS = 10.0

    # # Number of BS
    NB_BS = 3

    # Number of UT
    NB_UT = 10

    # Number of channel time samples
    NUM_SAMPLES = 64

    # Sampling frequency
    SAMPLING_FREQUENCY = 20e6

    # Map square resolution
    MAP_RES = 2000

    def setUpClass():
        dev = torch.device('cpu') 

        # Forcing the seed to make the tests deterministic
        seed = 43
        torch.manual_seed(seed)
        rng = torch.Generator(device=dev).manual_seed(seed)
        np.random.seed(seed)

        fc = TestChannelCoefficientsGenerator.CARRIER_FREQUENCY

        ccg = ChannelCoefficientsGenerator(
            fc,
            subclustering=True,
            rng=rng,
            dtype=torch.complex128,
            device=dev)
        TestChannelCoefficientsGenerator.ccg = ccg

        batch_size = TestChannelCoefficientsGenerator.BATCH_SIZE
        nb_ut = TestChannelCoefficientsGenerator.NB_UT
        nb_bs = TestChannelCoefficientsGenerator.NB_BS
        h_ut = TestChannelCoefficientsGenerator.H_UT
        h_bs = TestChannelCoefficientsGenerator.H_BS

        # rx_orientations = torch.empty((batch_size, nb_ut, 3), dtype=torch.float64).uniform_(0.0, 2*np.pi)
        # tx_orientations = torch.empty((batch_size, nb_bs, 3), dtype=torch.float64).uniform_(0.0, 2*np.pi)
        # ut_velocities = torch.empty((batch_size, nb_ut, 3), dtype=torch.float64).uniform_(0.0, 5.0)
        scen_map = np.zeros([TestChannelCoefficientsGenerator.MAP_RES, TestChannelCoefficientsGenerator.MAP_RES], dtype=int)

        ut_loc = generate_random_loc(batch_size, nb_ut, (100,TestChannelCoefficientsGenerator.MAP_RES),
                                     (100,TestChannelCoefficientsGenerator.MAP_RES), (h_ut, h_ut), dtype=torch.float64)
        bs_loc = generate_random_loc(batch_size, nb_bs, (0,100),
                                            (0,100), (h_bs, h_bs),
                                            dtype=torch.float64)

        scenario = SionnaScenario(ut_loc, bs_loc, scen_map, f_c=fc, seed=seed, dtype=torch.complex128, device=dev)
        TestChannelCoefficientsGenerator.scenario = scenario

        lsp_sampler = LSPGenerator(scenario, rng=rng)
        ray_sampler = RaysGenerator(scenario, rng=rng)
        lsp_sampler.topology_updated_callback()
        ray_sampler.topology_updated_callback()
        lsp = lsp_sampler()
        rays = ray_sampler(lsp)
        TestChannelCoefficientsGenerator.rays = rays
        TestChannelCoefficientsGenerator.lsp = lsp

        num_time_samples = TestChannelCoefficientsGenerator.NUM_SAMPLES
        sampling_frequency = TestChannelCoefficientsGenerator.SAMPLING_FREQUENCY
        c_ds = scenario.get_param("cDS")*1e-9
        if scenario.direction == "uplink":
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
        _, _, phi, sample_times = ccg(num_time_samples,
            sampling_frequency, lsp.k_factor, rays, scenario, c_ds,
            debug=True)
        TestChannelCoefficientsGenerator.phi = phi
        TestChannelCoefficientsGenerator.sample_times = sample_times
        TestChannelCoefficientsGenerator.c_ds = c_ds

    def max_rel_err(self, r, x):
        """Compute the maximum relative error, ``r`` being the reference value,
        ``x`` an esimate of ``r``."""
        err = np.abs(r-x)
        rel_err = np.where(np.abs(r) > 0.0, np.divide(err,np.abs(r)+1e-6), err)
        return np.max(rel_err)

    def unit_sphere_vector_ref(self, theta, phi):
        """Reference implementation: Unit to sphere vector"""
        uvec = np.stack([np.sin(theta)*np.cos(phi),
                            np.sin(theta)*np.sin(phi), np.cos(theta)],
                            axis=-1)
        uvec = np.expand_dims(uvec, axis=-1)
        return uvec

    def test_unit_sphere_vector(self):
        """Test 3GPP channel coefficient calculation: Unit sphere vector"""
        #
        batch_size = TestChannelCoefficientsGenerator.BATCH_SIZE
        theta = torch.randn([batch_size])
        phi = torch.randn([batch_size])
        uvec_ref = self.unit_sphere_vector_ref(theta.numpy(), phi.numpy())
        uvec = self.ccg._unit_sphere_vector(theta, phi).numpy()
        max_err = self.max_rel_err(uvec_ref, uvec)
        err_tol = TestChannelCoefficientsGenerator.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def forward_rotation_matrix_ref(self, orientations):
        """Reference implementation: Forward rotation matrix"""
        a, b, c = orientations[...,0], orientations[...,1], orientations[...,2]
        #
        R = np.zeros(list(a.shape) + [3,3])
        #
        R[...,0,0] = np.cos(a)*np.cos(b)
        R[...,1,0] = np.sin(a)*np.cos(b)
        R[...,2,0] = -np.sin(b)
        #
        R[...,0,1] = np.cos(a)*np.sin(b)*np.sin(c) - np.sin(a)*np.cos(c)
        R[...,1,1] = np.sin(a)*np.sin(b)*np.sin(c) + np.cos(a)*np.cos(c)
        R[...,2,1] = np.cos(b)*np.sin(c)
        #
        R[...,0,2] = np.cos(a)*np.sin(b)*np.cos(c) + np.sin(a)*np.sin(c)
        R[...,1,2] = np.sin(a)*np.sin(b)*np.cos(c) - np.cos(a)*np.sin(c)
        R[...,2,2] = np.cos(b)*np.cos(c)
        #
        return R

    def test_forward_rotation_matrix(self):
        """Test 3GPP channel coefficient calculation: Forward rotation matrix"""
        batch_size = TestChannelCoefficientsGenerator.BATCH_SIZE
        orientation = torch.randn([batch_size,3])
        R_ref = self.forward_rotation_matrix_ref(orientation.numpy())
        R = self.ccg._forward_rotation_matrix(orientation).numpy()
        max_err = self.max_rel_err(R_ref, R)
        err_tol = TestChannelCoefficientsGenerator.MAX_ERR
        self.assertLessEqual(max_err, err_tol)


    def reverse_rotation_matrix_ref(self, orientations):
        """Reference implementation: Reverse rotation matrix"""
        R = self.forward_rotation_matrix_ref(orientations)
        dim_ind = np.arange(len(R.shape))
        dim_ind = np.concatenate([dim_ind[:-2], [dim_ind[-1]], [dim_ind[-2]]],
                                    axis=0)
        R_inv = np.transpose(R, dim_ind)
        return R_inv

    def test_reverse_rotation_matrix(self):
        """Test 3GPP channel coefficient calculation: Reverse rotation matrix"""
        batch_size = TestChannelCoefficientsGenerator.BATCH_SIZE
        orientation = torch.randn([batch_size,3])
        R_ref = self.reverse_rotation_matrix_ref(orientation.numpy())
        R = self.ccg._reverse_rotation_matrix(orientation).numpy()
        max_err = self.max_rel_err(R_ref, R)
        err_tol = TestChannelCoefficientsGenerator.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def gcs_to_lcs_ref(self, orientations, theta, phi):
        """Reference implementation: GCS to LCS angles"""
        rho = self.unit_sphere_vector_ref(theta, phi)
        Rinv = self.reverse_rotation_matrix_ref(orientations)
        rho_prime = Rinv@rho

        x = np.array([1,0,0])
        x = np.expand_dims(x, axis=-1)
        x = np.broadcast_to(x, rho_prime.shape)

        y = np.array([0,1,0])
        y = np.expand_dims(y, axis=-1)
        y = np.broadcast_to(y, rho_prime.shape)

        z = np.array([0,0,1])
        z = np.expand_dims(z, axis=-1)
        z = np.broadcast_to(z, rho_prime.shape)

        theta_prime = np.sum(rho_prime*z, axis=-2)
        theta_prime = np.clip(theta_prime, -1., 1.)
        theta_prime = np.arccos(theta_prime)
        phi_prime = np.angle(np.sum(rho_prime*x, axis=-2)\
            + 1j*np.sum(rho_prime*y, axis=-2))

        theta_prime = np.squeeze(theta_prime, axis=-1)
        phi_prime = np.squeeze(phi_prime, axis=-1)

        return (theta_prime, phi_prime)

    def test_gcs_to_lcs(self):
        """Test 3GPP channel coefficient calculation: GCS to LCS"""
        batch_size = TestChannelCoefficientsGenerator.BATCH_SIZE
        orientation = torch.randn([batch_size,3], device=self.scenario.device)
        theta = torch.randn([batch_size], device=self.scenario.device)
        phi = torch.randn([batch_size], device=self.scenario.device)

        theta_prime_ref, phi_prime_ref = self.gcs_to_lcs_ref(orientation.numpy(force=True), theta.numpy(force=True),
                                                            phi.numpy(force=True))
        theta_prime, phi_prime = self.ccg._gcs_to_lcs(
            orientation.type(torch.float64),
            theta.type(torch.float64),
            phi.type(torch.float64))
        theta_prime = theta_prime.numpy(force=True)
        phi_prime = phi_prime.numpy(force=True)

        err_tol = TestChannelCoefficientsGenerator.MAX_ERR
        max_err = self.max_rel_err(theta_prime_ref, theta_prime)
        self.assertLessEqual(max_err, err_tol)
        max_err = self.max_rel_err(phi_prime_ref, phi_prime)
        self.assertLessEqual(max_err, err_tol)

    def compute_psi_ref(self, orientations, theta, phi):
        """Reference implementation: Compute psi angle"""
        a = orientations[...,0]
        b = orientations[...,1]
        c = orientations[...,2]

        real = np.sin(c)*np.cos(theta)*np.sin(phi-a)\
            + np.cos(c)*(np.cos(b)*np.sin(theta)\
                -np.sin(b)*np.cos(theta)*np.cos(phi-a))
        imag = np.sin(c)*np.cos(phi-a) + np.sin(b)*np.cos(c)*np.sin(phi-a)
        return np.angle(real+1j*imag)

    def l2g_response_ref(self, F_prime, orientations, theta, phi):
        """Reference implementation: L2G response"""

        psi = self.compute_psi_ref(orientations, theta, phi)

        mat = np.zeros(list(np.shape(psi)) + [2,2])
        mat[...,0,0] = np.cos(psi)
        mat[...,0,1] = -np.sin(psi)
        mat[...,1,0] = np.sin(psi)
        mat[...,1,1] = np.cos(psi)

        F = mat@np.expand_dims(F_prime, axis=-1)
        return F

    def test_l2g_response(self):
        """Test 3GPP channel coefficient calculation: L2G antenna response"""
        batch_size = TestChannelCoefficientsGenerator.BATCH_SIZE
        orientation = torch.randn([batch_size,3])
        theta = torch.randn([batch_size])
        phi = torch.randn([batch_size])
        F_prime = torch.randn([batch_size,2])

        F_ref = self.l2g_response_ref(F_prime.numpy(), orientation.numpy(), theta.numpy(), phi.numpy())
        F = self.ccg._l2g_response( F_prime.type(torch.float64),
                                    orientation.type(torch.float64),
                                    theta.type(torch.float64),
                                    phi.type(torch.float64)).numpy()

        max_err = self.max_rel_err(F_ref, F)
        err_tol = TestChannelCoefficientsGenerator.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def rot_pos_ref(self, orientations, positions):
        R = self.forward_rotation_matrix_ref(orientations)
        pos_r = R@positions
        return pos_r

    def rot_pos(self, orientations, positions):
        """Reference implementation: Rotate according to an orientation"""
        R = self.forward_rotation_matrix_ref(orientations)
        pos_r = R@positions
        return pos_r

    def test_rot_pos(self):
        """Test 3GPP channel coefficient calculation: Rotate position according
        to orientation"""
        batch_size = TestChannelCoefficientsGenerator.BATCH_SIZE
        orientations = torch.randn([batch_size,3])
        positions = torch.randn([batch_size,3, 1])

        pos_r_ref = self.rot_pos_ref(orientations.numpy(), positions.numpy())
        pos_r = self.ccg._rot_pos(  orientations.type(torch.float64),
                                    positions.type(torch.float64)).numpy()
        max_err = self.max_rel_err(pos_r_ref, pos_r)
        err_tol = TestChannelCoefficientsGenerator.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def step_11_get_tx_antenna_positions_ref(self, scenario):
        """Reference implementation: Positions of the TX array elements"""

        if scenario.direction == "uplink":
            tx_orientations = torch.zeros(scenario.batch_size, scenario.num_ut, 3, dtype=scenario._dtype_real).numpy() # [batch size, number of UTs, 3]
        else:
            tx_orientations = torch.zeros(scenario.batch_size, scenario.num_bs, 3, dtype=scenario._dtype_real).numpy() # [batch size, number of BSs, 3]

        # Antenna locations in LCS and reshape for broadcasting
        ant_loc_lcs = np.array([[0.,0.,0.]])
        ant_loc_lcs = np.expand_dims(np.expand_dims(
            np.expand_dims(ant_loc_lcs, axis=0), axis=1), axis=-1)

        # Antenna loc in GCS relative to BS location
        tx_orientations = np.expand_dims(tx_orientations, axis=2)
        ant_loc_gcs = np.squeeze(self.rot_pos_ref(tx_orientations, ant_loc_lcs),
                                 axis=-1)

        return ant_loc_gcs

    def test_step_11_get_tx_antenna_positions(self):
        """Test 3GPP channel coefficient calculation: Positions of the TX array
        elements"""
        tx_ant_pos_ref= self.step_11_get_tx_antenna_positions_ref(self.scenario)
        tx_ant_pos = self.ccg._step_11_get_tx_antenna_positions(self.scenario).numpy(force=True)
        tx_ant_pos = tx_ant_pos
        max_err = self.max_rel_err(tx_ant_pos_ref, tx_ant_pos)
        err_tol = TestChannelCoefficientsGenerator.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def step_11_get_rx_antenna_positions_ref(self, scenario):
        """Reference implementation: Positions of the RX array elements"""

        if scenario.direction == "uplink":
            rx_orientations = torch.zeros(scenario.batch_size, scenario.num_bs, 3, dtype=scenario._dtype_real).numpy() # [batch size, number of BSs, 3]
        else:
            rx_orientations = torch.zeros(scenario.batch_size, scenario.num_ut, 3, dtype=scenario._dtype_real).numpy() # [batch size, number of UTs, 3]

        # Antenna locations in LCS and reshape for broadcasting
        ant_loc_lcs = np.array([[0.,0.,0.]])
        ant_loc_lcs = np.expand_dims(np.expand_dims(
            np.expand_dims(ant_loc_lcs, axis=0), axis=1), axis=-1)

        # Antenna loc in GCS relative to UT location
        rx_orientations = np.expand_dims(rx_orientations, axis=2)
        ant_loc_gcs = np.squeeze(self.rot_pos_ref(rx_orientations, ant_loc_lcs),
                                    axis=-1)

        return ant_loc_gcs

    def test_step_11_get_rx_antenna_positions(self):
        """Test 3GPP channel coefficient calculation: Positions of the RX array
        elements"""
        rx_ant_pos_ref= self.step_11_get_rx_antenna_positions_ref(self.scenario)
        rx_ant_pos = self.ccg._step_11_get_rx_antenna_positions(self.scenario).numpy(force=True)
        rx_ant_pos = rx_ant_pos
        max_err = self.max_rel_err(rx_ant_pos_ref, rx_ant_pos)
        err_tol = TestChannelCoefficientsGenerator.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def step_11_phase_matrix_ref(self, Phi, kappa):
        """Reference implementation: Phase matrix"""
        xpr_scaling = np.sqrt(1./kappa)
        H_phase = np.zeros(list(Phi.shape[:-1]) + [2,2])\
            +1j*np.zeros(list(Phi.shape[:-1]) + [2,2])
        H_phase[...,0,0] = np.exp(1j*Phi[...,0])
        H_phase[...,0,1] = xpr_scaling*np.exp(1j*Phi[...,1])
        H_phase[...,1,0] = xpr_scaling*np.exp(1j*Phi[...,2])
        H_phase[...,1,1] = np.exp(1j*Phi[...,3])
        return H_phase

    def test_step_11_phase_matrix(self):
        """Test 3GPP channel coefficient calculation:
        Phase matrix calculation"""
        H_phase_ref = self.step_11_phase_matrix_ref(self.phi.numpy(force=True), self.rays.xpr.numpy(force=True))
        H_phase = self.ccg._step_11_phase_matrix(self.phi, self.rays).numpy(force=True)
        max_err = self.max_rel_err(H_phase_ref, H_phase)
        err_tol = TestChannelCoefficientsGenerator.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def step_11_field_matrix_ref(self, scenario, aoa, aod, zoa, zod, H_phase):
        """Reference implementation: Field matrix"""

        if scenario.direction == "uplink":
            tx_orientations = torch.zeros(scenario.batch_size, scenario.num_ut, 3).numpy() # [batch size, number of UTs, 3]
            rx_orientations = torch.zeros(scenario.batch_size, scenario.num_bs, 3).numpy() # [batch size, number of BSs, 3]
        else:
            rx_orientations = torch.zeros(scenario.batch_size, scenario.num_ut, 3).numpy() # [batch size, number of UTs, 3]
            tx_orientations = torch.zeros(scenario.batch_size, scenario.num_bs, 3).numpy() # [batch size, number of BSs, 3]
        tx_num_ant = 1
        rx_num_ant = 1

        # Convert departure angles to LCS
        tx_orientations = np.expand_dims(np.expand_dims(
            np.expand_dims(tx_orientations, axis=2), axis=2), axis=2)
        zod_prime, aod_prime = self.gcs_to_lcs_ref(tx_orientations, zod, aod)

        # Convert arrival angles to LCS
        rx_orientations = np.expand_dims(np.expand_dims(
            np.expand_dims(rx_orientations, axis=1), axis=3), axis=3)
        zoa_prime, aoa_prime = self.gcs_to_lcs_ref(rx_orientations, zoa, aoa)

        # Compute the TX antenna reponse in LCS and map it to GCS
        # Unity antenna
        F_tx_prime_pol1_1 = np.ones_like(zod_prime, dtype=np.float64)
        F_tx_prime_pol1_2 = np.zeros_like(aod_prime, dtype=np.float64)
        F_tx_prime_pol1 = np.stack([F_tx_prime_pol1_1, F_tx_prime_pol1_2],
            axis=-1)
        F_tx_pol1 = self.l2g_response_ref(F_tx_prime_pol1, tx_orientations,
                                            zod, aod)

        # Compute the RX antenna reponse in LCS and map it to GCS
        F_rx_prime_pol1_1 = np.ones_like(zoa_prime, dtype=np.float64)
        F_rx_prime_pol1_2 = np.zeros_like(aoa_prime, dtype=np.float64)
        F_rx_prime_pol1 = np.stack([F_rx_prime_pol1_1, F_rx_prime_pol1_2],
            axis=-1)
        F_rx_pol1 = self.l2g_response_ref(F_rx_prime_pol1, rx_orientations,
            zoa, aoa)

        # Compute prtoduct between the phase matrix and the TX antenna field.
        F_tx_pol1 = H_phase@F_tx_pol1

        # TX: Scatteing the antenna response
        # Single polarization case is easy, as one only needs to repeat the same
        # antenna response for all elements
        F_tx_pol1 = np.expand_dims(np.squeeze(F_tx_pol1, axis=-1), axis=-2)
        F_tx = np.tile(F_tx_pol1, [1,1,1,1,1,tx_num_ant,1])

        # RX: Scatteing the antenna response
        # Single polarization case is easy, as one only needs to repeat the same
        # antenna response for all elements
        F_rx_pol1 = np.expand_dims(np.squeeze(F_rx_pol1, axis=-1), axis=-2)
        F_rx = np.tile(F_rx_pol1, [1,1,1,1,1,rx_num_ant,1])
        # Computing H_field
        F_tx = np.expand_dims(F_tx, axis=-3)
        F_rx = np.expand_dims(F_rx, axis=-2)
        H_field = np.sum(F_tx*F_rx, axis=-1)
        return H_field

    def test_step_11_field_matrix(self):
        """Test 3GPP channel coefficient calculation:
        Field matrix calculation"""
        H_phase = self.step_11_phase_matrix_ref(self.phi.numpy(force=True), self.rays.xpr.numpy(force=True))
        H_field_ref = self.step_11_field_matrix_ref(self.scenario,
                                                    self.rays.aoa.numpy(force=True),
                                                    self.rays.aod.numpy(force=True),
                                                    self.rays.zoa.numpy(force=True),
                                                    self.rays.zod.numpy(force=True),
                                                    H_phase)

        H_field = self.ccg._step_11_field_matrix(self.scenario,
                                    self.rays.aoa.type(torch.float64),
                                    self.rays.aod.type(torch.float64),
                                    self.rays.zoa.type(torch.float64),
                                    self.rays.zod.type(torch.float64),
                                    torch.from_numpy(H_phase).to(self.scenario.device)).numpy(force=True)
        max_err = self.max_rel_err(H_field_ref, H_field)
        err_tol = TestChannelCoefficientsGenerator.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def step_11_array_offsets_ref(self, aoa, aod, zoa, zod, scenario):
        """Reference implementation: Array offset matrix"""

        # Arrival spherical unit vector
        r_hat_rx = np.squeeze(self.unit_sphere_vector_ref(zoa, aoa), axis=-1)
        r_hat_rx = np.expand_dims(r_hat_rx, axis=-2)

        # Departure spherical unit vector
        r_hat_tx =  np.squeeze(self.unit_sphere_vector_ref(zod, aod), axis=-1)
        r_hat_tx = np.expand_dims(r_hat_tx, axis=-2)

        # TX location vector
        d_bar_tx = self.step_11_get_tx_antenna_positions_ref(scenario)
        d_bar_tx = np.expand_dims(np.expand_dims(
            np.expand_dims(d_bar_tx, axis=2), axis=3), axis=4)

        # RX location vector
        d_bar_rx = self.step_11_get_rx_antenna_positions_ref(scenario)
        d_bar_rx = np.expand_dims(np.expand_dims(
                np.expand_dims(d_bar_rx, axis=1), axis=3), axis=4)

        lambda_0 = self.scenario.lambda_0

        # TX offset matrix

        tx_offset = np.sum(r_hat_tx*d_bar_tx, axis=-1)
        rx_offset = np.sum(r_hat_rx*d_bar_rx, axis=-1)

        tx_offset = np.expand_dims(tx_offset, -2)
        rx_offset = np.expand_dims(rx_offset, -1)
        antenna_offset = np.exp(1j*2*np.pi*(tx_offset+rx_offset)/lambda_0)

        return antenna_offset

    def test_step_11_array_offsets(self):
        """Test 3GPP channel coefficient calculation: Array offset matrix"""
        H_array_ref = self.step_11_array_offsets_ref(self.rays.aoa.numpy(force=True),
                                                     self.rays.aod.numpy(force=True),
                                                     self.rays.zoa.numpy(force=True),
                                                     self.rays.zod.numpy(force=True),
                                                     self.scenario)

        H_array = self.ccg._step_11_array_offsets(self.scenario,
                                self.rays.aoa.type(torch.float64),
                                self.rays.aod.type(torch.float64),
                                self.rays.zoa.type(torch.float64),
                                self.rays.zod.type(torch.float64)).numpy(force=True)

        max_err = self.max_rel_err(H_array_ref, H_array)
        err_tol = TestChannelCoefficientsGenerator.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def step_11_doppler_matrix_ref(self, scenario, aoa, zoa, t):
        """Reference implementation: Array offset matrix"""

        velocities = scenario.ut_velocities.numpy(force=True)

        lambda_0 = self.scenario.lambda_0

        # Arrival spherical unit vector
        r_hat_rx = np.squeeze(self.unit_sphere_vector_ref(zoa, aoa), axis=-1)

        # Velocity vec
        if scenario.direction == "downlink":
            velocities = np.expand_dims(velocities, axis=1)
        elif scenario.direction == 'uplink':
            velocities = np.expand_dims(velocities, axis=2)
        velocities = np.expand_dims(np.expand_dims(velocities, axis=3), axis=4)

        # Doppler matrix
        exponent = np.sum(r_hat_rx*velocities, axis=-1, keepdims=True)
        exponent = exponent/lambda_0
        exponent = 2*np.pi*exponent*t
        H_doppler = np.exp(1j*exponent)

        return H_doppler

    def test_step_11_doppler_matrix(self):
        """Test 3GPP channel coefficient calculation: Doppler matrix"""
        H_doppler_ref = self.step_11_doppler_matrix_ref(self.scenario,
                                                        self.rays.aoa.numpy(force=True),
                                                        self.rays.zoa.numpy(force=True),
                                                        self.sample_times.numpy(force=True))

        H_doppler = self.ccg._step_11_doppler_matrix(self.scenario,
                            self.rays.aoa.type(torch.float64),
                            self.rays.zoa.type(torch.float64),
                            self.sample_times.type(torch.float64)).numpy(force=True)

        max_err = self.max_rel_err(H_doppler_ref, H_doppler)
        err_tol = TestChannelCoefficientsGenerator.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def step_11_nlos_ref(self, phi, aoa, aod, zoa, zod, kappa, powers, t,
        scenario):
        """Reference implemenrtation: Compute the channel matrix of the NLoS
        component"""

        H_phase = self.step_11_phase_matrix_ref(phi, kappa)
        H_field = self.step_11_field_matrix_ref(scenario, aoa, aod, zoa, zod,
                                                H_phase)
        H_array = self.step_11_array_offsets_ref(aoa, aod, zoa, zod, scenario)
        H_doppler = self.step_11_doppler_matrix_ref(scenario, aoa, zoa, t)

        H_field = np.expand_dims(H_field, axis=-1)
        H_array = np.expand_dims(H_array, axis=-1)
        H_doppler = np.expand_dims(np.expand_dims(H_doppler, axis=-2), axis=-3)

        H_full = H_field*H_array*H_doppler

        power_scaling = np.sqrt(powers/aoa.shape[4])
        power_scaling = np.expand_dims(np.expand_dims(np.expand_dims(
                np.expand_dims(power_scaling, axis=4), axis=5), axis=6), axis=7)

        H_full = H_full*power_scaling

        return H_full

    def test_step_11_nlos_ref(self):
        """Test 3GPP channel coefficient calculation: Doppler matrix"""
        H_full_ref = self.step_11_nlos_ref( self.phi.numpy(force=True),
                                            self.rays.aoa.numpy(force=True),
                                            self.rays.aod.numpy(force=True),
                                            self.rays.zoa.numpy(force=True),
                                            self.rays.zod.numpy(force=True),
                                            self.rays.xpr.numpy(force=True),
                                            self.rays.powers.numpy(force=True),
                                            self.sample_times.numpy(force=True),
                                            self.scenario)

        H_full = self.ccg._step_11_nlos(self.phi.type(torch.float64),
                            self.scenario,
                            self.rays,
                            self.sample_times.type(torch.float64)).numpy(force=True)
        max_err = self.max_rel_err(H_full_ref, H_full)
        err_tol = TestChannelCoefficientsGenerator.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def step_11_reduce_nlos_ref(self, H_full, powers, delays, c_DS):
        """Reference implementation: Compute the channel matrix of the NLoS
        component 2"""

        # Sorting clusters in descending roder
        cluster_ordered = np.flip(np.argsort(powers, axis=3), axis=3)

        delays_ordered = np.take_along_axis(delays, cluster_ordered, axis=3)
        H_full_ordered = np.take_along_axis(H_full, cluster_ordered[...,None,None,None,None], axis=3)

        ## Weak clusters (all except first two)
        delays_weak = delays_ordered[:,:,:,2:]
        H_full_weak = np.sum(H_full_ordered[:,:,:,2:,:,:,:], axis=4)

        ## Strong clusters (first two)
        # Each strong cluster is split into 3 sub-cluster

        # Subcluster delays
        strong_delays = delays_ordered[:,:,:,:2]
        strong_delays = np.expand_dims(strong_delays, -1)
        delays_expension = np.array([[[[[0.0, 1.28, 2.56]]]]])
        c_DS = np.expand_dims(np.expand_dims(c_DS, axis=-1), axis=-1)
        strong_delays = strong_delays + delays_expension*c_DS
        strong_delays = np.reshape(strong_delays,
                                        list(strong_delays.shape[:-2]) + [-1])

        # Subcluster coefficient
        H_full_strong = H_full_ordered[:,:,:,:2,:,:,:]
        H_full_subcl_1 = np.sum(np.take(H_full_strong, [0,1,2,3,4,5,6,7,18,19],
                                axis=4), axis=4)
        H_full_subcl_2 = np.sum(np.take(H_full_strong, [8,9,10,11,16,17],
                                axis=4), axis=4)
        H_full_subcl_3 = np.sum(np.take(H_full_strong, [12,13,14,15],
                                axis=4), axis=4)
        H_full_strong_subcl = np.stack([H_full_subcl_1,H_full_subcl_2,
                                        H_full_subcl_3], axis=3)
        H_full_strong_subcl = np.transpose(H_full_strong_subcl,
                                            [0,1,2,4,3,5,6,7])
        H_full_strong_subcl = np.reshape(H_full_strong_subcl,
            np.concatenate([H_full_strong_subcl.shape[:3], [-1],
            H_full_strong_subcl.shape[5:]], axis=0))

        ## Putting together strong and weak clusters

        H_nlos = np.concatenate([H_full_strong_subcl, H_full_weak], axis=3)
        delays_nlos = np.concatenate([strong_delays, delays_weak], axis=3)

        ## Sorting
        delays_sorted_ind = np.argsort(delays_nlos, axis=3)
        delays_nlos = np.take_along_axis(delays_nlos, delays_sorted_ind, axis=3)
        H_nlos = np.take_along_axis(H_nlos, delays_sorted_ind[...,None,None,None], axis=3)

        return (H_nlos, delays_nlos)

    def test_step_11_reduce_nlos(self):
        """Test 3GPP channel coefficient calculation: NLoS channel matrix
        computation"""

        H_full_ref = self.step_11_nlos_ref( self.phi.numpy(force=True),
                                            self.rays.aoa.numpy(force=True),
                                            self.rays.aod.numpy(force=True),
                                            self.rays.zoa.numpy(force=True),
                                            self.rays.zod.numpy(force=True),
                                            self.rays.xpr.numpy(force=True),
                                            self.rays.powers.numpy(force=True),
                                            self.sample_times.numpy(force=True),
                                            self.scenario)

        H_nlos_ref, delays_nlos_ref = self.step_11_reduce_nlos_ref(
                                                    H_full_ref,
                                                    self.rays.powers.numpy(force=True),
                                                    self.rays.delays.numpy(force=True),
                                                    self.c_ds.numpy(force=True))

        H_nlos, delays_nlos = self.ccg._step_11_reduce_nlos(
            torch.from_numpy(H_full_ref).to(self.scenario.device), self.rays, self.c_ds)
        H_nlos = H_nlos.numpy(force=True)
        delays_nlos = delays_nlos.numpy(force=True)

        err_tol = TestChannelCoefficientsGenerator.MAX_ERR
        max_err = self.max_rel_err(H_nlos_ref, H_nlos)
        self.assertLessEqual(max_err, err_tol)

        max_err = self.max_rel_err(delays_nlos_ref, delays_nlos)
        self.assertLessEqual(max_err, err_tol)

    def step_11_los_ref(self, t, scenario):
        """Reference implementation: Compute the channel matrix of the NLoS
        component 2"""

        if scenario.direction == "uplink":
            aoa = torch.permute(torch.remainder(scenario.los_aod_rad, 2*torch.pi), [0, 2, 1]).numpy(force=True)
            aod = torch.permute(torch.remainder(scenario.los_aoa_rad, 2*torch.pi), [0, 2, 1]).numpy(force=True)
            zoa = torch.permute(torch.remainder(scenario.los_zod_rad, 2*torch.pi), [0, 2, 1]).numpy(force=True)
            zod = torch.permute(torch.remainder(scenario.los_zoa_rad, 2*torch.pi), [0, 2, 1]).numpy(force=True)
        else:
            aoa = scenario.los_aoa_rad
            aod = scenario.los_aod_rad
            zoa = scenario.los_zoa_rad
            zod = scenario.los_zod_rad

        # LoS departure and arrival angles
        los_aoa = np.expand_dims(np.expand_dims(aoa,
            axis=3), axis=4)
        los_zoa = np.expand_dims(np.expand_dims(zoa,
            axis=3), axis=4)
        los_aod = np.expand_dims(np.expand_dims(aod,
            axis=3), axis=4)
        los_zod = np.expand_dims(np.expand_dims(zod,
            axis=3), axis=4)

        # Field matrix
        H_phase = np.reshape(np.array([[1.,0.],
                                    [0.,-1.]]), [1,1,1,1,1,2,2])
        H_field = self.step_11_field_matrix_ref(scenario, los_aoa, los_aod,
                                                    los_zoa, los_zod, H_phase)

        # Array offset matrix
        H_array = self.step_11_array_offsets_ref(los_aoa, los_aod, los_zoa,
                                                            los_zod, scenario)

        # Doppler matrix
        H_doppler = self.step_11_doppler_matrix_ref(scenario, los_aoa,
                                                                    los_zoa, t)

        # Phase shift due to propagation delay
        d3D = scenario.distance_3d
        if scenario.direction == "uplink":
            d3D = torch.permute(d3D, [0, 2, 1]).numpy(force=True)
        lambda_0 = self.scenario.lambda_0
        H_delay = np.exp(1j*2*np.pi*d3D/lambda_0)

        # Combining all to compute channel coefficient
        H_field = np.expand_dims(np.squeeze(H_field, axis=4), axis=-1)
        H_array = np.expand_dims(np.squeeze(H_array, axis=4), axis=-1)
        H_doppler = np.expand_dims(H_doppler, axis=4)
        H_delay = np.expand_dims(np.expand_dims(np.expand_dims(
                np.expand_dims(H_delay, axis=3), axis=4), axis=5), axis=6)

        H_los = H_field*H_array*H_doppler*H_delay
        return H_los

    def test_step11_los(self):
        """Test 3GPP channel coefficient calculation: LoS channel matrix"""
        H_los_ref = self.step_11_los_ref(self.sample_times.numpy(force=True), self.scenario)

        H_los = self.ccg._step_11_los(self.scenario, self.sample_times).numpy(force=True)
        H_los = H_los

        max_err = self.max_rel_err(H_los_ref, H_los)
        err_tol = TestChannelCoefficientsGenerator.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def step_11_ref(self, phi, k_factor, aoa, aod, zoa, zod, kappa, powers,
                                                    delays, t, scenario, c_ds):
        """Reference implementation: Step 11"""

        ## NLoS
        H_full = self.step_11_nlos_ref(phi, aoa, aod, zoa, zod, kappa, powers,
                                                                t, scenario)
        H_nlos, delays_nlos = self.step_11_reduce_nlos_ref(H_full, powers,
                                                                delays, c_ds)

        ## LoS
        H_los = self.step_11_los_ref(t, scenario)

        k_factor = np.reshape(k_factor, list(k_factor.shape) + [1,1,1,1])
        los_scaling = np.sqrt(k_factor/(k_factor+1.))
        nlos_scaling = np.sqrt(1./(k_factor+1.))

        H_los_nlos = nlos_scaling*H_nlos
        H_los_los = los_scaling*H_los
        H_los_los = H_los_los + H_los_nlos[:,:,:,:1,...]
        H_los = np.concatenate([H_los_los, H_los_nlos[:,:,:,1:,...]], axis=3)

        ## Setting up the CIR according to the link configuration
        los_status = scenario.is_los
        if scenario.direction == "uplink":
            los_status = torch.permute(los_status, [0, 2, 1]).numpy(force=True)
        los_status = np.reshape(los_status, list(los_status.shape) + [1,1,1,1])
        H = np.where(los_status, H_los, H_nlos)

        return H, delays_nlos

    def test_step_11(self):
        """Test 3GPP channel coefficient calculation: Step 11"""
        H, delays_nlos = self.ccg._step_11(self.phi.type(torch.float64),
                                            self.scenario,
                                            self.lsp.k_factor,
                                            self.rays,
                                            self.sample_times.type(torch.float64),
                                            self.c_ds)
        H = H
        delays_nlos = delays_nlos

        H_ref, delays_nlos_ref = self.step_11_ref(self.phi.numpy(force=True),
                                                self.lsp.k_factor.numpy(force=True),
                                                self.rays.aoa.numpy(force=True),
                                                self.rays.aod.numpy(force=True),
                                                self.rays.zoa.numpy(force=True),
                                                self.rays.zod.numpy(force=True),
                                                self.rays.xpr.numpy(force=True),
                                                self.rays.powers.numpy(force=True),
                                                self.rays.delays.numpy(force=True),
                                                self.sample_times.numpy(force=True),
                                                self.scenario,
                                                self.c_ds.numpy(force=True))

        err_tol = TestChannelCoefficientsGenerator.MAX_ERR
        max_err = self.max_rel_err(H_ref, H.numpy(force=True))
        self.assertLessEqual(max_err, err_tol)
        max_err = self.max_rel_err(delays_nlos_ref, delays_nlos.numpy(force=True))
        self.assertLessEqual(max_err, err_tol)

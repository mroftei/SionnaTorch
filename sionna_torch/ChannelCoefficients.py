#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Class for sampling channel impulse responses following 3GPP TR38.901
specifications and giving LSPs and rays.
"""
from typing import Optional
import torch
import scipy.constants

class ChannelCoefficientsGenerator:
    # pylint: disable=line-too-long
    r"""
    Sample channel impulse responses according to LSPs rays.

    This class implements steps 10 and 11 from the TR 38.901 specifications,
    (section 7.5).

    Parameters
    ----------
    carrier_frequency : float
        Carrier frequency [Hz]

    subclustering : bool
        Use subclustering if set to `True` (see step 11 for section 7.5 in
        TR 38.901). CDL does not use subclustering. System level models (UMa,
        UMi, RMa) do.

    dtype : Complex DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `complex64`.

    Input
    -----
    num_time_samples : int
        Number of samples

    sampling_frequency : float
        Sampling frequency [Hz]

    k_factor : [batch_size, number of TX, number of RX]
        K-factor

    rays : Rays
        Rays from which to compute thr CIR

    scenario : SionnaScenario
        scenario of the network

    c_ds : [batch size, number of TX, number of RX]
        Cluster DS [ns]. Only needed when subclustering is used
        (``subclustering`` set to `True`), i.e., with system level models.
        Otherwise can be set to None.
        Defaults to None.

    debug : bool
        If set to `True`, additional information is returned in addition to
        paths coefficients and delays: The random phase shifts (see step 10 of
        section 7.5 in TR38.901 specification), and the time steps at which the
        channel is sampled.

    Output
    ------
    h : [batch size, num TX, num RX, num paths, num RX antenna, num TX antenna, num samples], complex
        Paths coefficients

    delays : [batch size, num TX, num RX, num paths], real
        Paths delays [s]

    phi : [batch size, number of BSs, number of UTs, 4], real
        Initial phases (see step 10 of section 7.5 in TR 38.901 specification).
        Last dimension corresponds to the four polarization combinations.

    sample_times : [number of time steps], float
        Sampling time steps
    """

    def __init__(self,  carrier_frequency,
                        subclustering,
                        rng: torch.Generator,
                        dtype=torch.complex64,
                        device: Optional[torch.device] = None):
        
        assert dtype.is_complex, "'dtype' must be complex type"
        self._dtype = dtype
        self._dtype_real = dtype.to_real()
        self.device = device

        self.rng = rng
        # Wavelength (m)
        self._lambda_0 = torch.tensor(scipy.constants.c/carrier_frequency, dtype=self._dtype_real, device=device)
        self._subclustering = subclustering

        # Sub-cluster information for intra cluster delay spread clusters
        # This is hardcoded from Table 7.5-5
        self._sub_cl_1_ind = torch.tensor([0,1,2,3,4,5,6,7,18,19], dtype=torch.int32, device=device)
        self._sub_cl_2_ind = torch.tensor([8,9,10,11,16,17], dtype=torch.int32, device=device)
        self._sub_cl_3_ind = torch.tensor([12,13,14,15], dtype=torch.int32, device=device)
        self._sub_cl_delay_offsets = torch.tensor([0, 1.28, 2.56], dtype=self._dtype_real, device=device)

        self.los_h_phase = torch.tensor([[1.+0j,0.+0j],[0.+0j,-1.+0j]], dtype=self._dtype, device=self.device)

    def __call__(self, num_time_samples, sampling_frequency, k_factor, rays,
                 scenario, c_ds=None, debug=False):
        # Sample times
        sample_times = (torch.arange(num_time_samples, dtype=self._dtype_real, device=self.device)/sampling_frequency)

        # Step 10
        phi = self._step_10(rays.aoa.shape)

        # Step 11
        h, delays = self._step_11(phi, scenario, k_factor, rays, sample_times, c_ds)

        # Return additional information if requested
        if debug:
            return h, delays, phi, sample_times

        return h, delays

    ###########################################
    # Utility functions
    ###########################################

    def _unit_sphere_vector(self, theta, phi):
        r"""
        Generate vector on unit sphere (7.1-6)

        Input
        -------
        theta : Arbitrary shape, float
            Zenith [radian]

        phi : Same shape as ``theta``, float
            Azimuth [radian]

        Output
        --------
        rho_hat : ``phi.shape`` + [3, 1]
            Vector on unit sphere

        """
        rho_hat = torch.stack([torch.sin(theta)*torch.cos(phi),
                            torch.sin(theta)*torch.sin(phi),
                            torch.cos(theta)], dim=-1)
        return torch.unsqueeze(rho_hat, -1)

    def _forward_rotation_matrix(self, orientations):
        r"""
        Forward composite rotation matrix (7.1-4)

        Input
        ------
            orientations : [...,3], float
                Orientation to which to rotate [radian]

        Output
        -------
        R : [...,3,3], float
            Rotation matrix
        """
        a, b, c = orientations[...,0], orientations[...,1], orientations[...,2]

        row_1 = torch.stack([torch.cos(a)*torch.cos(b),
            torch.cos(a)*torch.sin(b)*torch.sin(c)-torch.sin(a)*torch.cos(c),
            torch.cos(a)*torch.sin(b)*torch.cos(c)+torch.sin(a)*torch.sin(c)], dim=-1)

        row_2 = torch.stack([torch.sin(a)*torch.cos(b),
            torch.sin(a)*torch.sin(b)*torch.sin(c)+torch.cos(a)*torch.cos(c),
            torch.sin(a)*torch.sin(b)*torch.cos(c)-torch.cos(a)*torch.sin(c)], dim=-1)

        row_3 = torch.stack([-torch.sin(b),
            torch.cos(b)*torch.sin(c),
            torch.cos(b)*torch.cos(c)], dim=-1)

        rot_mat = torch.stack([row_1, row_2, row_3], dim=-2)
        return rot_mat

    def _rot_pos(self, orientations, positions):
        r"""
        Rotate the ``positions`` according to the ``orientations``

        Input
        ------
        orientations : [...,3], float
            Orientation to which to rotate [radian]

        positions : [...,3,1], float
            Positions to rotate

        Output
        -------
        : [...,3,1], float
            Rotated positions
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        return torch.matmul(rot_mat, positions)

    def _reverse_rotation_matrix(self, orientations):
        r"""
        Reverse composite rotation matrix (7.1-4)

        Input
        ------
        orientations : [...,3], float
            Orientations to rotate to  [radian]

        Output
        -------
        R_inv : [...,3,3], float
            Inverse of the rotation matrix corresponding to ``orientations``
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        rot_mat_inv = rot_mat.swapaxes(-2, -1)
        return rot_mat_inv

    def _gcs_to_lcs(self, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Compute the angles ``theta``, ``phi`` in LCS rotated according to
        ``orientations`` (7.1-7/8)

        Input
        ------
        orientations : [...,3] of rank K, float
            Orientations to which to rotate to [radian]

        theta : Broadcastable to the first K-1 dimensions of ``orientations``, float
            Zenith to rotate [radian]

        phi : Same dimension as ``theta``, float
            Azimuth to rotate [radian]

        Output
        -------
        theta_prime : Same dimension as ``theta``, float
            Rotated zenith

        phi_prime : Same dimensions as ``theta`` and ``phi``, float
            Rotated azimuth
        """

        rho_hat = self._unit_sphere_vector(theta, phi)
        rot_inv = self._reverse_rotation_matrix(orientations)
        rot_rho = torch.matmul(rot_inv, rho_hat)
        v1 = torch.tensor([0.,0.,1.], dtype=self._dtype_real, device=self.device)
        v1 = torch.reshape(v1, [1]*(len(rot_rho.shape)-1)+[3])
        v2 = torch.tensor([1+0j,1j,0], dtype=self._dtype, device=self.device)
        v2 = torch.reshape(v2, [1]*(len(rot_rho.shape)-1)+[3])
        z = torch.matmul(v1, rot_rho)
        z = torch.clip(z, -1.0, 1.0)
        theta_prime = torch.arccos(z)
        phi_prime = torch.angle((torch.matmul(v2, rot_rho.type_as(v2))))
        theta_prime = torch.squeeze(theta_prime, dim=(len(phi.shape), len(phi.shape)+1))
        phi_prime = torch.squeeze(phi_prime, dim=(len(phi.shape), len(phi.shape)+1))

        return (theta_prime, phi_prime)

    def _compute_psi(self, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Compute displacement angle :math:`Psi` for the transformation of LCS-GCS
        field components in (7.1-15) of TR38.901 specification

        Input
        ------
        orientations : [...,3], float
            Orientations to which to rotate to [radian]

        theta :  Broadcastable to the first K-1 dimensions of ``orientations``, float
            Spherical position zenith [radian]

        phi : Same dimensions as ``theta``, float
            Spherical position azimuth [radian]

        Output
        -------
            Psi : Same shape as ``theta`` and ``phi``, float
                Displacement angle :math:`Psi`
        """
        a = orientations[...,0]
        b = orientations[...,1]
        c = orientations[...,2]
        theta = theta
        phi = phi
        real = torch.sin(c)*torch.cos(theta)*torch.sin(phi-a)
        real += torch.cos(c)*(torch.cos(b)*torch.sin(theta)-torch.sin(b)*torch.cos(theta)*torch.cos(phi-a))
        imag = torch.sin(c)*torch.cos(phi-a) + torch.sin(b)*torch.cos(c)*torch.sin(phi-a)
        psi = torch.angle(real + 1j*imag)
        return psi

    def _l2g_response(self, f_prime, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Transform field components from LCS to GCS (7.1-11)

        Input
        ------
        f_prime : K-Dim Tensor of shape [...,2], float
            Field components

        orientations : K-Dim Tensor of shape [...,3], float
            Orientations of LCS-GCS [radian]

        theta : K-1-Dim Tensor with matching dimensions to ``f_prime`` and ``phi``, float
            Spherical position zenith [radian]

        phi : Same dimensions as ``theta``, float
            Spherical position azimuth [radian]

        Output
        ------
            F : K+1-Dim Tensor with shape [...,2,1], float
                The first K dimensions are identical to those of ``f_prime``
        """
        psi = self._compute_psi(orientations, theta, phi)
        row1 = torch.stack([torch.cos(psi), -torch.sin(psi)], dim=-1)
        row2 = torch.stack([torch.sin(psi), torch.cos(psi)], dim=-1)
        mat = torch.stack([row1, row2], dim=-2)
        f = torch.matmul(mat, torch.unsqueeze(f_prime, -1))
        return f

    def _step_11_get_tx_antenna_positions(self, scenario):
        r"""Compute d_bar_tx in (7.5-22), i.e., the positions in GCS of elements
        forming the transmit panel

        Input
        -----
        scenario : scenario
            Scenario of the network

        Output
        -------
        d_bar_tx : [batch_size, num TXs, num TX antenna, 3]
            Positions of the antenna elements in the GCS
        """
        # Get BS orientations got broadcasting
        if scenario.direction == "uplink":
            tx_orientations = torch.zeros(scenario.batch_size, scenario.num_ut, 3, dtype=scenario._dtype_real, device=self.device) # [batch size, number of UTs, 3]
        else:
            tx_orientations = torch.zeros(scenario.batch_size, scenario.num_bs, 3, dtype=scenario._dtype_real, device=self.device) # [batch size, number of BSs, 3]
        tx_orientations = torch.unsqueeze(tx_orientations, 2)

        # Get antenna element positions in LCS and reshape for broadcasting
        tx_ant_pos_lcs = torch.tensor([[0.,0.,0.]], dtype=self._dtype_real, device=self.device)
        tx_ant_pos_lcs = torch.reshape(tx_ant_pos_lcs, (1,1,*tx_ant_pos_lcs.shape,1))

        # Compute antenna element positions in GCS
        tx_ant_pos_gcs = self._rot_pos(tx_orientations, tx_ant_pos_lcs)
        tx_ant_pos_gcs = torch.reshape(tx_ant_pos_gcs, tx_ant_pos_gcs.shape[:-1])

        d_bar_tx = tx_ant_pos_gcs

        return d_bar_tx

    def _step_11_get_rx_antenna_positions(self, scenario):
        r"""Compute d_bar_rx in (7.5-22), i.e., the positions in GCS of elements
        forming the receive antenna panel

        Input
        -----
        scenario : SionnaScenario
            Scenario of the network

        Output
        -------
        d_bar_rx : [batch_size, num RXs, num RX antenna, 3]
            Positions of the antenna elements in the GCS
        """
        # Get UT orientations got broadcasting
        if scenario.direction == "uplink":
            rx_orientations = torch.zeros(scenario.batch_size, scenario.num_bs, 3, dtype=scenario._dtype_real, device=self.device) # [batch size, number of BSs, 3]
        else:
            rx_orientations = torch.zeros(scenario.batch_size, scenario.num_ut, 3, dtype=scenario._dtype_real, device=self.device) # [batch size, number of UTs, 3]
        rx_orientations = torch.unsqueeze(rx_orientations, 2)

        # Get antenna element positions in LCS and reshape for broadcasting
        rx_ant_pos_lcs = torch.tensor([[0.,0.,0.]], dtype=self._dtype_real, device=self.device)
        rx_ant_pos_lcs = torch.reshape(rx_ant_pos_lcs, (1,1,*rx_ant_pos_lcs.shape,1))

        # Compute antenna element positions in GCS
        rx_ant_pos_gcs = self._rot_pos(rx_orientations, rx_ant_pos_lcs)
        rx_ant_pos_gcs = torch.reshape(rx_ant_pos_gcs, rx_ant_pos_gcs.shape[:-1])

        d_bar_rx = rx_ant_pos_gcs

        return d_bar_rx

    def _step_10(self, shape):
        r"""
        Generate random and uniformly distributed phases for all rays and
        polarization combinations

        Input
        -----
        shape : Shape tensor
            Shape of the leading dimensions for the tensor of phases to generate

        Output
        ------
        phi : [shape] + [4], float
            Phases for all polarization combinations
        """
        phi = torch.rand(shape+(4,), generator=self.rng, dtype=self._dtype_real, device=self.device)
        (-torch.pi - torch.pi) * phi + torch.pi

        return phi

    def _step_11_phase_matrix(self, phi, rays):
        # pylint: disable=line-too-long
        r"""
        Compute matrix with random phases in (7.5-22)

        Input
        -----
        phi : [batch size, num TXs, num RXs, num clusters, num rays, 4], float
            Initial phases for all combinations of polarization

        rays : Rays
            Rays

        Output
        ------
        h_phase : [batch size, num TXs, num RXs, num clusters, num rays, 2, 2], complex
            Matrix with random phases in (7.5-22)
        """
        xpr = rays.xpr

        xpr_scaling = torch.sqrt(1/xpr) + 0j
        e0 = torch.exp(0.0 + 1j*phi[...,0])
        e3 = torch.exp(0.0 + 1j*phi[...,3])
        e1 = xpr_scaling*torch.exp(0.0 + 1j*phi[...,1])
        e2 = xpr_scaling*torch.exp(0.0 + 1j*phi[...,2])
        shape = e0.shape + (2,2)
        h_phase = torch.reshape(torch.stack([e0, e1, e2, e3], dim=-1), shape)

        return h_phase

    def _step_11_doppler_matrix(self, scenario, aoa, zoa, t):
        # pylint: disable=line-too-long
        r"""
        Compute matrix with phase shifts due to mobility in (7.5-22)

        Input
        -----
        scenario : SionnaScenario
            Scenario of the network

        aoa : [batch size, num TXs, num RXs, num clusters, num rays], float
            Azimuth angles of arrivals [radian]

        zoa : [batch size, num TXs, num RXs, num clusters, num rays], float
            Zenith angles of arrivals [radian]

        t : [number of time steps]
            Time steps at which the channel is sampled

        Output
        ------
        h_doppler : [batch size, num_tx, num rx, num clusters, num rays, num time steps], complex
            Matrix with phase shifts due to mobility in (7.5-22)
        """
        lambda_0 = self._lambda_0
        velocities = scenario.ut_velocities

        # Add an extra dimension to make v_bar broadcastable with the time
        # dimension
        # v_bar [batch size, num tx or num rx, 3, 1]
        v_bar = velocities
        v_bar = torch.unsqueeze(v_bar, dim=-1)

        # Depending on which end of the channel is moving, tx or rx, we add an
        # extra dimension to make this tensor broadcastable with the other end
        if scenario.direction == 'downlink': # moving_end == 'rx'
            # v_bar [batch size, 1, num rx, num tx, 1]
            v_bar = torch.unsqueeze(v_bar, 1)
        elif scenario.direction == 'uplink': # moving_end == 'tx'
            # v_bar [batch size, num tx, 1, num tx, 1]
            v_bar = torch.unsqueeze(v_bar, 2)

        # v_bar [batch size, 1, num rx, 1, 1, 3, 1]
        # or    [batch size, num tx, 1, 1, 1, 3, 1]
        v_bar = torch.unsqueeze(torch.unsqueeze(v_bar, -3), -3)

        # v_bar [batch size, num_tx, num rx, num clusters, num rays, 3, 1]
        r_hat_rx = self._unit_sphere_vector(zoa, aoa)

        # Compute phase shift due to doppler
        # [batch size, num_tx, num rx, num clusters, num rays, num time steps]
        exponent = 2*torch.pi/lambda_0*torch.sum(r_hat_rx*v_bar, -2)*t
        h_doppler = torch.exp(0.0 + 1j*exponent)

        # [batch size, num_tx, num rx, num clusters, num rays, num time steps]
        return h_doppler

    def _step_11_array_offsets(self, scenario, aoa, aod, zoa, zod):
        # pylint: disable=line-too-long
        r"""
        Compute matrix accounting for phases offsets between antenna elements

        Input
        -----
        scenario : SionnaScenario
            Scenario of the network

        aoa : [batch size, num TXs, num RXs, num clusters, num rays], float
            Azimuth angles of arrivals [radian]

        aod : [batch size, num TXs, num RXs, num clusters, num rays], float
            Azimuth angles of departure [radian]

        zoa : [batch size, num TXs, num RXs, num clusters, num rays], float
            Zenith angles of arrivals [radian]

        zod : [batch size, num TXs, num RXs, num clusters, num rays], float
            Zenith angles of departure [radian]
        Output
        ------
        h_array : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas], complex
            Matrix accounting for phases offsets between antenna elements
        """

        lambda_0 = self._lambda_0

        r_hat_rx = self._unit_sphere_vector(zoa, aoa)
        r_hat_rx = torch.squeeze(r_hat_rx, dim=len(r_hat_rx.shape)-1)
        r_hat_tx = self._unit_sphere_vector(zod, aod)
        r_hat_tx = torch.squeeze(r_hat_tx, dim=len(r_hat_tx.shape)-1)
        d_bar_rx = self._step_11_get_rx_antenna_positions(scenario)
        d_bar_tx = self._step_11_get_tx_antenna_positions(scenario)

        # Reshape tensors for broadcasting
        # r_hat_rx/tx have
        # shape [batch_size, num_tx, num_rx, num_clusters, num_rays,    3]
        # and will be reshaoed to
        # [batch_size, num_tx, num_rx, num_clusters, num_rays, 1, 3]
        r_hat_tx = torch.unsqueeze(r_hat_tx, -2)
        r_hat_rx = torch.unsqueeze(r_hat_rx, -2)

        # d_bar_tx has shape [batch_size, num_tx,          num_tx_antennas, 3]
        # and will be reshaped to
        # [batch_size, num_tx, 1, 1, 1, num_tx_antennas, 3]
        d_bar_tx = d_bar_tx[:,:,None,None,None]

        # d_bar_rx has shape [batch_size,    num_rx,       num_rx_antennas, 3]
        # and will be reshaped to
        # [batch_size, 1, num_rx, 1, 1, num_rx_antennas, 3]
        d_bar_rx = d_bar_rx[:,None,:,None,None]

        # Compute all tensor elements

        # As broadcasting of such high-rank tensors is not fully supported
        # in all cases, we need to do a hack here by explicitly
        # broadcasting one dimension:
        s = d_bar_rx.shape
        shape = s[0:1] + r_hat_rx.shape[1:2] + s[2:]
        d_bar_rx = torch.broadcast_to(d_bar_rx, shape)
        exp_rx = 2*torch.pi/lambda_0*torch.sum(r_hat_rx*d_bar_rx,
            axis=-1, keepdims=True)
        exp_rx = torch.exp(0.0 + 1j*exp_rx)

        # The hack is for some reason not needed for this term
        exp_tx = 2*torch.pi/lambda_0*torch.sum(r_hat_tx*d_bar_tx, axis=-1)
        exp_tx = torch.exp(0.0 + 1j*exp_tx)
        exp_tx = torch.unsqueeze(exp_tx, -2)

        h_array = exp_rx*exp_tx

        return h_array

    def _step_11_field_matrix(self, scenario, aoa, aod, zoa, zod, h_phase):
        # pylint: disable=line-too-long
        r"""
        Compute matrix accounting for the element responses, random phases
        and xpr

        Input
        -----
        scenario : SionnaScenario
            Scenario of the network

        aoa : [batch size, num TXs, num RXs, num clusters, num rays], float
            Azimuth angles of arrivals [radian]

        aod : [batch size, num TXs, num RXs, num clusters, num rays], float
            Azimuth angles of departure [radian]

        zoa : [batch size, num TXs, num RXs, num clusters, num rays], float
            Zenith angles of arrivals [radian]

        zod : [batch size, num TXs, num RXs, num clusters, num rays], float
            Zenith angles of departure [radian]

        h_phase : [batch size, num_tx, num rx, num clusters, num rays, num time steps], complex
            Matrix with phase shifts due to mobility in (7.5-22)

        Output
        ------
        h_field : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas], complex
            Matrix accounting for element responses, random phases and xpr
        """

        if scenario.direction == "uplink":
            tx_orientations = torch.zeros(scenario.batch_size, scenario.num_ut, 3, device=self.device) # [batch size, number of UTs, 3]
            rx_orientations = torch.zeros(scenario.batch_size, scenario.num_bs, 3, device=self.device) # [batch size, number of BSs, 3]
        else:
            rx_orientations = torch.zeros(scenario.batch_size, scenario.num_ut, 3, device=self.device) # [batch size, number of UTs, 3]
            tx_orientations = torch.zeros(scenario.batch_size, scenario.num_bs, 3, device=self.device) # [batch size, number of BSs, 3]

        # Transform departure angles to the LCS
        shape = tx_orientations.shape[:2] + (1,1,1) + tx_orientations.shape[-1:]
        tx_orientations = torch.reshape(tx_orientations, shape)

        # Transform arrival angles to the LCS
        rx_orientations = rx_orientations[:,None,:,None,None,:]

        # Compute transmitted and received field strength for all antennas
        # in the LCS  and convert to GCS
        f_tx_pol1_prime = torch.stack([torch.ones_like(zod, dtype=self._dtype_real, device=self.device), torch.zeros_like(aod, dtype=self._dtype_real, device=self.device)], axis=-1) # Unity antenna
        f_rx_pol1_prime = torch.stack([torch.ones_like(zoa, dtype=self._dtype_real, device=self.device), torch.zeros_like(aoa, dtype=self._dtype_real, device=self.device)], axis=-1) # Unity antenna

        f_tx_pol1 = self._l2g_response(f_tx_pol1_prime, tx_orientations,
            zod, aod)

        f_rx_pol1 = self._l2g_response(f_rx_pol1_prime, rx_orientations,
            zoa, aoa)

        # Fill the full channel matrix with field responses
        pol1_tx = torch.matmul(h_phase, f_tx_pol1 + 0.0j)

        num_ant_tx = 1
        # Each BS antenna gets the polarization 1 response
        f_tx_array = torch.tile(torch.unsqueeze(pol1_tx, 0), (num_ant_tx,) + (1,)*len(pol1_tx.shape))

        num_ant_rx = 1
        # Each UT antenna gets the polarization 1 response
        f_rx_array = torch.tile(torch.unsqueeze(f_rx_pol1, 0), (num_ant_rx,) + (1,)*len(f_rx_pol1.shape))
        f_rx_array = f_rx_array + 0j

        # Compute the scalar product between the field vectors through
        # reduce_sum and transpose to put antenna dimensions last
        h_field = torch.sum(torch.unsqueeze(f_rx_array, 1)*torch.unsqueeze(
            f_tx_array, 0), (-2,-1))
        h_field = torch.permute(h_field, torch.roll(torch.arange(len(h_field.shape)), -2, 0).tolist())

        return h_field

    def _step_11_nlos(self, phi, scenario, rays, t):
        # pylint: disable=line-too-long
        r"""
        Compute the full NLOS channel matrix (7.5-28)

        Input
        -----
        phi: [batch size, num TXs, num RXs, num clusters, num rays, 4], float
            Random initial phases [radian]

        scenario : SionnaScenario
            Scenario of the network

        rays : Rays
            Rays

        t : [num time samples], float
            Time samples

        Output
        ------
        h_full : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas, num time steps], complex
            NLoS channel matrix
        """

        h_phase = self._step_11_phase_matrix(phi, rays)
        h_field = self._step_11_field_matrix(scenario, rays.aoa, rays.aod,
                                                    rays.zoa, rays.zod, h_phase)
        # h_array = self._step_11_array_offsets(scenario, rays.aoa, rays.aod, rays.zoa, rays.zod)
        h_array = 1.0 # Always 1 since both unity gain antennas assumed
        h_doppler = self._step_11_doppler_matrix(scenario, rays.aoa, rays.zoa, t)

        h_full = torch.unsqueeze(h_field*h_array, -1) * torch.unsqueeze(
            torch.unsqueeze(h_doppler, -2), -2)

        power_scaling = torch.sqrt(rays.powers/h_full.shape[4]) + 0.0j
        h_full *= power_scaling[(...,)+(None,)*(len(h_full.shape)-len(power_scaling.shape))]

        return h_full

    def _step_11_reduce_nlos(self, h_full, rays, c_ds):
        # pylint: disable=line-too-long
        r"""
        Compute the final NLOS matrix in (7.5-27)

        Input
        ------
        h_full : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas, num time steps], complex
            NLoS channel matrix

        rays : Rays
            Rays

        c_ds : [batch size, num TX, num RX], float
            Cluster delay spread

        Output
        -------
        h_nlos : [batch size, num_tx, num rx, num clusters, num rx antennas, num tx antennas, num time steps], complex
            Paths NLoS coefficients

        delays_nlos : [batch size, num_tx, num rx, num clusters], float
            Paths NLoS delays
        """

        if self._subclustering:

            powers = rays.powers
            delays = rays.delays

            # Sort all clusters along their power
            strongest_clusters = torch.argsort(powers, dim=-1, descending=True)

            # Sort delays according to the same ordering
            delays_sorted = torch.take_along_dim(delays, strongest_clusters, dim=3)

            # Split into delays for strong and weak clusters
            delays_strong = delays_sorted[...,:2]
            delays_weak = delays_sorted[...,2:]

            # Compute delays for sub-clusters
            offsets = torch.reshape(self._sub_cl_delay_offsets,
                (len(delays_strong.shape)-1)*[1]+[-1]+[1])
            delays_sub_cl = (torch.unsqueeze(delays_strong, -2) +
                offsets*torch.unsqueeze(torch.unsqueeze(c_ds, dim=-1), dim=-1))
            delays_sub_cl = torch.flatten(delays_sub_cl, -2)

            # Select the strongest two clusters for sub-cluster splitting
            h_strong = torch.take_along_dim(h_full, strongest_clusters[...,:2,None,None,None,None], dim=3)

            # The other clusters are the weak clusters
            h_weak = torch.take_along_dim(h_full, strongest_clusters[...,2:,None,None,None,None], dim=3)

            # Sum specific rays for each sub-cluster
            h_sub_cl_1 = torch.sum(h_strong[:,:,:,:, self._sub_cl_1_ind], dim=4)
            h_sub_cl_2 = torch.sum(h_strong[:,:,:,:, self._sub_cl_2_ind], dim=4)
            h_sub_cl_3 = torch.sum(h_strong[:,:,:,:, self._sub_cl_3_ind], dim=4)

            # Sum all rays for the weak clusters
            h_weak = torch.sum(h_weak, dim=4)

            # Concatenate the channel and delay tensors
            h_nlos = torch.concatenate([h_sub_cl_1, h_sub_cl_2, h_sub_cl_3, h_weak],
                dim=3)
            delays_nlos = torch.concatenate([delays_sub_cl, delays_weak], dim=3)
        else:
            # Sum over rays
            h_nlos = torch.sum(h_full, dim=4)
            delays_nlos = rays.delays

        # Order the delays in ascending orders
        delays_ind = torch.argsort(delays_nlos, dim=-1)
        delays_nlos = torch.take_along_dim(delays_nlos, delays_ind, dim=3)

        # Order the channel clusters according to the delay, too
        h_nlos = torch.take_along_dim(h_nlos, delays_ind[...,None,None,None], dim=3)

        return h_nlos, delays_nlos

    def _step_11_los(self, scenario, t):
        # pylint: disable=line-too-long
        r"""Compute the LOS channels from (7.5-29)

        Intput
        ------
        scenario : SionnaScenario
            Network scenario

        t : [num time samples], float
            Number of time samples

        Output
        ------
        h_los : [batch size, num_tx, num rx, 1, num rx antennas, num tx antennas, num time steps], complex
            Paths LoS coefficients
        """

        if scenario.direction == "uplink":
            aoa = torch.permute(torch.remainder(scenario.los_aod_rad, 2*torch.pi), [0, 2, 1])
            aod = torch.permute(torch.remainder(scenario.los_aoa_rad, 2*torch.pi), [0, 2, 1])
            zoa = torch.permute(torch.remainder(scenario.los_zod_rad, 2*torch.pi), [0, 2, 1])
            zod = torch.permute(torch.remainder(scenario.los_zoa_rad, 2*torch.pi), [0, 2, 1])
        else:
            aoa = scenario.los_aoa_rad
            aod = scenario.los_aod_rad
            zoa = scenario.los_zoa_rad
            zod = scenario.los_zod_rad

         # LoS departure and arrival angles
        aoa = torch.unsqueeze(torch.unsqueeze(aoa, dim=3), dim=4)
        zoa = torch.unsqueeze(torch.unsqueeze(zoa, dim=3), dim=4)
        aod = torch.unsqueeze(torch.unsqueeze(aod, dim=3), dim=4)
        zod = torch.unsqueeze(torch.unsqueeze(zod, dim=3), dim=4)

        # Field matrix
        h_field = self._step_11_field_matrix(scenario, aoa, aod, zoa, zod, self.los_h_phase)

        # Array offset matrix
        # h_array = self._step_11_array_offsets(scenario, aoa, aod, zoa, zod)
        h_array = 1.0 # Always 1.0 since both unity gain antennas assumed

        # Doppler matrix
        h_doppler = self._step_11_doppler_matrix(scenario, aoa, zoa, t)

        # Phase shift due to propagation delay
        d3d = scenario.distance_3d
        if scenario.direction == "uplink":
            d3d = torch.permute(d3d, [0, 2, 1])
        lambda_0 = self._lambda_0
        h_delay = torch.exp(0.0 + 1j*2*torch.pi*d3d/lambda_0)

        # Combining all to compute channel coefficient
        h_field = torch.unsqueeze(torch.squeeze(h_field, dim=4), dim=-1)
        # h_array = torch.unsqueeze(torch.squeeze(h_array, dim=4), dim=-1)
        h_doppler = torch.unsqueeze(h_doppler, dim=4)
        h_delay = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(
            torch.unsqueeze(h_delay, dim=3), dim=4), dim=5), dim=6)

        h_los = h_field*h_array*h_doppler*h_delay
        return h_los

    def _step_11(self, phi, scenario, k_factor, rays, t, c_ds):
        # pylint: disable=line-too-long
        r"""
        Combine LOS and LOS components to compute (7.5-30)

        Input
        -----
        phi: [batch size, num TXs, num RXs, num clusters, num rays, 4], float
            Random initial phases

        scenario : SionnaScenario
            Network scenario

        k_factor : [batch size, num TX, num RX], float
            Rician K-factor

        rays : Rays
            Rays

        t : [num time samples], float
            Number of time samples

        c_ds : [batch size, num TX, num RX], float
            Cluster delay spread
        """

        h_full = self._step_11_nlos(phi, scenario, rays, t)
        h_nlos, delays_nlos = self._step_11_reduce_nlos(h_full, rays, c_ds)

        ####  LoS scenario
        h_los_los_comp = self._step_11_los(scenario, t)
        k_factor = k_factor[(...,)+(None,)*(len(h_los_los_comp.shape)-len(k_factor.shape))]
        k_factor = k_factor + 0j

        # Scale NLOS and LOS components according to K-factor
        h_los_los_comp = h_los_los_comp*torch.sqrt(k_factor/(k_factor+1))
        h_los_nlos_comp = h_nlos*torch.sqrt(1/(k_factor+1))

        # Add the LOS component to the zero-delay NLOS cluster
        h_los_cl = h_los_los_comp + torch.unsqueeze(
            h_los_nlos_comp[:,:,:,0,...], 3)

        # Combine all clusters into a single tensor
        h_los = torch.concatenate([h_los_cl, h_los_nlos_comp[:,:,:,1:,...]], dim=3)

        #### LoS or NLoS CIR according to link configuration
        los = scenario.is_los
        if scenario.direction == "uplink":
            los = torch.permute(scenario.is_los, [0, 2, 1])
        los_indicator = los[(...,)+ (None,)*4]
        h = torch.where(los_indicator, h_los, h_nlos)

        return h, delays_nlos

#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layer for applying channel responses to channel inputs in the time domain"""

from typing import Optional
import torch
import numpy as np
import scipy

class ApplyTimeChannel():
    # pylint: disable=line-too-long
    r"""ApplyTimeChannel(num_time_samples, l_tot, add_awgn=True, dtype=complex64, **kwargs)

    Apply time domain channel responses ``h_time`` to channel inputs ``x``,
    by filtering the channel inputs with time-variant channel responses.

    This class inherits from the Keras `Layer` class and can be used as layer
    in a Keras model.

    For each batch example, ``num_time_samples`` + ``l_tot`` - 1 time steps of a
    channel realization are required to filter the channel inputs.

    The channel output consists of ``num_time_samples`` + ``l_tot`` - 1
    time samples, as it is the result of filtering the channel input of length
    ``num_time_samples`` with the time-variant channel filter  of length
    ``l_tot``. In the case of a single-input single-output link and given a sequence of channel
    inputs :math:`x_0,\cdots,x_{N_B}`, where :math:`N_B` is ``num_time_samples``, this
    layer outputs

    .. math::
        y_b = \sum_{\ell = 0}^{L_{\text{tot}}} x_{b-\ell} \bar{h}_{b,\ell} + w_b

    where :math:`L_{\text{tot}}` corresponds ``l_tot``, :math:`w_b` to the additive noise, and
    :math:`\bar{h}_{b,\ell}` to the :math:`\ell^{th}` tap of the :math:`b^{th}` channel sample.
    This layer outputs :math:`y_b` for :math:`b` ranging from 0 to
    :math:`N_B + L_{\text{tot}} - 1`, and :math:`x_{b}` is set to 0 for :math:`b \geq N_B`.

    For multiple-input multiple-output (MIMO) links, the channel output is computed for each antenna
    of each receiver and by summing over all the antennas of all transmitters.

    Parameters
    ----------

    num_time_samples : int
        Number of time samples forming the channel input (:math:`N_B`)

    l_tot : int
        Length of the channel filter (:math:`L_{\text{tot}} = L_{\text{max}} - L_{\text{min}} + 1`)

    add_awgn : bool
        If set to `False`, no white Gaussian noise is added.
        Defaults to `True`.

    dtype : DType
        Complex datatype to use for internal processing and output.
        Defaults to `complex64`.

    Input
    -----

    (x, h_time, no) or (x, h_time):
        Tuple:

    x :  [batch size, num_tx, num_tx_ant, num_time_samples], complex
        Channel inputs

    h_time : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples + l_tot - 1, l_tot], complex
        Channel responses.
        For each batch example, ``num_time_samples`` + ``l_tot`` - 1 time steps of a
        channel realization are required to filter the channel inputs.

    no : Scalar or Tensor, float
        Scalar or tensor whose shape can be broadcast to the shape of the channel outputs: [batch size, num_rx, num_rx_ant, num_time_samples + l_tot - 1].
        Only required if ``add_awgn`` is set to `True`.
        The noise power ``no`` is per complex dimension. If ``no`` is a
        scalar, noise of the same variance will be added to the outputs.
        If ``no`` is a tensor, it must have a shape that can be broadcast to
        the shape of the channel outputs. This allows, e.g., adding noise of
        different variance to each example in a batch. If ``no`` has a lower
        rank than the channel outputs, then ``no`` will be broadcast to the
        shape of the channel outputs by adding dummy dimensions after the
        last axis.

    Output
    -------
    y : [batch size, num_rx, num_rx_ant, num_time_samples + l_tot - 1], complex
        Channel outputs.
        The channel output consists of ``num_time_samples`` + ``l_tot`` - 1
        time samples, as it is the result of filtering the channel input of length
        ``num_time_samples`` with the time-variant channel filter  of length
        ``l_tot``.
    """

    def __init__(self, num_time_samples, l_tot, rng:torch.Generator, add_awgn=True, device: Optional[torch.device] = None):
        self.rng = rng
        self.device = device
        self._add_awgn = add_awgn

        # The channel transfert function is implemented by first gathering from
        # the vector of transmitted baseband symbols
        # x = [x_0,...,x_{num_time_samples-1}]^T  the symbols that are then
        # multiplied by the channel tap coefficients.
        # We build here the matrix of indices G, with size
        # `num_time_samples + l_tot - 1` x `l_tot` that is used to perform this
        # gathering.
        # For example, if there are 4 channel taps
        # h = [h_0, h_1, h_2, h_3]^T
        # and `num_time_samples` = 10 time steps then G  would be
        #       [[0, 10, 10, 10]
        #        [1,  0, 10, 10]
        #        [2,  1,  0, 10]
        #        [3,  2,  1,  0]
        #        [4,  3,  2,  1]
        #        [5,  4,  3,  2]
        #        [6,  5,  4,  3]
        #        [7,  6,  5,  4]
        #        [8,  7,  6,  5]
        #        [9,  8,  7,  6]
        #        [10, 9,  8,  7]
        #        [10,10,  9,  8]
        #        [10,10, 10,  9]
        # Note that G is a Toeplitz matrix.
        # In this example, the index `num_time_samples`=10 corresponds to the
        # zero symbol. The vector of transmitted symbols is padded with one
        # zero at the end.
        first_colum = np.concatenate([  np.arange(0, num_time_samples),
                                        np.full([l_tot-1], num_time_samples)])
        first_row = np.concatenate([[0], np.full([l_tot-1], num_time_samples)])
        self._g = torch.from_numpy(scipy.linalg.toeplitz(first_colum, first_row)).to(device)

    def __call__(self, x, h_time, no=1e-14, bw=None):
        # Preparing the channel input for broadcasting and matrix multiplication
        x1 = torch.nn.functional.pad(x, (0,1))
        x1 = x1[:, None, None,...,None]

        x1 = torch.take_along_dim(x1, self._g[None,None,None,None,None], -2)

        # Apply the channel response
        y = torch.sum(h_time*x1, axis=-1)
        y = torch.sum(torch.sum(y, axis=4), axis=3)

        # Add AWGN if requested
        if self._add_awgn:
            # Create tensors of real-valued Gaussian noise for each complex dim.
            stddev = np.sqrt(1.0/2.0) # Half the variance for each dimension

            # Generate complex Gaussian noise with the right variance
            xr = torch.normal(mean=0.0, std=stddev, size=y.shape, generator=self.rng, dtype=torch.float32, device=self.device)
            xi = torch.normal(mean=0.0, std=stddev, size=y.shape, generator=self.rng, dtype=torch.float32, device=self.device)
            noise = xr + 1j*xi

            # Apply variance scaling
            noise *= np.sqrt(no)

            # Add noise to input
            p_0 = torch.mean(torch.abs(y) ** 2, -1)
            n_0 = torch.mean(torch.abs(noise) ** 2, -1)
            snr = 10*torch.log10(p_0 / n_0)

            y = y + noise
        else:
            snr = 10*torch.log10(torch.mean(torch.abs(y) ** 2, -1))

        if bw is not None: # Inband SNR
            snr -= 10*torch.log10(bw)

        return y, snr

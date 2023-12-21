import torch

def matrix_sqrt(A, numIters=100):
    """ Newton-Schulz iterations method to get matrix square root.
    Page 231, Eq 2.6b
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.6.8799&rep=rep1&type=pdf

    Args:
        A: the symmetric PSD matrix whose matrix square root be computed
        numIters: Maximum number of iterations.

    Returns:
        A^0.5

    Tensorflow Source:
        https://github.com/tensorflow/tensorflow/blob/df3a3375941b9e920667acfe72fb4c33a8f45503/tensorflow/contrib/opt/python/training/matrix_functions.py#L26C1-L73C42
    Torch Source:
        https://github.com/msubhransu/matrix-sqrt/blob/cc2289a3ed7042b8dbacd53ce8a34da1f814ed2f/matrix_sqrt.py#L74
    """

    normA = torch.linalg.matrix_norm(A, keepdim=True)
    err = normA + 1.0
    I = torch.eye(*A.shape[-2:], dtype=A.dtype)
    Z = torch.eye(*A.shape[-2:], dtype=A.dtype).expand_as(A)
    Y = A / normA
    for i in range(numIters):
        T = 0.5*(3.0*I - Z@Y)
        Y_new = Y@T
        Z_new = T@Z

        # This method require that we check for divergence every step.
        # Compute the error in approximation.
        mat_a_approx = (Y_new @ Y_new) * normA
        residual = A - mat_a_approx
        current_err = torch.linalg.matrix_norm(residual, keepdim=True) / normA
        if torch.any(current_err > err):
            break

        err = current_err
        Y = Y_new
        Z = Z_new

    sA = Y*torch.sqrt(normA)
    
    return sA

class LSP:
    r"""
    Class for conveniently storing LSPs

    Parameters
    -----------

    ds : [batch size, num tx, num rx], tf.float
        RMS delay spread [s]

    asd : [batch size, num tx, num rx], tf.float
        azimuth angle spread of departure [deg]

    asa : [batch size, num tx, num rx], tf.float
        azimuth angle spread of arrival [deg]

    sf : [batch size, num tx, num rx], tf.float
        shadow fading

    k_factor : [batch size, num tx, num rx], tf.float
        Rician K-factor. Only used for LoS.

    zsa : [batch size, num tx, num rx], tf.float
        Zenith angle spread of arrival [deg]

    zsd: [batch size, num tx, num rx], tf.float
        Zenith angle spread of departure [deg]
    """

    def __init__(self, ds, asd, asa, sf, k_factor, zsa, zsd):
        self.ds = ds
        self.asd = asd
        self.asa = asa
        self.sf = sf
        self.k_factor = k_factor
        self.zsa = zsa
        self.zsd = zsd


class LSPGenerator:
    """
    Sample large scale parameters (LSP) and pathloss given a channel scenario,
    e.g., UMa, UMi, RMa.

    This class implements steps 1 to 4 of the TR 38.901 specifications
    (section 7.5), as well as path-loss generation (Section 7.4.1) with O2I
    low- and high- loss models (Section 7.4.3).

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
    None

    Output
    ------
    An `LSP` instance storing realization of LSPs.
    """

    def __init__(self, scenario, rng: torch.Generator):
        self._scenario = scenario
        self.rng = rng

    def __call__(self):

        # LSPs are assumed to follow a log-normal distribution.
        # They are generated in the log-domain (where they follow a normal
        # distribution), where they are correlated as indicated in TR38901
        # specification (Section 7.5, step 4)

        s = torch.normal(mean=0.0, std=1.0, size=[self._scenario.batch_size,
            self._scenario.num_bs, self._scenario.num_ut, 7], generator=self.rng,
            dtype=self._scenario._dtype_real)

        ## Applyting cross-LSP correlation
        s = torch.unsqueeze(s, axis=4)
        s = self._cross_lsp_correlation_matrix_sqrt @ s
        s = torch.squeeze(s, axis=4)

        ## Applying spatial correlation
        s = torch.unsqueeze(torch.permute(s, [0, 1, 3, 2]), axis=3)
        b = self._spatial_lsp_correlation_matrix_sqrt
        b = torch.moveaxis(b, -1, -2) # Transpose last 2
        s = s @ b
        s = torch.permute(torch.squeeze(s, axis=3), [0, 1, 3, 2])

        ## Scaling and transposing LSPs to the right mean and variance
        lsp_log = self._scenario.lsp_log_std*s + self._scenario.lsp_log_mean

        ## Mapping to linear domain
        lsp = torch.pow(10.0, lsp_log)

        # Limit the RMS azimuth arrival (ASA) and azimuth departure (ASD)
        # spread values to 104 degrees
        # Limit the RMS zenith arrival (ZSA) and zenith departure (ZSD)
        # spread values to 52 degrees
        lsp = LSP(  ds        = lsp[:,:,:,0],
                    asd       = torch.clip(lsp[:,:,:,1], min=None, max=104.0),
                    asa       = torch.clip(lsp[:,:,:,2], min=None, max=104.0),
                    sf        = lsp[:,:,:,3],
                    k_factor  = lsp[:,:,:,4],
                    zsa       = torch.clip(lsp[:,:,:,5], min=None, max=52.0),
                    zsd       = torch.clip(lsp[:,:,:,6], min=None, max=52.0)
                    )

        return lsp

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

        # Pre-computing these quantities avoid unnecessary calculations at every
        # generation of new LSPs

        # Compute cross-LSP correlation matrix
        self._compute_cross_lsp_correlation_matrix()

        # Compute LSP spatial correlation matrix
        self._compute_lsp_spatial_correlation_sqrt()

    ########################################
    # Internal utility methods
    ########################################

    def _compute_cross_lsp_correlation_matrix(self):
        """
        Compute and store as attribute the square-root of the  cross-LSPs
        correlation matrices for each BS-UT link, and then the corresponding
        matrix square root for filtering.

        The resulting tensor is of shape
        [batch size, number of BSs, number of UTs, 7, 7)
        7 being the number of LSPs to correlate.

        Input
        ------
        None

        Output
        -------
        None
        """

        # The following 7 LSPs are correlated:
        # DS, ASA, ASD, SF, K, ZSA, ZSD
        # We create the correlation matrix initialized to the identity matrix
        cross_lsp_corr_mat = torch.eye(7, 7, dtype=self._scenario._dtype_real)
        cross_lsp_corr_mat = torch.tile(cross_lsp_corr_mat[None,None,None], (self._scenario.batch_size,self._scenario.num_bs,self._scenario.num_ut,1,1))

        # Internal function that adds to the correlation matrix ``mat``
        # ``cross_lsp_corr_mat`` the parameter ``parameter_name`` at location
        # (m,n)
        def _add_param(mat, parameter_name, m, n):
            # Mask to put the parameters in the right spot of the 7x7
            # correlation matrix
            indices = torch.tensor([[m,n],[n,m]])
            mask = torch.zeros((7,7), dtype=self._scenario._dtype_real)
            mask[indices[:, 0], indices[:, 1]] = torch.tensor([1.0, 1.0], dtype=self._scenario._dtype_real)
            mask = torch.reshape(mask, [1,1,1,7,7])
            # Get the parameter value according to the link scenario
            update = self._scenario.get_param(parameter_name).type(self._scenario._dtype_real)
            update = torch.unsqueeze(torch.unsqueeze(update, axis=3), axis=4)
            # Add update
            mat = mat + update*mask
            return mat

        # Fill off-diagonal elements of the correlation matrices
        # ASD vs DS
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrASDvsDS', 0, 1)
        # ASA vs DS
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrASAvsDS', 0, 2)
        # ASA vs SF
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrASAvsSF', 3, 2)
        # ASD vs SF
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrASDvsSF', 3, 1)
        # DS vs SF
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrDSvsSF', 3, 0)
        # ASD vs ASA
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrASDvsASA', 1,2)
        # ASD vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrASDvsK', 1, 4)
        # ASA vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrASAvsK', 2, 4)
        # DS vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrDSvsK', 0, 4)
        # SF vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrSFvsK', 3, 4)
        # ZSD vs SF
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSDvsSF', 3, 6)
        # ZSA vs SF
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSAvsSF', 3, 5)
        # ZSD vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSDvsK', 6, 4)
        # ZSA vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSAvsK', 5, 4)
        # ZSD vs DS
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSDvsDS', 6, 0)
        # ZSA vs DS
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSAvsDS', 5, 0)
        # ZSD vs ASD
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSDvsASD', 6,1)
        # ZSA vs ASD
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSAvsASD', 5,1)
        # ZSD vs ASA
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSDvsASA', 6,2)
        # ZSA vs ASA
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSAvsASA', 5,2)
        # ZSD vs ZSA
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSDvsZSA', 5,6)

        # Compute and store the square root of the cross-LSP correlation
        # matrix
        self._cross_lsp_correlation_matrix_sqrt = matrix_sqrt(
                cross_lsp_corr_mat)

    def _compute_lsp_spatial_correlation_sqrt(self):
        """
        Compute the square root of the spatial correlation matrices of LSPs.

        The LSPs are correlated accross users according to the distance between
        the users. Each LSP is spatially correlated according to a different
        spatial correlation matrix.

        The links involving different BSs are not correlated.
        UTs in different state (LoS, NLoS, O2I) are not assumed to be
        correlated.

        The correlation of the LSPs X of two UTs in the same state related to
        the links of these UTs to a same BS is

        .. math::
            C(X_1,X_2) = exp(-d/D_X)

        where :math:`d` is the distance between the UTs in the X-Y plane (2D
        distance) and D_X the correlation distance of LSP X.

        The resulting tensor if of shape
        [batch size, number of BSs, 7, number of UTs, number of UTs)
        7 being the number of LSPs.

        Input
        ------
        None

        Output
        -------
        None
        """

        # Tensors of bool indicating which pair of UTs to correlate.
        # Pairs of UTs that are correlated are those that share the same state
        # (LoS or NLoS).
        # LoS
        los_ut = self._scenario.is_los
        los_pair_bool = torch.logical_and(torch.unsqueeze(los_ut, axis=3),
                                       torch.unsqueeze(los_ut, axis=2))
        # NLoS
        nlos_ut = torch.logical_not(los_ut)
        nlos_pair_bool = torch.logical_and(torch.unsqueeze(nlos_ut, axis=3),
                                        torch.unsqueeze(nlos_ut, axis=2))

        # Stacking the correlation matrix
        # One correlation matrix per LSP
        filtering_matrices = []
        distance_scaling_matrices = []
        for parameter_name in ('corrDistDS', 'corrDistASD', 'corrDistASA',
            'corrDistSF', 'corrDistK', 'corrDistZSA', 'corrDistZSD'):
            # Matrix used for filtering and scaling the 2D distances
            # For each pair of UTs, the entry is set to 0 if the UTs are in
            # different states, -1/(correlation distance) otherwise.
            # The correlation distance is different for each LSP.
            filtering_matrix = torch.eye(self._scenario.num_ut, self._scenario.num_ut, dtype=self._scenario._dtype_real)
            filtering_matrix = torch.tile(filtering_matrix[None,None], (self._scenario.batch_size,self._scenario.num_bs,1,1))
            distance_scaling_matrix = self._scenario.get_param(parameter_name).type(self._scenario._dtype_real)
            distance_scaling_matrix = torch.tile(torch.unsqueeze(
                distance_scaling_matrix, axis=3),
                [1, 1, 1, self._scenario.num_ut])
            distance_scaling_matrix = -1./distance_scaling_matrix
            # LoS
            filtering_matrix = torch.where(los_pair_bool, 1.0, filtering_matrix)
            # NLoS
            filtering_matrix = torch.where(nlos_pair_bool, 1.0, filtering_matrix)
            # Stacking
            filtering_matrices.append(filtering_matrix)
            distance_scaling_matrices.append(distance_scaling_matrix)
        filtering_matrices = torch.stack(filtering_matrices, axis=2)
        distance_scaling_matrices = torch.stack(distance_scaling_matrices, axis=2)

        ut_dist_2d = self._scenario.matrix_ut_distance_2d
        # Adding a dimension for broadcasting with BS
        ut_dist_2d = torch.unsqueeze(torch.unsqueeze(ut_dist_2d, axis=1), axis=2)

        # Correlation matrix
        spatial_lsp_correlation = (torch.exp(
            ut_dist_2d*distance_scaling_matrices)*filtering_matrices)

        # Compute and store the square root of the spatial correlation matrix
        self._spatial_lsp_correlation_matrix_sqrt = matrix_sqrt(
                spatial_lsp_correlation)
        
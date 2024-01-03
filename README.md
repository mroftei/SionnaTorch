A torch rewrite of the 3gPP channe lmodels from NVIDIA's Sionna library (https://github.com/NVlabs/sionna). The following changes have been made:
- Support only vertically polarized omni-antennas
- Support only RMa and UMa models
- Allow mix of rural and urban models in a scenario based on receiver location
- Allow mix of propogation path terrain type for each receiver using Euclidean distance points allong transmission path
- No-Tensorflow code with all PyTorch code supporting devices (eg. GPU)

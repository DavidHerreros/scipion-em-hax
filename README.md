<h1 align='center'>Scipion HAX Plugin</h1>

<p align="center">
        
<img alt="Supported Python versions" src="https://img.shields.io/badge/Supported_Python_Versions-3.11-blue">
<img alt="GitHub Downloads (all assets, all releases)" src="https://img.shields.io/github/downloads/I2PC/Flexutils-Toolkit/total">
<img alt="GitHub License" src="https://img.shields.io/github/license/I2PC/Flexutils-Toolkit">

</p>

<p align="center">
        
<img alt="HAX" width="300" src="hax/logo.png">

</p>

Plugin to execute Hax package from Scipion.

# Installation

We recommend installing Hax in production mode using the Scipion Plugin manager or by running the command:

> [!WARNING]
> The following command assumes that you have defined an alias to the Scipion executable named `scipion3`

```bash

  scipion3 installp -p scipion-em-hax

```

If you are a developer, you might want to install the plugin in development mode. In this case, please clone this repository to your machine and install the plugin with the following command:

> [!WARNING]
> The following command assumes that you have defined an alias to the Scipion executable named `scipion3`

```bash

  scipion3 installp -p path/to/your/cloned/scipion-em-hax --devel

```

In both cases, the Plugin will automatically create a Conda environment to isolate Hax from other installations. So, you'll need to have Conda installed on your machine.

> [!WARNING]
> Hax currently supports NVIDIA drivers version: >= 525 (Cuda 12 will be installed along the package, so there is no need to have CUDA already installed in your system).

# References

- Herreros, D., Lederman, R.R., Krieger, J.M. et al. **Estimating conformational landscapes from Cryo-EM particles by 3D Zernike polynomials**. *Nat Commun* 14, 154 (2023). 
[![DOI:10.1038/s41467-023-35791-y](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1038/s41467-023-35791-y)
- Herreros, D., Kiska, J., Ramirez-Aportela, E. et al. **ZART: A Novel Multiresolution Reconstruction Algorithm with Motion-blur Correction for Single Particle Analysis**. *Journal of Molecular Biology* 435, 168088 (2023). 
[![DOI:10.1016/j.jmb.2023.168088](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1016/j.jmb.2023.168088)
- Herreros, D., Krieger, J.M., Fonseca, Y., et al. **Scipion Flexibility Hub: an integrative framework for advanced analysis of conformational heterogeneity in cryoEM**. *Acta Cryst. D* 79, 569-584 (2023). 
[![DOI:10.1107/S2059798323004497](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1107/S2059798323004497)
- Herreros, D., Mata, C.P., Noddings, C. et al. **Real-space heterogeneous reconstruction, refinement, and disentanglement of CryoEM conformational states with HetSIREN**. *Nat Commun* 16, 3751 (2025). 
[![DOI:10.1038/s41467-025-59135-0](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1038/s41467-025-59135-0)
- Herreros, D., Perez Mata, C., Sanchez Sorzano, C.O. et al. **Merging conformational landscapes in a single consensus space with FlexConsensus algorithm"**. *Nat Methods* (2023). 
[![DOI:10.1038/s41592-025-02841-w](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1038/s41592-025-02841-w)

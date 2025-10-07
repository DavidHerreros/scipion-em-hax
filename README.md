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

If you are a developer, you might want to installed the plugin in development mode. In this case, please, clone this repository to your machine and install the plugin with the following command:

> [!WARNING]
> The following command assumes that you have defined an alias to the Scipion executable named `scipion3`

```bash

  scipion3 installp -p path/to/your/cloned/scipion-em-hax --devel

```

In both cases, the Plugin will automatically create a Conda environment to isolate Hax from other installations. Thus, you must have Conda installed in your machine.

> [!WARNING]
> Hax currently supports NVIDIA drivers version: >= 525 (Cuda 12 will be installed along the package, so there is no need to have CUDA already installed in your system).

# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************


import os
import subprocess
import sys

from pwem import Config as emConfig

import pyworkflow.plugin as pwplugin


__version__ = "0.1.0"
_logo = "logo.png"
_references = []


class Plugin(pwplugin.Plugin):

    @classmethod
    def getEnvActivation(cls):
        return "conda activate hax"

    @classmethod
    def getProgram(cls, program, gpu, uses_project_manager=True):
        """ Return the program binary that will be used. """
        cmd = f'{cls.getCondaActivationCmd()} {cls.getEnvActivation()} && '
        if uses_project_manager:
            if gpu is not None:
                return cmd + f'project_manager --gpu {gpu} {program} '
            else:
                return cmd + f'project_manager {program} '
        else:
            return cmd + f'{program} '

    @classmethod
    def getCommand(cls, program, gpu, args):
        return cls.getProgram(program, gpu) + args

    @classmethod
    def getAnnotateSpaceFunctionsPath(cls):
        cmd = f'{cls.getCondaActivationCmd()} {cls.getEnvActivation()} && '
        cmd += 'python -c "import hax; import os; print(os.path.dirname(hax.__file__))"'

        result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=True)
        module_path = result.stdout.strip()

        return os.path.join(module_path, "viewers", "server_loading_functions", "load_model.py")

    def defineBinaries(cls, env):
        installation_commands = []
        conda_activation_command = cls.getCondaActivationCmd()
        isDevelInstall = "--devel" in sys.argv

        # Create conda environment
        conda_env_installed = "conda_env_installed"
        commands_conda_env = f"{conda_activation_command} conda create -n hax -y python=3.10 && touch {conda_env_installed}"
        installation_commands.append((commands_conda_env, conda_env_installed))

        # Install Hax
        hax_installed = "hax_installed"
        if isDevelInstall:
            print("Installing Hax from devel branch and editable mode...")

            hax_pip_package = f"git+https://github.com/DavidHerreros/Hax@devel --src {emConfig.EM_ROOT}"
        else:
            hax_pip_package = "git+https://github.com/DavidHerreros/Hax@master"  # TODO: Change this in the future to released package in Pypi
        commands_hax = f"{conda_activation_command} {cls.getEnvActivation()} && pip install {hax_pip_package} && touch {hax_installed}"
        installation_commands.append((commands_hax, hax_installed))

        env.addPackage('hax', version=__version__,
                       commands=installation_commands,
                       tar="void.tgz",
                       default=True)

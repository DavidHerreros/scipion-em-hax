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
import numpy as np

from hax import Plugin

def getZernike3DArguments(particles):
    server_functions_path = Plugin.getAnnotateSpaceFunctionsPath()

    args = f"--server_functions_path {server_functions_path}"

    return args


def getHetSIRENArguments(particles):
    server_functions_path = Plugin.getAnnotateSpaceFunctionsPath()

    args = f"--server_functions_path {server_functions_path} --pickled_nn {particles.getFlexInfo().getAttr('modelPath')}"

    return args

def getReducedSpaceArguments(particles, save_path):
    consensus_space = []
    for particle in particles.iterItems():
        consensus_space.append(particle.getZRed())
    consensus_space = np.asarray(consensus_space)

    np.savetxt(os.path.join(save_path, "reduced_space.txt"), consensus_space)

    args = f"--z_space_reduced {os.path.join(save_path, 'reduced_space.txt')}"

    return args
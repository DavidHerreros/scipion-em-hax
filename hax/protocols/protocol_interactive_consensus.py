# **************************************************************************
# *
# * Authors:     Eduardo García Delgado (eduardo.garcia@cnb.csic.es) [1]
# *              David Herreros Calero (dherreos@cnb.csic.es) [1]
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC [1]
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

import pyworkflow.protocol.params as params
from pyworkflow import VERSION_1
import pyworkflow.utils as pwutils

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import Volume, ParticleFlex, SetOfParticlesFlex

import hax
import hax.constants as const
from hax.utils import getOutputSuffix

class JaxProtInteractiveFlexConsensus(ProtAnalysis3D, ProtFlexBase):
    """ Protocol to filter particles based on a FlexConsensus network interactively """
    _label = 'interactive consensus - FlexConsensus'
    _lastUpdateVersion = VERSION_1
    OUTPUT_PREFIX = 'consensusParticles'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                             Select the one you want to use.")

        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")

        group = form.addGroup("Data")
        group.addParam('inputSet', params.PointerParam,
                       label="Input particles", pointerClass='SetOfParticlesFlex')

        form.addSection(label='Network')
        form.addParam('flexConsensusProtocol', params.PointerParam, label="FlexConsensus trained network",
                       pointerClass='JaxProtTrainFlexConsensus',
                       help="Previously executed 'train - FlexConsensus'. "
                            "This will allow to load the network trained in that protocol to be used during "
                            "the prediction")

        form.addParam('batchSize', params.IntParam, default=1024, label='Number of samples in batch',
                       help="Determines how many images will be load in the GPU at any moment during training (set by "
                            "default to 1024 - you can control GPU memory usage easily by tuning this parameter to fit your "
                            "hardware requirements - we recommend using tools like nvidia-smi to monitor and/or measure "
                            "memory usage and adjust this value - keep also in mind that bigger batch sizes might be "
                            "less precise when looking for very local motions")
        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.convertInputStep)
        self._insertFunctionStep(self.predictStep)

    def convertInputStep(self):
        particles = self.inputSet.get()
        data_path = self._getExtraPath("data")
        if not os.path.isdir(data_path):
            pwutils.makePath(data_path)

        progName = particles.getFlexInfo().getProgName()
        data_file = progName + "_1.npy"

        z_flex = []
        for particle in particles.iterItems():
            z_flex.append(particle.getZFlex())
        z_flex = np.vstack(z_flex)
        latent_space = os.path.join(data_path, data_file)
        np.save(latent_space, z_flex)

    def predictStep(self):
        data_path = self._getFlexConsensusProtocol()._getExtraPath("data")
        out_path = self._getFlexConsensusProtocol()._getExtraPath()
        batch_size = self.batchSize.get()
        particles = self.inputSet.get()
        progName = particles.getFlexInfo().getProgName()
        input_space = progName + f"_{1}:" + os.path.join(data_path, progName + f"_{1}.npy")

        args = ("--input_space %s --batch_size %d --output_path %s " % (input_space, batch_size, out_path))

        if self._getFlexConsensusProtocol().setManual:
            lat_dim = self._getFlexConsensusProtocol().latDim.get()
            args += '--lat_dim %d ' % lat_dim

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = ''

        program = hax.Plugin.getProgram("flexconsensus", gpu)
        self.runJob(program, args + f'--mode predict --reload {out_path}', numberOfMpi=1)

    def _createOutput(self):
        inputSet = self.inputSet.get()
        selected_idx = np.loadtxt(self._getExtraPath("selected_idx.txt"))

        suffix = getOutputSuffix(self, SetOfParticlesFlex)
        partSet = self._createSetOfParticlesFlex(suffix, progName=inputSet.getFlexInfo().getProgName())

        partSet.copyInfo(inputSet)
        partSet.setHasCTF(inputSet.hasCTF())
        partSet.setAlignmentProj()

        idx = 0
        for particle in inputSet.iterItems():
            if idx in selected_idx:
                outParticle = ParticleFlex(progName=inputSet.getFlexInfo().getProgName())
                outParticle.copyInfo(particle)
                partSet.append(outParticle)
            idx += 1

        name = self.OUTPUT_PREFIX + suffix
        args = {}
        args[name] = partSet
        self._defineOutputs(**args)
        self._defineSourceRelation(self.inputSet, partSet)

    # --------------------------- UTILS functions --------------------------------------------
    def _getFlexConsensusProtocol(self):
        return self.flexConsensusProtocol.get()

    # ----------------------- VALIDATE functions ----------------------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []

        flexConsensusSets = self._getFlexConsensusProtocol().inputSets
        in_lat_dim = self.inputSet.get().getFirstItem().getZFlex().size
        dim_match = False

        for particle_set in flexConsensusSets:
            lat_dim = particle_set.get().getFirstItem().getZFlex().size
            if lat_dim == in_lat_dim:
                dim_match = True
                break

        if not dim_match:
            errors.append("The input particles' flexible information does not match the flexible space "
                          "dimension of the provided FlexConsensus network. Please, provide a set of particles "
                          "computed with some of the following programs:\n")
            progNames = []
            for particle_set in flexConsensusSets:
                progName = particle_set.get().getFlexInfo().getProgName()
                if progName not in progNames:
                    progNames.append(progName)
                    errors.append(f"     -{progName}")

        return errors
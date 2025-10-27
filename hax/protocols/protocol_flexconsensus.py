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


import os, shutil
import numpy as np

from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params
from pyworkflow.object import String
from pyworkflow import VERSION_1
import pyworkflow.utils as pwutils

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import Volume, ParticleFlex

import xmipp3
from xmipp3.convert import createItemMatrix, setXmippAttributes, writeSetOfParticles, geometryFromMatrix, matrixFromGeometry

import hax
import hax.constants as const

class JaxProtTrainFlexConsensus(ProtAnalysis3D, ProtFlexBase):
    """ Protocol to train a FlexConsensus network """
    _label = 'train - FlexConsensus'
    _lastUpdateVersion = VERSION_1

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

        form.addParam('inputSets', params.MultiPointerParam, label="Input particles", pointerClass='SetOfParticlesFlex')

        group = form.addGroup("Latent Space", expertLevel=params.LEVEL_ADVANCED)
        group.addParam('setManual', params.BooleanParam, default=False, label='Set manually latent space dimension?',
                       expertLevel=params.LEVEL_ADVANCED,
                       help="If set to No, consensus space dimensions will be set automatically to the minimum dimension "
                            "of all the input spaces.")

        group.addParam('latDim', params.IntParam, label='Latent space dimension',
                       expertLevel=params.LEVEL_ADVANCED, condition="setManual",
                       help="Dimension of the FlexConsensus bottleneck (latent space dimension)")

        form.addSection(label='Network')
        form.addParam('fineTune', params.BooleanParam, default=False, label='Fine tune previous network?',
                      help='If fineTune, a previously trained deepPose network will be fine tuned based on the '
                           'new input parameters.')

        group = form.addGroup("Network hyperparameters")
        group.addParam('epochs', params.IntParam, default=100, label='Number of training epochs')

        group.addParam('batch_size', params.IntParam, default=1024, label='Number of samples in batch',
                       help="Number of samples that will be used simultaneously for every training step. "
                            "We do not recommend to change this value unless you experience memory errors. "
                            "In this case, value should be decreased.")

        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.convertInputStep)
        self._insertFunctionStep(self.trainingPredictStep)

    def convertInputStep(self):
        data_path = self._getExtraPath("data")
        if not os.path.isdir(data_path):
            pwutils.makePath(data_path)

        idx = 0
        for inputSet in self.inputSets:
            particle_set = inputSet.get()

            progName = particle_set.getFlexInfo().getProgName()
            data_file = progName + f"_{idx}.npy"

            z_flex = []
            for particle in particle_set.iterItems():
                z_flex.append(particle.getZFlex())
            z_flex = np.vstack(z_flex)
            latent_space = os.path.join(data_path, data_file)
            np.save(latent_space, z_flex)

            idx += 1

    def trainingPredictStep(self):
        data_path = self._getExtraPath("data")
        out_path = self._getExtraPath()
        batch_size = self.batch_size.get()
        epochs = self.epochs.get()
        lat_dim = self.latDim.get()
        args = "--epochs %d --batch_size %d --output_path %s " % (epochs, batch_size, out_path)

        for file in os.listdir(data_path):
            args += '--input_space %s ' % os.path.join(data_path, file)

        if self.setManual:
            args += '--lat_dim %d ' % lat_dim

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = ''

        program = hax.Plugin.getProgram("flexconsensus", gpu)
        self.runJob(program,
                    args + f'--mode train --reload {self._getExtraPath("FlexConsensus")}'
                    if self.fineTune else args + '--mode train',
                    numberOfMpi=1)
        self.runJob(program, args + f'--mode predict --reload {self._getExtraPath("FlexConsensus")}', numberOfMpi=1)

    def createOutputStep(self):
        inputParticles = self.inputParticles.get()
        out_path_vols = self._getExtraPath('volumes')
        model_path = self._getExtraPath('FlexConsensus')

        latents_path = self._getExtraPath("latents")
        if not os.path.isdir(latents_path):
            pwutils.makePath(latents_path)
        latents_npy = [file for file in self._getExtraPath() if file.endswith('_consensus.npy')]
        latent_space = []
        for latent_npy in latents_npy:
            new_latent_npy = os.path.join(latents_path, latent_npy)
            shutil.move(self._getExtraPath(latent_npy), new_latent_npy)
            latent_space.append(np.load(new_latent_npy))

        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticlesFlex(progName=const.FLEXCONSENSUS)

        partSet.copyInfo(inputSet)
        partSet.setHasCTF(inputSet.hasCTF())
        partSet.setAlignmentProj()

        idx = 0
        for particle in inputSet.iterItems():
            outParticle = ParticleFlex(progName=const.FLEXCONSENSUS)
            outParticle.copyInfo(particle)
            outParticle.setZFlex(latent_space[idx])
            partSet.append(outParticle)
            idx += 1

        partSet.getFlexInfo().modelPath = String(model_path)

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = ''

        for file in latents_path:
            args = "--latents_file %s --output_path %s" % (file, out_path_vols)
            program = hax.Plugin.getProgram("decode_states_from_latents", gpu)
            self.runJob(program, args, numberOfMpi=1)

        outVols = self._createSetOfVolumes()
        outVols.setSamplingRate(inputParticles.getSamplingRate())
        for idx in range(latent_space.shape[0]):
            outVol = Volume()
            outVol.setSamplingRate(inputParticles.getSamplingRate())

            ImageHandler().scaleSplines(os.path.join(out_path_vols, f"decoded_volume_{idx:04d}.mrc"),
                                        os.path.join(out_path_vols, f"decoded_volume_{idx:04d}.mrc"),
                                        finalDimension=inputParticles.getXDim(), overwrite=True)

            ImageHandler().setSamplingRate(os.path.join(out_path_vols, f"decoded_volume_{idx:04d}.mrc"),
                                           inputParticles.getSamplingRate())

            outVol.setLocation(os.path.join(out_path_vols, f"decoded_volume_{idx:04d}.mrc"))
            outVols.append(outVol)

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(inputParticles, partSet)

        self._defineOutputs(outputVolumes=outVols)
        self._defineTransformRelation(inputParticles, outVols)

        # --------------------------- INFO functions -----------------------------

    def _summary(self):
        summary = []
        logFile = os.path.abspath(self._getLogsPath()) + "/run.stdout"
        with open(logFile, "r") as fi:
            for ln in fi:
                if ln.startswith("GPU memory has"):
                    summary.append(ln)
                    break
        return summary

    # --------------------------- UTILS functions --------------------------------------------

    # ----------------------- VALIDATE functions ----------------------------------------
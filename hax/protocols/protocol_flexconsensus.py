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
from pwem.objects import Volume, ParticleFlex

import hax
import hax.constants as const

class JaxProtTrainFlexConsensus(ProtAnalysis3D, ProtFlexBase):
    """
    Protocol to train a FlexConsensus network. This protocol learns a common
    latent representation from multiple flexible particle sets so that
    heterogeneous conformational information produced by different flexible
    analysis methods or independent datasets can be projected into a shared
    consensus space.

    AI Generated:

    Train FlexConsensus (JaxProtTrainFlexConsensus) — User Manual

    This protocol is designed to integrate several flexible cryo-EM latent
    spaces into a single common representation. In many practical workflows,
    different flexible reconstruction or heterogeneity analysis methods produce
    their own latent coordinates. Although each latent space may capture
    meaningful structural variability, direct comparison between them is often
    difficult because the coordinate systems are method-dependent. FlexConsensus
    addresses this problem by learning a consensus embedding that preserves the
    biologically relevant variability shared across the input spaces.

    The protocol takes as input multiple sets of flexible particles. Each input
    particle set must already contain latent coordinates produced by a previous
    flexible analysis protocol. During preparation, the protocol extracts the
    latent vectors from every particle and stores them as numerical arrays. In
    biological terms, this means that the network does not operate directly on
    particle images or volumes, but rather on previously learned descriptors of
    conformational variability.

    An important concept in this protocol is the consensus latent dimension.
    When left in automatic mode, the protocol determines the dimensionality of
    the consensus space from the input spaces, choosing a dimension compatible
    with all of them. This is generally the safest strategy for most users
    because it avoids imposing an overly complex representation. Advanced users
    may define the latent dimension manually when they want tighter control over
    the compression level or when prior biological knowledge suggests the
    expected complexity of the conformational landscape.

    The neural network training is controlled through a compact set of
    hyperparameters. The number of epochs determines how long the model learns
    the shared structure of the latent spaces. The batch size controls memory
    usage and optimization stability, which can be relevant when very large
    particle sets are used. The learning rate governs how quickly the consensus
    model adapts during optimization. In practice, conservative learning rates
    are usually preferred because latent-space integration often benefits from
    stable convergence rather than aggressive optimization.

    The protocol also supports fine tuning. When enabled, an existing
    FlexConsensus model can be reused and refined with additional datasets or
    updated latent representations. This is particularly useful in iterative
    biological projects where new particle subsets are added progressively or
    when consensus representations are refined after upstream flexible analysis
    improves.

    During execution, all input latent spaces are first exported and organized
    as independent numerical datasets. The training stage then learns a common
    latent manifold that relates all input spaces. After training, a prediction
    stage projects every input particle into that learned consensus space. This
    prediction step is important because it provides a direct correspondence
    between the original particle sets and the shared representation.

    The output consists of one particle set for each input set. Each output
    particle preserves the original particle metadata and alignment information,
    but now includes a new reduced latent coordinate corresponding to the
    consensus embedding. From a biological perspective, this makes it possible
    to compare conformational states across datasets or across different
    flexible reconstruction methods using a common coordinate system.

    In addition to the consensus coordinates, the protocol reports two
    quantitative quality measures for each particle. The consensus error
    estimates how well the particle is represented in the common latent space,
    while the representation error reflects how faithfully the original latent
    information is preserved. These quantities can be useful for downstream
    interpretation, for example when identifying particles that are poorly
    represented in the shared manifold or when evaluating whether certain
    datasets contribute incompatible structural information.

    In practical cryo-EM analysis, this protocol is especially valuable when
    several flexible analyses need to be interpreted jointly. For example,
    different reconstruction methods may emphasize different aspects of
    structural variability, or independent datasets may capture related but not
    identical conformational landscapes. FlexConsensus provides a principled
    way to unify those descriptions and examine their common structural trends.

    A useful practical strategy is to begin with automatic latent dimensionality
    and default optimization settings, then inspect the consensus and
    representation errors. If the consensus space appears biologically coherent,
    the resulting latent coordinates can be used for clustering, visualization,
    or comparison of conformational trajectories across experiments. In this
    sense, the protocol is not only a dimensionality reduction tool but also a
    bridge for interpreting heterogeneous structural variability in a unified
    biological framework.
    """
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

        form.addParam('inputSets', params.MultiPointerParam, label="Input particles", pointerClass='SetOfParticlesFlex',
                      important=True)

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
                      help='When set to Yes, you will be able to provide a previously trained FlexConsensus network to refine it with new '
                           'data. If set to No, you will train a new FlexConsensus network from scratch.')

        group = form.addGroup("Network hyperparameters")
        group.addParam('epochs', params.IntParam, default=100, label='Number of training epochs')

        group.addParam('batch_size', params.IntParam, default=1024, label='Number of samples in batch',
                       help="Determines how many images will be load in the GPU at any moment during training (set by "
                            "default to 1024 - you can control GPU memory usage easily by tuning this parameter to fit your "
                            "hardware requirements - we recommend using tools like nvidia-smi to monitor and/or measure "
                            "memory usage and adjust this value - keep also in mind that bigger batch sizes might be "
                            "less precise when looking for very local motions")

        group.addParam('learningRate', params.FloatParam, default=1e-5, label='Learning rate',
                       help="The learning rate (lr) sets the speed of learning. Think of the model as trying to find the "
                            "lowest point in a valley; the lr is the size of the step it takes on each attempt. A large "
                            "lr (e.g., 0.01) is like taking huge leaps — it's fast but can be unstable, overshoot the "
                            "lowest point, or cause NAN errors. A small lr (e.g., 1e-6) is like taking tiny shuffles — "
                            "it's stable but very slow and might get stuck before reaching the bottom. A good default is "
                            "often 0.0001. If training fails or errors explode, try making the lr 10 times smaller (e.g., "
                            "0.001 --> 0.0001).")

        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.convertInputStep)
        self._insertFunctionStep(self.trainingPredictStep)
        self._insertFunctionStep(self.createOutputStep)

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
        learningRate = self.learningRate.get()
        epochs = self.epochs.get()
        lat_dim = self.latDim.get()

        input_spaces = []
        idx = 0
        for inputSet in self.inputSets:
            particle_set = inputSet.get()

            progName = particle_set.getFlexInfo().getProgName()
            input_spaces.append(progName + f"_{idx}:" + os.path.join(data_path, progName + f"_{idx}.npy"))

            idx += 1

        args = ("--input_space %s --epochs %d --batch_size %d --output_path %s --learning_rate %s " %
                (" ".join(input_spaces), epochs, batch_size, out_path, learningRate))

        if self.setManual:
            args += '--lat_dim %d ' % lat_dim

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = None

        program = hax.Plugin.getProgram("flexconsensus", gpu)
        if not os.path.isdir(self._getExtraPath("FlexConsensus")):
            self.runJob(program,
                        args + f'--mode train --reload {self._getExtraPath()}'
                        if self.fineTune else args + '--mode train',
                        numberOfMpi=1)
        self.runJob(program, args + f'--mode predict --reload {self._getExtraPath()}', numberOfMpi=1)

    def createOutputStep(self):
        idx = 0
        for inputSet in self.inputSets:
            inputSet = inputSet.get()
            progName = inputSet.getFlexInfo().getProgName()

            outputSet = self._createSetOfParticlesFlex(progName=progName, suffix=f"_{idx}")
            outputSet.copyInfo(inputSet)
            outputSet.setHasCTF(inputSet.hasCTF())
            outputSet.setAlignmentProj()

            consensus_space = np.load(self._getExtraPath(progName + f"_{idx}_consensus.npy"))
            consensus_space_error = np.load(self._getExtraPath(progName + f"_{idx}_consensus_error.npy"))
            representation_error = np.load(self._getExtraPath(progName + f"_{idx}_representation_error.npy"))

            idl = 0
            for particle in inputSet.iterItems():
                outParticle = ParticleFlex(progName=const.FLEXCONSENSUS)
                outParticle.copyInfo(particle)
                outParticle.setZRed(consensus_space[idl])
                outParticle.getFlexInfo().setAttr("consensus_error", float(consensus_space_error[idl]))
                outParticle.getFlexInfo().setAttr("representation_error", float(representation_error[idl]))
                outputSet.append(outParticle)
                idl += 1

            self._defineOutputs(**{f"outputParticles_{progName}_{idx}": outputSet})
            self._defineTransformRelation(inputSet, outputSet)

            idx += 1

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
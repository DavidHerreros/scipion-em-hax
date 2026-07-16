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
    """
    Interactively filters particles in a flexible latent space using a previously trained
    FlexConsensus neural network. The protocol evaluates the latent descriptors associated
    with each particle and selects only those that satisfy the consensus learned by the
    network, producing a refined particle subset suitable for downstream structural analysis.

    AI Generated:

    Interactive Consensus - FlexConsensus (JaxProtInteractiveFlexConsensus) — User Manual

        Overview

        The Interactive FlexConsensus protocol is designed to perform particle selection in
        flexible cryo-EM datasets using a previously trained FlexConsensus neural network.
        Instead of operating directly on particle images, the protocol works in the latent
        conformational space associated with each particle. This makes it especially useful
        in workflows where conformational variability has already been characterized using
        flexible reconstruction methods such as HetSIREN, ReconSIREN, Zernike3D, or related
        approaches.

        From a biological perspective, this protocol helps identify particles that belong to
        coherent conformational regions while excluding particles whose latent descriptors are
        inconsistent with the consensus model. In practice, this provides a principled way to
        clean heterogeneous datasets before downstream refinement, classification, variability
        analysis, or structural interpretation.

        Inputs and General Workflow

        The protocol requires two inputs. The first is a set of flexible particles, meaning a
        particle set where each particle already contains a latent descriptor representing its
        conformational state. The second is a previously trained FlexConsensus network, which
        defines the latent-space manifold used for evaluating whether particles belong to a
        consistent conformational population.

        During execution, the protocol first extracts the latent coordinates from all input
        particles and stores them in a numerical latent-space file. These coordinates are then
        passed to the trained FlexConsensus model, which predicts which particles satisfy the
        learned consensus criteria. The final output is generated interactively by reading the
        selected particle indices and constructing a new subset containing only the accepted
        particles.

        Biological Interpretation

        In flexible cryo-EM analysis, latent spaces often contain mixtures of meaningful
        conformational states together with transitional particles, noisy assignments, or
        regions that are poorly sampled. The role of this protocol is not to create new
        conformational information, but to identify particles that occupy latent-space regions
        compatible with the previously learned consensus representation.

        For biological users, this means that the output typically represents a more coherent
        structural population. This can be especially valuable when preparing particles for
        high-resolution refinement, focused heterogeneity analysis, or interpretation of
        continuous conformational landscapes. Importantly, the protocol preserves the original
        particle metadata and flexible descriptors, so the filtered particles remain directly
        compatible with later flexible-analysis steps.

        Batch Size and Computational Behavior

        The main user-adjustable runtime parameter is the batch size. This controls how many
        latent vectors are processed simultaneously on the GPU during prediction. Larger batch
        sizes usually improve throughput but increase GPU memory consumption. Since the
        protocol operates only on latent descriptors rather than full particle images, memory
        requirements are generally modest compared with image-based neural-network protocols.

        In most biological workflows, the default value is sufficient. Larger values may be
        beneficial for very large particle sets, while smaller values may help when GPU memory
        is limited.

        Latent-Space Compatibility

        A critical requirement is that the latent-space dimensionality of the input particles
        matches at least one of the latent spaces used to train the provided FlexConsensus
        model. The protocol explicitly checks this before execution.

        Biologically, this means that the particles must originate from a compatible flexible
        analysis framework. If the dimensionality does not match, the protocol refuses to run,
        preventing biologically meaningless predictions caused by incompatible latent
        representations.

        Outputs and Their Meaning

        The output is a new flexible particle set containing only the particles selected by
        the consensus network. The output preserves the flexible descriptors, alignment
        information, and all relevant metadata inherited from the original input set.

        In practical terms, the resulting subset can be interpreted as a consensus-filtered
        population of particles that better represents the conformational manifold learned by
        the FlexConsensus model. These particles can be directly used for subsequent
        refinement, clustering, structural comparison, or focused biological interpretation.

        Practical Recommendations

        In typical biological applications, this protocol is most useful after a flexible
        analysis has revealed a broad conformational landscape and the user wishes to isolate
        the most coherent region of that space. It is especially valuable when noisy latent
        assignments or poorly populated conformational tails interfere with downstream
        analysis.

        When using this protocol, it is good practice to ensure that the FlexConsensus model
        was trained on particle sets that are biologically comparable to the current input.
        The more consistent the training data and prediction data are, the more meaningful the
        resulting consensus selection will be.

        Final Perspective

        For cryo-EM studies of structural heterogeneity, particle filtering in latent space is
        often more biologically informative than filtering based solely on image similarity.
        The Interactive FlexConsensus protocol provides a practical mechanism to translate
        learned conformational consensus into a physically meaningful subset of particles.

        In this sense, it acts as a bridge between flexible manifold learning and classical
        downstream cryo-EM refinement, allowing users to focus subsequent analysis on
        particles that are both computationally consistent and biologically interpretable.
    """
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
        data_file = progName + ".npy"

        z_flex = []
        for particle in particles.iterItems():
            z_flex.append(particle.getZFlex())
        z_flex = np.vstack(z_flex)
        latent_space = os.path.join(data_path, data_file)
        np.save(latent_space, z_flex)

    def predictStep(self):
        data_path = self._getExtraPath("data")
        batch_size = self.batchSize.get()
        particles = self.inputSet.get()
        progName = particles.getFlexInfo().getProgName()
        input_space = progName + ":" + os.path.join(data_path, progName + ".npy")

        args = ("--input_space %s --batch_size %d --output_path %s " % (input_space, batch_size, self._getExtraPath()))

        if self._getFlexConsensusProtocol().setManual:
            lat_dim = self._getFlexConsensusProtocol().latDim.get()
            args += '--lat_dim %d ' % lat_dim

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = ''

        program = hax.Plugin.getProgram("flexconsensus", gpu)
        self.runJob(program, args + f'--mode predict --reload {self._getFlexConsensusProtocol()._getExtraPath()}', numberOfMpi=1)

    def _createOutput(self):
        inputSet = self.inputSet.get()
        selected_idx = np.loadtxt(self._getExtraPath("selected_idx.txt")).astype(int)
        particle_ids = list(inputSet.getIdSet())

        suffix = getOutputSuffix(self, SetOfParticlesFlex)
        partSet = self._createSetOfParticlesFlex(suffix, progName=inputSet.getFlexInfo().getProgName())

        partSet.copyInfo(inputSet)
        partSet.setHasCTF(inputSet.hasCTF())
        partSet.setAlignmentProj()

        for idx in selected_idx:
            partSet.append(inputSet[particle_ids[idx]].clone())

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
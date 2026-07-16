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
    Protocol to filter particles based on a FlexConsensus network interactively.

    AI Generated:

    Interactive Consensus – FlexConsensus (JaxProtInteractiveFlexConsensus) — User Manual
        Overview

        The Interactive Consensus – FlexConsensus protocol applies a previously trained
        FlexConsensus network to a new set of particles in order to project them into a
        consensus latent representation and enable interactive particle selection.

        Its main purpose is not to train a new consensus model, but to reuse an already
        learned consensus space so that new particles can be evaluated, explored, and
        filtered according to their position in that common latent manifold.

        In cryo-EM workflows, this protocol is especially useful when several heterogeneous
        datasets have already been unified through FlexConsensus and the user wants to
        classify, inspect, or select particles from an additional dataset in a way that is
        consistent with the previously learned conformational landscape.

        Inputs and General Workflow

        The protocol requires two main inputs:

            1. Input particles
               A SetOfParticlesFlex containing particles with previously computed latent
               coordinates.

            2. A trained FlexConsensus network
               This must come from a previously executed
               "train - FlexConsensus" protocol.

        The workflow is divided into three conceptual stages:

            1. Latent extraction
               The latent coordinates stored in the input particles are extracted and saved
               into a temporary NumPy file.

            2. Consensus prediction
               The trained FlexConsensus model projects those latent coordinates into the
               learned consensus representation.

            3. Interactive filtering
               After interactive selection, the chosen particles are written into a new
               output particle set.

        Latent Space Compatibility

        The most important requirement of this protocol is latent-space compatibility.

        The input particles must belong to a latent representation whose dimensionality
        matches at least one of the latent spaces used during the original FlexConsensus
        training.

        This does not necessarily mean that the particles must come from the same program,
        but their latent dimensionality must be compatible with one of the spaces known by
        the trained network.

        During validation, the protocol checks this explicitly.

        If no compatible latent dimension is found, execution is stopped.

        From a biological perspective, this requirement ensures that new particles are
        projected into a consensus space that remains meaningful with respect to the
        original conformational landscape.

        Prediction Step

        During prediction, the protocol calls the previously trained FlexConsensus model
        in prediction mode only.

        No training or parameter update is performed.

        The latent coordinates of the input particles are passed through the trained
        network, producing the consensus-space representation needed for interactive
        inspection.

        If the original FlexConsensus training used a manually fixed latent dimension,
        that same dimension is passed again here to preserve consistency.

        GPU Execution

        The protocol supports GPU acceleration.

        If GPU execution is enabled, the selected GPU is passed directly to the external
        FlexConsensus prediction program.

        GPU acceleration is particularly useful when projecting very large particle sets,
        since the prediction step may involve thousands or millions of latent samples.

        Batch Size

        The batch size controls how many latent samples are processed simultaneously.

        Larger values improve throughput but require more GPU memory.

        Smaller values reduce memory pressure and may be preferable on limited hardware.

        In practical cryo-EM workflows, the default batch size is usually appropriate,
        although very large datasets may require tuning depending on available GPU memory.

        Interactive Particle Selection

        After prediction, the protocol is designed for interactive particle filtering.

        A subset of particles is selected externally from the consensus representation,
        and the selected particle indices are stored in:

            selected_idx.txt

        The protocol then creates a new output particle set containing only the selected
        particles.

        Output Generation

        The output is a new SetOfParticlesFlex.

        For every selected particle:

            - original particle metadata are preserved
            - acquisition information is preserved
            - CTF information is preserved
            - alignment information is preserved

        The protocol does not alter the particle content itself.

        It only filters the original particle population according to the interactive
        selection performed in consensus space.

        Biological Interpretation

        From a biological perspective, this protocol provides a practical way to isolate
        regions of conformational interest inside a previously learned consensus manifold.

        Typical applications include:

            - selecting particles corresponding to specific conformational states
            - isolating continuous transitions along flexible trajectories
            - removing particles that occupy poorly populated or noisy regions
            - extracting subsets for downstream refinement or focused analysis

        Because filtering is performed in a consensus latent space rather than directly in
        image space, the selected subsets often reflect biologically meaningful
        conformational organization.

        Practical Recommendations

        In routine use, it is generally advisable to:

            - use a FlexConsensus model trained on biologically representative datasets
            - ensure that the input particles belong to a compatible latent space
            - inspect the consensus embedding visually before selecting particles
            - use conservative selections when preparing particles for downstream refinement

        If the input particles originate from a latent representation that differs too much
        from the training data, projections into consensus space may become difficult to
        interpret biologically.

        Validation Rules

        Before execution, the protocol verifies that:

            - the latent dimensionality of the input particles matches at least one of the
              latent spaces used during FlexConsensus training

        If no compatible latent space is found, execution stops and the protocol reports
        which source programs were used to train the available consensus network.

        Final Perspective

        Interactive Consensus – FlexConsensus is best understood as a latent-space
        selection tool rather than a learning protocol.

        It allows new particles to be interpreted within an already established
        conformational consensus landscape, making it particularly valuable for
        exploratory cryo-EM heterogeneity analysis, targeted particle selection,
        and biologically guided dataset refinement.
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
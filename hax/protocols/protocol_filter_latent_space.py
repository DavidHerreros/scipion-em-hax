# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreos@cnb.csic.es) [1]
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

from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params
from pyworkflow.object import Float
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_1
from pyworkflow.utils import getExt, makePath

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import Volume, Particle

import xmipp3
from xmipp3.convert import writeSetOfParticles, matrixFromGeometry

import hax

class JaxProtFilterLatents(ProtAnalysis3D, ProtFlexBase):
    """
    Protocol to filter automatically latent conformational spaces.

    AI Generated:

    Filter Latent Space (JaxProtFilterLatents) — User Manual
        Overview

        The Filter Latent Space protocol analyzes a previously computed latent conformational space and
        automatically removes particles whose latent coordinates behave as outliers with respect to the
        global conformational landscape. Its main purpose is to retain particles that belong to the
        coherent manifold learned during flexible reconstruction while excluding isolated points that may
        arise from poor alignments, reconstruction artifacts, or rare unstable conformations.

        In practical cryo-EM flexible analysis, latent spaces often contain a dense continuous region
        representing meaningful structural variability together with scattered points that do not follow
        the same geometric organization. This protocol provides a convenient way to clean that latent
        representation before downstream analysis such as clustering, trajectory exploration, volume
        decoding, or conformational interpretation.

        Inputs and General Workflow

        The protocol requires as input a set of flexible particles, meaning particles that already carry
        latent coordinates produced by a previous flexible reconstruction protocol. These latent vectors
        represent each particle in a reduced conformational space where geometric proximity usually
        reflects structural similarity.

        During execution, the protocol first extracts the latent coordinates associated with every input
        particle and stores them as a numerical latent matrix. This matrix is then passed to the
        filtering engine, which evaluates the local neighborhood structure of each point in latent space.
        After filtering is complete, only the particles whose latent coordinates satisfy the selected
        acceptance criterion are copied into a new output particle set.

        Because the output preserves the original particle metadata, this protocol behaves as a
        conformational selection step rather than a new reconstruction step. It therefore fits naturally
        between flexible embedding and later structural interpretation.

        Neighborhood-Based Detection of Outliers

        The central idea of the protocol is that meaningful conformational states tend to occupy locally
        coherent regions of latent space. Each latent coordinate is compared to its nearest neighbors in
        order to estimate whether it behaves like a regular member of the manifold or as an isolated
        outlier.

        The number of neighbors controls the scale at which the latent landscape is examined. Smaller
        values emphasize very local structure and are useful when the conformational manifold contains
        fine branches, subtle transitions, or narrow pathways. Larger values provide a broader view of
        the latent organization and can be advantageous when the latent space is smooth and globally
        continuous.

        From a biological perspective, choosing too few neighbors may preserve tiny isolated clusters
        that reflect noise rather than real structural states, whereas choosing too many neighbors may
        oversmooth the landscape and remove legitimate but sparsely populated conformations.

        Outlier Threshold and Biological Interpretation

        The filtering decision is controlled by an outlier distance threshold expressed as a Z-score.
        Particles whose latent coordinates fall within the accepted threshold are retained, while those
        exceeding the threshold are discarded.

        Lower thresholds produce stricter filtering. This is often useful when the latent space contains
        obvious scattered points, strong reconstruction artifacts, or noisy regions that interfere with
        interpretation. Higher thresholds are more permissive and may be preferable when exploring
        heterogeneous systems where biologically meaningful states are expected to be rare or only weakly
        populated.

        In biological practice, this parameter should be interpreted conservatively. A particle that
        appears as an outlier in latent space is not necessarily biologically irrelevant. Rare
        intermediate states, transient motions, or low-population conformations may naturally occupy
        sparse regions. For this reason, aggressive filtering should generally be followed by visual
        inspection or independent validation.

        Computational Considerations

        The protocol allows the user to control batch size during latent filtering. This parameter mainly
        affects GPU memory usage and computational throughput. Larger batches generally improve execution
        efficiency on modern accelerators, whereas smaller batches can help when hardware resources are
        limited.

        GPU execution is supported but optional. Since the protocol operates directly on latent vectors
        rather than raw particle images, its computational demands are usually moderate compared with
        reconstruction or neural network training.

        Outputs and Their Interpretation

        The output is a new flexible particle set containing only the particles whose latent coordinates
        passed the filtering criterion. Each output particle preserves the metadata and latent
        representation of its original counterpart, but the overall set becomes restricted to the
        selected conformational manifold.

        This filtered set is particularly useful for downstream analyses that are sensitive to outliers.
        Clustering becomes more stable, latent visualization becomes easier to interpret, and decoded
        volumes tend to reflect cleaner conformational trends rather than isolated artifacts.

        Importantly, the protocol does not modify latent coordinates. It only performs selection.
        Therefore, the resulting dataset remains directly compatible with subsequent flexible analysis
        tools.

        Practical Recommendations

        In most biological workflows, it is advisable to begin with moderate neighborhood values and a
        conservative threshold, then inspect the latent distribution after filtering. When the original
        latent space contains obvious isolated points far from the main manifold, filtering usually
        improves interpretability substantially.

        For highly flexible proteins, however, caution is essential. Sparse latent regions can sometimes
        correspond to genuine conformational intermediates rather than noise. In these situations, a
        permissive threshold is often preferable during exploratory analysis, followed by stricter
        filtering only after the structural meaning of the landscape becomes clearer.

        Final Perspective

        For cryo-EM flexible analysis, latent filtering is best understood as a refinement of the
        conformational landscape rather than a simple cleanup operation. By removing isolated latent
        outliers while preserving the coherent manifold, the protocol helps reveal the dominant
        structural organization of heterogeneous datasets. Used carefully, it improves downstream
        interpretability without replacing biological judgment about which conformational states are
        meaningful.
    """
    _label = 'filter latent space'
    _lastUpdateVersion = VERSION_1

    # --------------------------- DEFINE param functions -----------------------
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
        group.addParam('inputParticles', params.PointerParam, label="Input particles",
                       pointerClass='SetOfParticlesFlex', important=True)

        group.addParam('batchSize', params.IntParam, default=1024, label='Number of images in batch',
                       help="Determines how many images will be load in the GPU at any moment during training (set by "
                            "default to 1024 - you can control GPU memory usage easily by tuning this parameter to fit your "
                            "hardware requirements - we recommend using tools like nvidia-smi to monitor and/or measure "
                            "memory usage and adjust this value - keep also in mind that bigger batch sizes might be "
                            "less precise when looking for very local motions")

        group = form.addGroup("Filter parameters")
        group.addParam("neighbours", params.IntParam, default=10, label="Number of neighbours",
                      help="Number of nearest neighbours to be search for each landscape sample. Smaller "
                           "values will better approximate the locality features of a sample, while global "
                           "values will capture more general landscape features.")

        group.addParam('outliersThreshold', params.FloatParam, default=1.0,
                      label="Outliers distance threshold",
                      help='Z-Score value from 0 to infinite. Only coordinates with a Z-Score smaller than '
                           'or equal to the threshold will be kept in the output')

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.writeLatentsStep)
        self._insertFunctionStep(self.filterLatentsStep)
        self._insertFunctionStep(self.createOutputStep)

    def writeLatentsStep(self):
        inputParticles = self.inputParticles.get()

        # Save latents to .npy file
        latents = []
        for particle in inputParticles.iterItems():
            latents.append(particle.getZFlex())
        latents = np.vstack(latents)
        latents_file = self._getExtraPath("latents.npy")
        np.save(latents_file, latents)


    def filterLatentsStep(self):
        latents_file = self._getExtraPath("latents.npy")
        thr = self.outliersThreshold.get()
        n_neighbours = self.neighbours.get()
        batch_size = self.batchSize.get()


        args = (f"--latents {latents_file} --thr {thr} --n_neighbours {n_neighbours} --return_ids "
                f"--batch_size {batch_size} --output_path {self._getExtraPath()}")

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = None

        program = hax.Plugin.getProgram("filter_latents", gpu)
        self.runJob(program, args, numberOfMpi=1)

    def createOutputStep(self):
        inputSet = self.inputParticles.get()

        filter_ids = np.load(self._getExtraPath("filtered_latents.npy")).astype(int)
        particle_ids = list(inputSet.getIdSet())

        partSet = self._createSetOfParticlesFlex(progName=inputSet.getFlexInfo().getProgName())
        partSet.copyInfo(inputSet)
        partSet.setHasCTF(inputSet.hasCTF())

        for idx in filter_ids:
            partSet.append(inputSet[particle_ids[idx]].clone())

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(inputSet, partSet)


    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        summary = []
        return summary

    # ----------------------- VALIDATE functions -----------------------
    def _validate(self):
        """ Try to find errors on define params. """
        errors = []
        return errors

    def _warnings(self):
        warnings = []
        return warnings

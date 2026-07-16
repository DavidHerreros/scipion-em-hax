# **************************************************************************
# *
# * Authors:     David Herreros (dherreros@cnb.csic.es)
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


import numpy as np
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params

from pwem.objects import Volume
from pwem.protocols import ProtReconstruct3D

from xmipp3.convert import writeSetOfParticles

import hax
import hax.constants as const


class JaxProtReconstructMoDART(ProtReconstruct3D):
    """
    Reconstructs a 3D volume from a set of aligned particle projections using the MoDART algorithm.
    The protocol estimates a density map directly from experimental images and can optionally start
    from an initial reference volume, apply masking constraints, incorporate symmetry information,
    and perform motion-aware reconstruction when flexible-particle information is available.

    AI Generated:

    Reconstruct MoDART (JaxProtReconstructMoDART) — User Manual

        Overview

        The Reconstruct MoDART protocol generates a three-dimensional density map from a set of
        aligned cryo-EM particle images. Its purpose is to convert the collection of two-dimensional
        projections into a volumetric reconstruction that represents the underlying molecular
        structure. In standard cryo-EM workflows, this corresponds to the stage where particle
        alignment information is transformed into a physically interpretable map.

        For biological users, this protocol is useful both when obtaining a first reconstruction from
        a curated particle set and when refining a reconstruction using additional prior information.
        The resulting volume can serve as a map for structural interpretation, downstream refinement,
        or comparison with other conformational states.

        Inputs and General Workflow

        The protocol requires a set of particles with projection alignment parameters. These alignment
        parameters define the orientation of each particle and are therefore essential for computing a
        meaningful reconstruction. The input particle metadata are first converted into an internal
        metadata representation suitable for the reconstruction engine.

        Reconstruction then proceeds by combining the aligned particle projections according to the
        specified acquisition geometry and the selected correction settings. The final output is a 3D
        map written with the same sampling rate as the input dataset, ensuring consistency with the
        original experimental scale.

        Initial Map and Reconstruction Mask

        An optional initial map can be provided to guide reconstruction. When present, this volume is
        used as the starting point of the iterative estimation process. This can be useful when prior
        structural knowledge exists or when continuing analysis from a previous reconstruction.

        A reconstruction mask may also be supplied. The mask restricts the region in which density is
        estimated, which can improve computational efficiency and reduce the influence of empty solvent
        regions. From a biological perspective, the mask should include the molecular envelope while
        excluding unnecessary background. Poorly chosen masks may artificially constrain the
        reconstruction or suppress genuine peripheral density.

        CTF Handling

        The protocol supports several ways of handling the contrast transfer function. The selected
        mode determines whether the CTF is ignored, applied to simulated projections, corrected by
        Wiener filtering, or assumed to have been corrected previously.

        In practical biological workflows, this choice should match the preprocessing history of the
        particles. Maintaining consistency between the particle images and the reconstruction model is
        important for obtaining interpretable density and avoiding systematic artifacts.

        Symmetry

        Symmetry can be imposed during reconstruction through the symmetry group parameter. For
        asymmetric particles, the correct choice is c1. For symmetric complexes, applying the proper
        symmetry often improves signal-to-noise ratio and can substantially enhance the quality of the
        final map.

        Biologically, symmetry should only be imposed when it reflects the true structural state of
        the specimen. Applying incorrect symmetry may force nonphysical averaging and obscure relevant
        asymmetries or flexible regions.

        Reconstruction Modes

        The protocol offers two main operating modes. In standard reconstruction mode, all particles
        contribute to the estimation of a single final volume. This is the typical choice for routine
        map generation.

        In gold-standard mode, the dataset is internally split into two independent halves and two
        half-maps are reconstructed. These half-maps are essential for downstream resolution
        estimation, Fourier shell correlation analysis, and local resolution calculations. For most
        publication-oriented cryo-EM analyses, gold-standard reconstruction is usually the preferred
        option because it enables more rigorous validation of map quality.

        Motion Correction

        When the input particles originate from flexible-analysis workflows such as HetSIREN or
        Zernike3Deep, the protocol can optionally perform motion-aware reconstruction. In this mode,
        particle-specific conformational information is used to reduce motion-blur effects during map
        estimation.

        From a biological standpoint, this option can be particularly valuable for flexible complexes
        in which conformational variability would otherwise smear fine structural details. However, it
        should only be enabled when the particle set contains compatible flexibility metadata.

        Performance and Data Loading

        The reconstruction can be executed using GPU acceleration, which is the intended operating
        mode for most practical datasets. Batch size controls how many particle images are processed at
        the same time and therefore directly affects GPU memory usage.

        Particle loading can be configured either through RAM loading or memory-mapped disk access.
        Loading images into RAM generally provides the highest throughput when memory allows it. For
        larger datasets, using an SSD or NVMe scratch folder can significantly improve performance by
        reducing disk I/O bottlenecks.

        Outputs and Interpretation

        The primary output is a reconstructed volume sampled at the same pixel size as the input
        particles. In gold-standard mode, two additional half-maps are also produced and associated
        with the final volume.

        Biologically, the resulting map should be interpreted as a reconstruction consistent with the
        supplied particle orientations, preprocessing assumptions, and reconstruction constraints. The
        map quality therefore depends strongly on the accuracy of alignment, the suitability of masking
        and symmetry choices, and the degree of conformational heterogeneity in the particle set.

        Practical Recommendations

        For most routine reconstructions, it is often advisable to begin with standard reconstruction
        mode, use c1 symmetry unless symmetry is well established, and avoid excessive masking unless
        the molecular boundaries are known with confidence.

        When flexible particles are available, motion correction can improve local detail, especially
        in heterogeneous systems. For final-resolution assessment and publication-quality analysis,
        gold-standard reconstruction should generally be preferred.

        Final Perspective

        For cryo-EM users, MoDART reconstruction is the stage where aligned particle information is
        transformed into an interpretable three-dimensional density map. Its reliability depends not
        only on computational settings but also on biologically informed choices regarding symmetry,
        masking, initial references, and flexibility handling. Careful parameter selection therefore
        has a direct impact on the structural conclusions drawn from the resulting reconstruction.
    """
    _label = 'reconstruct MoDART'

    def __init__(self, **args):
        ProtReconstruct3D.__init__(self, **args)

    #--------------------------- DEFINE param functions --------------------------------------------   
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addHidden(
            params.USE_GPU,
            params.BooleanParam,
            default=True,
            label="Use GPU for execution",
            help="This protocol has both CPU and GPU implementation.\
                                         Select the one you want to use.",
        )

        form.addHidden(
            params.GPU_LIST,
            params.StringParam,
            default="0",
            expertLevel=params.LEVEL_ADVANCED,
            label="Choose GPU IDs",
            help="Add a list of GPU devices that can be used",
        )

        group = form.addGroup("Data")
        group.addParam('inputParticles', params.PointerParam, pointerClass='SetOfParticles, SetOfParticlesFlex',
                      pointerCondition='hasAlignmentProj',
                      label="Input particles",  
                      help='Select the input images from the project.')

        group.addParam('initialMap', params.PointerParam, pointerClass='Volume',
                      label="Initial map",
                      allowsNull=True,
                      help='If provided, this map will be used as the initialization of the reconstruction '
                           'process. Otherwise, an empty volume will be used')

        group.addParam('recMask', params.PointerParam, pointerClass='VolumeMask',
                      allowsNull=True,
                      label="Reconstruction mask",
                      help="Mask used to restrict the reconstruction space to increase performance.")

        group.addParam('ctfType', params.EnumParam, choices=['None', 'Apply', 'Wiener', 'Precorrect'],
                       default=1, label="CTF correction type",
                       display=params.EnumParam.DISPLAY_HLIST,
                       expertLevel=params.LEVEL_ADVANCED,
                       help="* *None*: CTF will not be considered\n"
                            "* *Apply*: CTF is applied to the projection generated from the reference map\n"
                            "* *Wiener*: input particle is CTF corrected by a Wiener fiter\n"
                            "* *Precorrect: similar to Wiener but CTF has already been corrected")

        group = form.addGroup("Motion correction")
        group.addParam('doMotionCorrection', params.BooleanParam, default=False,
                      condition="inputParticles and isinstance(inputParticles, SetOfParticlesFlex)",
                      label="Correct motion blurred artifacts?",
                      help="Correct the conformation of the particles during the reconstruct process "
                           "to reduce motion blurred artifacts and increase resolution. Note that this "
                           "option requires that the particles come from HetSIREN (mass transport) or Zernike3Deep. "
                           "Otherwise, the parameter should be set to 'No'")

        group = form.addGroup("Symmetry")
        group.addParam('symmetryGroup', params.StringParam, default="c1",
                      label='Symmetry group',
                      help='If no symmetry is present, give c1')

        group = form.addGroup("Reconstruction modes")
        group.addParam('mode', params.EnumParam, choices=['Reconstruct', 'Gold standard'],
                      default=0, display=params.EnumParam.DISPLAY_HLIST,
                      label="Reconstruction mode",
                      help="\t * Reconstruct: usual reconstruction of a single volume using all the images "
                           "in the dataset\n"
                           "\t * Gold standard: volumes halves reconstruction for FSC and local resolution "
                           "computations\n")

        form.addSection(label='Data loading')
        form.addParam('lazyLoad', params.BooleanParam, default=False,
                      expertLevel=params.LEVEL_ADVANCED, label='Lazy loading into RAM',
                      help='If provided, images will be loaded to RAM. This is recommended if you want the best performance '
                           'and your dataset fits in your RAM memory. If this flag is not provided, '
                           'images will be memory mapped. When this happens, the program will trade disk space for performance. '
                           'Thus, during the execution additional disk space will be used and the performance '
                           'will be slightly lower compared to loading the images to RAM. Disk usage will be back to normal '
                           'once the execution has finished.')

        form.addParam('scratchFolder', params.PathParam,
                      condition="not lazyLoad",
                      label='Path to SSD scratch folder',
                      help='If you are not loading the images to RAM, we strongly recommend to provide here a path to a folder in '
                           'a SSD/NVME disk to speed up the data loading. In general, you can expected a decrease in training performance when '
                           'loading images > 256px on a HDD disk.')

        group = form.addGroup("Network hyperparameters")
        group.addParam('batchSize', params.IntParam, default=64, label='Number of images in batch',
                       help="Determines how many images will be load in the GPU at any moment during training (set by "
                            "default to 64 - you can control GPU memory usage easily by tuning this parameter to fit your "
                            "hardware requirements - we recommend using tools like nvidia-smi to monitor and/or measure "
                            "memory usage and adjust this value - keep also in mind that bigger batch sizes might be "
                            "less precise when looking for very local motions")

        form.addParallelSection(threads=4, mpi=1)

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'imgsFn': self._getExtraPath('input_particles.xmd'),
            'outputVol': self._getExtraPath('modart_map.mrc'),
            'outputHalfFirst': self._getExtraPath('modart_first_half.mrc'),
            'outputHalfSecond': self._getExtraPath('modart_second_half.mrc'),
        }
        self._updateFilenamesDict(myDict)

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep(self.writeMetaDataStep)
        self._insertFunctionStep(self.reconstructStep)
        self._insertFunctionStep(self.createOutputStep)


    #--------------------------- STEPS functions --------------------------------------------
    def writeMetaDataStep(self):
        imgsFn = self._getFileName('imgsFn')
        inputParticles = self.inputParticles.get()
        writeSetOfParticles(inputParticles, imgsFn)

    def reconstructStep(self):
        imgsFn = self._getFileName('imgsFn')
        sr = self.inputParticles.get().getSamplingRate()
        symmetryGroup = self.symmetryGroup.get()
        batchSize = self.batchSize.get()
        output_path = self._getExtraPath()

        args = (f"--md {imgsFn} --sr {sr} --symmetry_group {symmetryGroup} --batch_size {batchSize} "
                f"--output_path {output_path} ")

        if self.initialMap.get():
            args += f"--vol {self.initialMap.get().getFileName()} "

        if self.recMask.get():
            args += f"--mask {self.recMask.get().getFileName()} "

        if self.ctfType != 0:
            if self.ctfType.get() == 1:
                args += "--ctf_type apply "
            elif self.ctfType.get() == 2:
                args += "--ctf_type wiener "
            elif self.ctfType.get() == 3:
                args += "--ctf_type precorrect "
        else:
            args += "--ctf_type None "

        if self.mode.get() == 1:
            args += "--reconstruct_halves "

        if self.doMotionCorrection.get():
            args += f"--motion_correction {self.inputParticles.get().getFlexInfo().modelPath.get()} "

        if self.lazyLoad:
            args += "--load_images_to_ram "
        else:
            if self.scratchFolder.get() is not None:
                args += f"--ssd_scratch_folder {self.scratchFolder.get()} "

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = None

        program = hax.Plugin.getProgram("modart", gpu)
        self.runJob(program, args, numberOfMpi=1)

    def createOutputStep(self):
        volume = Volume()
        volume.setFileName(self._getFileName('outputVol'))
        volume.setSamplingRate(self.inputParticles.get().getSamplingRate())

        if self.mode.get() == 1:
            volume.setHalfMaps([self._getFileName('modart_first_half'), self._getFileName('modart_second_half')])

        # Set correct sampling rate in volume header
        ImageHandler().setSamplingRate(self._getExtraPath("modart_map.mrc"), self.inputParticles.get().getSamplingRate())

        self._defineOutputs(outputVolume=volume)
        self._defineSourceRelation(self.inputParticles, volume)

    #--------------------------- INFO functions -------------------------------------------- 
    def _summary(self):
        """ Should be overriden in subclasses to 
        return summary message for NORMAL EXECUTION. 
        """
        return []

    def _validate(self):
        errors = []

        mask = self.recMask.get()
        if mask:
            data = ImageHandler(mask.getFileName()).getData()
            if not np.all(np.logical_or(data == 0, data == 1)):
                errors.append("Mask provided is not binary. Please, provide a binary mask")

        return errors

    #--------------------------- UTILS functions --------------------------------------------
    def _validate(self):
        """ Try to find errors on define params. """
        errors = []

        if self.doMotionCorrection.get():
            correction_method = self.inputParticles.get().getFlexInfo().getProgName()

            if correction_method not in [const.HETSIREN, const.ZERNIKE3D]:
                errors.append("Motion correction method not recognized. If you want to use motion correction, the particles "
                              "provided must come from HetSIREN (mass transport) or Zernike3Deep protocols.")

        return errors
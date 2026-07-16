# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreos@cnb.csic.es)
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
from glob import glob

import numpy as np
from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_2_0
from pyworkflow.utils import getExt

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.constants import ALIGN_PROJ, ALIGN_NONE
from pwem.objects import Volume, ParticleFlex

from xmipp3.convert import createItemMatrix, setXmippAttributes, writeSetOfParticles, \
    geometryFromMatrix, matrixFromGeometry
import xmipp3

import hax
import hax.constants as const


class JaxProtAngularAlignmentReconSiren(ProtAnalysis3D, ProtFlexBase):
    """
    Performs ab initio angular assignment and 3D reconstruction of particle images using the ReconSIREN neural network.
    The protocol can operate either from completely unaligned particle data or from particles that already contain
    projection alignment information. In the first case, the network jointly learns particle orientations, in-plane
    shifts, and a volume representation directly from the images. In the second case, the protocol behaves as a
    refinement procedure, improving existing angular assignments while optionally refining a supplied starting map.

    AI Generated:

    Angular Alignment ReconSIREN (JaxProtAngularAlignmentReconSiren) — User Manual

        Overview

        This protocol applies the ReconSIREN deep-learning framework to estimate angular assignments and reconstruct
        one or more 3D volumes from a set of single-particle cryo-EM images. Its main objective is to recover a
        consistent geometric description of the dataset by learning particle orientations, translational shifts, and
        an underlying 3D density model. In practical cryo-EM workflows, this protocol is especially useful when the
        dataset lacks reliable alignment information, when a coarse initial reconstruction must be improved, or when
        heterogeneous structural states are expected.

        From a biological perspective, the protocol is designed for early-stage structure determination as well as
        refinement-oriented workflows. It can start from raw particle images alone, or it can use an existing volume
        as structural prior knowledge. This flexibility makes it suitable both for exploratory reconstruction and for
        more controlled refinement scenarios.

        Inputs and General Workflow

        The protocol requires a set of input particles. These particles may already contain projection alignment
        parameters, but this is not mandatory. If no prior alignment exists, ReconSIREN estimates orientations and
        in-plane shifts from scratch. If alignment information is already available, the protocol can refine those
        parameters while simultaneously improving the reconstructed volume.

        An optional starting volume may also be supplied. When present, the network uses this map as an initial
        structural reference instead of learning the initial density entirely from the particle images. This is
        particularly valuable when a low-resolution reference already exists or when the goal is refinement rather
        than de novo reconstruction.

        Before training, the protocol prepares metadata, rescales the input particles if requested, converts input
        maps and masks into internal working files, and ensures that all data are brought to a consistent box size
        and sampling representation.

        Particle Downsampling and Computational Strategy

        A central practical parameter is the target box size. The protocol can downsample particles before training,
        which reduces memory consumption and computational cost. This is especially useful when working with large
        particle boxes or limited GPU memory.

        Biologically, downsampling should be understood as a computational tradeoff. Smaller box sizes usually improve
        speed and memory efficiency, but they also limit the highest recoverable resolution. For exploratory runs,
        coarse angular assignment, or large datasets, moderate downsampling is often entirely appropriate. For
        high-resolution refinement, however, aggressive downsampling should be avoided.

        The effective sampling rate used during training is adjusted accordingly so that geometric consistency between
        image size and physical sampling is preserved.

        Volume Initialization and Masking

        If an initial volume is provided, it is resized to match the working dimensions used by the network. The
        protocol can then refine this map during learning. If no initial volume is given, ReconSIREN infers the
        initial density representation directly from the particle images.

        An optional binary mask may also be provided. This mask defines the region of space where density is allowed
        to exist and therefore strongly influences reconstruction behavior. From a biological standpoint, masking is
        often critical. A well-designed mask focuses learning on meaningful molecular regions while excluding solvent
        and irrelevant background.

        When no initial volume is available, a compact spherical mask is often appropriate for globular particles. For
        membrane proteins, nanodiscs, or elongated assemblies, larger masks may be more suitable. When refining a
        known map, a threshold-derived binary mask usually provides the most stable behavior.

        CTF Handling

        The protocol supports several CTF strategies. It can ignore CTF effects entirely, apply the CTF to projections
        generated from the current volume estimate, or perform Wiener-based correction schemes. The correct choice
        depends on how the input particles were prepared upstream.

        In practical biological workflows, consistency with earlier preprocessing steps is important. If particles
        already contain CTF-corrected information, using an inappropriate correction mode can degrade reconstruction
        quality or bias alignment estimation.

        Alignment Refinement and Volume Learning

        If the input particles already contain projection assignments, the protocol can refine the current alignment
        rather than estimating it from scratch. This is useful when previous alignment exists but is not yet
        sufficiently accurate.

        When an input volume is available, the user may also choose whether the volume itself should continue being
        optimized. If volume refinement is disabled, ReconSIREN only estimates angular assignments and shifts while
        keeping the supplied reference map fixed. This is particularly relevant when a trusted high-resolution model
        already exists and only particle alignment needs improvement.

        From a biological perspective, this distinction is important. Allowing volume refinement increases model
        flexibility but may introduce unwanted structural drift if the dataset is limited or noisy. Keeping the map
        fixed provides stronger structural regularization.

        Symmetry Considerations

        The protocol allows the user to specify cyclic or dihedral symmetry groups. Symmetry information is taken into
        account during learning of both angular assignments and the reconstructed density.

        Even when symmetry is provided, the final particle orientations remain expressed in a symmetry-broken c1
        representation. This is often advantageous in downstream workflows because the estimated orientations can be
        directly reused in later reconstruction or refinement protocols.

        Training Procedure

        ReconSIREN performs a neural-network training stage followed by a prediction stage. During training, the model
        learns a compact latent description of particle variability together with orientation, shift, and density
        information. The training process is controlled mainly by the number of epochs, batch size, and learning rate.

        The protocol also supports fine tuning. When enabled, a previously trained network is reloaded and further
        optimized using the current dataset. This is useful when expanding datasets, refining previous results, or
        continuing interrupted optimization.

        From a practical standpoint, the batch size mostly controls GPU memory consumption, whereas the learning rate
        controls convergence stability. If training becomes unstable, lowering the learning rate is often the most
        effective corrective action.

        Data Loading and Performance

        The protocol supports two data-loading modes. Images may be loaded directly into RAM for maximum speed, or
        they may be memory mapped from disk. When disk-based access is used, a fast SSD or NVMe scratch directory is
        strongly recommended.

        For large particle datasets, data-loading strategy can noticeably affect runtime. Although this does not
        directly change biological results, it can strongly influence the practicality of large-scale processing.

        Outputs and Their Interpretation

        After prediction, the protocol produces an updated particle set, a reconstructed consensus volume, and
        optionally a set of heterogeneous volumes.

        The output particle set contains newly estimated angular parameters, in-plane shifts, and a latent-space
        descriptor associated with each particle. These latent variables encode structural variability learned by the
        network and can be useful for downstream heterogeneity analysis.

        The consensus volume represents the principal reconstructed density. Before export, it is rescaled back to the
        original particle box size and assigned the correct sampling rate so that it can be directly used in later
        Scipion processing steps.

        Additional heterogeneous maps may also be produced. These represent alternative structural states learned by
        the network and can provide biologically meaningful insight into conformational variability, compositional
        heterogeneity, or continuous structural motion.

        Validation and Practical Considerations

        The protocol validates the reconstruction mask before execution. Masks must be binary. Non-binary masks can
        lead to ambiguous spatial constraints and therefore unstable optimization behavior.

        In routine biological practice, it is often advisable to begin with moderate downsampling and a conservative
        mask, then inspect the reconstructed volume and angular consistency before increasing resolution demands.
        When prior structural knowledge exists, supplying a reasonable initial volume and enabling refinement usually
        improves convergence and stability.

        Final Perspective

        ReconSIREN is not simply a reconstruction tool but a joint geometric-learning framework. It simultaneously
        infers particle orientations, translational corrections, latent structural variability, and a 3D density
        representation. For cryo-EM users, its main value lies in its ability to recover meaningful structural
        organization even when conventional alignment information is weak, absent, or partially unreliable.
    """
    _label = 'angular align - ReconSIREN'
    _lastUpdateVersion = VERSION_2_0

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
        group.addParam('inputParticles', params.PointerParam, label="Input particles", pointerClass='SetOfParticles',
                       important=True,
                       help="If your particles do not have alignment information, ReconSIREN will learn the pose and in plane shifts "
                            "from scratch. Otherwise, the current alignment of the particles will be refined.")

        group.addParam('inputVolume', params.PointerParam, allowsNull=True,
                       label="Starting volume", pointerClass='Volume',
                       help="If not provided, ReconSIREN will learn from scratch the initial volume form the images. If a "
                            "volume is provided, ReconSIREN will perform a refinement of the volume as it improves the angles.")

        group.addParam('inputVolumeMask', params.PointerParam,
                       label="Reconstruction mask", pointerClass='VolumeMask', allowsNull=True,
                       help="ReconSIREN will use the values within the provided mask to determine the mass available to be used "
                            "to reconstruct/refine the initial volume (NOTE: masks provided here MUST be BINARY):\n\n"
                            "If no volume is provided\n"
                            "In this case, we recommend providing a circular mask with enough mass to reconstruct the protein captured in the "
                            "input images. As a rule of thumb, we recommend a circular mask with radius 0.25 * box_length unless your protein "
                            "is embedded in a membrane or nano-disc. In this case, we recommend a circular mask with radius 0.5 * box_length\n\n"
                            "If a volume is provided\n"
                            "If you have provided an input volume, we recommend a mask generated by thresholding the input volume. You may also dilate "
                            "the mask to give ReconSIREN additional mass and increase its accuracy.")

        group.addParam('boxSize', params.IntParam, default=64,
                       label='Downsample particles to this box size',
                       help='If your GPU is not able to fit your particles in memory, you can play with this parameter to downsample your images. '
                            'In this way, you will be able to fit your particles at the expense of losing resolution in the volumes generated by the '
                            'network.')

        group.addParam('ctfType', params.EnumParam, choices=['None', 'Apply', 'Wiener', 'Precorrect'],
                       default=1, label="CTF correction type",
                       display=params.EnumParam.DISPLAY_HLIST,
                       expertLevel=params.LEVEL_ADVANCED,
                       help="* *None*: CTF will not be considered\n"
                            "* *Apply*: CTF is applied to the projection generated from the reference map\n"
                            "* *Wiener*: input particle is CTF corrected by a Wiener fiter\n"
                            "* *Precorrect: similar to Wiener but CTF has already been corrected")

        group.addParam('refineCurrent', params.BooleanParam, default=False,
                       condition='inputParticles and inputParticles.hasAlignmentProj',
                       label="Refine current particle alignments?")

        group.addParam('refineVolume', params.BooleanParam, default=True,
                       condition='inputVolume',
                       label="Refine current volume?",
                       help="When this parameter is provided, ReconSIREN will just learn an angular assignment with shifts without learning any map. This is usually useful when a reference volume with "
                            "high resolution is provided (e.g. coming from an atomic model) and no refinement of the map is needed.")

        group = form.addGroup("Symmetry")
        group.addParam('symmetry', params.StringParam, default="c1", label='Symmetry group',
                       help="If your protein has any kind of symmetry, you may pass it here so that it is considered while learning the angular assignment "
                            "and the volume (NOTE: only c* and d* symmetry groups are currently supported - the parameter is lower case sensitive - even if "
                            "symmetry is provided, the network will learn a symmetry broken set of angles in c1. Therefore, the angles can be directly used "
                            "in a reconstruction/refinement.)")

        form.addSection(label='Network')
        form.addParam('fineTune', params.BooleanParam, default=False, label='Fine tune previous network?',
                      help='When set to Yes, you will be able to provide a previously trained ReconSIREN network to refine it with new '
                           'data. If set to No, you will train a new ReconSIREN network from scratch.')

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
        group.addParam('epochs', params.IntParam, default=50,
                       label='Number of training epochs',
                       help="Number of epochs to train the network (i.e. how many times to loop over the whole dataset "
                            "of images - set to default to 50 - as a rule of thumb, consider 50 to 100 epochs enough "
                            "for 100k images / if your dataset is bigger or smaller, scale this value proportionally to it")

        group.addParam('batchSize', params.IntParam, default=64, label='Number of images in batch',
                       help="Determines how many images will be load in the GPU at any moment during training (set by "
                            "default to 8 - you can control GPU memory usage easily by tuning this parameter to fit your "
                            "hardware requirements - we recommend using tools like nvidia-smi to monitor and/or measure "
                            "memory usage and adjust this value - keep also in mind that bigger batch sizes might be "
                            "less precise when looking for very local motions")

        group.addParam('learningRate', params.FloatParam, default=1e-4, label='Learning rate',
                       help="The learning rate (lr) sets the speed of learning. Think of the model as trying to find the "
                            "lowest point in a valley; the lr is the size of the step it takes on each attempt. A large "
                            "lr (e.g., 0.01) is like taking huge leaps — it's fast but can be unstable, overshoot the "
                            "lowest point, or cause NAN errors. A small lr (e.g., 1e-6) is like taking tiny shuffles — "
                            "it's stable but very slow and might get stuck before reaching the bottom. A good default is "
                            "often 0.0001. If training fails or errors explode, try making the lr 10 times smaller (e.g., "
                            "0.001 --> 0.0001).")
        form.addParallelSection(threads=4, mpi=0)

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'imgsFn': self._getExtraPath('input_particles.xmd'),
            'predictedFn': self._getExtraPath('predicted_pose_shifts.xmd'),
            'fnVol': self._getExtraPath('volume.mrc'),
            'fnVolMask': self._getExtraPath('mask.mrc'),
            'fnOutDir': self._getExtraPath(),
            'fnSymMatrices': self._getExtraPath("sym_matrices.npy")
        }
        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep(self.writeMetaDataStep)
        self._insertFunctionStep(self.trainingPredictStep)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions -----------------------
    def writeMetaDataStep(self):
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        fnVolMask = self._getFileName('fnVolMask')

        inputParticles = self.inputParticles.get()
        Xdim = inputParticles.getXDim()
        newXdim = self.boxSize.get()
        vol_mask_dim = newXdim

        if self.inputVolume.get():
            ih = ImageHandler()
            inputVolume = self.inputVolume.get().getFileName()
            ih.convert(self._getXmippFileName(inputVolume), fnVol)
            curr_vol_dim = ImageHandler(self._getXmippFileName(inputVolume)).getDimensions()[-1]
            if curr_vol_dim != vol_mask_dim:
                self.runJob("xmipp_image_resize",
                            "-i %s --fourier %d " % (fnVol, vol_mask_dim), numberOfMpi=1,
                            env=xmipp3.Plugin.getEnviron())
            ih.setSamplingRate(fnVol, inputParticles.getSamplingRate())

        if self.inputVolumeMask.get():
            ih = ImageHandler()
            inputMask = self.inputVolumeMask.get().getFileName()
            if inputMask:
                ih.convert(self._getXmippFileName(inputMask), fnVolMask)
                curr_mask_dim = ImageHandler(self._getXmippFileName(inputMask)).getDimensions()[-1]
                if curr_mask_dim != vol_mask_dim:
                    self.runJob("xmipp_image_resize",
                                "-i %s --dim %d --interp nearest" % (fnVolMask, vol_mask_dim), numberOfMpi=1,
                                env=xmipp3.Plugin.getEnviron())

        writeSetOfParticles(inputParticles, imgsFn)

        if newXdim != Xdim:
            params = "-i %s -o %s --save_metadata_stack %s --fourier %d" % \
                     (imgsFn,
                      self._getTmpPath('scaled_particles.stk'),
                      self._getExtraPath('scaled_particles.xmd'),
                      newXdim)
            if self.numberOfMpi.get() > 1:
                params += " --mpi_job_size %d" % int(inputParticles.getSize() / self.numberOfMpi.get())
            self.runJob("xmipp_image_resize", params, numberOfMpi=self.numberOfMpi.get(),
                        env=xmipp3.Plugin.getEnviron())
            moveFile(self._getExtraPath('scaled_particles.xmd'), imgsFn)

    def trainingPredictStep(self):
        md_file = self._getFileName('imgsFn')
        vol_file = self._getFileName('fnVol')
        mask_file = self._getFileName('fnVolMask')
        out_path = self._getExtraPath()
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        symmetry = self.symmetry.get()
        batch_size = self.batchSize.get()
        learningRate = self.learningRate.get()
        epochs = self.epochs.get()
        newXdim = self.boxSize.get()
        correctionFactor = self.inputParticles.get().getXDim() / newXdim
        sr = correctionFactor * self.inputParticles.get().getSamplingRate()
        args = "--md %s --sr %f --epochs %d --batch_size %d --learning_rate %s --output_path %s --symmetry_group %s " \
               % (md_file, sr, epochs, batch_size, learningRate, out_path, symmetry)

        if self.inputVolume.get():
            args += '--vol %s ' % vol_file

        if self.inputVolumeMask.get():
            args += '--mask %s ' % mask_file

        if self.refineCurrent.get():
            args += '--refine_current_assignment '

        if not self.refineVolume.get():
            args += "--do_not_learn_volume "

        if self.ctfType != 0:
            if self.ctfType.get() == 1:
                args += '--ctf_type apply '
            elif self.ctfType.get() == 2:
                args += '--ctf_type wiener '
            elif self.ctfType.get() == 3:
                args += '--ctf_type precorrect '
        else:
            args += '--ctf_type None '

        if self.lazyLoad:
            args += '--load_images_to_ram '
        else:
            if self.scratchFolder.get() is not None:
                args += '--ssd_scratch_folder %s ' % self.scratchFolder.get()

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = None

        program = hax.Plugin.getProgram("reconsiren", gpu)
        if not os.path.isdir(self._getExtraPath("ReconSIREN")):
            self.runJob(program,
                        args + f'--mode train --reload {self._getExtraPath()}'
                        if self.fineTune else args + '--mode train',
                        numberOfMpi=1)
        self.runJob(program, args + f'--mode predict --reload {self._getExtraPath()}', numberOfMpi=1)

    def createOutputStep(self):
        out_path_vol = self._getExtraPath('reconsiren_map.mrc')
        md_file = self._getFileName('predictedFn')
        Xdim = self.inputParticles.get().getXDim()
        self.newXdim = self.boxSize.get()

        metadata = XmippMetaData(md_file)
        correctionFactor = Xdim / self.newXdim
        latent_space = np.asarray([np.fromstring(item, sep=',') for item in metadata[:, 'latent_space']])
        euler_rot = metadata[:, 'angleRot']
        euler_tilt = metadata[:, 'angleTilt']
        euler_psi = metadata[:, 'anglePsi']
        shift_x = correctionFactor * metadata[:, 'shiftX']
        shift_y = correctionFactor * metadata[:, 'shiftY']

        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticlesFlex(progName=const.RECONSIREN)

        partSet.copyInfo(inputSet)
        partSet.setHasCTF(inputSet.hasCTF())
        partSet.setAlignmentProj()
        inverseTransform = partSet.getAlignment() == ALIGN_PROJ

        idx = 0
        for particle in inputSet.iterItems():
            outParticle = ParticleFlex(progName=const.HETSIREN)
            outParticle.copyInfo(particle)
            outParticle.setZFlex(latent_space[idx])

            # Set new transformation matrix
            tr = matrixFromGeometry(np.array([shift_x[idx], shift_y[idx], 0.0]),
                                    np.array([euler_rot[idx], euler_tilt[idx], euler_psi[idx]]),
                                    inverseTransform)
            outParticle.getTransform().setMatrix(tr)

            partSet.append(outParticle)
            idx += 1

        outVols = self._createSetOfVolumes()
        outVols.setSamplingRate(inputSet.getSamplingRate())
        outVol = Volume()
        outVol.setSamplingRate(inputSet.getSamplingRate())

        ImageHandler().scaleSplines(out_path_vol, out_path_vol, finalDimension=inputSet.getXDim(), overwrite=True)
        ImageHandler().setSamplingRate(out_path_vol, inputSet.getSamplingRate())

        outVol.setLocation(out_path_vol)
        outVols.append(outVol)

        outHetVols = self._createSetOfVolumes(suffix='Het')
        outHetVols.setSamplingRate(inputSet.getSamplingRate())
        for file in glob(self._getExtraPath('reconsiren_hetmap*')):
            outHetVol = Volume()
            outHetVol.setSamplingRate(inputSet.getSamplingRate())

            ImageHandler().scaleSplines(file, file, finalDimension=inputSet.getXDim(), overwrite=True)
            ImageHandler().setSamplingRate(file, inputSet.getSamplingRate())

            outHetVol.setLocation(file)
            outHetVols.append(outHetVol)

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(inputSet, partSet)

        self._defineOutputs(outputVolumes=outVols)
        self._defineTransformRelation(inputSet, outVols)

        self._defineOutputs(outputHetVolumes=outHetVols)
        self._defineTransformRelation(inputSet, outHetVols)


    # --------------------------- UTILS functions -----------------------
    def _getXmippFileName(self, filename):
        if getExt(filename) == ".mrc":
            filename += ":mrc"
        return filename

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

    # ----------------------- VALIDATE functions -----------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []

        mask = self.inputVolumeMask.get()
        if mask is not None:
            data = ImageHandler(mask.getFileName()).getData()
            if not np.all(np.logical_and(data >= 0, data <= 1)):
                errors.append("Mask provided is not binary. Please, provide a binary mask")

        return errors

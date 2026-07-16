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
    Performs ab initio reconstruction and angular assignment of particles using the ReconSIREN neural network.
    The protocol can either estimate particle orientations and reconstruct a 3D map from scratch, or refine
    an existing volume and/or pre-existing particle alignment information.

    AI Generated:

    Angular Align – ReconSIREN (JaxProtAngularAlignmentReconSiren) — User Manual
        Overview

        The Angular Align – ReconSIREN protocol estimates particle orientations, in-plane shifts, and a
        corresponding 3D reconstruction directly from experimental particle images using the ReconSIREN
        neural network.

        Its main goal is to jointly solve two problems that are central in cryo-EM:
        assigning a projection direction to each particle and reconstructing the underlying 3D map.
        Depending on the inputs, the protocol can operate in two modes:

        - Ab initio mode, where no prior volume is provided and both the angular assignment and the
          initial reconstruction are learned from scratch.

        - Refinement mode, where an initial reference volume and/or previous particle alignment are
          available and the network refines them.

        In biological practice, this protocol is useful when studying new datasets without an available
        structural model, when refining difficult heterogeneous data, or when improving an initial
        reconstruction while simultaneously refining particle poses.

        Inputs and General Workflow

        The protocol requires a set of input particles.

        If particles do not contain alignment information, ReconSIREN learns orientations and shifts
        directly from the images.

        If particles already contain projection alignment parameters, the protocol can optionally refine
        those assignments.

        An optional starting volume may also be provided.

        - Without an input volume, the network estimates both particle orientations and a new 3D map
          from scratch.

        - With an input volume, the protocol performs refinement of angular assignment and optionally
          refinement of the volume itself.

        This flexibility makes the protocol suitable both for early exploratory reconstruction and for
        later refinement stages in cryo-EM workflows.

        Reconstruction Mask

        A reconstruction mask can optionally be provided.

        The mask determines which spatial region contains the mass available to reconstruct or refine
        the volume.

        This parameter is biologically important because it strongly influences what the network
        interprets as signal.

        The mask must be binary.

        Practical recommendations depend on the workflow:

        If no initial volume is provided:
        a circular mask is usually recommended. A practical default is approximately one quarter of the
        box size radius for compact particles. Larger masks may be preferable for membrane proteins,
        nanodiscs, or particles occupying a large fraction of the box.

        If an input volume is provided:
        a binary mask generated from the reference volume is generally preferable. Slight dilation of
        the mask can help the network preserve peripheral density while refining the reconstruction.

        Poor masks can introduce instability, excessive noise learning, or biologically misleading maps.

        Downsampling and Computational Strategy

        Input particles may be downsampled to a smaller box size before training.

        This reduces GPU memory requirements and can substantially accelerate execution.

        In practical terms, downsampling is useful when working with large particles or limited GPU
        resources. The trade-off is reduced spatial detail in the reconstructed volumes.

        For exploratory analyses, moderate downsampling is often acceptable. For final biological
        interpretation, users should remember that aggressive downsampling may limit recoverable
        structural detail.

        CTF Handling

        The protocol supports several strategies for contrast transfer function handling:

        - None: CTF is ignored.
        - Apply: CTF is applied to the projection generated from the current model.
        - Wiener: input particles are Wiener corrected.
        - Precorrect: assumes particles were already corrected beforehand.

        From a biological perspective, the correct choice depends on upstream preprocessing.

        For most standard workflows, the default Apply mode is a sensible starting point.
        Wiener correction may help when the dataset has already been prepared accordingly.
        Incorrect CTF handling can degrade both angular assignment and reconstruction quality.

        Alignment Refinement

        If the input particles already contain alignment parameters, the protocol can refine the current
        angular assignment.

        This is particularly useful when particles come from previous classification, initial model
        estimation, or external reconstruction workflows.

        Instead of learning orientations from scratch, the network starts from the current assignment
        and improves it.

        In practice, this often accelerates convergence and stabilizes difficult datasets.

        Volume Refinement

        When an input volume is provided, the user can decide whether the map itself should be refined.

        If refinement is enabled, ReconSIREN updates both the angular assignment and the volume.

        If refinement is disabled, the protocol only estimates orientations and shifts relative to the
        fixed reference volume.

        This second option can be especially useful when the provided reference is already high quality,
        for example one derived from a high-resolution reconstruction or an atomic model.

        Symmetry

        The protocol allows the use of cyclic and dihedral symmetry groups.

        Symmetry is incorporated during learning of angular assignment and reconstruction.

        Biologically, this can significantly improve stability and reconstruction quality for symmetric
        macromolecular complexes.

        Even when symmetry is used, the protocol stores particle orientations as symmetry-broken
        assignments in c1. This makes the resulting orientations directly usable in downstream
        refinement and reconstruction protocols.

        Neural Network Training

        ReconSIREN is trained directly on the particle images.

        The main hyperparameters are:

        - Number of epochs, controlling how many times the dataset is revisited.
        - Batch size, controlling GPU memory usage and training throughput.
        - Learning rate, controlling optimization step size.

        In practical cryo-EM use, the default settings usually provide a good starting point.

        Larger datasets often require more epochs.
        Limited GPU memory may require reducing the batch size.
        If training becomes unstable or diverges, reducing the learning rate is usually the first
        recommended adjustment.

        Fine Tuning

        The protocol can also fine-tune a previously trained ReconSIREN model.

        This is useful when incorporating additional particles, extending previous analyses, or refining
        an already converged network with related data.

        Instead of starting from random initialization, training resumes from an existing network state.

        Outputs and Their Interpretation

        After execution, the protocol produces three main outputs.

        Output particles:
        Each particle receives updated angular parameters, shifts, and a latent representation learned
        by the network. These particles can be directly used in downstream flexible analysis or further
        reconstruction workflows.

        Output reconstructed volume:
        A consensus 3D map is generated and rescaled to the original particle box size.

        Output heterogeneous volumes:
        Additional heterogeneous maps may also be produced when present. These represent alternative
        structural states learned during reconstruction and may help reveal conformational variability.

        All output volumes preserve the sampling rate of the input particles.

        Practical Recommendations

        For new datasets without a prior structural model, running the protocol in ab initio mode is
        often the natural starting point.

        When a reliable initial reconstruction exists, providing it usually improves convergence and
        angular stability.

        A carefully chosen binary mask often provides one of the strongest improvements, especially for
        flexible or elongated complexes.

        Downsampling can greatly reduce computational cost during exploratory runs, but final biological
        interpretation should preferably rely on minimally downsampled data.

        When particles already contain approximate alignment information, enabling refinement of current
        assignments generally improves efficiency.

        Final Perspective

        ReconSIREN combines angular assignment and reconstruction into a single learning framework.

        For cryo-EM users, this means the protocol is not simply estimating orientations: it is jointly
        learning the geometric organization of the particles and the corresponding 3D structural model.

        Its strength is especially evident in difficult datasets where traditional reconstruction and
        orientation assignment become strongly coupled problems.
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

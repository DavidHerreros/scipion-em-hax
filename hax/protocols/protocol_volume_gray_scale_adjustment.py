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

from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_1
from pyworkflow.utils import getExt, makePath

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import Volume, Particle

import xmipp3
from xmipp3.convert import writeSetOfParticles, matrixFromGeometry

import hax

class JaxProtVolumeAdjustment(ProtAnalysis3D, ProtFlexBase):
    """
    Adjusts the gray-scale values of a reference 3D volume so that its projections become more consistent
    with an experimental particle dataset. The protocol uses a neural-network-based volume adjustment
    strategy in which a starting map is iteratively refined in intensity space rather than through
    geometric deformation. Its main purpose is to correct mismatches in density amplitude or local
    gray-scale behavior that can arise when a reference map and the experimental particle images do not
    share the same statistical intensity characteristics.

    AI Generated:

    Volume Adjustment (JaxProtVolumeAdjustment) — User Manual

        Overview

        The Volume Adjustment protocol estimates a gray-scale correction model that transforms an input
        volume into a new map whose simulated projections better agree with the experimental particle
        images. In practical cryo-EM analysis, this is useful when a reference reconstruction already
        captures the correct structural arrangement but differs from the particle data in density scale,
        contrast, or regional intensity balance. Instead of changing the molecular shape, the protocol
        learns how to adjust the voxel values so the reference becomes more compatible with the observed
        dataset.

        For biological users, this is particularly relevant when using a map obtained from another
        reconstruction pipeline, another refinement stage, or another experimental condition. In those
        cases, the protocol can help standardize the density representation before downstream procedures
        such as interpretation, comparison, or additional model-based analysis.

        Inputs and Data Preparation

        The protocol requires an input particle set and a starting reference volume. The particles define
        the experimental observations, while the volume provides the structural template whose gray-scale
        values will be corrected. During preparation, the reference map is converted into an internal
        working format and resized if needed so that its dimensions match those of the particle images.
        The sampling rate is preserved so the resulting adjusted map remains physically meaningful.

        An optional binary mask can also be provided. When present, the adjustment focuses on the masked
        region, allowing the network to concentrate on biologically relevant density while reducing the
        influence of solvent or poorly informative regions. If no mask is supplied, the protocol can still
        proceed, but including a biologically sensible mask often improves robustness and interpretability.

        Particle metadata are also exported before training. If the input dataset contains additional
        labels, such as subtomogram annotations, these labels are preserved and passed into the training
        workflow.

        CTF Handling

        The protocol can optionally account for the contrast transfer function. Several modes are
        available, including applying the CTF to simulated projections, Wiener-based correction, or using
        already precorrected images. In routine practice, this choice should reflect the state of the
        particle dataset before entering the protocol.

        When the particle images have already been corrected in earlier preprocessing steps, using the
        appropriate correction mode helps preserve consistency between the experimental data and the
        projections generated from the reference map. If no correction is desired, the protocol can ignore
        CTF effects entirely.

        Network Training and Prediction

        The core of the protocol consists of a neural network trained to infer gray-scale adjustments from
        the relationship between experimental particles and projections of the reference volume. Training
        uses standard hyperparameters such as latent dimension, batch size, number of epochs, and learning
        rate.

        The latent dimension controls the size of the internal representation used by the network. Higher
        values allow more expressive corrections but may also increase model complexity. The batch size
        determines GPU memory usage and therefore often becomes the first parameter to adjust when working
        with limited hardware resources.

        The learning rate controls optimization stability. Large values can cause unstable training or
        numerical divergence, while smaller values improve robustness but slow convergence. In most
        practical situations, the default value provides a good starting point.

        A fine-tuning mode is available for continuing from a previously trained network rather than
        starting from scratch. This is useful when expanding an existing analysis with additional data or
        refining a previous adjustment without discarding prior learning.

        Adjustment Modes

        The protocol can estimate corrections either per voxel or as a global scalar adjustment. Per-voxel
        prediction provides spatially varying corrections and is generally the more flexible mode when
        local intensity mismatches are expected. Global prediction instead estimates a simpler overall
        adjustment and may be preferable when the discrepancy between map and particles is mostly uniform.

        From a biological perspective, per-voxel prediction is often more informative for heterogeneous
        density distributions, whereas global correction may be sufficient for simpler normalization tasks.

        Performance and Data Loading

        Data loading can be configured either by loading particles into RAM or by memory mapping from disk.
        Loading into RAM generally provides better performance when the dataset fits comfortably in memory.
        Memory mapping is more conservative in RAM usage but can become I/O limited.

        For large particle stacks, especially with box sizes above typical moderate dimensions, using a
        fast SSD or NVMe scratch directory can significantly improve throughput. GPU execution is supported
        and is the intended execution mode for most practical datasets.

        Outputs and Interpretation

        After training and prediction, the protocol generates a single adjusted volume. The output retains
        the structural geometry of the original reference but contains updated gray-scale values estimated
        from the experimental dataset. The sampling rate and general metadata are preserved so the adjusted
        volume can be used naturally in downstream workflows.

        Biologically, the resulting map should be interpreted as a density-consistent version of the input
        reference rather than as a newly reconstructed structure. The protocol does not infer new
        conformations or recover missing structural information. Instead, it modifies how the existing map
        represents density so that it better matches the particle observations.

        Practical Recommendations

        In most workflows, it is advisable to start with a high-quality reference map and an appropriate
        binary mask. The more biologically relevant the masked region, the more likely the adjustment will
        focus on meaningful signal rather than solvent background.

        If training appears unstable, reducing the learning rate or the batch size is usually the most
        effective first intervention. When extending previous analyses, fine-tuning often provides a faster
        and more stable path than retraining from scratch.

        Final Perspective

        For cryo-EM users, volume gray-scale adjustment provides a practical bridge between an existing
        reference map and a new experimental dataset. Rather than replacing reconstruction, it refines the
        compatibility between map and particles at the density level. When applied carefully, it can
        improve consistency for interpretation, downstream comparison, and subsequent flexible analysis.
    """
    _label = 'predict - Volume Adjustment '
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
        group.addParam('inputParticles', params.PointerParam, label="Input particles", pointerClass='SetOfParticles',
                       important=True)

        group.addParam('inputVolume', params.PointerParam,
                       label="Starting volume", pointerClass='Volume',
                       help="Reference volume needed to be adjusted according to the projections. If adjustment prediction setting "
                            "is set to True, these projections will be needed for its estimation.")

        group.addParam('inputVolumeMask', params.PointerParam,
                       label="Reconstruction mask", pointerClass='VolumeMask', allowsNull=True,
                       help="This mask helps focusing the adjustment prediction on a region of interest "
                            "(if not provided, a inscribed spherical mask will be used).")

        group.addParam('ctfType', params.EnumParam, choices=['None', 'Apply', 'Wiener', 'Precorrect'],
                       default=1, label="CTF correction type",
                       display=params.EnumParam.DISPLAY_HLIST,
                       expertLevel=params.LEVEL_ADVANCED,
                       help="* *None*: CTF will not be considered\n"
                            "* *Apply*: CTF is applied to the projection generated from the reference map\n"
                            "* *Wiener*: input particle is CTF corrected by a Wiener fiter\n"
                            "* *Precorrect: similar to Wiener but CTF has already been corrected")

        form.addSection(label='Network')
        form.addParam('fineTune', params.BooleanParam, default=False, label='Fine tune previous network?',
                      help='When set to Yes, you will be able to provide a previously trained network to refine it with new '
                           'data. If set to No, you will train a new network from scratch.')

        form.addParam('predictsPerVoxel', params.BooleanParam, default=False,
                      label='Predict per voxel adjustment?', expertLevel=params.LEVEL_ADVANCED,
                      help='If not provided, the adjustment will be estimated per voxel - otherwise, adjustment will be estimated for the whole volume.')

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
        group.addParam('latDim', params.IntParam, default=10, label='Latent space dimension',
                       expertLevel=params.LEVEL_ADVANCED,
                       help="Dimension of the network bottleneck (latent space dimension)")

        group.addParam('epochs', params.IntParam, default=50,
                       label='Number of training epochs',
                       help="Number of epochs to train the network (i.e. how many times to loop over the whole dataset "
                            "of images - set to default to 50 - as a rule of thumb, consider 50 to 100 epochs enough "
                            "for 100k images / if your dataset is bigger or smaller, scale this value proportionally to it")

        group.addParam('batchSize', params.IntParam, default=8, label='Number of images in batch',
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

        form.addParallelSection(threads=0, mpi=4)

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'imgsFn': self._getExtraPath('input_particles.xmd'),
            'predictedFn': self._getExtraPath('predicted_latents.xmd'),
            'fnVol': self._getExtraPath('volume.mrc'),
            'fnVolMask': self._getExtraPath('mask.mrc'),
        }
        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep(self.writeMetaDataStep)
        self._insertFunctionStep(self.trainingPredictStep)
        self._insertFunctionStep(self.createOutputStep)

    def writeMetaDataStep(self):
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        fnVolMask = self._getFileName('fnVolMask')

        inputParticles = self.inputParticles.get()
        Xdim = inputParticles.getXDim()

        ih = ImageHandler()
        inputVolume = self.inputVolume.get().getFileName()
        ih.convert(self._getXmippFileName(inputVolume), fnVol)
        curr_vol_dim = ImageHandler(self._getXmippFileName(inputVolume)).getDimensions()[-1]
        if curr_vol_dim != Xdim:
            self.runJob("xmipp_image_resize",
                        "-i %s --fourier %d " % (fnVol, Xdim), numberOfMpi=1,
                        env=xmipp3.Plugin.getEnviron())
        ih.setSamplingRate(fnVol, inputParticles.getSamplingRate())

        if self.inputVolumeMask.get():
            ih = ImageHandler()
            inputMask = self.inputVolumeMask.get().getFileName()
            if inputMask:
                ih.convert(self._getXmippFileName(inputMask), fnVolMask)
                curr_mask_dim = ImageHandler(self._getXmippFileName(inputMask)).getDimensions()[-1]
                if curr_mask_dim != Xdim:
                    self.runJob("xmipp_image_resize",
                                "-i %s --dim %d --interp nearest" % (fnVolMask, Xdim), numberOfMpi=1,
                                env=xmipp3.Plugin.getEnviron())

        writeSetOfParticles(inputParticles, imgsFn)

        # Write extra attributes (if needed)
        md = XmippMetaData(imgsFn)
        if hasattr(inputParticles.getFirstItem(), "_xmipp_subtomo_labels"):
            labels = np.asarray([int(particle._xmipp_subtomo_labels) for particle in inputParticles.iterItems()])
            md[:, "subtomo_labels"] = labels
        md.write(imgsFn, overwrite=True)

    def trainingPredictStep(self):
        md_file = self._getFileName('imgsFn')
        vol_file = self._getFileName('fnVol')
        mask_file = self._getFileName('fnVolMask')
        out_path = self._getExtraPath()
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        batch_size = self.batchSize.get()
        learningRate = self.learningRate.get()
        epochs = self.epochs.get()
        latDim = self.latDim.get()
        sr = self.inputParticles.get().getSamplingRate()
        args = "--md %s --vol %s --sr %f --lat_dim %d --epochs %d --batch_size %d --learning_rate %f --output_path %s " \
               % (md_file, vol_file, sr, latDim, epochs, batch_size, learningRate, out_path)

        if self.inputVolumeMask.get():
            args += '--mask %s ' % mask_file

        if self.ctfType != 0:
            if self.ctfType.get() == 1:
                args += '--ctf_type apply '
            elif self.ctfType.get() == 2:
                args += '--ctf_type wiener '
            elif self.ctfType.get() == 3:
                args += '--ctf_type precorrect '
        else:
            args += '--ctf_type None '

        if not self.predictsPerVoxel:
            args += '--predicts_value '

        if self.lazyLoad:
            args += '--load_images_to_ram '
        else:
            if self.scratchFolder.get() is not None:
                args += '--ssd_scratch_folder %s ' % self.scratchFolder.get()

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = None

        program = hax.Plugin.getProgram("volume_gray_scale_adjustment", gpu)
        if not os.path.isdir(self._getExtraPath("volumeAdjustment")):
            self.runJob(program,
                        args + f'--mode train --reload {self._getExtraPath()}'
                        if self.fineTune else args + '--mode train',
                        numberOfMpi=1)
        self.runJob(program, args + f'--mode predict --reload {self._getExtraPath()}', numberOfMpi=1)

    def createOutputStep(self):
        volSet = self.inputVolume.get()
        out_path = os.path.join(self._getExtraPath(), "adjusted_volume.mrc")

        outVol = Volume()
        outVol.setSamplingRate(volSet.getSamplingRate())
        outVol.copyInfo(volSet)
        outVol.setLocation(out_path)

        self._defineOutputs(outputVolume=outVol)
        self._defineTransformRelation(volSet, outVol)

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
    def _validate(self):
        """ Try to find errors on define params. """
        errors = []

        vol = self.inputVolume.get()
        mask = self.inputVolumeMask.get()

        if vol is None:
            errors.append("A volume is required. Please, select a valid and adequate volume for your dataset")

        if mask is not None:
            data = ImageHandler(mask.getFileName()).getData()
            if not np.all(np.logical_and(data >= 0, data <= 1)):
                errors.append("Mask provided is not binary. Please, provide a binary mask")

        return errors

    def _warnings(self):
        warnings = []

        return warnings

    # --------------------------- UTILS functions -----------------------

    def _getXmippFileName(self, filename):
        if getExt(filename) == ".mrc":
            filename += ":mrc"
        return filename
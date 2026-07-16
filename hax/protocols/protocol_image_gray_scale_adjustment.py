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
from pyworkflow.object import Float
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_1
from pyworkflow.utils import getExt, makePath

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import Volume, Particle

import xmipp3
from xmipp3.convert import writeSetOfParticles, matrixFromGeometry

import hax

class JaxProtImageAdjustment(ProtAnalysis3D, ProtFlexBase):
    """
    Protocol for image gray values adjustment with the Image Gray Scale Adjustment algorithm.

    AI Generated:

    Image Adjustment (JaxProtImageAdjustment) — User Manual
        Overview

        The Image Adjustment protocol estimates and applies gray-scale corrections to a set of
        experimental particle images using a reference 3D volume. Its main purpose is to reduce
        systematic intensity inconsistencies between experimental projections and reference-based
        projections, improving downstream compatibility for reconstruction, alignment, or neural
        network–based heterogeneous analysis.

        In cryo-EM workflows, this protocol is especially useful when particle images show
        acquisition-dependent contrast differences, normalization mismatches, or intensity
        distortions that can negatively affect learning-based reconstruction methods.

        Inputs and General Workflow

        The protocol requires a set of input particles and a reference volume.

        The reference volume is used to generate projections that serve as intensity references
        during network training. These reference projections are compared against the experimental
        particle images so that the network can learn how gray values should be adjusted.

        Optionally, a binary mask can also be provided. The mask restricts the learning process
        to a biologically relevant region of the volume, preventing solvent or background regions
        from dominating the intensity estimation.

        The general workflow consists of three stages:

            1. Input preparation
               The reference volume is converted and resized if necessary so that its dimensions
               match the particle box size. If a mask is provided, it is also resized accordingly.

            2. Network training and prediction
               A neural network is trained to estimate gray-scale corrections between particle
               images and volume projections.

            3. Output generation
               Adjusted particle images are written to a new output set while preserving particle
               metadata and acquisition information.

        Reference Volume Preparation

        The reference volume is mandatory.

        Before training, the protocol automatically checks whether the input volume matches the
        particle dimensions. If not, the volume is resized in Fourier space to preserve frequency
        information as much as possible.

        From a biological perspective, the quality of the reference map strongly influences the
        quality of the correction. A well-resolved and biologically representative map generally
        produces more reliable gray-scale estimation.

        If the selected reference does not adequately represent the particle population, the
        learned correction may become biased.

        Masking Strategy

        The optional reconstruction mask helps the protocol focus only on the region of interest.

        This is particularly useful when particles contain substantial solvent background,
        micelle signal, support film contamination, or flexible peripheral regions that should
        not dominate the intensity correction.

        If no mask is provided, the protocol still runs, but the network may use the full volume
        projection during optimization.

        The mask must be binary. Non-binary masks are explicitly rejected during validation.

        Correction Modes

        The protocol supports two adjustment prediction modes.

        Per-projection adjustment
            In this mode, the network predicts a global correction for each image.
            Two scalar parameters are estimated for every particle:

                - a_adjustment
                - b_adjustment

            This mode is computationally lighter and usually sufficient when intensity
            differences are mostly global.

        Per-pixel adjustment
            In this mode, the network predicts local gray-scale corrections at pixel level.

            This mode is more flexible and can capture local intensity variation, but it is
            computationally more demanding and may require more training data for stable behavior.

        In practical cryo-EM use, per-projection correction is usually a good starting point,
        while per-pixel adjustment is more appropriate when images show strong local contrast
        variability.

        CTF Handling

        The protocol allows several strategies for handling the Contrast Transfer Function (CTF).

        None
            The CTF is ignored.

        Apply
            The CTF is applied to the projection generated from the reference volume.

        Wiener
            Experimental images are Wiener-corrected before adjustment.

        Precorrect
            The protocol assumes the data were already CTF-corrected upstream.

        In biological workflows, choosing the correct CTF mode is important because intensity
        behavior strongly depends on whether CTF effects remain present in the particle images.

        Training Parameters

        Several network hyperparameters control learning behavior.

        Latent space dimension
            Defines the bottleneck size of the neural network.

        Epochs
            Controls how many passes are performed over the dataset.

        Batch size
            Determines GPU memory usage and the degree of stochasticity during optimization.

        Learning rate
            Controls convergence speed and stability.

        In routine practice, the default parameters usually provide a reasonable starting point.
        Larger datasets may benefit from more epochs, while GPU memory limitations are typically
        addressed by reducing batch size.

        Memory Management

        The protocol supports two data-loading strategies.

        Lazy loading into RAM
            Images are loaded directly into memory when possible, offering the best performance.

        SSD scratch mode
            If RAM loading is disabled, temporary data can be placed on a fast SSD or NVMe device.

        This becomes particularly important for large particle datasets or large image box sizes,
        where I/O performance may become a major bottleneck.

        GPU Execution

        The protocol supports GPU acceleration.

        If enabled, the selected GPU device is passed directly to the external
        image_gray_scale_adjustment program.

        GPU acceleration is especially important because both training and prediction are neural
        network operations and may become computationally expensive for large datasets.

        Outputs and Their Interpretation

        The protocol generates a new SetOfParticles as output.

        For each output particle:

            - The adjusted image is stored as the new particle location.
            - Original metadata and acquisition parameters are preserved.

        When per-projection adjustment is selected, each particle additionally stores:

            - a_adjustment
            - b_adjustment

        These values can be interpreted as global intensity correction coefficients applied to
        that particular image.

        Practical Recommendations

        In typical cryo-EM workflows, it is often advisable to begin with:

            - a biologically representative reference volume
            - a binary mask covering the stable core
            - per-projection correction
            - default learning parameters

        If particles show strong local intensity heterogeneity, per-pixel adjustment may provide
        better correction but should be used with additional caution.

        When working with noisy datasets, selecting an appropriate mask often improves the
        biological relevance of the learned corrections more than tuning network hyperparameters.

        Validation Rules

        Before execution, the protocol verifies two critical conditions:

            - A reference volume must be provided
            - If a mask is supplied, it must be strictly binary

        These checks help prevent unstable or biologically meaningless training behavior.

        Final Perspective

        Image gray-scale adjustment is not simply a numerical normalization step.

        In cryo-EM analysis, intensity consistency strongly affects the quality of downstream
        reconstruction, heterogeneity analysis, and machine-learning–based inference.

        A carefully chosen reference volume, appropriate masking, and realistic correction mode
        are the main elements for obtaining biologically meaningful adjusted particle images.
    """
    _label = 'predict - Image Adjustment '
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
                       help="Reference volume needed to generate the projections to be adjusted. If adjustment prediction setting "
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

        form.addParam('predictsPerPixel', params.BooleanParam, default=False,
                      label='Predict per pixel adjustment?', expertLevel=params.LEVEL_ADVANCED,
                      help='If not provided, the adjustment will be estimated per pixel - otherwise, adjustment will be estimated per projection.')

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

        if not self.predictsPerPixel:
            args += '--predict_value '

        if self.lazyLoad:
            args += '--load_images_to_ram '
        else:
            if self.scratchFolder.get() is not None:
                args += '--ssd_scratch_folder %s ' % self.scratchFolder.get()

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = None

        program = hax.Plugin.getProgram("image_gray_scale_adjustment", gpu)
        if not os.path.isdir(self._getExtraPath("imageAdjustment")):
            self.runJob(program,
                        args + f'--mode train --reload {self._getExtraPath()}'
                        if self.fineTune else args + '--mode train',
                        numberOfMpi=1)
        self.runJob(program, args + f'--mode predict --reload {self._getExtraPath()}', numberOfMpi=1)

    def createOutputStep(self):
        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticles()
        out_md = os.path.join(self._getExtraPath(), "adjusted_images.xmd")

        md = XmippMetaData(out_md)

        partSet.copyInfo(inputSet)
        partSet.setHasCTF(inputSet.hasCTF())
        partSet.setAlignmentProj()

        idx = 0
        for particle in inputSet.iterItems():
            outParticle = Particle()
            outParticle.copyInfo(particle)

            outParticle.setLocation(md[idx, "image"])

            if not self.predictsPerPixel:
                outParticle.a_adjustment = Float(md[idx, "adjustment_a"])
                outParticle.b_adjustment = Float(md[idx, "adjustment_b"])

            partSet.append(outParticle)
            idx += 1

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(inputSet, partSet)

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
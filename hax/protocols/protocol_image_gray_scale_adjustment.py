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
    Adjusts particle image gray-scale values using a neural-network-based image intensity
    normalization strategy driven by projections generated from a reference volume. The
    protocol learns how experimental particle intensities differ from synthetic projections
    and estimates corrections that make the particle dataset more internally consistent for
    downstream reconstruction, alignment, or flexible analysis.

    AI Generated:

    Image Adjustment (JaxProtImageAdjustment) — User Manual

        Overview

        The Image Adjustment protocol is intended to correct systematic gray-value
        discrepancies between experimental particle images and projections derived from a
        reference volume. In practical cryo-EM workflows, particles may differ in global
        intensity scaling, offset, or local gray-value behavior because of acquisition
        conditions, preprocessing differences, detector normalization, or dataset merging.
        These differences can degrade the consistency between experimental images and
        reference-based modeling.

        This protocol uses a neural-network-based image gray-scale adjustment algorithm to
        learn those discrepancies directly from the data. It predicts how each particle
        should be adjusted so that the experimental observations better match the reference
        projections. Biologically, this does not change the structural information contained
        in the particles, but it improves the numerical consistency of the dataset, which
        can facilitate more stable downstream refinement and interpretation.

        Inputs and General Workflow

        The protocol requires a particle set together with a reference volume. The reference
        volume is essential because it provides the projections used as the target
        statistical representation during training. Without this volume, the protocol cannot
        estimate meaningful gray-scale corrections.

        An optional binary reconstruction mask may also be provided. When used, the network
        focuses the estimation on a biologically relevant region of the projection rather
        than on the full image. This is often advantageous when particles contain large
        solvent regions, disordered peripheral densities, or substantial background
        variation.

        Before training begins, the protocol converts the reference volume into an internal
        format, rescales it if necessary so that its dimensions match the input particle
        box size, and ensures that the sampling rate remains consistent. If a mask is
        provided, the same dimensional consistency is enforced. Particle metadata are then
        written into the internal metadata representation used by the network.

        Adjustment Modes

        The protocol supports two related adjustment strategies.

        In projection-level adjustment mode, the network predicts a global linear
        correction for each particle. In this case, the output includes two parameters
        describing the gray-value transformation. These values can be interpreted as the
        global intensity rescaling and offset needed to bring the experimental image into
        better agreement with the corresponding reference projection.

        In per-pixel adjustment mode, the protocol estimates a more flexible correction
        directly at the pixel level. This mode is more expressive and may better capture
        local intensity differences, although it can also be more computationally demanding
        and potentially more sensitive to noise.

        From a biological standpoint, projection-level adjustment is usually sufficient for
        routine normalization problems, whereas per-pixel adjustment becomes more relevant
        when local detector artifacts, uneven background behavior, or subtle image-domain
        distortions are suspected.

        CTF Treatment

        The protocol allows optional consideration of the contrast transfer function. This
        determines how reference projections are matched to the experimental particles.

        When CTF correction is disabled, the protocol ignores the microscope transfer
        function. This may be acceptable in exploratory tests but is generally less
        realistic.

        When enabled, the reference projections can be compared under different correction
        assumptions, including direct application of the CTF, Wiener filtering, or
        precorrected data. The biologically appropriate choice depends on how the input
        particles were preprocessed earlier in the workflow.

        Network Training and Computational Behavior

        The protocol trains a neural network using the provided particles, the reference
        volume, and the selected optimization parameters. The user can control latent-space
        dimension, number of epochs, batch size, and learning rate.

        The latent dimension controls the internal compact representation used by the model.
        In most biological applications, this parameter mainly affects the flexibility of
        the learned correction rather than directly encoding structural heterogeneity.

        The number of epochs determines how long the network learns from the dataset. Larger
        datasets often require more epochs, while smaller datasets may converge more
        quickly.

        Batch size controls GPU memory usage. Increasing batch size usually improves
        throughput but may require substantially more memory.

        Learning rate determines optimization stability. If the network becomes unstable or
        produces numerical divergence, reducing the learning rate is usually the first
        corrective action.

        Fine Tuning and Data Loading

        The protocol supports both training from scratch and fine tuning of a previously
        trained network. Fine tuning is useful when working with related datasets acquired
        under similar experimental conditions, since previously learned intensity behavior
        can often transfer effectively.

        Data loading can be performed either directly in RAM or through lazy loading with
        optional SSD scratch storage. In large-scale cryo-EM facilities, using fast SSD
        storage often provides a favorable compromise between speed and memory usage.

        Outputs and Their Interpretation

        After prediction, the protocol produces a new particle set containing the adjusted
        images. The original particle metadata and alignment information are preserved.

        In projection-level adjustment mode, each output particle also stores the estimated
        gray-value adjustment parameters. These parameters are often useful for diagnostic
        interpretation, since they provide a direct indication of how strongly each
        particle needed to be corrected.

        From a practical biological perspective, the adjusted particle set is usually best
        interpreted as a normalized dataset with improved statistical consistency relative
        to the supplied structural reference. This can be particularly useful before
        reconstruction, refinement, or flexible analysis when intensity mismatches would
        otherwise introduce unwanted variability.

        Validation and Practical Considerations

        The protocol requires a valid reference volume. If no volume is provided, execution
        is prevented because no meaningful projection-based comparison can be performed.

        If a mask is supplied, it must be binary. Non-binary masks are rejected because the
        protocol assumes a strict distinction between included and excluded regions during
        optimization.

        In routine biological workflows, it is usually advisable to begin with global
        projection-level adjustment using a biologically reliable reference volume and a
        conservative mask focused on the stable core of the structure. Per-pixel prediction
        should generally be reserved for cases where global normalization is clearly
        insufficient.

        Final Perspective

        Image intensity normalization is often treated as a purely technical preprocessing
        step, but in practice it can strongly influence downstream structural analysis.
        Better consistency between particle images and reference projections can improve the
        robustness of learning-based methods and reduce unwanted non-structural variation.

        For this reason, the Image Adjustment protocol is best understood not as a
        structural reconstruction method, but as a quantitative harmonization step that
        helps ensure that subsequent cryo-EM analysis focuses more on biological signal and
        less on acquisition-related intensity differences.
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
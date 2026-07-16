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
from sklearn.cluster import KMeans

from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params
from pyworkflow.object import String
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_1
from pyworkflow.utils import getExt, makePath

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import Volume, ParticleFlex
from pwem import ALIGN_PROJ

import xmipp3
from xmipp3.convert import createItemMatrix, setXmippAttributes, writeSetOfParticles, geometryFromMatrix, matrixFromGeometry

import hax
import hax.constants as const

class JaxProtPredictHetSiren(ProtAnalysis3D, ProtFlexBase):
    """
    Predicts particle poses and latent coordinates using a previously trained
    HetSIREN network, and optionally decodes representative heterogeneous volumes.

    AI Generated:

    Predict HetSIREN (JaxProtPredictHetSiren) — User Manual
        Overview

        The Predict HetSIREN protocol applies a previously trained HetSIREN model
        to a new set of particles in order to estimate per-particle orientations,
        in-plane shifts, and latent coordinates describing conformational variability.

        From a biological perspective, this protocol is mainly used when a HetSIREN
        model has already learned the structural heterogeneity of a system and the
        user wants to project additional particles into that learned conformational
        landscape. This is particularly useful when processing new datasets,
        validating a trained model on independent particles, or assigning particles
        to previously identified structural states.

        Inputs and General Workflow

        The protocol requires two main inputs:

        1. A set of particles to be predicted.
        2. A previously trained HetSIREN protocol.

        The trained HetSIREN protocol provides all structural context required for
        prediction, including:

        - the trained neural network,
        - latent dimensionality,
        - input reference volume,
        - reconstruction mask,
        - CTF handling strategy,
        - downsampling settings,
        - reconstruction mode.

        In practice, this means prediction is fully constrained by the original
        training configuration. New particles are therefore interpreted in the same
        conformational space learned during training.

        Metadata Preparation

        Before prediction, the protocol converts the input particles into Xmipp
        metadata format.

        If the original HetSIREN training used downsampling, the protocol rescales
        the new particles to the same box size. This step is critical because the
        prediction network expects exactly the same dimensional input as during
        training.

        If the trained HetSIREN protocol contains a reference volume, that volume
        is copied and resized if necessary.

        If a binary mask was used during training, the same mask is also copied and
        resized. If no mask was originally provided, the protocol automatically
        creates a spherical mask.

        Biologically, this ensures that the new particles are interpreted under
        the same structural assumptions as the training dataset.

        Prediction Step

        The prediction stage runs the HetSIREN neural network in prediction mode.

        For each particle, the network estimates:

        - latent coordinates,
        - Euler angles,
        - translational shifts.

        The prediction uses exactly the same physical interpretation learned during
        training, including sampling rate correction if downsampling was applied.

        The protocol automatically inherits several key settings from the trained
        model:

        CTF handling
            The same CTF treatment used during training is reused during prediction.
            This is essential because changes in CTF preprocessing may alter the
            statistical consistency between training and prediction particles.

        Reconstruction mode
            If the trained network used mass transport, prediction also uses mass
            transport.
            If implicit architecture was enabled, that option is reused.
            If local reconstruction was used instead, the same local reconstruction
            mode is applied.

        From a biological perspective, these inherited settings guarantee that the
        predicted particles remain embedded in the same structural model learned by
        the original network.

        Memory and I/O Strategy

        The protocol provides two ways of loading particle data:

        Lazy loading into RAM
            Particles are loaded directly into memory.
            This is faster when enough RAM is available.

        SSD scratch folder
            If lazy loading is disabled, temporary memory-mapped files can be stored
            on a fast SSD or NVMe device.

        For large cryo-EM datasets, this option can strongly affect runtime.
        Although it does not change biological interpretation, it can make prediction
        substantially more practical on large particle sets.

        Output Particles

        After prediction, the protocol creates a new flexible particle set.

        Each output particle contains:

        - the original particle information,
        - predicted latent coordinates,
        - updated rigid-body transformation,
        - links to the trained HetSIREN model,
        - reference map information,
        - reference mask information,
        - CTF mode metadata.

        The latent coordinates define the particle position in the conformational
        space learned during training.

        The rigid-body transformation is reconstructed from the predicted Euler
        angles and translational shifts.

        Biologically, the resulting particle set becomes a direct representation of
        how the new particles populate the learned heterogeneity landscape.

        Decoding Representative Volumes

        A particularly useful feature of this protocol is the decoding of
        representative heterogeneous volumes.

        The predicted latent coordinates are clustered using K-means into
        20 representative conformational centers.

        These cluster centers are then decoded into 3D volumes using the trained
        HetSIREN decoder.

        The decoded maps are finally resized back to the original particle box size
        and assigned the proper sampling rate.

        From a biological point of view, these decoded volumes provide a compact
        visualization of the main conformational states sampled by the new dataset.

        This is especially useful for:

        - inspecting dominant structural states,
        - comparing conformational distributions across datasets,
        - exploring whether new particles populate previously unseen regions of
          conformational space.

        Validation

        The protocol validates the binary mask inherited from the training protocol.

        If a mask is provided, it must contain only values between 0 and 1.

        This validation is important because non-binary masks may lead to ambiguous
        structural constraints during prediction.

        Practical Recommendations

        In routine cryo-EM applications, the most important recommendation is to use
        prediction only with particles that are compatible with the original
        training dataset.

        Good practice includes:

        - similar particle preprocessing,
        - compatible box size,
        - similar sampling rate,
        - similar molecular composition.

        If the new particles differ strongly from the training data, the predicted
        latent coordinates may still be mathematically valid but biologically
        misleading.

        When comparing multiple datasets, this protocol becomes especially valuable
        because all particles are projected into the same learned conformational
        reference frame.

        Final Perspective

        Predict HetSIREN is not simply a pose prediction tool.

        It is best understood as a transfer step that projects new experimental
        particles into an already learned heterogeneity model.

        For biological interpretation, this allows users to compare datasets,
        recover representative conformational states, and study how new particles
        distribute within an existing structural landscape.
    """
    _label = 'predict - HetSIREN'
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
        group.addParam('inputParticles', params.PointerParam, label="Input particles to predict",
                       pointerClass='SetOfParticles')

        form.addSection(label='Network')
        form.addParam('hetsirenProtocol', params.PointerParam, label="HetSIREN trained network",
                       pointerClass='JaxProtFlexibleAlignmentHetSiren',
                       help="Previously executed 'angular align - HetSIREN'. "
                            "This will allow to load the network trained in that protocol to be used during "
                            "the prediction")

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

        form.addParam('batchSize', params.IntParam, default=8, label='Number of images in batch',
                       help="Determines how many images will be load in the GPU at any moment during training (set by "
                            "default to 8 - you can control GPU memory usage easily by tuning this parameter to fit your "
                            "hardware requirements - we recommend using tools like nvidia-smi to monitor and/or measure "
                            "memory usage and adjust this value - keep also in mind that bigger batch sizes might be "
                            "less precise when looking for very local motions")

        form.addParallelSection(threads=4, mpi=0)

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
        self._insertFunctionStep(self.predictStep)
        self._insertFunctionStep(self.createOutputStep)

    def writeMetaDataStep(self):
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        fnVolMask = self._getFileName('fnVolMask')

        inputParticles = self.inputParticles.get()
        Xdim = inputParticles.getXDim()
        newXdim = self._getHetSirenProtocol().boxSize.get() if self._getHetSirenProtocol().doDownsample.get() else Xdim
        vol_mask_dim = newXdim

        if self._getHetSirenProtocol().inputVolume.get():
            ih = ImageHandler()
            inputVolume = self._getHetSirenProtocol().inputVolume.get().getFileName()
            ih.convert(self._getXmippFileName(inputVolume), fnVol)
            curr_vol_dim = ImageHandler(self._getXmippFileName(inputVolume)).getDimensions()[-1]
            if curr_vol_dim != vol_mask_dim:
                self.runJob("xmipp_image_resize",
                            "-i %s --fourier %d " % (fnVol, vol_mask_dim), numberOfMpi=1,
                            env=xmipp3.Plugin.getEnviron())
            ih.setSamplingRate(fnVol, inputParticles.getSamplingRate())

        if self._getHetSirenProtocol().inputVolumeMask.get():
            ih = ImageHandler()
            inputMask = self._getHetSirenProtocol().inputVolumeMask.get().getFileName()
            if inputMask:
                ih.convert(self._getXmippFileName(inputMask), fnVolMask)
                curr_mask_dim = ImageHandler(self._getXmippFileName(inputMask)).getDimensions()[-1]
                if curr_mask_dim != vol_mask_dim:
                    self.runJob("xmipp_image_resize",
                                "-i %s --dim %d --interp nearest" % (fnVolMask, vol_mask_dim), numberOfMpi=1,
                                env=xmipp3.Plugin.getEnviron())
        else:
            ImageHandler().createCircularMask(fnVolMask, boxSize=vol_mask_dim, is3D=True)

        writeSetOfParticles(inputParticles, imgsFn)

        # Write extra attributes (if needed)
        md = XmippMetaData(imgsFn)
        if hasattr(inputParticles.getFirstItem(), "_xmipp_subtomo_labels"):
            labels = np.asarray([int(particle._xmipp_subtomo_labels) for particle in inputParticles.iterItems()])
            md[:, "subtomo_labels"] = labels
        md.write(imgsFn, overwrite=True)

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

    def predictStep(self):
        md_file = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        mask_file = self._getFileName('fnVolMask')
        out_path = self._getHetSirenProtocol()._getExtraPath()
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        batch_size = self._getHetSirenProtocol().batchSize.get()
        latDim = self._getHetSirenProtocol().latDim.get()
        newXdim = self._getHetSirenProtocol().boxSize.get()  if self._getHetSirenProtocol().doDownsample.get() else self.inputParticles.get().getXDim()
        correctionFactor = self.inputParticles.get().getXDim() / newXdim
        sr = correctionFactor * self.inputParticles.get().getSamplingRate()
        args = "--md %s --sr %f --lat_dim %d --batch_size %d  --output_path %s " \
               % (md_file, sr, latDim, batch_size, self._getExtraPath())

        if self._getHetSirenProtocol().inputVolume.get():
            args += '--vol %s ' % fnVol

        if self._getHetSirenProtocol().inputVolumeMask.get():
            args += '--mask %s ' % mask_file

        if self._getHetSirenProtocol().ctfType.get() != 0:
            if self._getHetSirenProtocol().ctfType.get() == 1:
                args += '--ctf_type apply '
            elif self._getHetSirenProtocol().ctfType.get() == 2:
                args += '--ctf_type wiener '
            elif self._getHetSirenProtocol().ctfType.get() == 3:
                args += '--ctf_type precorrect '

        if self.lazyLoad.get():
            args += '--load_images_to_ram '
        else:
            if self.scratchFolder.get() is not None:
                args += '--ssd_scratch_folder %s ' % self.scratchFolder.get()

        if self._getHetSirenProtocol().massTransport.get():
            args += '--transport_mass '

            if self._getHetSirenProtocol().isImplicit:
                args += '--implicit_network '

        elif self._getHetSirenProtocol().localRecon.get():
            args += '--local_reconstruction '

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = None

        program = hax.Plugin.getProgram("hetsiren", gpu)
        self.runJob(program, args + f'--mode predict --reload {out_path}', numberOfMpi=1)

    def createOutputStep(self):
        out_path_vols = self._getExtraPath('volumes')
        model_path = self._getHetSirenProtocol()._getExtraPath('HetSIREN')
        md_file = self._getFileName('predictedFn')
        out_path = self._getHetSirenProtocol()._getExtraPath()
        Xdim = self.inputParticles.get().getXDim()
        self.newXdim = self._getHetSirenProtocol().boxSize.get()
        correctionFactor = Xdim / self.newXdim

        metadata = XmippMetaData(md_file)
        latent_space = np.asarray([np.fromstring(item, sep=',') for item in metadata[:, 'latent_space']])
        euler_rot = metadata[:, 'angleRot']
        euler_tilt = metadata[:, 'angleTilt']
        euler_psi = metadata[:, 'anglePsi']
        shift_x = correctionFactor * metadata[:, 'shiftX']
        shift_y = correctionFactor * metadata[:, 'shiftY']

        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticlesFlex(progName=const.HETSIREN)
        
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

        partSet.getFlexInfo().modelPath = String(model_path)

        if self._getHetSirenProtocol().inputVolume.get():
            inputVolume = self._getHetSirenProtocol().inputVolume.get().getFileName()
            partSet.getFlexInfo().refMap = String(inputVolume)

        if self._getHetSirenProtocol().inputVolumeMask.get():
            inputMask = self._getHetSirenProtocol().inputVolumeMask.get().getFileName()
            partSet.getFlexInfo().refMask = String(inputMask)

        if self._getHetSirenProtocol().ctfType.get() != 0:
            if self._getHetSirenProtocol().ctfType.get() == 1:
                partSet.getFlexInfo().ctfType = String("apply")
            elif self._getHetSirenProtocol().ctfType.get() == 1:
                partSet.getFlexInfo().ctfType = String("wiener")
            elif self._getHetSirenProtocol().ctfType.get() == 2:
                partSet.getFlexInfo().ctfType = String("precorrect")
                
        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = None

        latents_kmeans = KMeans(n_clusters=20).fit(latent_space).cluster_centers_

        latents_file_txt = os.path.join(out_path, 'latents.txt')
        np.savetxt(latents_file_txt, latents_kmeans)

        if not os.path.isdir(out_path_vols):
            makePath(out_path_vols)

        args = "--latents_file %s --output_path %s --reload %s" % (latents_file_txt, out_path_vols, model_path)
        program = hax.Plugin.getProgram("decode_states_from_latents", gpu)
        self.runJob(program, args, numberOfMpi=1)

        outVols = self._createSetOfVolumes()
        outVols.setSamplingRate(inputSet.getSamplingRate())
        for idx in range(latents_kmeans.shape[0]):
            outVol = Volume()
            outVol.setSamplingRate(inputSet.getSamplingRate())

            ImageHandler().scaleSplines(os.path.join(out_path_vols, f"decoded_volume_{idx:04d}.mrc"),
                                        os.path.join(out_path_vols, f"decoded_volume_{idx:04d}.mrc"),
                                        finalDimension=inputSet.getXDim(), overwrite=True)

            ImageHandler().setSamplingRate(os.path.join(out_path_vols, f"decoded_volume_{idx:04d}.mrc"),
                                           inputSet.getSamplingRate())

            outVol.setLocation(os.path.join(out_path_vols, f"decoded_volume_{idx:04d}.mrc"))
            outVols.append(outVol)

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(inputSet, partSet)

        self._defineOutputs(outputVolumes=outVols)
        self._defineTransformRelation(inputSet, outVols)

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
        
        mask = self._getHetSirenProtocol().inputVolumeMask.get()

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
    
    def _getHetSirenProtocol(self):
        return self.hetsirenProtocol.get()
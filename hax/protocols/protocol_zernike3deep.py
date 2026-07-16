import os
import numpy as np


import xmipp3
from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params
from pyworkflow import VERSION_1
import pyworkflow.utils as pwutils

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import Volume, ParticleFlex

from xmipp3.convert import writeSetOfParticles, matrixFromGeometry
from pyworkflow.utils import getExt, makePath
from pyworkflow.utils.path import moveFile
from pwem.constants import ALIGN_PROJ, ALIGN_NONE
from pyworkflow.object import String, Boolean, Float

from sklearn.cluster import KMeans

import hax
import hax.constants as const


class JaxProtTrainZernike3Deep(ProtAnalysis3D, ProtFlexBase):
    """
    Trains a Zernike3Deep neural network to learn continuous conformational
    variability directly from cryo-EM particle images by deforming a reference
    volume toward the experimental observations. The protocol combines deep
    latent-space learning with a Zernike3D deformation model, allowing the
    reconstruction of flexible molecular landscapes rather than a single static
    structure.

    AI Generated:

    Train Zernike3Deep (JaxProtTrainZernike3Deep) — User Manual
        Overview

        The Train Zernike3Deep protocol is designed for the analysis of
        structural flexibility in cryo-EM datasets. Instead of assuming that
        all particles correspond to one rigid macromolecular conformation, this
        protocol learns a continuous latent representation describing how the
        reference map can deform in order to explain the experimental images.

        In biological practice, this is particularly useful when the dataset
        contains smooth conformational changes, domain motions, breathing
        effects, or other continuous structural rearrangements that are not
        well represented by discrete classification alone. Rather than dividing
        particles into a small number of classes, the protocol places every
        particle in a learned latent space, where nearby particles correspond
        to similar conformations.

        Inputs and Data Preparation

        The protocol requires a set of aligned particles together with a
        reference volume that acts as the structural starting point. During
        training, the network attempts to deform this reference so that its
        projections become consistent with the experimental particle images.

        An optional binary mask can also be provided. Biologically, the mask is
        important because it restricts deformation to the molecular region of
        interest. This is especially useful when flexible domains are embedded
        within large solvent regions or when only a specific portion of the
        structure is expected to move. If no mask is supplied, the protocol
        automatically generates one.

        If computational memory is limiting, the protocol can downsample the
        particle images before training. This can make otherwise intractable
        datasets feasible on smaller GPUs. The tradeoff is a reduction in the
        final spatial detail of reconstructed conformations. In practice, this
        option is often useful during exploratory analysis or early parameter
        optimization.

        CTF Handling

        The protocol supports several strategies for dealing with the contrast
        transfer function. The CTF may be ignored, applied to the forward
        projection, corrected through Wiener filtering, or treated as already
        precorrected.

        From a biological perspective, the choice depends on the state of the
        input particles and on the upstream processing workflow. Correct CTF
        treatment is especially important when subtle conformational differences
        are expected, since incorrect handling may cause the latent space to
        capture imaging artifacts rather than biologically meaningful motions.

        Neural Network Training

        At the core of the protocol is a neural network that learns a latent
        representation of conformational variability. Each particle is mapped
        into a latent vector whose dimensionality is controlled by the latent
        dimension parameter. Smaller latent spaces enforce simpler global
        motions, whereas larger latent spaces allow richer and more complex
        deformations.

        Training proceeds for a user-defined number of epochs and uses mini-
        batch stochastic optimization. The batch size directly affects GPU
        memory usage and may also influence the sensitivity of the model to
        highly localized motions. Smaller batches can sometimes capture subtle
        flexibility more effectively, whereas larger batches usually improve
        throughput.

        The learning rate governs the stability of optimization. In practical
        cryo-EM applications, overly aggressive learning rates may lead to
        unstable training or divergence, whereas very small learning rates may
        slow convergence substantially. For most biological datasets, the
        default value provides a reliable starting point.

        Zernike3D Basis and Deformation Freedom

        Structural deformations are parameterized using a Zernike3D basis. The
        Zernike degree and spherical harmonics degree jointly determine the
        expressive power of the deformation model.

        Biologically, low values favor smooth, large-scale collective motions,
        such as hinge-like rearrangements or rigid-body shifts of domains.
        Higher values allow finer local rearrangements, but they also increase
        flexibility and therefore the risk of fitting noise or poorly supported
        image features.

        For this reason, the choice of basis complexity should ideally reflect
        the expected scale of the biological motion rather than simply maximizing
        mathematical freedom.

        Gaussian Volume Representation

        Before latent learning begins, the reference map is approximated by a
        set of Gaussian components. The protocol can estimate automatically how
        many Gaussians are needed, or the user may define a fixed number.

        This representation provides a compact deformable model of the input
        structure. In practice, fixing the number of Gaussians can be useful
        when the user wants to impose prior biological knowledge, such as
        relating the representation to residue count, domain granularity, or
        expected structural complexity.

        Checkpointing and Performance Considerations

        Long training jobs can be resumed from a previous checkpoint. This is
        particularly useful for large datasets, multi-day runs, or iterative
        optimization of hyperparameters.

        The protocol also provides different data-loading strategies. When
        enough RAM is available, loading particles directly into memory usually
        gives the best training speed. Otherwise, memory-mapped loading is
        possible, and the use of fast SSD or NVMe scratch storage can greatly
        improve throughput.

        Outputs and Structural Interpretation

        After training, the protocol predicts a latent coordinate for every
        particle. These latent vectors are stored in a flexible particle set,
        together with updated projection geometry and metadata linking the
        particles to the learned model, the reference map, and the applied mask.

        Biologically, each particle becomes associated with a specific location
        in conformational space. Nearby particles correspond to similar
        molecular states, while distant particles represent larger structural
        differences.

        To facilitate interpretation, the protocol clusters the latent space
        into representative states and decodes a set of volumes corresponding
        to the latent-space cluster centers. These decoded maps provide a
        discrete visualization of the continuous landscape learned by the
        network.

        This step is often especially valuable for biological interpretation,
        since it allows users to inspect representative conformations, identify
        dominant structural transitions, and connect latent-space organization
        with known biochemical states or functional mechanisms.

        Practical Interpretation

        In exploratory structural biology, this protocol is particularly useful
        when discrete classification produces blurred classes or when a
        continuum of conformations is suspected. It often reveals motions that
        are difficult to isolate by traditional classification methods,
        including smooth opening and closing events, flexible domain rotations,
        and correlated collective rearrangements.

        The most meaningful biological interpretation usually comes not from
        the latent coordinates alone, but from combining latent-space
        organization, decoded representative volumes, and prior knowledge of
        molecular function.

        Final Perspective

        For cryo-EM studies of conformational heterogeneity, Zernike3Deep
        provides a powerful framework to move beyond static reconstruction. By
        learning how a reference structure continuously deforms across the
        experimental particle set, the protocol enables a richer and often more
        biologically faithful description of molecular dynamics.
    """

    _label = "train - Zernike3Deep"
    _lastUpdateVersion = VERSION_1

    def _defineParams(self, form):
        # ------------------------
        # INPUT SECTION
        # ------------------------
        form.addSection(label="Input")

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

        group.addParam(
            "inputParticles",
            params.PointerParam,
            label="Input particles",
            pointerClass="SetOfParticles",
            important=True,
        )

        group.addParam(
            "inputVolume",
            params.PointerParam,
            pointerClass="Volume",
            label="Input volume",
            important=True,
            help="Volume to be warped toward the images.",
        )

        group.addParam(
            "inputVolumeMask",
            params.PointerParam,
            pointerClass="VolumeMask",
            label="Mask (optional)",
            allowsNull=True,
            help="Binary mask enclosing all mass to be moved. "
            "If empty, the volume will be auto-masked.",
        )

        group.addParam('doDownsample', params.BooleanParam, default=False,
                       label='Downsample the images before training?',
                       help='If your GPU is not able to fit your particles in memory, you can play with this parameter to downsample your images. '
                            'In this way, you will be able to fit your particles at the expense of losing resolution in the volumes generated by the '
                            'network.'
        )

        group.addParam(
            "boxSize",
            params.IntParam,
            condition="doDownsample",
            default=128,
            label="Downsample particles to this box size",
            help="If your GPU is not able to fit your particles in memory, you can play with this parameter to downsample your images. "
            "In this way, you will be able to fit your particles at the expense of losing resolution in the volumes generated by the "
            "network.",
        )

        group.addParam(
            "ctfType",
            params.EnumParam,
            choices=["None", "apply", "wiener", "precorrect"],
            default=1,
            label="CTF handling",
            help=(
                "* *None*: CTF will not be considered\n"
                "* *Apply*: CTF is applied to the projection generated from the reference map\n"
                "* *Wiener*: input particle is CTF corrected by a Wiener fiter\n"
                "* *Precorrect: similar to Wiener but CTF has already been corrected"
            ),
        )

        # ------------------------
        # TRAINING GROUP
        # ------------------------
        form.addSection(label="Training")

        form.addParam(
            "restoreCheckpoint",
            params.BooleanParam,
            default=False,
            label="Continue from previous run",
            help="During this training, this protocol saves the current state of"
            " the network. If you enable this option, the training will continue"
            " from a previous run, instead of starting from scratch.",
        )

        form.addParam('lazyLoad', params.BooleanParam, default=False,
                      expertLevel=params.LEVEL_ADVANCED, label='Lazy loading into RAM',
                      help='If provided, images will be loaded to RAM. This is recommended if you want the best performance '
                           'and your dataset fits in your RAM memory. If this flag is not provided, '
                           'images will be memory mapped. When this happens, the program will trade disk space for performance. '
                           'Thus, during the execution additional disk space will be used and the performance '
                           'will be slightly lower compared to loading the images to RAM. Disk usage will be back to normal '
                           'once the execution has finished.'
        )

        form.addParam('scratchFolder', params.PathParam,
                      condition="not lazyLoad",
                      label='Path to SSD scratch folder',
                      help='If you are not loading the images to RAM, we strongly recommend to provide here a path to a folder in '
                           'a SSD/NVME disk to speed up the data loading. In general, you can expected a decrease in training performance when '
                           'loading images > 256px on a HDD disk.'
        )

        train = form.addGroup("Network hyperparameters")
        train.addParam(
            "latDim",
            params.IntParam,
            default=8,
            label="Latent dimension",
            help="Dimensionality of the latent space.",
        )

        train.addParam("epochs", params.IntParam, default=50, label='Number of training epochs',
                       help="Number of epochs to train the network (i.e. how many times to loop over the whole dataset "
                             "of images - set to default to 50 - as a rule of thumb, consider 50 to 100 epochs enough "
                             "for 100k images / if your dataset is bigger or smaller, scale this value proportionally to it"
        )

        train.addParam("batchSize", params.IntParam, default=8, label='Number of images in batch',
                       help="Determines how many images will be load in the GPU at any moment during training (set by "
                            "default to 8 - you can control GPU memory usage easily by tuning this parameter to fit your "
                            "hardware requirements - we recommend using tools like nvidia-smi to monitor and/or measure "
                            "memory usage and adjust this value - keep also in mind that bigger batch sizes might be "
                            "less precise when looking for very local motions"
        )

        train.addParam(
            "learningRate", params.FloatParam, default=1e-4, label="Learning rate",
            help="The learning rate (lr) sets the speed of learning. Think of the model as trying to find the "
                 "lowest point in a valley; the lr is the size of the step it takes on each attempt. A large "
                 "lr (e.g., 0.01) is like taking huge leaps — it's fast but can be unstable, overshoot the "
                 "lowest point, or cause NAN errors. A small lr (e.g., 1e-6) is like taking tiny shuffles — "
                 "it's stable but very slow and might get stuck before reaching the bottom. A good default is "
                 "often 0.0001. If training fails or errors explode, try making the lr 10 times smaller (e.g., "
                 "0.001 --> 0.0001)."
        )

        train = form.addGroup("Zernike3D basis parameters")
        train.addParam("L1", params.IntParam, default=7, label='Zernike degree',
                       help="This parameters controls the degree of the Zernike polynomials used to compute the Zernike3D "
                            "basis. The larger the value, the larger the freedom to find a given conformational change. "
                            "This value must be >= to the spherical harmonics degree."
                       )

        train.addParam("L2", params.IntParam, default=7, label='Spherical harmonics degree',
                       help="This parameters controls the degree of the spherical harmonics used to compute the Zernike3D "
                            "basis. The larger the value, the larger the freedom to find a given conformational change."
                       )

        # ------------------------
        # RECONSTURCTION GROUP
        # ------------------------
        form.addSection(label='Reconstruction')

        form.addParam('numberGaussians', params.IntParam, label='Number of gaussians (fixed)',
                      expertLevel=params.LEVEL_ADVANCED,
                      allowsNull=True,
                      help="Before training the network, Zernike3Deep will try to fit a set of Gaussians in the reference volume to recreate it. "
                           "The default criterium is to automatically determine the number of Gaussians neede to reproduce the reference volume "
                           "with high-fidelity. However, if you prefer to fix the number of Gaussians in advance based on your own criterium (e.g., "
                           "the number of residues in your protein), you can set this parameter. When set, the Zernike3Deep will fit this fixed number of Gaussians "
                           "so that the reproduce the reference volume as well as possible.")

    def _createFilenameTemplates(self):
        """Centralize how files are called"""
        myDict = {
            "imgsFn": self._getExtraPath("input_particles.xmd"),
            "predictedFn": self._getExtraPath("predicted_latents.xmd"),
            "fnVol": self._getExtraPath("volume.mrc"),
            "fnVolMask": self._getExtraPath("mask.mrc"),
        }

        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions -----------------------
    def writeMetadataStep(self):

        fnVol = self._getFileName("fnVol")
        fnVolMask = self._getFileName("fnVolMask")
        imgsFn = self._getFileName("imgsFn")

        inputParticles = self.inputParticles.get()
        Xdim = inputParticles.getXDim()
        newXdim = self.boxSize.get()  if self.doDownsample.get() else Xdim
        vol_mask_dim = newXdim

        if self.inputVolume.get():
            ih = ImageHandler()

            inputVolume = self.inputVolume.get().getFileName()
            ih.convert(self._getXmippFileName(inputVolume), fnVol)

            curr_vol_dim = ImageHandler(
                self._getXmippFileName(inputVolume)
            ).getDimensions()[-1]

            if curr_vol_dim != vol_mask_dim:
                self.runJob(
                    "xmipp_image_resize",
                    "-i %s --fourier %d " % (fnVol, vol_mask_dim),
                    numberOfMpi=1,
                    env=xmipp3.Plugin.getEnviron(),
                )

            ih.setSamplingRate(fnVol, inputParticles.getSamplingRate())

        if self.inputVolumeMask.get():
            ih = ImageHandler()
            inputMask = self.inputVolumeMask.get().getFileName()
            if inputMask:
                ih.convert(self._getXmippFileName(inputMask), fnVolMask)
                curr_mask_dim = ImageHandler(
                    self._getXmippFileName(inputMask)
                ).getDimensions()[-1]
                if curr_mask_dim != vol_mask_dim:
                    self.runJob(
                        "xmipp_image_resize",
                        "-i %s --dim %d --interp nearest" % (fnVolMask, vol_mask_dim),
                        numberOfMpi=1,
                        env=xmipp3.Plugin.getEnviron(),
                    )
        else:
            ImageHandler().createCircularMask(
                fnVolMask, boxSize=vol_mask_dim, is3D=True
            )

        writeSetOfParticles(inputParticles, imgsFn)

        # Write extra attributes (if needed)
        md = XmippMetaData(imgsFn)
        if hasattr(inputParticles.getFirstItem(), "_xmipp_subtomo_labels"):
            labels = np.asarray(
                [
                    int(particle._xmipp_subtomo_labels)
                    for particle in inputParticles.iterItems()
                ]
            )
            md[:, "subtomo_labels"] = labels
        md.write(imgsFn, overwrite=True)

        if newXdim != Xdim:
            params = "-i %s -o %s --save_metadata_stack %s --fourier %d" % (
                imgsFn,
                self._getTmpPath("scaled_particles.stk"),
                self._getExtraPath("scaled_particles.xmd"),
                newXdim,
            )
            if self.numberOfMpi.get() > 1:
                params += " --mpi_job_size %d" % int(
                    inputParticles.getSize() / self.numberOfMpi.get()
                )
            self.runJob(
                "xmipp_image_resize",
                params,
                numberOfMpi=self.numberOfMpi.get(),
                env=xmipp3.Plugin.getEnviron(),
            )
            moveFile(self._getExtraPath("scaled_particles.xmd"), imgsFn)

    def trainingStep(self):
        md_file = self._getFileName("imgsFn")
        vol_file = self._getFileName("fnVol")
        mask_file = self._getFileName("fnVolMask")
        out_path = self._getExtraPath()

        if not os.path.isdir(out_path):
            os.mkdir(out_path)

        batch_size = self.batchSize.get()
        learningRate = self.learningRate.get()
        epochs = self.epochs.get()
        latDim = self.latDim.get()

        newXdim = self.boxSize.get() if self.doDownsample.get() else self.inputParticles.get().getXDim()
        correctionFactor = self.inputParticles.get().getXDim() / newXdim
        sr = correctionFactor * self.inputParticles.get().getSamplingRate()

        args = (
            "--md %s --sr %f --lat_dim %d --epochs %d --batch_size %d --learning_rate %s --L1 %d --L2 %d --vol %s "
            "--mask %s --output_path %s "
            % (
                md_file,
                sr,
                latDim,
                epochs,
                batch_size,
                learningRate,
                self.L1.get(),
                self.L2.get(),
                vol_file,
                mask_file,
                out_path,
            )
        )

        if self.ctfType != 0:
            if self.ctfType.get() == 1:
                args += "--ctf_type apply "
            elif self.ctfType.get() == 2:
                args += "--ctf_type wiener "
            elif self.ctfType.get() == 3:
                args += "--ctf_type precorrect "
        else:
            args += "--ctf_type None "

        if self.lazyLoad:
            args += "--load_images_to_ram "
        else:
            if self.scratchFolder.get() is not None:
                args += '--ssd_scratch_folder %s ' % self.scratchFolder.get()

        if self.numberGaussians.get() is not None:
            args += '--num_gaussians %d ' % self.numberGaussians.get()

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = None

        program = hax.Plugin.getProgram("zernike3deep", gpu)

        if not os.path.isdir(self._getExtraPath("Zernike3Deep")):
            self.runJob(
                program,
                (
                    args + f"--mode train --reload {self._getExtraPath()}"
                    if self.restoreCheckpoint
                    else args + "--mode train"
                ),
                numberOfMpi=1,
            )

        self.runJob(
            program,
            args + f"--mode predict --reload {self._getExtraPath()}",
            numberOfMpi=1,
        )

    def createOutputStep(self):
        inputParticles = self.inputParticles.get()
        out_path_vols = self._getExtraPath("volumes")
        model_path = self._getExtraPath("Zernike3Deep")
        md_file = self._getFileName("predictedFn")
        out_path = self._getExtraPath()
        Xdim = inputParticles.getXDim()
        self.newXdim = self.boxSize.get()

        metadata = XmippMetaData(md_file)
        correctionFactor = Xdim / self.newXdim
        latent_space = np.asarray(
            [np.fromstring(item, sep=",") for item in metadata[:, "latent_space"]]
        )
        euler_rot = metadata[:, "angleRot"]
        euler_tilt = metadata[:, "angleTilt"]
        euler_psi = metadata[:, "anglePsi"]
        shift_x = correctionFactor * metadata[:, "shiftX"]
        shift_y = correctionFactor * metadata[:, "shiftY"]

        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticlesFlex(progName=const.ZERNIKE3D)

        partSet.copyInfo(inputSet)
        partSet.setHasCTF(inputSet.hasCTF())
        partSet.setAlignmentProj()
        inverseTransform = partSet.getAlignment() == ALIGN_PROJ

        idx = 0
        for particle in inputSet.iterItems():
            outParticle = ParticleFlex(progName=const.ZERNIKE3D)
            outParticle.copyInfo(particle)
            outParticle.setZFlex(latent_space[idx])

            # Set new transformation matrix
            tr = matrixFromGeometry(
                np.array([shift_x[idx], shift_y[idx], 0.0]),
                np.array([euler_rot[idx], euler_tilt[idx], euler_psi[idx]]),
                inverseTransform,
            )
            outParticle.getTransform().setMatrix(tr)

            partSet.append(outParticle)
            idx += 1

        partSet.getFlexInfo().modelPath = String(model_path)

        inputVolume = self.inputVolume.get().getFileName()
        partSet.getFlexInfo().refMap = String(inputVolume)

        if self.inputVolumeMask.get().getFileName() is not None:
            inputMask = self.inputVolumeMask.get().getFileName()
        else:
            inputMask = self._getExtraPath("mask.mrc")
        partSet.refMask = String(inputMask)

        if self.ctfType.get() != 0:
            if self.ctfType.get() == 1:
                partSet.getFlexInfo().ctfType = String("apply")
            elif self.ctfType.get() == 1:
                partSet.getFlexInfo().ctfType = String("wiener")
            elif self.ctfType.get() == 2:
                partSet.getFlexInfo().ctfType = String("precorrect")

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = None

        latents_kmeans = KMeans(n_clusters=20).fit(latent_space).cluster_centers_

        latents_file_txt = os.path.join(out_path, "latents.txt")
        np.savetxt(latents_file_txt, latents_kmeans)

        if not os.path.isdir(out_path_vols):
            makePath(out_path_vols)

        args = "--latents_file %s --output_path %s --reload %s" % (
            latents_file_txt,
            out_path_vols,
            self._getExtraPath("Zernike3Deep"),
        )
        program = hax.Plugin.getProgram("decode_states_from_latents", gpu)
        self.runJob(program, args, numberOfMpi=1)

        outVols = self._createSetOfVolumes()
        outVols.setSamplingRate(inputParticles.getSamplingRate())
        for idx in range(latents_kmeans.shape[0]):
            outVol = Volume()
            outVol.setSamplingRate(inputParticles.getSamplingRate())

            ImageHandler().scaleSplines(
                os.path.join(out_path_vols, f"decoded_volume_{idx:04d}.mrc"),
                os.path.join(out_path_vols, f"decoded_volume_{idx:04d}.mrc"),
                finalDimension=inputParticles.getXDim(),
                overwrite=True,
            )

            ImageHandler().setSamplingRate(
                os.path.join(out_path_vols, f"decoded_volume_{idx:04d}.mrc"),
                inputParticles.getSamplingRate(),
            )

            outVol.setLocation(
                os.path.join(out_path_vols, f"decoded_volume_{idx:04d}.mrc")
            )
            outVols.append(outVol)

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(inputParticles, partSet)

        self._defineOutputs(outputVolumes=outVols)
        self._defineTransformRelation(inputParticles, outVols)

    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep(self.writeMetadataStep)
        self._insertFunctionStep(self.trainingStep)
        self._insertFunctionStep(self.createOutputStep)

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

        mask = self.inputVolumeMask.get()

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

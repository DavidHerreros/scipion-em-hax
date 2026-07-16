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
from xmipp3.convert import writeSetOfParticles, matrixFromGeometry

import hax
import hax.constants as const

class JaxProtFlexibleAlignmentHetSiren(ProtAnalysis3D, ProtFlexBase):
    """
    Protocol for angular alignment with heterogeneous reconstruction using the
    HetSIREN algorithm. This protocol estimates particle orientations while
    simultaneously learning a latent representation of structural variability,
    allowing continuous heterogeneous states to be modeled directly from
    cryo-EM particle images.

    AI Generated:

    Flexible Alignment with HetSIREN (JaxProtFlexibleAlignmentHetSiren) — User Manual

    This protocol performs angular alignment and heterogeneous reconstruction by
    combining image alignment with a neural representation of structural
    flexibility. Instead of assuming that all particles correspond to a single
    rigid structure, HetSIREN learns a low-dimensional latent space in which
    each particle is assigned a continuous coordinate describing its structural
    state. In practical cryo-EM analysis, this makes the protocol especially
    useful for proteins or complexes that undergo conformational transitions,
    domain rearrangements, or continuous motions that cannot be represented
    adequately by a small number of discrete classes.

    The protocol starts from a set of input particles and optionally from a
    reference volume. When a reference map is provided, HetSIREN uses it as an
    initial structural prior and progressively learns how that map must deform
    in order to explain the heterogeneity present in the experimental data.
    When no initial map is provided, the network learns the heterogeneous
    representation directly from the particles. This makes the method suitable
    both for exploratory analyses, where little prior knowledge is available,
    and for refinement workflows, where an existing reconstruction already
    provides a biologically meaningful starting point.

    A reconstruction mask can also be supplied. Its role depends strongly on
    the selected reconstruction mode. In mass transport mode, the mask defines
    the region whose density can be displaced across the volume. This is useful
    when the main biological question concerns large-amplitude motions or
    significant domain displacements. In standard heterogeneous reconstruction
    mode, the mask instead defines the region of interest in which structural
    variability is analyzed. For biological interpretation, the most useful
    masks are usually those that include the structurally relevant core while
    excluding solvent regions and irrelevant peripheral density. Since the
    protocol validates that masks are binary, masks should always be prepared
    accordingly before execution.

    The protocol optionally downsamples particle images before training. This
    option is primarily intended to reduce GPU memory requirements when working
    with large box sizes. Downsampling makes training substantially more
    accessible on limited hardware, although it reduces the effective spatial
    resolution of the learned volumes. During metadata preparation, input
    volumes and masks are automatically resized to the working box size, and
    particle metadata are updated consistently so that later alignment and
    reconstruction remain geometrically coherent.

    Training is controlled through a set of neural network hyperparameters.
    These include the latent dimensionality, which determines how much
    conformational variability can be represented, the number of epochs, the
    batch size, and the learning rate. In biological terms, the latent
    dimension defines the expressive capacity of the heterogeneity model:
    smaller values encourage compact descriptions of motion, while larger
    values allow more complex structural variability but may also capture
    noise if used excessively. The protocol also includes a denoising
    regularization term that encourages smoother reconstructions and can be
    particularly helpful when analyzing noisy datasets.

    The protocol supports several strategies for handling contrast transfer
    effects. Depending on the selected option, the reference projections can be
    CTF-filtered, particles can be Wiener-corrected, or the algorithm can
    assume that CTF correction has already been applied previously. Choosing an
    appropriate CTF strategy is important because alignment accuracy depends
    strongly on the consistency between particle images and the model
    projections generated during training.

    Two related reconstruction philosophies are available. In mass transport
    mode, the network explicitly models density displacement, which is often
    more appropriate for large coherent motions. An optional implicit
    architecture can be enabled in this regime to obtain richer local motion
    descriptions, although at a higher GPU memory cost. When mass transport is
    disabled, the protocol performs heterogeneous reconstruction instead. In
    that case, local reconstruction can be enabled to focus the analysis on a
    specific flexible region of the macromolecule rather than distributing
    representational capacity over the full volume.

    During execution, the protocol first prepares all metadata, resampling
    volumes, masks, and particle stacks when necessary. It then launches
    HetSIREN training and subsequently prediction. Prediction produces, for
    every particle, an updated orientation, in-plane shifts, and a latent
    coordinate. These latent coordinates are stored in the output particle set
    together with the refined alignment parameters, making the output directly
    usable for downstream flexible analysis, clustering, or visualization.

    A second biologically important output is a representative set of decoded
    volumes. After prediction, the latent space is clustered using k-means,
    and the centers of those clusters are decoded into volumes that represent
    characteristic conformational states sampled by the dataset. These decoded
    maps are automatically rescaled to the original particle box size and
    assigned the correct sampling rate. For biological interpretation, these
    volumes often provide a convenient first overview of the principal motions
    present in the experiment.

    In practice, this protocol is most valuable when the goal is not only to
    refine particle orientations but also to understand structural variability
    in a continuous manner. When the biological system is expected to move
    along smooth conformational trajectories rather than occupy sharply
    separated states, the latent representation learned by HetSIREN often
    provides a more faithful description than traditional discrete
    classification approaches.

    A good practical strategy is to begin with moderate latent dimensionality,
    conservative denoising, and a biologically meaningful mask when the region
    of motion is known. If strong large-scale motions are expected, mass
    transport mode is often the most informative starting point. For more
    localized flexibility or refinement of a known domain, local
    reconstruction generally offers easier interpretation. As with all
    heterogeneous cryo-EM analyses, the most reliable conclusions arise when
    latent organization, decoded volumes, and known structural biology are
    interpreted together rather than in isolation.
    """
    _label = 'flexible align - HetSIREN'
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

        group.addParam('inputVolume', params.PointerParam, allowsNull=True,
                       label="Starting volume", pointerClass='Volume',
                       help="If not provided, HetSIREN will learn from scratch how to recover heterogeneous states. If a "
                            "volume is provided, HetSIREN will use that volume as its starting state, and learn how to "
                            "modify it towards the heterogeneous states present in the data.\n\n"
                            "When should a map be provided?\n"
                            "    - If transport mass mode is set to Yes (check Advanced parameter in Reconstruction tab), it is "
                            "advisable to provide an input map\n"
                            "    - If transport mass mode is set to Yes and "
                            "If transport mass mode is set to No and local reconstruction is enabled (check Advanced "
                            "parameter in Reconstruction tab), it is highly recommended to provide a map\n\n"
                            "When is not advisable to provide a map?\n"
                            "If transport mass and local reconstruction are ser to No (check Advanced parameter in "
                            "Reconstruction tab), we recommend leaving this  parameter empty to give HetSIREN more freedom "
                            "to denoise and learn accurate states.")

        group.addParam('inputVolumeMask', params.PointerParam,
                       label="Reconstruction mask", pointerClass='VolumeMask', allowsNull=True,
                       help="Reconstruction mask meaning will depend on the mode chosen (NOTE: masks provided here MUST "
                            "be BINARY):\n\n"
                            "Transport mass is set to Yes\n"
                            "In this case, the provided mask will determine which voxels can be moved inside the volume box. "
                            "If you have provided an input volume, you can provide here a tight mask or a dilated tight mask, as HetSIREN "
                            "will be able to move those voxels anywhere needed to recover a given heterogeneous state. If don't provide an input "
                            "volume, we recommend providing a mask with enough mass to recover all expected states. A good rule of thumb "
                            "for this case is to provide a spherical mask with radius 0.25 * box_size.\n\n"
                            "Transport mass is set to No\n"
                            "If local reconstruction is set to No, we recommend to leave this parameter empty. If local reconstruction is "
                            "enabled, you should provide here a mask enclosing your region of interest AND any region where the motion of the "
                            "protein is expected to happen. Remember that, in Reconstruction mode (i.e., Transport mass is set to No), HetSIREN "
                            "will assume voxels as fixed in space. Any motions happening outside the binary mask will NOT be considered.")

        group.addParam('doDownsample', params.BooleanParam, default=False,
                       label='Downsample the images before training?',
                       help='If your GPU is not able to fit your particles in memory, you can play with this parameter to downsample your images. '
                            'In this way, you will be able to fit your particles at the expense of losing resolution in the volumes generated by the '
                            'network.')

        group.addParam('boxSize', params.IntParam, default=128,
                       condition="doDownsample",
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

        form.addSection(label='Network')
        form.addParam('fineTune', params.BooleanParam, default=False, label='Fine tune previous network?',
                      help='When set to Yes, you will be able to provide a previously trained HetSIREN network to refine it with new '
                           'data. If set to No, you will train a new HetSIREN network from scratch.')

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
                       help="Dimension of the HetSIREN bottleneck (latent space dimension)")

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

        group = form.addGroup("Network regularization")
        group.addParam('denoisingStrength', params.FloatParam, default=0.0, label='Denoising strength',
                       expertLevel=params.LEVEL_ADVANCED,
                       help="Determines how strongly HetSIREN will learn to remove noise from the resulting volumes. "
                            "Increasing the value of this parameter will result in a stronger regularization of the noise,"
                            " but it may affect the protein signal as well. (NOTE: We recommend setting this parameter in "
                            "the range 0.0001 to 0.1)")

        form.addSection(label='Reconstruction')
        form.addParam('massTransport', params.BooleanParam, default=True,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Mass transportation',
                      help='When set, HetSIREN will be able to "move" the mass inside the mask '
                           'instead of just reconstructing the volume. This implies that HetSIREN will estimate the motion '
                           'to be applied to the points within the provided mask, instead of considering them fixed in space. '
                           'This approach is useful when working with large box sizes that '
                           'do not fit in GPU memory, or when a more through analysis of motions is desired. '
                           'If False, HetSIREN program will perform heterogeneous reconstruction.')

        form.addParam('isImplicit', params.BooleanParam, expertLevel=params.LEVEL_ADVANCED, default=True,
                      condition='massTransport',
                      label='Use implicit architecture?',
                      help='HetSIREN implicit architecture is able to report better local motions present in the protein at the '
                           'expense of a larger GPU memory consumption. If you are running out of memory in the GPU, setting this '
                           'parameter to False might help reducing the memory burden.')

        form.addParam('numberGaussians', params.IntParam, label='Number of gaussians (fixed)',
                      expertLevel=params.LEVEL_ADVANCED,
                      allowsNull=True, condition="inputVolume",
                      help="Before training the network, HetSIREN will try to fit a set of Gaussians in the reference volume to recreate it. "
                           "The default criterium is to automatically determine the number of Gaussians neede to reproduce the reference volume "
                           "with high-fidelity. However, if you prefer to fix the number of Gaussians in advance based on your own criterium (e.g., "
                           "the number of residues in your protein), you can set this parameter. When set, the HetSIREN will fit this fixed number of Gaussians "
                           "so that the reproduce the reference volume as well as possible.")

        form.addParam('localRecon', params.BooleanParam, expertLevel=params.LEVEL_ADVANCED, default=False,
                       condition='not massTransport',
                       label='Local reconstruction',
                       help='When set, HetSIREN will turn to local heterogeneous reconstruction/refinement mod, '
                            'focusing the analysis of heterogeneity to a region of interest enclosed by the provided reference mask.')

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
        newXdim = self.boxSize.get() if self.doDownsample.get() else Xdim
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
        denoisingStrength = self.denoisingStrength.get()
        newXdim = self.boxSize.get() if self.doDownsample.get() else self.inputParticles.get().getXDim()
        correctionFactor = self.inputParticles.get().getXDim() / newXdim
        sr = correctionFactor * self.inputParticles.get().getSamplingRate()
        args = "--md %s --sr %f --lat_dim %d --epochs %d --batch_size %d --learning_rate %s --output_path %s --denoising_strength %f " \
               % (md_file, sr, latDim, epochs, batch_size, learningRate, out_path, denoisingStrength)

        if self.inputVolume.get():
            args += '--vol %s ' % vol_file

            if self.numberGaussians.get() is not None:
                args += '--num_gaussians %d ' % self.numberGaussians.get()

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

        if self.lazyLoad:
            args += '--load_images_to_ram '
        else:
            if self.scratchFolder.get() is not None:
                args += '--ssd_scratch_folder %s ' % self.scratchFolder.get()

        if self.massTransport:
            args += '--transport_mass '

            if self.isImplicit:
                args += '--implicit_network '

        elif self.localRecon:
            args += '--local_reconstruction '

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = None

        program = hax.Plugin.getProgram("hetsiren", gpu)
        if not os.path.isdir(self._getExtraPath("HetSIREN")):
            self.runJob(program,
                        args + f'--mode train --reload {self._getExtraPath()}'
                        if self.fineTune else args + '--mode train',
                        numberOfMpi=1)
        self.runJob(program, args + f'--mode predict --reload {self._getExtraPath()}', numberOfMpi=1)

    def createOutputStep(self):
        out_path_vols = self._getExtraPath('volumes')
        model_path = self._getExtraPath('HetSIREN')
        md_file = self._getFileName('predictedFn')
        out_path = self._getExtraPath()
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

        if self.inputVolume.get():
            inputVolume = self.inputVolume.get().getFileName()
            partSet.getFlexInfo().refMap = String(inputVolume)

        if self.inputVolumeMask.get():
            inputMask = self.inputVolumeMask.get().getFileName()
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
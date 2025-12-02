# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
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
import shutil
import numpy as np
from glob import glob
from xmipp_metadata.image_handler import ImageHandler

from pyworkflow import NEW, VERSION_1
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.protocol.params import PointerParam, IntParam, BooleanParam, StringParam, USE_GPU, GPU_LIST
import pyworkflow.utils as pwutils
from pyworkflow.utils.properties import Message
from pyworkflow.gui.dialog import askYesNo
from pyworkflow.object import Integer

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import ClassFlex, VolumeFlex, SetOfClassesFlex

import hax
from hax.utils import getOutputSuffix
from hax.annotate_space_functions.annotate_space_arguments import getReducedSpaceArguments


class JaxProtAnnotateSpace(ProtAnalysis3D, ProtFlexBase):
    """ Interactive annotation of conformational spaces """

    _label = 'annotate space'
    _devStatus = NEW
    _lastUpdateVersion = VERSION_1
    OUTPUT_PREFIX = 'flexible3DClasses'
    OUTPUT_PREFIX_CLASSES = 'flexible3DClasses'
    OUTPUT_PREFIX_VOLUMES = 'flexible3DVolumes'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addHidden(USE_GPU, BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                                     Select the one you want to use.")
        form.addHidden(GPU_LIST, StringParam, default='0',
                       expertLevel=LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")
        form.addParam('particles', PointerParam, label="Particles to annotate",
                      pointerClass='SetOfParticlesFlex', important=True)
        form.addParam('boxSize', IntParam, label="Box size",
                      condition="particles and particles.getFlexInfo().getProgName() == 'CryoDRGN'",
                      help="Volumes generated from the CryoDrgn network will be resampled to the "
                           "chosen box size (only for the visualization).")

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.launchVolumeSlicer, interactive=True)

    def _createOutput(self):
        particles = self.particles.get()
        sr = particles.getSamplingRate()
        partIds = np.array(list(particles.getIdSet())).astype(int)
        progName = particles.getFlexInfo().getProgName()

        # Get FlexHub set creation functions
        createFn = self._createSetOfClassesFlex
        createFnSet = self._createSetOfVolumesFlex

        # Create SetOfFlexClasses
        suffix = getOutputSuffix(self, SetOfClassesFlex)
        flexClasses = createFn(self.particles, suffix, progName=progName)
        flexSetVols = createFnSet(progName=progName, suffix=suffix)
        flexSetVols.setSamplingRate(sr)

        # Folder with saved results
        layers_folder = self._getExtraPath(os.path.join("Intermediate_results", "selections_layers"))

        # Populate set of classes
        clInx = 1
        newId = 1
        for layer_folder in os.listdir(layers_folder):
            # Folder with saved volumes and particle indices
            representative_path = os.path.join(layers_folder, layer_folder, "representative.mrc")
            particle_ids_path = os.path.join(layers_folder, layer_folder, "particle_indices.txt")

            if "KMeans" not in layer_folder:
                newClass = ClassFlex()
                newClass.copyInfo(particles)
                newClass.setObjId(clInx)
                newClass.setHasCTF(particles.hasCTF())
                newClass.setAcquisition(particles.getAcquisition())
                representative = VolumeFlex(progName=progName)
                representative.setLocation(representative_path)
                if hasattr(representative, "setSamplingRate"):
                    representative.setSamplingRate(sr)

                # Check if representative needs resampling and set correct sampling rate in header
                ImageHandler().scaleSplines(representative_path, representative_path, finalDimension=particles.getXDim(), overwrite=True)
                ImageHandler().setSamplingRate(representative_path, sr)

                newClass.setRepresentative(representative)

                flexSetVols.append(representative)
                flexClasses.append(newClass)

                # Populate images
                enabledClass = flexClasses[newClass.getObjId()]
                enabledClass.enableAppend()
                particle_ids = partIds[np.loadtxt(particle_ids_path, dtype=int)]
                for particle_id in particle_ids:
                    item = particles[int(particle_id)]
                    item._xmipp_subtomo_labels = Integer(clInx)
                    item.setObjId(newId)
                    enabledClass.append(item)
                    newId += 1

                flexClasses.update(enabledClass)
                clInx += 1

        # Save new output
        name_classes = self.OUTPUT_PREFIX_CLASSES + "_" + suffix
        name_volumes = self.OUTPUT_PREFIX_VOLUMES + "_" + suffix
        args = {}
        args[name_classes] = flexClasses
        args[name_volumes] = flexSetVols

        self._defineOutputs(**args)
        self._defineSourceRelation(particles, flexClasses)

    # --------------------------- STEPS functions -----------------------------
    def launchVolumeSlicer(self):
        particles = self.particles.get()
        self.num_vol = 0

        # Check whether intermediate results' folder has been created
        if not os.path.isdir(self._getExtraPath("Intermediate_results")):
            pwutils.makePath(self._getExtraPath("Intermediate_results"))

        # Get Z space
        z_space = []
        for particle in particles.iterItems():
            z_space.append(particle.getZFlex())
        z_space = np.asarray(z_space)

        # Generate files to call command line
        file_z_space = self._getExtraPath(os.path.join("Intermediate_results", "z_space.txt"))
        np.savetxt(file_z_space, z_space)

        # Get common arguments
        path = os.path.abspath(self._getExtraPath("Intermediate_results"))
        args = f"--z_space {file_z_space} --path {path} --mode {particles.getFlexInfo().getProgName()}"

        # Provide ChimeraX path if plugin is installed
        try:
            from chimera import Plugin as chimeraPlugin
            chimerax_bin_path = os.path.join(chimeraPlugin.getHome(), "bin", "ChimeraX")
            args += f" --chimerax_binary {chimerax_bin_path}"
        except ImportError:
            print("ChimeraX plugin is not installed. Annotate space connection with ChimeraX will not be available")

        # GPU id
        gpu = ','.join([str(elem) for elem in self.getGpuList()])

        # Check if program has a reduced space saved (like in FlexConsensus)
        if particles.getFirstItem().getZRed() is not None:
            args += f" {getReducedSpaceArguments(particles, self._getExtraPath())}"

        # Program specific arguments
        progName = particles.getFlexInfo().getProgName()
        if progName == "Zernike3D":
            from hax.annotate_space_functions.annotate_space_arguments import getZernike3DArguments
            args += f" {getZernike3DArguments(particles)}"
        elif progName == "HetSIREN":
            from hax.annotate_space_functions.annotate_space_arguments import getHetSIRENArguments
            args += f" {getHetSIRENArguments(particles)}"
        elif progName == "Dynamight":
            from relion.dynamight.annotate_space_arguments import getAnnotateSpaceArguments
            if gpu:
                args += f" {getAnnotateSpaceArguments(particles, gpu_id=gpu)}"
            else:
                args += f" {getAnnotateSpaceArguments(particles)}"

        program = "annotate_space"
        program = hax.Plugin.getProgram(program, gpu=gpu)
        self.runJob(program, args)
        if len(glob(self._getExtraPath(os.path.join("Intermediate_results", "selections_layers*/")))) > 0 and \
           askYesNo(Message.TITLE_SAVE_OUTPUT, Message.LABEL_SAVE_OUTPUT, None):
            self._createOutput()

    # --------------------------- OUTPUT functions -----------------------------
    def deleteOutput(self, output):
        attrName = self.findAttributeName(output)
        output_id = attrName.split("_")[-1]
        volumes_path = self._getExtraPath(f"Output_Volumes_{output_id}")
        shutil.rmtree(volumes_path)
        super().deleteOutput(output)

    def allowsDelete(self, obj):
        return True

    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        summary = []
        if self.getOutputsSize() >= 1:
            for _, outClasses in self.iterOutputAttributes():
                summary.append("Output *%s*:" % outClasses.getNameId().split('.')[1])
                summary.append("    * Total annotated classes: *%s*" % outClasses.getSize())
        else:
            summary.append("Output annotated classes not ready yet")
        return summary

    def _methods(self):
        return [
            "Interactive annotation of conformational spaces",
        ]

    # ----------------------- VALIDATE functions ----------------------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        particles = self.particles.get()

        # Check CryoDRGN boxsize parameter is set as it is mandatory (TODO: Can we set this automatically in each program with custom parameters?)
        if particles.getFlexInfo().getProgName() == 'CryoDRGN':
            if self.boxSize.get() is None:
                errors.append("Boxsize parameter needs to be set to an integer value smaller than or equal "
                              "to the boxsize used internally to train the CryoDRGN network")
            elif self.boxSize.get() % 2 != 0:
                errors.append("Boxsize parameter needs to be an even value")

        return errors

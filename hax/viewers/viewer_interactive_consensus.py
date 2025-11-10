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

from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
import pyworkflow.protocol.params as params

from hax.protocols.protocol_interactive_consensus import JaxProtInteractiveFlexConsensus
from hax.viewers.functions.interactive_histogram import InteractiveHist

class JaxFlexConsensusView(ProtocolViewer):
    """ Interactive FlexConsensus filtering """
    _label = 'viewer FlexConsensus'
    _targets = [JaxProtInteractiveFlexConsensus]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _choices = ["Consensus error", "Representation error"]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)

    def _defineParams(self, form):
        form.addSection(label='Show deformation')
        form.addParam('histChoice', params.EnumParam,
                      choices=self._choices, default=0,
                      label='Error histogram to display', display=params.EnumParam.DISPLAY_COMBO,
                      help="\t Consensus error: Error distribution computed directly in FlexConsensus space\n" \
                           "\t Representation error: Error distribution computed when decoding FlexConsensus space "
                           "towards the input spaces")
        form.addParam('doShowHist', params.LabelParam,
                      label="Display the selected histogram in interactive mode")

    def _getVisualizeDict(self):
        self.chosen = self._choices[self.histChoice.get()]
        return {'doShowHist': self._doShowHist}

    # ------------------- Interactive histogram method -------------------
    def _doShowHist(self, param=None):
        if self.chosen == self._choices[0]:
            file = [f for f in os.listdir(self.protocol._getExtraPath()) if f.endswith('consensus_error')]
            data = np.load(self.protocol._getExtraPath(file))
        elif self.chosen == self._choices[1]:
            file = [f for f in os.listdir(self.protocol._getExtraPath()) if f.endswith('representation_error')]
            data = np.load(self.protocol._getExtraPath(file))

        hist = InteractiveHist(data, self.protocol)
        hist.show()

    # ------------------- ------------------- -------------------
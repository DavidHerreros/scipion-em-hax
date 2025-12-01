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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import seaborn as sns
import pandas as pd
from pyworkflow import Config

sns.set_theme()

class InteractiveHist(object):

    def __init__(self, data, protocol):
        self.data = pd.Series(data, name="data")
        self.protocol = protocol
        self.threshold = None
        self.idx_selected = None
        self.total_selected = None

    def update(self, val):
        self.threshold = val
        self.idx_selected = np.argwhere(self.data.values <= self.threshold)
        self.total_selected = self.idx_selected.size
        self.limit_line.set_xdata([val, val])
        self.axs.set_title("Total images selected: %d" % self.total_selected)
        self.fig.canvas.draw_idle()

    def _plotButton(self):
        axcreateSubset = plt.axes([0.75, 0.02, 0.2, 0.050])
        color = Config.SCIPION_MAIN_COLOR if hasattr(Config, "SCIPION_MAIN_COLOR") else "maroon"
        axcreateSubset.text(0.5, 0.5, 'Subset Particles',
                            verticalalignment='center',
                            horizontalalignment='center',
                            transform=axcreateSubset.transAxes, color='white')
        bcreateSubset = Button(axcreateSubset, '',
                               color=color,
                               hovercolor='maroon')
        bcreateSubset.on_clicked(self.createSubset)
        return bcreateSubset

    def createSubset(self, event):
        np.savetxt(self.protocol._getExtraPath("selected_idx.txt"), self.idx_selected)
        self.protocol._createOutput()

    def show(self):
        self.fig, self.axs = plt.subplots(1, 1, figsize=(10, 5))
        self.fig.subplots_adjust(bottom=0.25)

        sns.kdeplot(self.data, ax=self.axs, fill=True, linewidth=0, color="green")

        self.axs.set_xlabel("RMSE")
        self.axs.set_ylabel("Density")
        slider_ax = self.fig.add_axes([0.20, 0.1, 0.60, 0.03])

        self.slider = Slider(slider_ax, "Threshold", self.data.min(), self.data.max())
        self.limit_line = self.axs.axvline(self.slider.val, color='k')
        self.threshold = self.slider.val
        self.idx_selected = np.argwhere(self.data.values <= self.threshold)
        self.total_selected = self.idx_selected.size
        self.axs.set_title("Total images selected: %d" % self.total_selected)
        self.bcreateSubset = self._plotButton()
        self.slider.on_changed(self.update)

        plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import hex2color, rgb2hex
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import iqr
import pathlib

# Import functions for calculations
from functions import *

cpal = {'Olive': '#647314', 'Green': '#5eb342', 'Teal': '#48bcbc', 
        'Blue': '#00488d', 'Purple': '#792373', 'Red': '#c5373d',
        'Orange': '#b34900', 'Yellow': '#9b740a', 'Skin': '#734e3d', 
        'Tone': '#5e5948', 'Gray': '#435369'}

mpl.rcParams['font.sans-serif'] = 'Arial'
tickwidth = 0.5
mpl.rcParams.update({'font.size': 8, 'lines.linewidth':1, 'axes.linewidth':tickwidth,
                     'xtick.major.width':tickwidth, 'xtick.minor.width':tickwidth,
                     'ytick.major.width':tickwidth, 'ytick.minor.width':tickwidth})

figpath = (pathlib.Path().absolute().resolve().parents[1]) / ('OneDrive - Chalmers\Python\ev_waveguide_characterization3\Figures')

def plot_histograms(ax, y, lims, binmode, h, cc, alpha, lw, pde, pde_res, pde_lw, pde_alpha, labels):
    if binmode == 'auto':
        h = np.zeros_like(y)
        for i in range(len(y)):
            h[i] = 2*iqr(y[i])/(len(y[i])**(1/3))
        h = np.nanmax(h)
        print(h)

    bins = np.arange(lims[0], lims[1], h)

    for i in range(len(y)):
        yh, xh = np.histogram(y[i], density=True, bins=bins)
        print(labels[i])
        ax[i].hist(y[i], density=True, color=cc[i], bins=bins, alpha=alpha, label=labels[i])
        ax[i].step(xh[:-1], yh, color=cc[i], where='post', linewidth=lw)
        if pde == True:
            xx = np.linspace(lims[0], lims[1], pde_res)
            ypde = y[i][(y[i] < lims[1]) & (y[i] > lims[0])]
            kde = gaussian_kde(ypde)
            y2 = kde(xx)
            ax[i].plot(xx, y2, color='k', linewidth=pde_lw, alpha=pde_alpha)
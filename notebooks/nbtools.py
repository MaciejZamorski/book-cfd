import os
import sys

module_path = os.path.abspath(os.path.join('../code/'))
if module_path not in sys.path:
    sys.path.append(module_path)

import time
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.style.use('../code/style.mplstyle')

def init_plot(x, y):    
    fig, ax = plt.subplots(figsize=(5,4))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    return fig, ax

def update_plot(y, fig, line):
    line.set_ydata(y)
    display.clear_output(wait=True)
    display.display(fig)
    time.sleep(0.001)

def get_colors(n, start=0.2, end=1, rev=True):
    if rev:
        return plt.cm.gist_heat_r(np.linspace(start, end, n))
    return plt.cm.gist_heat(np.linspace(start, end, n))  
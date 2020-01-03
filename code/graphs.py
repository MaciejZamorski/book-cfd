# coding: utf-8
import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.interpolate as spi
import matplotlib.patches as mpatch
import matplotlib.path as mpath
import matplotlib as mpl


def _rotate_vector(v, rad):
    M = np.array([[np.cos(rad), -np.sin(rad)], 
                  [np.sin(rad), np.cos(rad)]])
    return v @ M


def _get_colors(n, start=0.2, end=1, rev=True):
    if rev:
        return plt.cm.gist_heat_r(np.linspace(start, end, n))
    return plt.cm.gist_heat(np.linspace(start, end, n))   

def convert_to_graph(fig, ax, log=False):
    # Taken from
    # https://stackoverflow.com/questions/17646247/how-to-make-fuller-axis-arrows-with-matplotlib
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['bottom', 'right', 'top', 'left']:
        ax.spines[side].set_visible(False)

    # removing the axis ticks
    plt.xticks([])  # labels
    plt.yticks([])
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.xaxis.set_ticks_position('none')  # tick markers
    ax.yaxis.set_ticks_position('none')
    
    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./40.*(ymax-ymin)
    hl = 1./40.*(xmax-xmin)
    lw = 0.5  # axis line width
    ohg = 0.3  # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin) * height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin) * width/height

    if not log:
        # draw x and y axis
        ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw=lw, 
                head_width=hw, head_length=hl, overhang=ohg, 
                length_includes_head=True, clip_on=False)

        ax.arrow(xmin, ymin, 0, ymax-ymin, fc='k', ec='k', lw=lw, 
                head_width=yhw, head_length=yhl, overhang=ohg, 
                length_includes_head=True, clip_on=False)

    return fig, ax


def remove_axes(fig, ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis=u'both', which=u'both',length=0)
    for side in ['bottom', 'right', 'top', 'left']:
        ax.spines[side].set_visible(False)
    return fig, ax


def ensemble_average():
    """Draws plot for ensemble average of RANS section"""
    fig, ax = plt.subplots(figsize=(6*0.7, 3.5*0.7))

    # Ensure graph is the same if needed
    np.random.seed(1)

    npts = 200
    navg = 20000
    maxt = 10

    x = np.linspace(0, maxt, npts)

    # Ensemble averaging
    y = np.random.random(npts) * 0.5*np.sin(np.pi/5*x) + \
        0.5*np.random.random(npts) + 0.75
    ax.plot(x, y, 'k', lw=1, label=r'$u_i(\vec{x}, t)$')

    ymean = 0
    for i in range(navg):
        np.random.seed(i)
        ymean += np.random.random(npts) * 0.5*np.sin(np.pi/5*x) + \
            0.5*np.random.random(npts) + 0.75
    ymean = ymean/navg
    ax.plot(x, ymean, '--k', lw=1, label=r'$u_E(\vec{x}, t)$')

    ax.set_ylim([0, 2])
    ax.set_xlim([0, maxt])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$u$')

    fig, ax = convert_to_graph(fig, ax)
    plt.legend()
    plt.savefig('ch7_ensemble_averaging.pdf')
    plt.close(fig)


def time_average():
    """Draws a time-averaged solution for the Reynolds averaging section"""
    fig, ax = plt.subplots(figsize=(6*0.7, 3.5*0.7))

    # Ensure graph is the same if needed
    np.random.seed(1)

    npts = 200
    maxt = 10

    x = np.linspace(0, maxt, npts)

    # Time averaging (statistically-stationary)
    y = 0.5*np.random.random(npts) + 0.75
    ax.plot(x, y, lw=1, color='black', label=r'$u_i(\vec{x}, t)$')
    ax.axhline(np.mean(y), linestyle='--', color='black', 
               lw=1, label=r'$u_T(\vec{x})$')

    ax.set_ylim([0, 2])
    ax.set_xlim([0, maxt])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$u$')

    fig, ax = convert_to_graph(fig, ax)
    plt.legend()
    plt.savefig('ch7_time_averaging.pdf')
    plt.close(fig)


def discrete_scheme():
    """
    Draws a 1d representation of the FVM and FDM methods
    l : domain length
    dx : mesh spacing
    ds : slope line length
    dax : axes extra distance
    dy : vertical distance to slope/average marker
    """

    # Changing these values requires manual
    # modification of the tick labels

    l = 1.2
    dx = 0.2
    ds = 0.04
    dax = dx/2
    dy = 0

    for which in ['fv', 'fd']:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        # Create a random initial curve
        npts = 9
        np.random.seed(3)
        x0 = np.linspace(0, l, npts)
        y0 = 0.4*np.random.rand(npts)+0.3
        y0[0] += 0.3

        # Approximate with a polynomial
        y = np.polyfit(x0, y0, 6)
        dydx = np.polyder(y)

        # Compute data
        xcurve = np.linspace(0, l, 100)
        ycurve = np.polyval(y, xcurve)

        # Plot curve
        ax.plot(xcurve, ycurve, lw=1, color='black')

        if which == 'fd':
            xnod = np.linspace(0, l, int(l/dx))
            ynod = np.polyval(y, xnod)
            ch = 10
            # Draw vertical lines
            ax.vlines(xnod, ynod*0, ynod, lw=0.7, color='gray', linestyle='--')

            # Plot nodes
            ax.plot(xnod, ynod, 'o', markersize=5, color='black')
            ax.plot(xnod, ynod*0, 'o', markersize=4, color='black')

            # Get value of slope at nodes
            dydx_nodes = np.polyval(dydx, xnod)

            # Plot slopes
            for i, p in enumerate(xnod):
                xm = np.linspace(p-ds, p+ds, 10)
                ym = (xm-xnod[i])*dydx_nodes[i] + ynod[i] + dy
                plt.plot(xm, ym, lw=2, color='red')

        elif which == 'fv':
            xnod = np.linspace(0, l, int(l/dx)+1)
            ynod = np.polyval(y, xnod)
            ch = 11
            yavg = []
            ymax = []
            for i, p in enumerate(xnod[:-1]):
                xdata = np.linspace(xnod[i], xnod[i+1], 100)
                avgval = np.mean(np.polyval(y, xdata))
                yavg.append(avgval)

                if i == 0:
                    ymax.append(avgval)
                else:
                    ymax.append(np.max([yavg[-2], yavg[-1]]))

                rectangle = plt.Rectangle(
                    (xnod[i], 0), (xnod[i+1]-xnod[i]), avgval, 
                    hatch='////', facecolor='#ededed', edgecolor='#d6d6d6')
                ax.add_artist(rectangle)

            ax.hlines(yavg, xnod[:-1], xnod[1:], lw=2, color='red')
            # Draw vertical lines
            ax.vlines(xnod, ynod*0, [*ymax, yavg[-1]], 
                      lw=0.7, color='gray', linestyle='--')

        # delta x annotation
        ax.text((xnod[3] + xnod[2])/2, np.min(ynod)/2*1.1, 
                r'$\Delta x$', horizontalalignment='center')
        ax.annotate("", xy=(xnod[3], np.min(ynod)/2), xytext=(xnod[2], np.min(
            ynod)/2), arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color='black'), )

        ax.set_ylim([-dax/2, l-dax])
        ax.set_xlim([-dax/2, l+dax])
        ax.set_ylabel('$u(x)$')
        fig, ax = convert_to_graph(fig, ax)

        if which == 'fd':
            plt.xticks(np.linspace(0, l, int(
                l/dx)), [r'$x_{i-2}$', r'$x_{i-1}$', 
                r'$x_{i}$', r'$x_{i+1}$', r'$x_{i+2}$'])
        elif which == 'fv':
            plt.xticks(np.linspace(0+dx/2, l-dx/2, int(l/dx)), 
                       [r'$\Omega_{i-2}$', r'$\Omega_{i-1}$', 
                       r'$\Omega_{i}$', r'$\Omega_{i+1}$', r'$\Omega_{i+2}$'])

        fig.savefig(f'ch{ch}_{which}_scheme.pdf')
        plt.close(fig)


def stability_advection():
    """Draws the unit circle with eigenvalues of advection equation"""
    dx = 1.0

    colors = plt.cm.gist_heat_r(np.linspace(0.2, 1, 6))
    kappas = np.linspace(0, 2*np.pi, 500)

    for which in ['explicit', 'implicit']:
        fig, ax = plt.subplots()

        # which = 'us'

        if which == 'explicit':
            def ampf(cfl, kappa): return 1-cfl+cfl*np.exp(1j*kappa*dx)
            center = (0, 0)
            xlim = [-1.2, 1.2]
            r = 1
            maxval = 1.0
            cfls = [0.5, 0.6, 0.7, 0.8, 0.9]
            ylim = [-1.1, 1.3]
        elif which == 'implicit':
            def ampf(cfl, kappa): return 1/(1+cfl*(1-np.exp(-1j*kappa*dx)))
            center = (0.5, 0)
            xlim = [-1.2, 1.2]
            r = 0.5
            maxval = '\infty'
            cfls = [0.5, 1.0, 2.0, 4.0, 8.0]
            ylim = [-1.1, 1.3]

        circle = plt.Circle((0, 0), 1, facecolor='#e6e6e6', 
                            edgecolor='#c2c2c2', linestyle='--', hatch='////')
        ax.add_artist(circle)

        handles = []
        for i, cfl in enumerate(cfls):
            y = []
            for kappa in kappas:
                y.append(ampf(cfl, kappa))

            opts = {
                'color': colors[i], 
                'lw': 1.1, 
                'label': f'$\\sigma={cfl}$'
            }

            line, = ax.plot(np.real(y), np.imag(y), **opts)
            handles.append(line)

        circle = plt.Circle(center, r, fill=False, linestyle='--')
        handles.append(circle)
        ax.add_artist(circle)

        # Create fake plot for circle legend
        ax.plot([], [], label=f'$\\sigma={maxval}$', 
                color='black', lw='1', linestyle='--')
        ax.axhline(y=0, lw=0.7, color='black')
        ax.axvline(x=0, lw=0.7, color='black')
        ax.set_xlabel(r'Re$\left(e^{a\Delta t}\right)$')
        ax.set_ylabel(r'Im$\left(e^{a\Delta t}\right)$')
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_aspect('equal', )
        plt.legend(loc=9, ncol=3, handlelength=0.8, borderaxespad=0)

        plt.tight_layout()
        fig.savefig(f'ch12_{which}_advection.pdf')
        plt.close(fig)


def stability_diffusion():
    """Draws the unit circle with eigenvalues of difffusion equation"""
    dx = 1.0

    colors = plt.cm.gist_heat_r(np.linspace(0.2, 1, 4))
    kappas = np.linspace(0, 2*np.pi, 20)

    for which in reversed(['explicit', 'implicit']):
        fig, ax = plt.subplots()

        if which == 'explicit':
            def ampf(cfl, kappa): return 1+cfl * \
                (np.exp(-1j*kappa*dx)-2+np.exp(1j*kappa*dx))
            cfls = [0.1, 0.3, 0.5]
            markers = ['o', 's', 'd', 'p']
        elif which == 'implicit':
            def ampf(cfl, kappa): return 1 / \
                (1-cfl*(np.exp(-1j*kappa*dx)-2+np.exp(1j*kappa*dx)))
            cfls = [1.0, 5.0, 10.0]
            markers = ['o', 's', 'd', 'p']

        circle = plt.Circle((0, 0), 1, facecolor='#e6e6e6', 
                            edgecolor='#c2c2c2', linestyle=None, hatch='////')
        ax.add_artist(circle)

        handles = []
        for i, cfl in enumerate(cfls):
            y = []
            for kappa in kappas:
                y.append(ampf(cfl, kappa))

            opts = {
                'color': colors[i], 
                'lw': 1.1, 
                'label': f'$\\sigma={cfl}$', 
                'marker': markers[i], 
            }

            line, = ax.plot(np.real(y), np.imag(y), **opts)
            handles.append(line)

        ax.axhline(y=0, lw=0.7, color='black')
        ax.axvline(x=0, lw=0.7, color='black')
        ax.set_xlabel(r'Re$\left(e^{a\Delta t}\right)$')
        ax.set_ylabel(r'Im$\left(e^{a\Delta t}\right)$')
        ax.set_ylim([-1.1, 1.3])
        ax.set_xlim([-1.2, 1.2])
        ax.set_aspect('equal', )
        plt.legend(loc=9, ncol=3, handlelength=0.8, borderaxespad=0)
        plt.tight_layout()

        fig.savefig(f'ch12_{which}_diffusion.pdf')
        plt.close(fig)


def fv_arbitrary_volume():
    """Draws a volume of arbitrary shape for derivation of FVM"""
    pts = np.array([[0, 0], 
                    [0.25, 0.45], 
                    [0.5, 0.40], 
                    [0.75, 0.4], 
                    [1, 0], 
                    [0.75, -0.35], 
                    [0.5, -0.5], 
                    [0.38, -0.36], 
                    [0, 0]])
    pts = pts*0.3
    tck, u = spi.splprep(pts.T, u=None, s=0.0, per=1)
    u = np.linspace(u.min(), u.max(), 100)
    # Get spline coordinates
    xs, ys = spi.splev(u, tck, der=0)

    # Get its derivatives
    xd, yd = spi.splev(u, tck, der=1)

    # Get normal vector for point p
    p = 43
    v = -np.array([xd[p], yd[p]])
    nhat = _rotate_vector(v, np.pi/2)
    nhat = nhat / 2
    fv = _rotate_vector(v, -np.pi/8)*1.2

    xc = (xs.max() + xs.min())/2
    yc = (ys.max() + ys.min())/2

    fig, ax = plt.subplots(figsize=(4, 3))

    # Plot flux and normal vectors
    ax.quiver(xs[p], ys[p], nhat[0], nhat[1], 
              units='xy', scale=15, scale_units='xy')
    ax.quiver(xs[p], ys[p], -fv[0], -fv[1], units='xy', 
              scale=15, scale_units='xy', color=blue)

    ax.plot(xs, ys, 'k', lw=1)
    # ax.fill(xs, ys, color='#dedede')
    ax.plot(xs[p], ys[p], 'o', color='black', markersize=4)

    ax.annotate(r'$\Omega$', xy=(xc, yc))
    ax.annotate(r'$S$', xy=pts[5]+np.array([0.015, -0.015]))

    xp = np.array([xs[p], ys[p]])
    ax.annotate(r'$\hat n$', xy=xp+np.array([0.001, +0.015]))
    ax.annotate(r'$\vec F$', xy=xp+np.array([0.030, -0.022]))

    # Remove the axes
    ax.set_xlim([-0.01, 0.36])
    ax.set_ylim([-0.16, 0.16])
    ax.margins(0.2)
    fig, ax = remove_axes(fig, ax)
    plt.show()
    fig.savefig(f'ch11_fv_arbitrary_volume.pdf')
    plt.close(fig)


def fv_trivolume():
    """Draws a triangular FV volume with fluxes and solution on it"""
    L = 1
    t = np.pi/3
    cost = np.cos(t)
    sint = np.sin(t)

    # Vertices
    pts = [[0, 0], [L*cost, L*sint], [L, 0], [0, 0]]
    pts = np.array(pts)

    # Face center
    cpts = [[L/2*cost, L/2*sint], [L/2*(1+cost), L/2*sint], [L/2, 0]]
    cpts = np.array(cpts)

    # Get surface vectors
    svec = pts[1:] - pts[:-1]

    # Get normal vectors
    nhat = -_rotate_vector(svec, np.pi/2)
    nhat = nhat/np.max(nhat, axis=0)

    # Get random flux vectors
    np.random.seed(5)
    fvec = (nhat + 0.5*np.random.rand(3, 2))*1.6

    # Modify vectors 2 and 3
    fvec[-2] *= -1
    # fvec[-1] *= 1.4

    fig, ax = plt.subplots(figsize=(3.2, 4))
    ax.quiver(cpts[:, 0], cpts[:, 1], nhat[:, 0], 
              nhat[:, 1], units='xy', scale=12)
    ax.quiver(cpts[:, 0], cpts[:, 1], fvec[:, 0], 
              fvec[:, 1], color=blue, units='xy', scale=9)
    ax.plot(pts[:, 0], pts[:, 1], 'k', lw=1)

    ax.annotate(r'$\hat n_1$', xy=cpts[0]+np.array([-0.1, -0.035]))
    ax.annotate(r'$\hat n_2$', xy=cpts[1]+np.array([+0.07, -0.035]))
    ax.annotate(r'$\hat n_3$', xy=cpts[2]+np.array([-0.1, -0.1]))

    ax.annotate(r'$\vec F_1$', xy=cpts[0]+np.array([-0.065, +0.18]))
    ax.annotate(r'$\vec F_2$', xy=cpts[1]+np.array([-0.06, -0.18]))
    ax.annotate(r'$\vec F_3$', xy=cpts[2]+np.array([+0.03, -0.14]))

    ax.annotate(r'$S_1$', xy=cpts[0]+np.array([+0.085, +0.01]), ha='center')
    ax.annotate(r'$S_2$', xy=cpts[1]+np.array([-0.085, +0.01]), ha='center')
    ax.annotate(r'$S_3$', xy=cpts[2]+np.array([0, +0.02]), ha='center')

    # Draw solution point
    xc = L/(np.sqrt(3))
    r = xc/2
    ax.plot(L/2, r, 'o', color=red)
    ax.annotate(r'$\vec{\overline{u}}$', xy=(L/2, r+0.04), ha='center')

    ax.set_ylim([-0.4, 1])
    fig, ax = remove_axes(fig, ax)

    fig.savefig('ch11_fv_cellavg.pdf')
    plt.close(fig)


def fv_riemann_diagram():
    """Draws Riemann problem using two triangles"""
    fig, ax = plt.subplots(figsize=(4, 3))

    # Define node coordinates
    x = [[0, 0.4], 
         [1.1, 0], 
         [2, 0.8], 
         [0.9, 1]]
    x = np.array(x)

    # Draw triangles
    ax.plot(x[:, 0], x[:, 1], 'k', lw=1)
    ax.plot(x[[0, 3], 0], x[[0, 3], 1], 'k', lw=1)
    ax.plot(x[[1, 3], 0], x[[1, 3], 1], 'k', lw=1)

    # Get location of flux vector
    xfv = (x[3] + x[1])/2

    # Draw flux vector
    ax.quiver(xfv[0], xfv[1], 1, 0.5, color=blue, units='xy', scale=2.5)
    ax.annotate(r'$\vec{F}$', xy=(xfv[0]+0.15, xfv[1]+0.1))

    # Draw solution points
    ax.plot([xfv[0]-0.25, xfv[0]+0.25], 
            [xfv[1]-0.1, xfv[1]-0.1], 'o', color=red)
    ax.annotate(r'$\vec{\overline{u}}_L$', xy=(
        xfv[0]-0.25, xfv[1]-0.25), ha='center')
    ax.annotate(r'$\vec{\overline{u}}_R$', xy=(
        xfv[0]+0.25, xfv[1]-0.25), ha='center')

    fig, ax = remove_axes(fig, ax)
    fig.savefig('ch11_fv_riemann_diagram.pdf')
    # plt.show()
    plt.close(fig)

def fv_1d_advection():
    """Draw diagram to derive FV advection"""
    x = np.array([0, 1/3, 2/3, 1])+0.05
    u = np.array([0.8, 0.5, 0.7])
    xc = 0.5*(x[:-1] + x[1:])

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.vlines(x, 0, 1, lw=0.8, color='gray', linestyle='--')
    ax.hlines(u, x[:-1], x[1:], lw=1.2, color=red)

    # Draw normals and fluxes
    ax.quiver(x[[1, 2]], u[[1, 1]], [-1, 1], [0, 0])
    ax.quiver(x[[1, 2]]-0.1, [1, 1], [2, 2], [0, 0], units='xy', scale=8, color=blue)
    ax.quiver(xc[1]-0.075, 0.7, 1.5, 0, units='xy', scale=8)

    ax.annotate(r'$\overline u_{i-1}$', 
                xy=(xc[0], u[0]-0.05), ha='center', va='top')
    ax.annotate(r'$\overline u_{i}$', 
                xy=(xc[1], u[1]-0.05), ha='center', va='top')
    ax.annotate(r'$\overline u_{i+1}$',
                xy=(xc[2], u[2]-0.05), ha='center', va='top')
    ax.annotate(r'$\vec F_1(\overline u_{i-1}, \overline u_i)$', 
                xy=(x[1], 1.14), ha='center', va='top')
    ax.annotate(r'$\vec F_2(\overline u_i, \overline u_{i+1})$', 
                xy=(x[2], 1.14), ha='center', va='top')
    ax.annotate(r'$\hat n_1$', xy=(x[1]-0.1, u[1]), 
                ha='center', va='center')
    ax.annotate(r'$\hat n_2$', xy=(x[2]+0.1, u[1]), 
                ha='center', va='center')
    ax.annotate(r'$\alpha$', xy=(xc[1], 0.8), 
                ha='center', va='top')
    ax.annotate(r'$\Delta x$', xy=(xc[1], 0.1), 
                ha='center', va='top')
    ax.annotate(r'', xy=(x[1], 0.12), xytext=(x[2], 0.12), 
                arrowprops=dict(arrowstyle="<|-|>", 
                lw=0.7, color='black'), )

    ax.plot(xc, u, 'o', color=red)
    # ax.margins(0.1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u$')
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 1.2])
    fig, ax = convert_to_graph(fig, ax)
    ax.tick_params(which='major', pad=10)
    plt.xticks(xc, [r'$\Omega_{i-1}$', r'$\Omega_{i}$', 
               r'$\Omega_{i+1}$'], va='center' )
    fig.tight_layout()
    # plt.show()
    fig.savefig('ch11_fv_advection.pdf')
    plt.close(fig)

def fv_1d_diffusion():
    """Draw diagram to derive FV diffusion"""
    x = np.array([0, 1/3, 2/3, 1])+0.05
    u = np.array([0.8, 0.5, 0.7])
    xc = 0.5*(x[:-1] + x[1:])

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.vlines(x, 0, 1, lw=0.8, color='gray', linestyle='--')
    ax.hlines(u, x[:-1], x[1:], lw=1.2, color=red)
    
    dudxp = np.polyfit(xc[:-1], u[:-1], 1)
    dudxx = np.linspace(xc[0], xc[1], 2)
    dudxy = np.polyval(dudxp, dudxx)
    ax.plot(dudxx, dudxy, lw=0.8, linestyle='--', color=blue)

    dudxp = np.polyfit(xc[1:], u[1:], 1)
    dudxx = np.linspace(xc[1], xc[2], 2)
    dudxy = np.polyval(dudxp, dudxx)
    ax.plot(dudxx, dudxy, lw=0.8, linestyle='--', color=blue)

    # Draw normals
    ax.quiver(x[[1, 2]], u[[1, 1]], [-1, 1], [0, 0])

    ax.annotate(r'$\overline u_{i-1}$', 
                xy=(xc[0], u[0]-0.05), ha='center', va='top')
    ax.annotate(r'$\overline u_{i}$', 
                xy=(xc[1], u[1]-0.05), ha='center', va='top')
    ax.annotate(r'$\overline u_{i+1}$', 
                xy=(xc[2], u[2]-0.05), ha='center', va='top')
    ax.annotate(r'$\left(\frac{\partial u}{\partial x}\right)_1$', 
                xy=(x[1]+0.1, 0.8), ha='left', va='center')
    ax.annotate(r'$\left(\frac{\partial u}{\partial x}\right)_2$', 
                xy=(x[2]+0.1, 0.85), ha='left', va='center')
    ax.annotate(r'$\hat n_1$', xy=(x[1]-0.1, u[1]), 
                ha='center', va='center')
    ax.annotate(r'$\hat n_2$', xy=(x[2]+0.1, u[1]), 
                ha='center', va='center')
    ax.annotate(r'$\Delta x$', xy=(xc[1], 0.1), 
                ha='center', va='top')
    ax.annotate(r'', xy=(x[1], 0.12), xytext=(x[2], 0.12), 
                arrowprops=dict(arrowstyle="<|-|>", 
                lw=0.7, color='black'), )
    ax.annotate(r'', xy=(x[2]+0.01, 0.62), xytext=(x[2]+0.1, 0.85), 
                arrowprops=dict(arrowstyle="-|>", 
                lw=0.7, color='black'), )
    ax.annotate(r'', xy=(x[1]+0.01, 0.65), xytext=(x[1]+0.1, 0.8), 
                arrowprops=dict(arrowstyle="-|>", 
                lw=0.7, color='black'), )

    ax.plot(xc, u, 'o', color=red)
    # ax.margins(0.1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u$')
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 1.2])
    fig, ax = convert_to_graph(fig, ax)
    ax.tick_params(which='major', pad=10)
    plt.xticks(xc, [ r'$\Omega_{i-1}$', 
                       r'$\Omega_{i}$', r'$\Omega_{i+1}$'], va='center' )
    fig.tight_layout()
    # plt.show()
    fig.savefig('ch11_fv_diffusion.pdf')
    plt.close(fig)

def lsc_advection_dye():
    """Draws dye particle for advection at t=0 and t>0"""
    fig, ax = plt.subplots(figsize=(4,3))
    ax.fill([0, 1, 1, 0], [0.4, 0.4, 0.5, 0.5], color='#dbdbdb')
    ax.fill([0, 1, 1, 0], [0.1, 0.1, 0.2, 0.2], color='#dbdbdb')
    ax.hlines([0.1, 0.2, 0.4, 0.5], 0, 1)
    ax.plot(0.1, 0.45, 'o', 0.9, 0.15, 'o', color='black')
    ax.vlines([0.1, 0.9], [0, 0], [0.45, 0.15], linestyle='--', lw=0.7)
    ax.annotate('', xy=(0.1, 0.0), xytext=(0.9, 0.0), arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color='black'))
    ax.text(0.5, 0.01, r'$d=\alpha t$', ha='center')
    ax.text(1.05, 0.45, r'$t=0$', va='center')
    ax.text(1.05, 0.15, r'$t>0$', va='center')
    ax.set_xlim([0, 1.15])
    fig, ax = remove_axes(fig, ax)
    fig.savefig('ch11_lsc_advection_dye.pdf')

def lsc_characteristics():
    """Draws characteristic lines of the scalar advection equation"""
    fig, ax = plt.subplots(figsize=(4, 3))
    tp = np.linspace(0, 1, 100)
    alpha = 0.5
    nc = 7

    # Draw characteristic lines and x0
    x0s = np.linspace(0, 1, nc)
    for x0 in x0s:
        xp = x0 + alpha*tp
        ax.plot(xp, tp, 'k', lw=1)
        ax.plot(x0, 0 , 'ok', markersize=2)
    
    # Draw slope marker
    s1 = 40
    s2 = 55
    ax.plot(xp[[s1, s2, s2]], tp[[s1, s1, s2]], 'k', zorder=-10, lw=1)
    xc = xp[s2] + 0.05
    yc = np.mean(tp[[s1, s2]])
    ax.annotate(r'$\alpha > 0$', xy=(xc, yc), va='center')

    fig, ax = convert_to_graph(fig, ax)   
    ax.set_ylabel('$t$')
    ax.set_xlabel('$x$')

    fig.savefig('ch11_lsc_characteristics.pdf')

    
def lsys_characteristics():
    """Draws characteristic lines of linear hyperbolic systems"""
    fig, ax = plt.subplots(figsize=(4, 3))
    nc = 6

    # Draw characteristic lines and x0
    alphas = np.linspace(-1, 1, nc)
    for i, alpha in enumerate(alphas):
        x = np.sin(alpha)
        y = np.cos(alpha)
        ax.plot([0, x], [0, y], 'k', lw=1)

        # Annotate lines
        if i < 2:
            text = f'$\lambda_{i+1}$'
        elif i == nc-2:
            text = f'$\lambda_{{m-1}}$'
        elif i == nc-1:
            text = f'$\lambda_{{m}}$'
        else:
            text = '..'
        ax.text(x*1.04, y*1.04, text, ha='center')

    ax.plot([-1, 1], [0, 0], 'k', lw=0.75)
    fig, ax = remove_axes(fig, ax)   
    # plt.show()
    fig.savefig('ch11_lsys_characteristics.pdf')

def riem_lsc_u0():
    """Draws the initial condition for the scalar Riemann problem"""
    fig, ax = plt.subplots(figsize=(4, 3))

    y = np.array([0.25, 0.75])
    start = np.array([0, 0.5])
    end = np.array([0.5, 1])
    ax.hlines(y, start, end)
    ax.axvline(x=0.5, ymin=0, ymax=0.75, color='k', lw=0.7, linestyle='--',)

    ax.text(np.mean(start), y[0]+0.025, '$u_L$', ha='center')
    ax.text(np.mean(end), y[1]+0.025, '$u_R$', ha='center')
    
    ax.set_ylim([0, 1])
    ax.set_xlabel('$x=0$')
    ax.set_ylabel('$u$')
    fig, ax = convert_to_graph(fig, ax)
    fig.savefig('ch11_riem_lsc_u0.pdf')
    plt.show()

def riem_lsc_characteristics():
    """Draws characteristic line at x=0 for the linear scalar Riemann problem"""
    fig, ax = plt.subplots(figsize=(4, 3))
    tp = np.linspace(0, 1, 100)
    alpha = 3*np.pi/16
    nc = 7

    # Draw characteristic lines and x0
    xp = np.cos(alpha)
    tp = np.sin(alpha)
    ax.plot([0, xp], [0, tp], 'k', lw=1)
    
    # Annotate domain
    ax.text(xp/4, 2*tp/3, '$u_L$')
    ax.text(2*xp/3, tp/4, '$u_R$')
    ax.text(xp/2, tp/2, '$u(x_0=0)$', ha='center', va='bottom',
            rotation=alpha*180/np.pi, rotation_mode='anchor')
    ax.set_ylim(0, 0.5)
    fig, ax = convert_to_graph(fig, ax)   
    ax.set_aspect('equal')
    ax.set_ylabel('$t$')
    ax.set_xlabel('$x$')
    fig.savefig('ch11_riem_lsc_characteristics.pdf')

def riem_lsys_characteristics():
    """Draws characteristic lines of the linear system Riemann problem"""
    fig, ax = plt.subplots(figsize=(4, 3))
    nc = 6

    # Draw characteristic lines and x0
    alphas = [-1, -0.55, 0.55, 1]
    for i, alpha in enumerate(alphas):
        x = np.sin(alpha)
        y = np.cos(alpha)
        
        # Annotate lines
        text = f'$\lambda_{i+1}$'

        ax.plot([0, x], [0, y], 'k', lw=1)
        ax.text(x*1.04, y*1.04, text, ha='center')
    
    ax.fill([0, 2*np.sin(-1), 0, 2*np.sin(-0.55), 0], 
            [2, 2*np.cos(-1), 0, 2*np.cos(-0.55), 2], color='#e8e8e8')

    ax.fill([0, 2*np.sin(-0.55), 0, 2*np.sin(0.55), 0], 
        [2, 2*np.cos(-0.55), 0, 2*np.cos(0.55), 2], color=gray)

    ax.fill([0, 2*np.sin(0.55), 0, 2*np.sin(1), 0], 
        [2, 2*np.cos(0.55), 0, 2*np.cos(1), 2], color='#c4c4c4')

    ax.text(0.2, 0.708, r'$P_m$', ha='center', va='bottom')
    ax.text(0.6, 0.55, r'$P_{m,L}^*$', ha='center', va='bottom')
    ax.text(-0.6, 0.55, r'$P_{m,R}^*$', ha='center', va='bottom')
    ax.plot(0.2, 0.7, '*k', )
    ticks = []
    for alpha in alphas:
        x = 0.2 + 0.7*np.tan(alpha)
        ax.plot([0.2, x], [0.7, 0], '--k', lw=0.7)
        ticks.append(x)
    ax.text(-0.65, 0.15, r'$\vec u_L$', ha='center')
    ax.text(0.65, 0.15, r'$\vec u_R$', ha='center')
    ax.plot([-1, 1.35], [0, 0], 'k', lw=0.75)
    fig, ax = remove_axes(fig, ax)   
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([-1, 1.4])
    plt.xticks(ticks, [r'$x-\lambda_4 t$', r'$x-\lambda_3 t$', r'$x-\lambda_2 t$', r'$x-\lambda_1 t$'])
    ax.tick_params(axis=u'both', which=u'both',length=0)
    fig.savefig('ch11_riem_lsys_characteristics.pdf')

def riem_linacoustics_characteristics():
    """Draws sound wave characteristic curves for linear acoustics example"""
    fig, ax = plt.subplots(figsize=(4,3))
    c = -1
    ax.plot([0, np.sin(c)], [0, np.cos(c)], 'k', lw=1)
    ax.plot([0, np.sin(-c)], [0, np.cos(-c)], 'k', lw=1)
    ax.text(np.sin(c), np.cos(c)+0.01, r'$-c$', ha='center')
    ax.text(np.sin(-c), np.cos(c)+0.01, r'$c$', ha='center')
    ax.plot([-1, 1], [0, 0], 'k', lw=1)
    ax.text(0, -0.01, '$x=0$', va='top', ha='center')
    ax.text(-0.5, 0.15, r'$\rho_L$', va='bottom', ha='center')
    ax.text(-0.5, 0.1, r'$v_L$', va='bottom', ha='center')
    ax.text(0.5, 0.15, r'$\rho_R$', va='bottom', ha='center')
    ax.text(0.5, 0.1, r'$v_R$', va='bottom', ha='center')
    ax.text(0, 0.25, r'$\rho_m$', va='bottom', ha='center')
    ax.text(0, 0.2, r'$v_m$', va='bottom', ha='center')
    ax.fill([0, 2*np.sin(-1), 0, 2*np.sin(1), 0], 
            [2, 2*np.cos(-1), 0, 2*np.cos(1), 2], color=gray)
    fig, ax = remove_axes(fig, ax)
    ax.set_ylim([-0.01, 0.6])
    ax.set_xlim([-1, 1])
    fig.savefig('ch11_riem_linacoustics_characteristics.pdf')

def riem_linacoustics_u0():
    """Draws the initial condition for the linear acoustics example (Riemann)"""
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot([0, 1], [0, 0], blue,)
    ax.plot([0, 0.5], [0.5, 0.5], red,)
    ax.plot([0.5, 1.0], [0.25, 0.25], red,)
    ax.plot([0.5, 0.5], [0, 0.5], '--k', lw=0.7)

    ax.text(0.25, 0.508, r'$\rho_L$', ha='center')
    ax.text(0.75, 0.258, r'$\rho_R$', ha='center')
    ax.text(0.25, 0.008, r'$v_L$', ha='center')
    ax.text(0.75, 0.008, r'$v_R$', ha='center')

    ax.set_xlim([0, 1.01])
    ax.set_ylim(0, 0.65)

    fig, ax = convert_to_graph(fig, ax)
    ax.set_ylim(-0.01, 0.65)

    ax.set_ylabel(r'$\vec u$')
    plt.xticks([0.5], ['$x=0$'])
    ax.tick_params(axis=u'both', which=u'both',length=0)

    fig.savefig('ch11_riem_linacoustics_u0.pdf')

def riem_linacoustics_sol():
    """Draws solution to the Riemann problem example (linear acoustics)"""
    fig, ax = plt.subplots(figsize=(4, 3))

    #u_L
    ax.plot([0, 0.25], [0, 0], blue,)
    #u_R
    ax.plot([0.75, 1.0], [0, 0], blue,)
    #rho_L
    ax.plot([0, 0.25], [0.5, 0.5], red,)
    #rho_R
    ax.plot([0.75, 1.0], [0.25, 0.25], red,)

    #u_m
    ax.plot([0.25, 0.75], [0.125, 0.125], blue)
    #rho_m
    ax.plot([0.25, 0.75], [0.375, 0.375], red)
    # Discontinuity lines
    ax.plot([0.25, 0.25], [0, 0.5], '--k', lw=0.7)
    ax.plot([0.75, 0.75], [0, 0.375], '--k', lw=0.7)

    ax.text(0.125, 0.508, r'$\rho_L$', ha='center')
    ax.text(0.875, 0.258, r'$\rho_R$', ha='center')
    ax.text(0.125, 0.008, r'$v_L$', ha='center')
    ax.text(0.875, 0.008, r'$v_R$', ha='center')
    ax.text(0.5, 0.130, r'$v_m$', ha='center')
    ax.text(0.5, 0.380, r'$\rho_m$', ha='center')
    ax.set_xlim([0, 1.01])
    ax.set_ylim(0, 0.65)
    fig, ax = convert_to_graph(fig, ax)
    ax.set_ylim(-0.01, 0.65)
    plt.xticks([0.25, 0.75], ['$-ct$', '$ct$'])
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.set_ylabel(r'$\vec u$')
    fig.savefig('ch11_riem_linacoustics_sol.pdf')

def reynolds_transport():
    """Draws abitrary shape for the Reynolds transport theorem"""

    # Define control points
    pts = np.array([[0, 0], 
                    [0.25, 0.45], 
                    [0.5, 0.48], 
                    [0.75, 0.4], 
                    [1, 0], 
                    [0.75, -0.35], 
                    [0.5, -0.5], 
                    [0.38, -0.55], 
                    [0, 0]])

    pts = pts*2
    fig, ax = plt.subplots()
    pts2 = np.copy(pts)
    pts2[:, 0] = pts2[:, 0] - 0.15
    pts2[:, 1] = pts2[:, 1] - 0.2*np.cos(pts2[:, 1])*np.sin(pts2[:, 1])-0.15
    pts[[0,-1], 0] += 0.03 

    # Draw closed shape at t=0
    tck, u2 = spi.splprep(pts2.T, u=None, s=0.0, per=1)
    u2 = np.linspace(u2.min(), u2.max(), 100)
    xs2, ys2 = spi.splev(u2, tck, der=0)
    ax.plot(xs2, ys2, 'k', lw=1.5, zorder=10)

    # # Get its derivatives
    xd, yd = spi.splev(u2, tck, der=1)

    # Get normal vector at point p
    p = 1
    v = -np.array([xd[p], yd[p]])
    nhat = _rotate_vector(v, np.pi/2)

    # Plot flux and normal vectors
    ax.quiver(xs2[p], ys2[p], nhat[0], nhat[1], width=0.015, units='xy', 
              scale=50, scale_units='xy')

    # Draw closed shape at t = t + dt
    tck, u = spi.splprep(pts.T, u=None, s=0.0, per=1)
    u = np.linspace(u.min(), u.max(), 100)
    xs, ys = spi.splev(u, tck, der=0)
    ax.plot(xs, ys, 'k--', lw=0.7, color=darkgray)

    for i in range(len(xs)):
        if i % 3 == 0:
            ax.arrow(xs2[i], ys2[i], xs[i]-xs2[i], ys[i]-ys2[i], 
                    fc=darkgray, lw=0.5, head_width=0.03, head_length=0.04, 
                    length_includes_head=True, color=darkgray)

    # Annotate graph
    ax.text(pts2[3,0], pts2[3,1]*0.99, '$S$', ha='right', va='top')
    ax.text(np.mean(pts2[:,0]), np.mean(pts2[:,1]), 
            '$\Omega$', ha='center', va='center')
    ax.text(1.2*np.min(pts2[:,0]), np.min(pts2[:,1]), 
            '$t=0$', ha='left', va='center')
    ax.text(0.9*np.max(pts[:,0]), 0.9*np.max(pts[:,1]), 
            '$t+dt$', ha='center', va='center')
    ax.arrow(0.8*np.mean(pts2[:,0]), np.mean(pts2[:,1]*0.8), 0.25, 0.25, 
             fc='k', lw=1, head_width=0.03, head_length=0.04, 
             length_includes_head=True, color='k')
    ax.text(0.8*np.mean(pts2[:,0]), np.mean(pts2[:,1]*0.8), r'$\vec V(x,t)$', 
            va='bottom',rotation=45, rotation_mode='anchor')
    ax.text(xs2[p]-0.085, ys2[p]+0.05, r'$\hat n$')
            
    # Remove the axes   
    ax.margins(0.2)
    fig, ax = remove_axes(fig, ax)
    fig.savefig('ch4_reytrans_volume.pdf')

def exact_solutions():
    """Draws the exact solution for the advection, diffusion and burgers eqs"""
    x = np.linspace(0, 1, 150)
    y = np.exp(-(x-1/2)**2/(0.1**2))
    fig, ax = plt.subplots()
    ax.plot(x, y, 'k')
    ax.plot(x, np.roll(y, 30), 'k--', lw=1)
    plt.show()

def _draw_eddie(c, r, ai=np.pi/2, af=2*np.pi, deg=False, sc=150, cf=7.0, color='k'):
    """Draws arc with center c, radius r from angle ai to angle af in rad"""

    if deg:
        if af > 360:
            af += -360
            ai += -360
        ai = ai*np.pi/180
        af = af*np.pi/180

    x = [c[0] + np.cos(angle)*r for angle in np.linspace(ai, af, 60)]
    y = [c[1] + np.sin(angle)*r for angle in np.linspace(ai, af, 60)]

    # Arrow head correction due to FancyArrowPatch issue
    # Determine spacing in both directions
    dy, dx = y[-1]-y[-2], x[-1]-x[-2]

    # Compute slope angle and find distance
    m = np.arctan2(dy, dx)
    d = np.sqrt(dx**2 + dy**2)

    # Extend length of path so that arrow head seems to start at end
    x.append(x[-1] + cf*abs(d)*np.cos(m))
    y.append(y[-1] + cf*abs(d)*np.sin(m))

    # Define path, then create arrow from path
    path = mpath.Path(np.array([x, y], dtype='float').T)
    patch = mpatch.FancyArrowPatch(path=path, arrowstyle='-|>',
                                   fc=color, mutation_scale=sc*r, color=color)
    return patch

def energy_cascade():
    """Draws the turbulent energy cascade"""

    fig, ax = plt.subplots(figsize=(5, 4))
    scale_x = 2.5
    shift_x = 0.05
    shift_y = 0.075

    # Enegy cascade curve
    x = np.array([0, 0.04, 0.15, 0.6, 0.62, 0.66, 0.68])
    eq = 1.416-x[3]*5/3
    y = np.array([0.85, 0.95, 1.0, eq, eq-0.025, eq-0.19, 0])
    x += shift_x

    # Production region
    poly = np.polyfit(x[:3], y[:3], 2)
    xp = np.linspace(x[0], x[2], 20)
    yp = np.polyval(poly, xp)
    ax.plot(xp*scale_x, yp+shift_y, 'k')

    # -5/3 slope
    poly = np.polyfit(x[2:4], y[2:4], 1)
    xp = np.linspace(x[2], x[3], 20)
    yp = np.polyval(poly, xp)
    ax.plot(xp*scale_x, yp+shift_y, 'k')

    # Dissipation region
    poly = np.polyfit(x[-4:], y[-4:], 3)
    xp = np.linspace(x[-4], x[-1], 20)
    yp = np.polyval(poly, xp)
    ax.plot(xp*scale_x, yp+shift_y, 'k')

    # Draw eddies
    rmax = 0.075
    rmed = 0.05
    rmin = 0.02
    r = [rmax, rmax, rmax, rmed, rmed, rmed, rmin, rmin, rmin, rmin, rmin]

    c = np.array([[0.25, 1.2],
                 [0.45, 1.2],
                 [0.35, 1.35],
                 [0.85, 0.9],
                 [1.00, 0.9],
                 [0.925, 1.0],
                 [1.55, 0.58],
                 [1.60, 0.59],
                 [1.65, 0.57],
                 [1.57, 0.65],
                 [1.62, 0.66]])

    c[:, 0] += shift_x*3
    c[:, 1] += shift_y

    # Define start and end angles of each eddie
    ai = [np.pi/2, -np.pi/2, 0, -np.pi/2, 0, np.pi/2, 
          np.pi/2, -np.pi/2, 0, np.pi/2, -np.pi/2, 0]
    af = [2*np.pi, np.pi, 3*np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, 
          2*np.pi, np.pi, 3*np.pi/2, 2*np.pi, np.pi, 3*np.pi/2]
    
    for i, (ci, ri, aii, afi) in enumerate(zip(c, r, ai, af)):
        patch = _draw_eddie(ci, ri, ai=aii, af=afi)
        ax.add_patch(patch)

    # Annotate graph
    ax.text(0.55+shift_x*3, 1.2+shift_y, r'$l$')
    ax.text(1.7+shift_x*3, 0.60+shift_y, r'$\eta$')
    ax.margins(x=0.2)
    
    ax.text(1.7+shift_x*3, 1.2, 'Viscous\ndissipation', ha='center', va='bottom')
    ax.arrow(1.7+shift_x*3, 1.15, 0, -0.3,  lw=0.7, head_width=0.03, head_length=0.04, fc='k' )
    ax.set_ylim(0, 1.65)
    ax.set_xlim(0, max(xp)*scale_x+0.1)
    fig, ax = convert_to_graph(fig, ax)
    
    
    ax.set_ylabel(r'$\log E(\kappa)$')
    ax.set_xlabel(r'$\log \kappa$')
    ax.set_aspect('equal')
    fig.savefig('ch7_energy_cascade.pdf')
    plt.show()

def law_of_the_wall():
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Viscous layer relation
    y_subv = np.linspace(1, 15, 40)
    u_subv = y_subv
    ax.plot(np.log(y_subv), u_subv, 'k--', lw=1)

    # Log law relation
    y_log = np.linspace(6, 1000, 40)
    u_log = 1/0.41 * np.log(y_log) + 5.2
    ax.plot(np.log(y_log), u_log, 'k--', lw=1)

    # Use Spalding's relation
    u = np.linspace(1, 25, 100)
    y = u + 0.1108*(np.exp(0.4*u) - 1 - 0.4*u - \
        (0.4*u)**2/2 - (0.4*u)**3/6 - (0.4*u)**4/24)
    idx = y <= 1001
    ax.plot(np.log(y[idx]), u[idx], 'k')
    ax.vlines(np.log([5, 30]), 0, 28, lw=1, linestyle=':')

    # Viscous sub-layer annotations
    x, y = np.log(np.sqrt(5)), 25
    ax.text(x, y, 'Viscous\nsub-layer', ha='center', va='center')
    ax.annotate(r'$u^+=y^+$', xytext=(x, 7), xy=(np.log(4), 4), ha='center',
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color='black'))

    # Buffer layer
    ax.text(np.log(np.sqrt(5*30)), 25, 'Buffer\nlayer', 
            ha='center', va='center')

    # Outer layer
    x, y = np.log(np.sqrt(30*1000)), 25
    ax.text(x, y, 'Log-law\nregion', ha='center', va='center')
    ax.annotate(r'$\frac{1}{\kappa}\ln y^+ + B$', 
                xytext=(x, 7), xy=(np.log(100), 16.5),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color='black'))

    ax.set_ylim(0, 30)
    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel(r'$u^+$')
    fig, ax = convert_to_graph(fig, ax)
    fig.savefig('ch7_law_of_the_wall.pdf')
    # plt.show()

def dispersion_dissipation_diagram():
    x = np.linspace(0, 1, 100)

    # Dissipation
    fig, ax = plt.subplots(figsize=(5, 3))
    y = np.sin(2*np.pi*x)
    ax.plot(x, y, 'k', lw=1)
    ax.plot(x, 0.75*y, 'k--', lw=1, color=red)
    # ax.set_title('Dissipation')
    ax.set_ylabel('$u$')
    ax.text(1.02, 0, '$x$', va='center')
    fig, ax = convert_to_graph(fig, ax)
    ax.set_xlim(0, 1.1)
    fig.savefig('ch13_dissipation_diagram.pdf')
    # Dispersion
    fig, ax = plt.subplots(figsize=(5, 3))
    y = np.sin(2*np.pi*x)
    ax.plot(x, y, 'k', lw=1)
    ax.plot(x+0.1,y, 'k--', lw=1, color=red)
    # ax.set_title('Dispersion')
    ax.set_ylabel('$u$')
    ax.text(1.12, 0, '$x$', va='center')
    fig, ax = convert_to_graph(fig, ax)
    ax.set_xlim(0, 1.1)
    fig.savefig('ch13_dispersion_diagram.pdf')

    plt.show()

def spectral_dissipation_adv_explct():
    sigmas = np.linspace(0.5, 1, 6)
    kmdx = np.linspace(0, np.pi, 100)
    fig, ax = plt.subplots()
    colors = _get_colors(len(sigmas))
    for sig, color in zip(sigmas, colors):
        amp = np.abs(1-sig+sig*np.exp(-1j*kmdx))
        ax.plot(kmdx, amp, color=color, label=f'$\sigma={sig:.1f}$')
    ax.legend()
    ax.set_xlabel(r'$\kappa_m \Delta x$')
    ax.set_ylabel(r'$|e^{a\Delta t}|$')
    plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], 
               ['0', r'$\frac{\pi}{4}$',  r'$\frac{\pi}{2}$',  r'$\frac{3\pi}{4}$', r'$\pi$'])
    ax.tick_params(axis='x', which='major', pad=10)
    fig.tight_layout()
    fig.savefig('ch13_dissipation_adv_explct.pdf')
    # plt.show()

def spectral_dissipation_adv_implct():
    sigmas = np.linspace(0, 10, 6)
    kmdx = np.linspace(0, np.pi, 100)
    fig, ax = plt.subplots()
    colors = _get_colors(len(sigmas))
    for sig, color in zip(sigmas, colors):
        amp = np.abs(1/(1+sig*(1-np.exp(-1j*kmdx))))
        ax.plot(kmdx, amp, color=color, label=f'$\sigma={sig:.0f}$')
    ax.legend()
    ax.set_xlabel(r'$\kappa_m \Delta x$')
    ax.set_ylabel(r'$|e^{a\Delta t}|$')
    plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], 
               ['0', r'$\frac{\pi}{4}$',  r'$\frac{\pi}{2}$',  r'$\frac{3\pi}{4}$', r'$\pi$'])
    ax.tick_params(axis='x', which='major', pad=10)
    fig.tight_layout()
    fig.savefig('ch13_dissipation_adv_implct.pdf')
    # plt.show()

def spectral_dispersion_adv_explct():
    sigmas = np.linspace(0, 1, 6)
    kmdx = np.linspace(0, np.pi, 100)
    fig, ax = plt.subplots()
    colors = _get_colors(len(sigmas))
    for sig, color in zip(sigmas, colors):
        y = np.imag(np.log(1-sig+sig*np.exp(-1j*kmdx)))
        ax.plot(kmdx, y, color=color, label=f'$\sigma={sig:.2f}$')
    ax.legend()
    ax.set_xlabel(r'$\kappa_m \Delta x$')
    ax.set_ylabel(r'$\Im\left(|e^{a\Delta t}|\right)$')
    plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], 
               ['0', r'$\frac{\pi}{4}$',  r'$\frac{\pi}{2}$',  r'$\frac{3\pi}{4}$', r'$\pi$'])
    ax.tick_params(axis='x', which='major', pad=10)
    fig.tight_layout()
    fig.savefig('ch13_dispersion_adv_explct.pdf')
    # plt.show()

def taylor_series():
    x = sp.Symbol('x')
    x0 = 0
    f = sp.sin(x)
    fig, ax = plt.subplots()
    xp = np.linspace(-np.pi, np.pi, 100)
    ax.plot(xp, np.sin(xp), '--k', label='$sin(x)$')
    n = 10
    fx = 0
    colors = _get_colors(n, end=0.9)
    for i in range(n):
        n_term = sp.diff(f, x, i).subs(x, x0)*(x-x0)**i/(math.factorial(i))
        if n_term != 0 or i == 0:
            fx += n_term
            y = [fx.subs('x', xpp) for xpp in xp]
            ax.plot(xp, y, label=f'$f_{{{i}}}(x)$', color=colors[i], lw=0.9)

    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
               [r'$-\pi$', r'$-\frac{\pi}{2}$', '$0$',
                r'$\frac{\pi}{2}$', r'$\pi$'])
    ax.tick_params(axis='x', which='major', pad=10)
    ax.vlines(x0, -1.2, 1.2, lw=0.5, linestyle='--')
    ax.set_ylim(-1.2, 1.2)
    ax.legend()
    fig.savefig('ch9_taylor_series_sin.pdf')
    # plt.show()

def heated_plate_flow():
    fig1, ax1 = plt.subplots(figsize=(5.8, 2.8))
    fig2, ax2 = plt.subplots(figsize=(5.4, 2.8))
    mpl.rcParams['hatch.linewidth'] = 0.5

    for j, (fig, ax) in enumerate(zip([fig1, fig2], [ax1, ax2])):
        ax.plot([0, 1], [0, 0], 'k', lw=1)
        ax.fill([0, 1, 1, 0], [0, 0, -0.03, -0.03], 'w', hatch='/////', lw=0.5)

        x = [0, 0.3, 1.5]
        y = [0, 0.3, 0.5]

        poly = np.polyfit(x, y, 2)
        
        x = np.linspace(0, 1, 100)
        y = 0.5*np.polyval(poly, x)
        ax.plot(x, y, color=blue)

        ax.set_aspect('equal')
        ax.text(0.35, 0.22, 'cold')
        ax.text(0.95, 0.31, r'$\delta$', va='bottom', ha='center')
        ax.text(0.5, -0.1, 'hot', ha='center', va='bottom')
        ax.set_ylim(-0.2, 0.4)

        fig, ax = remove_axes(fig, ax)

        if j == 1:
            # Draw velocity profile
            p = int(len(x)/2)
            ax.vlines([x[p]], [0], [y[p]], darkgray)
            yp = np.linspace(0, y[p], 50)
            xp = np.sqrt(yp/8) + x[p]
            ax.plot(xp, yp, darkgray)

            for i in range(len(xp)):
                if (i+1) % 5 == 0:
                    ax.arrow(x[p], yp[i], xp[i]-x[p], 0, 
                            fc=darkgray, lw=0.5, head_width=0.01, head_length=0.02, 
                            length_includes_head=True, color=darkgray)
            ax.text(0.75, 0.11, 'Turbulent\ndiffusion', ha='left', va='center')
            ax.arrow(0.72, 0.03, 0, 0.19, fc='k', lw=0.8, head_width=0.025, 
                     head_length=0.04, length_includes_head=True, color='k')
        else:
            c = [[0.25, 0], 
                 [0.3, 0.1125], 
                 [0.42, 0.065],
                 [0.54, 0.14],
                 [0.622, 0.125],
                 [0.75, 0.16],
                 [0.85, 0.16]]
            r = [0.1, 0.1, 0.105, 0.105, 0.11, 0.11, 0.115]
            ai = [170+5*i if i % 2 == 0 else 235-5*i for i in range(7)] 
            af = [75-7*i if i % 2 == 0 else 350+5*i for i in range(7)]
            for ci, ri, aii, afii in zip(c, r, ai, af):
                patch = _draw_eddie(ci, ri, aii, afii, True, 120, color=darkgray)
                ax.add_patch(patch)

            ax.arrow(1.0, 0.278, 0, -0.08, fc='k', lw=0.8, head_width=0.02, 
                     head_length=0.035, length_includes_head=True, color='k')
            ax.arrow(0.9, 0.07, 0, 0.08, fc='k', lw=0.8, head_width=0.02, 
                     head_length=0.035, length_includes_head=True, color='k')

            ax.text(1.02, 0.238, 'cold down', ha='left', va='center')
            ax.text(0.92, 0.11, 'hot up', ha='left', va='center')
            ax.set_xlim(0, 1.2)

        fig.tight_layout()
        fig.savefig(f'ch7_heated_plate_flow_{j}.pdf')
    # plt.show()

def _time_stepping_curve(coords=False):
    x = [0, 0.1, 0.25, 0.5, 0.75, 1]
    y = [0.4, 0.3, 0.29, 0.5, 0.7, 0.6]
    poly = np.polyfit(x, y, 5)
    dpoly = np.polyder(poly)
    if coords:
        return poly, dpoly, x, y
    return poly, dpoly
    
def explicit_euler_method():
    # Get time-stepping curve and plot it
    poly, dpoly = _time_stepping_curve()

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.linspace(0, 1, 100)
    y = np.polyval(poly, x)
    ax.plot(x, y, darkgray)

    # Locate ut and ut+dt (utt)
    xt, ut = 0.25, np.polyval(poly, 0.25)
    xtt, utt = 0.75, np.polyval(poly, 0.75)
    dt = xtt - xt
    
    # Draw vertical lines
    ax.vlines([xt, xtt], 0, [ut, utt], lw=0.8, linestyle='--', color=darkgray)
    
    # Find slope angle at ut
    dut = np.polyval(dpoly, xt)
    theta_t = np.arctan(dut)

    # Draw connecting line to Euler solution
    xt_xtt = [xt, xtt]
    ut_utt = [ut, ut+dt*np.tan(theta_t)]
    ax.plot(xt_xtt, ut_utt, '--', color=red, lw=1.0)
    ax.plot(xt_xtt[1], ut_utt[1], 'o', color=red, markersize=4)

    # Draw slope and ut marker
    ds = 0.1
    dx, dy = ds/2*np.cos(theta_t), ds/2*np.sin(theta_t)
    xp = [xt-dx, xt+dx]
    yp = [ut-dy, ut+dy]
    ax.plot(xp, yp, red, lw=2.5)
    ax.plot(xt, ut, 'ok', markersize=3)

    ax.set_ylim(0, 0.9)
    ax.text(xt, ut*1.1, '$u^t$', ha='center', va='bottom')
    ax.text(xtt*1.03, ut_utt[1], '$u^{t+1}$', ha='left', va='center')
    ax.annotate(r'$\approx\frac{\partial u}{\partial t}\Big\vert_{t}$', 
                xy=(xt*1.1, ut*0.98), xytext=(xt*1.1, ut*0.5), va='top', 
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color='black'))
    fig, ax = convert_to_graph(fig, ax)
    plt.xticks([xt, xtt], ['$t$', r'$t+1$'])
    ax.set_ylabel('$u$')
    fig.savefig('ch15_explicit_euler.pdf')
    plt.show()

def heuns_method():
    # Get time-stepping curve and plot it
    poly, dpoly, xc, yc = _time_stepping_curve(True)

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.linspace(0, 1, 100)
    y = np.polyval(poly, x)
    ax.plot(x, y, darkgray)

    # Locate ut and ut+dt (utt)
    xt, ut = 0.25, np.polyval(poly, 0.25)
    xtt, utt = 0.75, np.polyval(poly, 0.75)
    dt = xtt - xt
    
    # Draw vertical lines
    ax.vlines([xt, xtt], 0, [ut, utt], lw=0.8, linestyle='--', color=darkgray)
    
    # Find slope angle at ut
    dut = np.polyval(dpoly, xt)
    theta_t = np.arctan(dut)

    # Draw connecting line to predicted solution
    xt_xtt = [xt, xtt]
    ut_utt = [ut, ut+dt*np.tan(theta_t)]
    ax.plot(xt_xtt, ut_utt, '--', color=darkgray, lw=0.8)
    ax.plot(xt_xtt[1], ut_utt[1], 'ok', markersize=3, zorder=3)

    # Draw slope and ut marker
    ds = 0.1
    dx, dy = ds/2*np.cos(theta_t), ds/2*np.sin(theta_t)
    xp = [xt-dx, xt+dx]
    yp = [ut-dy, ut+dy]
    ax.plot(xp, yp, darkgray, lw=2.5)
    ax.plot(xt, ut, 'ok', markersize=4, zorder=3)

    # Find slope at predicted solution
    theta_t2 = theta_t*2
    # Draw slope at predicted solution
    dx2, dy2 = ds/2*np.cos(theta_t2), ds/2*np.sin(theta_t2)
    ax.plot([xtt-dx2, xtt+dx2], [ut_utt[1]-dy2, ut_utt[1]+dy2], darkgray, lw=2.5, zorder=2)
    utt_tilde = ut_utt[1]

    # Draw new slope at ut
    theta_heun = (theta_t + theta_t2)/2
    ds = 0.1
    dx, dy = ds/2*np.cos(theta_heun), ds/2*np.sin(theta_heun)
    xp = [xt-dx, xt+dx]
    yp = [ut-dy, ut+dy]
    
    ax.plot(xp, yp, red, lw=2.5, zorder=2)

    # Draw connecting like to Heuns solution
    utt_heun = ut+dt*np.tan(theta_heun)
    ut_utt_heun = [ut, utt_heun ]
    ax.plot(xt_xtt, ut_utt_heun, '--', color=red, lw=1.0)
    ax.plot(xt_xtt[1], utt_heun, 'o', color=red, markersize=4)

    ax.text(xt, ut*1.1, '$u^t$', ha='center', va='bottom')
    ax.text(xtt*1.03, utt_heun, r'$u^{t+1}$', ha='left', va='center')
    ax.annotate(r'$\approx\frac{\partial u}{\partial t}\Big\vert_{t}$', 
                xy=(xt*1.2, ut*1.09), xytext=(xt*1.1, ut*0.5), va='top', 
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color='black'))
    ax.annotate(r'$\approx\frac{\partial \tilde u}{\partial t}\Big\vert_{t+1}$', 
                xy=(xtt*1.03, utt_tilde*1.02), xytext=(xtt*1.1, utt_tilde*0.75), va='top', 
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color='black'))
    ax.annotate(r'$\frac{R(u) + R(\tilde u)}{2}$', 
                xy=(xt*0.85, ut*0.9), xytext=(xt*0.7, ut*0.5), va='top', ha='right', 
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color='black'))

    ax.set_ylim(0, 0.9)
    fig, ax = convert_to_graph(fig, ax)
    plt.xticks([xt, xtt], ['$t$', r'$t+1$'])
    ax.set_ylabel('$u$')
    fig.savefig('ch15_heuns_method.pdf')
    plt.show()

def midpoint_method():
    # Get time-stepping curve and plot it
    poly, dpoly, xc, yc = _time_stepping_curve(True)

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.linspace(0, 1, 100)
    y = np.polyval(poly, x)
    ax.plot(x, y, darkgray)

    # Locate ut and ut+dt/2 and ut+dt (utt) in the original polynomial
    xt, ut = 0.25, np.polyval(poly, 0.25)
    xtt, utt = 0.75, np.polyval(poly, 0.75)
    xth = (xt+xtt)/2 
    uth = np.polyval(poly, ut)
    dt = xtt - xt
    
    # Find slope angle at ut
    dut = np.polyval(dpoly, xt)
    theta_t = np.arctan(dut)

    # Draw connecting line to solution at t+1/2
    uth_mid = ut+dt/2*np.polyval(dpoly, xt)

    ax.plot([xt, xth], [ut, uth_mid], '--', color=darkgray, lw=0.8, zorder=-1)
    ax.plot(xth, uth_mid, 'ok', markersize=3, zorder=3)

    # Draw slope and marker at ut
    ds = 0.1
    dx, dy = ds/2*np.cos(theta_t), ds/2*np.sin(theta_t)
    xp = [xt-dx, xt+dx]
    yp = [ut-dy, ut+dy]
    ax.plot(xp, yp, darkgray, lw=2.5)
    ax.plot(xt, ut, 'ok', markersize=3, zorder=3)

    # Get slope and draw at uth_mid
    fh = np.polyval(dpoly, xth)
    theta_mid = np.arctan(fh)
    dxm, dym = ds/2*np.cos(theta_mid), ds/2*np.sin(theta_mid)
    ax.plot([xth-dxm, xth+dxm], [uth_mid-dym, uth_mid+dym], red, lw=2.5, zorder=0)

    # Re-draw uth_mid slope at ut
    ax.plot([xt-dxm, xt+dxm], [ut-dym, ut+dym], red, lw=2.5)

    # Connect ut to final solution
    utt_mid = ut + dt*np.polyval(dpoly, xth)
    ax.plot([xt, xtt], [ut, utt_mid], '--', color=red, lw=1.0, zorder=-1)
    ax.plot(xtt, utt_mid, 'o', color=red, markersize=4, zorder=3)

    # Draw vertical lines
    ax.vlines([xt, xth, xtt], 0, [ut, uth_mid, utt_mid], 
              lw=0.8, linestyle='--', zorder=-1, color=darkgray)


    ax.text(xt, ut*1.1, '$u^t$', ha='center', va='bottom')
    ax.text(xtt*1.03, utt_mid, r'$u^{t+1}$', ha='left', va='center')
    ax.text(xth*1.05, uth_mid, r'$\tilde u$', ha='left', va='center')
    ax.annotate(r'$\approx\frac{\partial u}{\partial t}\Big\vert_{t}$', 
                xy=(xt*0.82, ut*0.95), xytext=(xt*0.6, ut*0.7), va='top', ha='right',
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color='black'))
    ax.annotate(r'$\approx\frac{\partial \tilde u}{\partial t}\Big\vert_{t+1}$', 
                xy=(xth*0.97, uth_mid*0.96), xytext=(xth*1.1, uth_mid*0.75), va='top', 
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color='black'))
    ax.annotate(r'', 
                xy=(xt*0.9, ut*0.9), xytext=(xth*1.1, uth_mid*0.75), va='top', 
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color='black'))

    ax.set_ylim(0, 0.9)
    fig, ax = convert_to_graph(fig, ax)
    plt.xticks([xt, xth, xtt], ['$t$', r'$t+1/2$', r'$t+1$'])
    ax.set_ylabel('$u$')
    fig.savefig('ch15_midpoint_method.pdf')
    plt.show()

def rk_methods():
    # Get time-stepping curve and plot it
    poly, dpoly, xc, yc = _time_stepping_curve(True)

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.linspace(0, 1, 100)
    y = np.polyval(poly, x)
    ax.plot(x, y, darkgray)

    # Locate ut and ut+dt/2 and ut+dt (utt) in the original polynomial
    xt, ut = 0.25, np.polyval(poly, 0.25)
    xtt, utt = 0.75, np.polyval(poly, 0.75)
    xth = (xt+xtt)/2 
    uth = np.polyval(poly, ut)
    dt = xtt - xt

    k1 = np.polyval(dpoly, xt)
    k2 = np.polyval(dpoly, xt+dt/2)
    k3 = np.polyval(dpoly, xt+dt/2)
    k4 = np.polyval(dpoly, xt+dt)

    angles = np.arctan([k1, k2, k3, k4])

    ut1 = ut 
    ut2 = ut + dt/2*k1
    ut3 = ut + dt/2*k2
    ut4 = ut + dt*k3

    utt_rk = ut + dt/6*(k1+2*k2+2*k3+k4)
    ax.plot(xtt, utt_rk, 'o', markersize=4, color=red)

    ds = 0.1
    xvals =  [xt, xt+dt/2, xt+dt/2, xtt]
    utvals = [ut1, ut2, ut3, ut4]
    for a, xp, utn in zip(angles, xvals, utvals):
        dx, dy = ds/2*np.cos(a), ds/2*np.sin(a)
        xs = [xp-dx, xp+dx]
        ys = [utn-dy, utn+dy]
        ax.plot(xs, ys, darkgray, lw=2.5)
        ax.plot(xp, utn, 'ok', markersize=3, zorder=3)
        ax.plot([xt, xp], [ut1, utn], '--', color='#737373', lw=0.8, zorder=-1)
    
    # Plot final slope
    a = np.arctan(1/6*sum([k1, 2*k2, 2*k3, k4]))
    dx, dy = ds/2*np.cos(a), ds/2*np.sin(a)
    xs = [xt-dx, xt+dx]
    ys = [ut-dy, ut+dy]
    ax.plot(xs, ys, lw=2.5, color=red)

    ax.text(xvals[0], utvals[0]+0.02, '$u_1$', ha='center', va='bottom')
    ax.text(xvals[1]+0.03, utvals[1], '$u_2$', ha='left', va='center')
    ax.text(xvals[2], utvals[2]+0.02, '$u_3$', ha='center', va='bottom')
    ax.text(xvals[3], utvals[3]+0.01, '$u_4$', ha='center', va='bottom')
    ax.text(xtt+0.02, utt_rk+0.03, '$u^{t+1}$', ha='left', va='center')
    del xvals[1]
    del utvals[1]

    ax.vlines(xvals, 0, utvals, linestyle='--', lw=0.8, color='#737373')
    ax.plot([xt, xtt], [ut, utt_rk], '--', color=red, lw=1.0)
    ax.annotate(r'$\sum{b_i R(u_i)}$', 
                xy=(xt*0.9, ut*0.9), xytext=(xt*1.1, ut*0.75), va='top', 
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color='black'))
    
    
    ax.set_ylim(0, 0.9)
    fig, ax = convert_to_graph(fig, ax)
    plt.xticks([xt, xth, xtt], ['$t$', r'$t+1/2$', r'$t+1$'])
    ax.set_ylabel('$u$')

    fig.savefig('ch15_rk_method.pdf')
    plt.show()
    plt.show()


def logistic_map():
    n = 10000
    a = np.linspace(1, 4.0, n)
    x = 1e-5 * np.ones(n)
    fig, ax = plt.subplots()
    for i in range(1000):
        x = a*x*(1-x)
        if i > 975:
            ax.plot(a, x, ',k', rasterized=True)

    ax.set_xlabel('$a$')
    ax.set_ylabel('$x$')
    fig.savefig('ch7_logistic_map.pdf')
    # plt.show()


def main():

    # Chapter: Conservation Laws
    # reynolds_transport()
    # exact_solutions()

    # Chapter: Turbulence
    # logistic_map()
    # energy_cascade()
    # law_of_the_wall()

    # Reynolds averaging
    # time_average()
    # ensemble_average()
    # heated_plate_flow()

    # FD, FV schemes
    # discrete_scheme()
    # fv_arbitrary_volume()
    # fv_trivolume()
    # fv_riemann_diagram()
    # fv_1d_advection()
    # fv_1d_diffusion()

    # Chapter: Taylor-Series
    # taylor_series()

    # # Stability plots
    # stability_advection()
    # stability_diffusion()
    # dispersion_dissipation_diagram()

    # Chapter: Spectral Properties
    # spectral_dissipation_adv_explct()
    # spectral_dissipation_adv_implct()
    # spectral_dispersion_adv_explct()

    # Chapter: Time-stepping methods
    # explicit_euler_method()
    heuns_method()
    # midpoint_method()
    # rk_methods()

    # Hyperbolic problems
    # lsc_advection_dye()
    # lsc_characteristics()
    # lsys_characteristics()
    # riem_lsc_u0()
    # riem_lsc_characteristics()
    # riem_lsys_characteristics()
    # riem_linacoustics_characteristics()
    # riem_linacoustics_u0()
    # riem_linacoustics_sol()


if __name__ == '__main__':
    plt.style.use('style.mplstyle')
    blue = '#001aab'
    red = '#bd0c00'
    gray = '#dbdbdb'
    darkgray = '#4a4a4a'
    main()
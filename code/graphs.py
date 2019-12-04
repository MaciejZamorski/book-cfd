# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import sympy
import scipy.interpolate as spi


def _rotate_vector(v, rad):
    M = np.array([[np.cos(rad), -np.sin(rad)], 
                  [np.sin(rad), np.cos(rad)]])
    return v @ M


def convert_to_graph(fig, ax):
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

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw=lw, 
             head_width=hw, head_length=hl, overhang=ohg, 
             length_includes_head=True, clip_on=False)

    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw=lw, 
             head_width=yhw, head_length=yhl, overhang=ohg, 
             length_includes_head=True, clip_on=False)

    return fig, ax


def remove_axes(fig, ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ['bottom', 'right', 'top', 'left']:
        ax.spines[side].set_visible(False)
    return fig, ax


def ensemble_average():
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
    
def linear_scalar_characteristics():
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
   
    fig.savefig('ch11_linear_scalar_characteristics.pdf')


def main():
    # Reynolds averaging
    # time_average()
    # ensemble_average()

    # FD, FV schemes
    # discrete_scheme()
    # fv_arbitrary_volume()
    # fv_trivolume()
    # fv_riemann_diagram()
    # fv_1d_advection()
    # fv_1d_diffusion()

    # Hyperbolic problems
    linear_scalar_characteristics()

    # # Stability plots
    # stability_advection()
    # stability_diffusion()


if __name__ == '__main__':
    plt.style.use('style.mplstyle')
    blue = '#001aab'
    red = '#bd0c00'
    main()
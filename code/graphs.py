# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import sympy

def convert_to_graph(fig, ax):
    # Taken from
    # https://stackoverflow.com/questions/17646247/how-to-make-fuller-axis-arrows-with-matplotlib
    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)

    # removing the axis ticks
    plt.xticks([]) # labels 
    plt.yticks([])
    ax.xaxis.set_ticks_position('none') # tick markers
    ax.yaxis.set_ticks_position('none')

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./40.*(ymax-ymin) 
    hl = 1./40.*(xmax-xmin)
    lw = 0.5 # axis line width
    ohg = 0.3 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw=lw, 
             head_width=hw, head_length=hl, overhang=ohg, 
             length_includes_head=True, clip_on=False) 

    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw=lw, 
             head_width=yhw, head_length=yhl, overhang=ohg, 
             length_includes_head=True, clip_on=False)
    
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
    y = np.random.random(npts) * 0.5*np.sin(np.pi/5*x) + 0.5*np.random.random(npts) + 0.75
    ax.plot(x, y, 'k', lw=1, label=r'$u_i(\vec{x},t)$')

    ymean = 0
    for i in range(navg):
        np.random.seed(i)
        ymean += np.random.random(npts) * 0.5*np.sin(np.pi/5*x) + 0.5*np.random.random(npts) + 0.75
    ymean = ymean/navg
    ax.plot(x, ymean, '--k', lw=1, label=r'$u_E(\vec{x},t)$')

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
    ax.plot(x, y, lw=1, color='black', label=r'$u_i(\vec{x},t)$')
    ax.axhline(np.mean(y), linestyle='--', color='black', lw=1, label=r'$u_T(\vec{x})$')

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

                rectangle = plt.Rectangle((xnod[i],0), (xnod[i+1]-xnod[i]), avgval, hatch='////', facecolor='#ededed', edgecolor='#d6d6d6')
                ax.add_artist(rectangle)

            ax.hlines(yavg, xnod[:-1], xnod[1:], lw=2, color='red')
            # Draw vertical lines
            ax.vlines(xnod,  ynod*0, [*ymax, yavg[-1]], lw=0.7, color='gray', linestyle='--')

        # delta x annotation
        ax.text((xnod[3] + xnod[2])/2, np.min(ynod)/2*1.1, r'$\Delta x$', horizontalalignment='center' )
        ax.annotate("", xy=(xnod[3], np.min(ynod)/2), xytext=(xnod[2], np.min(ynod)/2),  arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color='black'),)

        ax.set_ylim([-dax/2, l-dax])
        ax.set_xlim([-dax/2, l+dax])
        ax.set_ylabel('$u(x)$')
        fig, ax = convert_to_graph(fig, ax)

        if which == 'fd':
            plt.xticks(np.linspace(0, l, int(l/dx)), [r'$x_{i-2}$', r'$x_{i-1}$', r'$x_{i}$', r'$x_{i+1}$', r'$x_{i+2}$'])
        elif which == 'fv':
            plt.xticks(np.linspace(0+dx/2, l-dx/2, int(l/dx)), [r'$\Omega_{i-2}$', r'$\Omega_{i-1}$', r'$\Omega_{i}$', r'$\Omega_{i+1}$', r'$\Omega_{i+2}$'])
        
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
            ampf = lambda cfl, kappa: 1-cfl+cfl*np.exp(1j*kappa*dx)
            center = (0, 0)
            xlim = [-1.2, 1.2]
            r = 1
            maxval = 1.0
            cfls = [0.5, 0.6, 0.7, 0.8, 0.9]
            ylim = [-1.1, 1.3]
        elif which =='implicit':
            ampf = lambda cfl, kappa: 1/(1+cfl*(1-np.exp(-1j*kappa*dx)))
            center = (0.5, 0)
            xlim = [-1.2, 1.2]
            r = 0.5
            maxval = '\infty'
            cfls = [0.5, 1.0, 2.0, 4.0, 8.0]
            ylim = [-1.1, 1.3]

        circle = plt.Circle((0, 0), 1, facecolor='#e6e6e6', edgecolor='#c2c2c2', linestyle='--', hatch='////')
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
        ax.plot([],[], label=f'$\\sigma={maxval}$', color='black', lw='1', linestyle='--')
        ax.axhline(y=0, lw=0.7, color='black')
        ax.axvline(x=0, lw=0.7, color='black')
        ax.set_xlabel(r'Re$\left(e^{a\Delta t}\right)$')
        ax.set_ylabel(r'Im$\left(e^{a\Delta t}\right)$')
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_aspect('equal', )
        plt.legend(loc=9, ncol=3,  handlelength=0.8, borderaxespad=0)

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
            ampf = lambda cfl, kappa: 1+cfl*(np.exp(-1j*kappa*dx)-2+np.exp(1j*kappa*dx))
            cfls = [0.1, 0.3, 0.5]
            markers = ['o', 's', 'd', 'p']
        elif which =='implicit':
            ampf = lambda cfl, kappa: 1/(1-cfl*(np.exp(-1j*kappa*dx)-2+np.exp(1j*kappa*dx)))
            cfls = [1.0, 5.0, 10.0]
            markers = ['o', 's', 'd', 'p']

        circle = plt.Circle((0, 0), 1, facecolor='#e6e6e6', edgecolor='#c2c2c2', linestyle=None, hatch='////')
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
        plt.legend(loc=9, ncol=3,  handlelength=0.8, borderaxespad=0)
        plt.tight_layout()

        fig.savefig(f'ch12_{which}_diffusion.pdf')
        plt.close(fig)

def main():
    # Reynolds averaging 
    time_average()
    ensemble_average()

    # FD, FV schemes
    discrete_scheme()

    # Stability plots
    stability_advection()
    stability_diffusion()


if __name__ == '__main__':
    plt.style.use('style.mplstyle')
    main()

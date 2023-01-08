import matplotlib.pyplot as plt
import os

def PlotTemplate(fontsize=15):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes3d.grid'] = True
    plt.rcParams['axes.xmargin']=0
    plt.rcParams['axes.labelsize'] = 1.5 * fontsize
    plt.rcParams['axes.titlesize'] = 2 * fontsize
    plt.rcParams['xtick.labelsize'] = 0.7 * fontsize
    plt.rcParams['ytick.labelsize'] = 0.7 * fontsize

def SaveAllActiveFigures(Prefix: str = "Figure"):
    if not os.path.exists("Figures"):
        os.mkdir("Figures")
    for fignum in plt.get_fignums():
        plt.figure(fignum)
        plt.savefig(f"Figures/{Prefix}_{fignum:02d}.png", dpi=400, bbox_inches='tight')
        # plt.savefig(f"Figures/{fignum}.eps", format='eps')
        print(f"Figures/{Prefix}_{fignum:02d}.png Saved.")

def ColorbarSubplot(colormapObj, figObj, vmin, vmax, position_ax, ylabel=None, fraction=0.05):
    """
    :param colormapObj: mpl.cm.plasma
    :return:
    """
    Cmap = colormapObj  # rainbow bwr
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = figObj.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=Cmap),
                           ax=position_ax,
                           fraction=fraction,
                           ticks=np.linspace(vmin, vmax, 6, endpoint=True))
    cbar.axes1.set_ylabel(ylabel, rotation=270, labelpad=12)

def IncreaseLegendLinewidth(leg, linewidth: float = 3):
    for legobj in leg.legendHandles:
        legobj.set_linewidth(linewidth)

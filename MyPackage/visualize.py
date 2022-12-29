import matplotlib.pyplot as plt

def PlotTemplate(fontsize):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['mathtext.fontset'] = 'stix'
    # plt.rcParams['text.usetex']=True

def SaveAllActiveFigures(IndexingName: str = "Figure"):
    if not os.path.exists("Figures"):
        os.mkdir("Figures")
    for fignum in plt.get_fignums():
        plt.figure(fignum)
        plt.savefig(f"Figures/{IndexingName}_{fignum:02d}.png", dpi=400, bbox_inches='tight')
        # plt.savefig(f"Figures/{fignum}.eps", format='eps')
        print(f"Figures/{IndexingName}_{fignum:02d}.png Saved.")

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

def IncreaseLegendLinewidth(leg, linewidth: float = 2):
    for legobj in leg.legendHandles:
        legobj.set_linewidth(linewidth)
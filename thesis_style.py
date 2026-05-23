"""
thesis_style.py  --  Unified, journal-quality matplotlib style for the whole TFM.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

COL_W = 3.40          # single column  (\columnwidth)
FULL_W = 7.00         # full text width (\textwidth, use inside figure*)

COLORS = {
    'blue':   '#0077BB', 'orange': '#EE7733', 'teal':   '#009988',
    'red':    '#CC3311', 'magenta':'#EE3377', 'cyan':   '#33BBEE',
    'grey':   '#BBBBBB', 'black':  '#000000',
}
_CYCLE = [COLORS['blue'], COLORS['orange'], COLORS['teal'], COLORS['red'],
          COLORS['magenta'], COLORS['cyan']]

def apply_thesis_style():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'figure.titlesize': 9,
        'lines.linewidth': 1.2, 'lines.markersize': 4.0,
        'axes.linewidth': 0.8, 'patch.linewidth': 0.8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
        'xtick.minor.size': 2.0, 'ytick.minor.size': 2.0,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'legend.frameon': True, 'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8', 'legend.handlelength': 1.6,
        'legend.borderpad': 0.4, 'legend.labelspacing': 0.3,
        'axes.grid': False, 'grid.linewidth': 0.5, 'grid.alpha': 0.25,
        'grid.linestyle': ':',
        'axes.prop_cycle': mpl.cycler(color=_CYCLE),
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02, 'savefig.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

def panel_label(ax, text, loc='upper left', dx=0.0, dy=0.0, **kw):
    pos = {
        'upper left': (0.03, 0.97, 'left', 'top'),
        'upper right': (0.97, 0.97, 'right', 'top'),
        'lower left': (0.03, 0.03, 'left', 'bottom'),
        'lower right': (0.97, 0.03, 'right', 'bottom'),
    }[loc]
    x, y, ha, va = pos
    ax.text(x + dx, y + dy, text, transform=ax.transAxes,
            ha=ha, va=va, fontsize=9, fontweight='bold', **kw)

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
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
        'figure.titlesize': 9,
        'lines.linewidth': 0.8, 'lines.markersize': 3.0,
        'axes.linewidth': 0.7, 'patch.linewidth': 0.7,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.minor.size': 2.0, 'ytick.minor.size': 2.0,
        'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
        'legend.frameon': True, 'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8', 'legend.handlelength': 1.4,
        'legend.borderpad': 0.35, 'legend.labelspacing': 0.3,
        'axes.grid': False, 'grid.linewidth': 0.5, 'grid.alpha': 0.25,
        'grid.linestyle': ':',
        'axes.prop_cycle': mpl.cycler(color=_CYCLE),
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02, 'savefig.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

def panel_label(ax, text, loc='outside', dx=0.0, dy=0.0, **kw):
    """Add a bold panel label to an axes.

    loc='outside' (default): label just outside upper-left corner — never
        overlaps plot content or legends.
    loc='inside':  white label inside upper-left corner — for heatmaps where
        outside placement is clipped or looks odd.
    """
    if loc == 'inside':
        ax.text(0.03 + dx, 0.97 + dy, text, transform=ax.transAxes,
                ha='left', va='top', fontsize=9, fontweight='bold',
                color='white', **kw)
    else:
        # Default: just outside upper-left — never overlaps content
        ax.text(-0.12 + dx, 1.04 + dy, text, transform=ax.transAxes,
                fontsize=9, fontweight='bold',
                va='bottom', ha='right', **kw)

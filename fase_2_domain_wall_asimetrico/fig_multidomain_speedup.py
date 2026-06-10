#!/usr/bin/env python3
"""fig_multidomain_speedup.py  --  Multi-domain speedup + 5-state CTAP figure."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, NullFormatter

from thesis_style import apply_thesis_style, panel_label, COLORS, FULL_W

apply_thesis_style()

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Data ──────────────────────────────────────────────────────────────────────
sp = pd.read_csv(os.path.join(HERE, 'paso2_speedup.csv'))
ts = pd.read_csv(os.path.join(HERE, 'paso3_5state.csv'))

T1 = float(sp.loc[sp['D'] == 1, 'T_est'].iloc[0])
sp = sp.copy()
sp['speedup'] = T1 / sp['T_est']

T_sim  = float(ts['t'].iloc[-1])
t_norm = ts['t'].values / T_sim

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FULL_W, FULL_W * 0.47))
gs  = GridSpec(1, 2, figure=fig, width_ratios=[0.78, 1.22], wspace=0.42)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])

# ─────────────────────────────────────────────────────────────────────────────
# Panel (a): speedup T1/TD vs D
# ─────────────────────────────────────────────────────────────────────────────
Ds  = sp['D'].values.astype(int)
spd = sp['speedup'].values
nsg = sp['n_seg'].values.astype(int)

ax_a.semilogy(Ds, spd, 'o-',
              color=COLORS['blue'], lw=0.9, ms=4.5,
              markerfacecolor='white', markeredgewidth=1.1,
              markeredgecolor=COLORS['blue'], clip_on=False)

# n_seg labels — placed to the right of each marker except D=4 (left)
nseg_y_scale = [1.9, 2.0, 2.5, 0.18]   # relative y offset per marker
nseg_x_off   = [0.12, 0.12, 0.12, -0.12]
nseg_ha      = ['left', 'left', 'left', 'right']
for d, s, n, ys, xo, ha in zip(Ds, spd, nsg, nseg_y_scale, nseg_x_off, nseg_ha):
    ax_a.text(d + xo, s * ys, fr'$\ell_\mathrm{{s}}={n}$',
              fontsize=6, ha=ha, va='center', color='#555555')

# Annotation at D=4
factor = int(round(spd[-1]))
ax_a.annotate(rf'$\times {factor}$',
              xy=(4, spd[-1]), xytext=(2.9, spd[-1] * 0.055),
              fontsize=7.5, ha='center',
              arrowprops=dict(arrowstyle='->', lw=0.6,
                              color='black', shrinkA=2, shrinkB=2))

ax_a.set_xlabel(r'Number of domains $D$')
ax_a.set_ylabel(r'Speedup $T_1/T_D$')
ax_a.set_xticks([1, 2, 3, 4])
ax_a.set_xlim(0.55, 4.55)
ax_a.set_ylim(0.4, 6e4)
ax_a.yaxis.set_minor_locator(LogLocator(subs=range(2, 10), numticks=40))
ax_a.yaxis.set_minor_formatter(NullFormatter())
panel_label(ax_a, '(a)')

# ─────────────────────────────────────────────────────────────────────────────
# Panel (b): 5-state populations vs t/T
# ─────────────────────────────────────────────────────────────────────────────
c = COLORS

# Dark (near-zero) states first so they sit under the main curves
ax_b.plot(t_norm, ts['P_S1'].values, '--', color=c['grey'],   lw=0.7,
          label=r'$P_{S_1}$')
ax_b.plot(t_norm, ts['P_S3'].values, ':',  color=c['grey'],   lw=0.7,
          label=r'$P_{S_3}$')
# Transient intermediate
ax_b.plot(t_norm, ts['P_S2'].values, '-',  color=c['orange'], lw=1.0,
          label=r'$P_{S_2}$')
# Main transfer
ax_b.plot(t_norm, ts['P_L'].values,  '-',  color=c['blue'],   lw=1.2,
          label=r'$P_L$')
ax_b.plot(t_norm, ts['P_R'].values,  '-',  color=c['red'],    lw=1.2,
          label=r'$P_R$')

ax_b.set_xlabel(r'$t/T$')
ax_b.set_ylabel(r'Population')
ax_b.set_xlim(0, 1)
ax_b.set_ylim(-0.04, 1.08)
ax_b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

# Legend in the upper centre (both P_L and P_R dip below ~0.5 there)
ax_b.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99),
            ncol=5, fontsize=6.5,
            borderpad=0.32, columnspacing=0.55,
            handlelength=1.3, handletextpad=0.35)
panel_label(ax_b, '(b)')

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(HERE, 'fig_multidomain_speedup.pdf')
fig.savefig(out)
print(f"Saved: {out}")

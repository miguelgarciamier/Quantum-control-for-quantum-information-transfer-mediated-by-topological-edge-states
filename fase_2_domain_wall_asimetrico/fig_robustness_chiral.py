#!/usr/bin/env python3
"""fig_robustness_chiral.py -- Chiral protection: spectral + transfer robustness."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from thesis_style import apply_thesis_style, panel_label, COLORS, FULL_W

apply_thesis_style()

HERE        = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.normpath(os.path.join(HERE, '..', 'tesis', 'figures'))
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
sp = pd.read_csv(os.path.join(HERE, 'paso4_chiral_spectral.csv'))
fd = pd.read_csv(os.path.join(HERE, 'paso4_fig_data.csv'))
fd['std_F'] = pd.to_numeric(fd['std_F'], errors='coerce').fillna(0.0)

A_NN = sp[sp['case'] == 'A_NN'].sort_values('value')
B    = sp[sp['case'] == 'B'].sort_values('value')
C    = sp[sp['case'] == 'C'].sort_values('value')

off = fd[fd['condition'] == 'CTAP_NN_offdiag'].sort_values('sigma')
on  = fd[fd['condition'] == 'CTAP_NN_onsite'].sort_values('sigma')
nnn = fd[fd['condition'] == 'CTAP_NNN_offdiag'].sort_values('sigma')

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FULL_W, FULL_W * 0.46))
gs  = GridSpec(1, 2, figure=fig, wspace=0.52)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
c = COLORS

# ─────────────────────────────────────────────────────────────────────────────
# Panel (a): <|E0|> vs sigma (A_NN, B) and vs kappa (C) — log Y, twin X
# ─────────────────────────────────────────────────────────────────────────────
sig_A = A_NN['value'].values;  E0_A = A_NN['mean_E0'].values
sig_B = B['value'].values;     E0_B = np.maximum(B['mean_E0'].values, 1e-19)
kap_C = C['value'].values;     E0_C = np.maximum(C['mean_E0'].values, 1e-19)

ax_a.set_yscale('log')

ax_a.plot(sig_A, E0_A,
          'o--', color=c['blue'], lw=0.9, ms=4.0,
          markerfacecolor='white', markeredgewidth=1.0,
          label=r'off-diag, NN  ($\kappa\!=\!0$)')
ax_a.plot(sig_B, E0_B,
          's-', color=c['orange'], lw=0.9, ms=4.0,
          markerfacecolor='white', markeredgewidth=1.0,
          label=r'onsite, NN  ($\kappa\!=\!0$)')

# Twin top x-axis for kappa (C series)
ax_a2 = ax_a.twiny()
ax_a.tick_params(top=False)          # suppress mirrored sigma ticks on top spine
ax_a2.plot(kap_C, E0_C,
           '^-', color=c['teal'], lw=0.9, ms=4.0,
           markerfacecolor='white', markeredgewidth=1.0,
           label=r'NNN  ($\sigma\!=\!0$)')

ax_a.set_xlabel(r'Disorder $\sigma$')
ax_a.set_ylabel(r'$\langle |E_0| \rangle$')
ax_a2.set_xlabel(r'NNN coupling $\kappa$', color=c['teal'], labelpad=4)
ax_a2.tick_params(axis='x', colors=c['teal'])
ax_a2.spines['top'].set_edgecolor(c['teal'])

ax_a.set_xlim(-0.01, 0.32)
ax_a2.set_xlim(-0.007, 0.215)
ax_a.set_ylim(5e-19, 0.55)
ax_a.set_xticks([0, 0.10, 0.20, 0.30])
ax_a2.set_xticks([0, 0.05, 0.10, 0.15, 0.20])

lines1, labs1 = ax_a.get_legend_handles_labels()
lines2, labs2 = ax_a2.get_legend_handles_labels()
ax_a.legend(lines1 + lines2, labs1 + labs2,
            loc='center right', bbox_to_anchor=(0.97, 0.50),
            fontsize=6.5, borderpad=0.4, handlelength=1.5, handletextpad=0.4)
panel_label(ax_a, '(a)', dy=0.08)   # extra dy to clear top x-label

# ─────────────────────────────────────────────────────────────────────────────
# Panel (b): <F> vs sigma — error bars + QEC reference line
# ─────────────────────────────────────────────────────────────────────────────
sig_f = off['sigma'].values

ax_b.axhline(0.995, color='#888888', lw=0.7, ls=':', zorder=0,
             label=r'$f_0=0.995$')

kw = dict(capsize=2.5, capthick=0.6, lw=0.9, ms=4.5,
          markerfacecolor='white', markeredgewidth=1.0, elinewidth=0.7)

ax_b.errorbar(sig_f, off['mean_F'].values, yerr=off['std_F'].values,
              fmt='o-', color=c['blue'],
              label=r'NN off-diag  ($\kappa\!=\!0$)', **kw)
ax_b.errorbar(sig_f, on['mean_F'].values, yerr=on['std_F'].values,
              fmt='s-', color=c['orange'],
              label=r'NN onsite  ($\kappa\!=\!0$)', **kw)
ax_b.errorbar(sig_f, nnn['mean_F'].values, yerr=nnn['std_F'].values,
              fmt='^-', color=c['teal'],
              label=r'NNN off-diag  ($\kappa\!=\!J/3$)', **kw)

ax_b.set_xlabel(r'Disorder $\sigma$')
ax_b.set_ylabel(r'$\langle f \rangle$')
ax_b.set_xlim(-0.01, 0.22)
ax_b.set_ylim(-0.06, 1.09)
ax_b.set_xticks([0, 0.05, 0.10, 0.20])
ax_b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

ax_b.legend(loc='lower left', fontsize=6.5,
            borderpad=0.4, handlelength=1.5, handletextpad=0.4)
panel_label(ax_b, '(b)')

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(FIGURES_DIR, 'fig_robustness_chiral.pdf')
fig.savefig(out)
print(f"Saved: {out}")

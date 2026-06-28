#!/usr/bin/env python3
"""run_sta_figs.py -- Regenerate fig9, fig10, fig11 (experiments 5-7 only).

Extracts the relevant setup and experiment blocks from domain_wall_asimetrico.py
without running experiments 1-4 or 8.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from thesis_style import apply_thesis_style, FULL_W, panel_label
apply_thesis_style()

from domain_wall_asimetrico import (
    build_ssh_multidomain,
    v_pulse_standard, v_pulse_sta, v_pulse_optimal_sta, v_pulse_linear,
    time_evolve_symmetric,
    scan_fidelity_vs_time, scan_fidelity_vs_time_symmetric,
    plot_transfer_heatmap,
)

FIGURES_DIR = os.path.normpath(os.path.join(_HERE, '..', 'tesis', 'figures'))
os.makedirs(FIGURES_DIR, exist_ok=True)

# Shared setup (mirrors main())
w     = 1.0
v_tr  = 0.5
dt    = 0.5
L_total = 21
wall_positions = {
    'Center (~1/2)': 11,
    'Third (~1/3)':   7,
    'Quarter (~1/4)': 5,
}

# =========================================================================
# EXPERIMENT 5: fig9_sta_pulse_comparison.pdf
# =========================================================================
print("=" * 72)
print("EXPERIMENT 5: Pulse Comparison -> fig9_sta_pulse_comparison.pdf")
print("=" * 72)

N_dom = 2
ell   = 4
L_sym = N_dom * (ell + 1) + 1
print(f"  Symmetric chain: N={N_dom}, ell={ell}, L={L_sym}")

pulse_configs = {
    'Standard (sin², t_prep=15)': {
        'func': v_pulse_standard, 'kwargs': {'t_prep': 15.0},
        'color': '#0077BB', 'ls': '-'
    },
    'Standard (sin², t_prep=10)': {
        'func': v_pulse_standard, 'kwargs': {'t_prep': 10.0},
        'color': '#33BBEE', 'ls': '-'
    },
    'Standard (sin², t_prep=5)': {
        'func': v_pulse_standard, 'kwargs': {'t_prep': 5.0},
        'color': '#88CCEE', 'ls': '-'
    },
    'Ramp-shaping (α=2, t_prep=10)': {
        'func': v_pulse_sta, 'kwargs': {'t_prep': 10.0, 'alpha': 2.0},
        'color': '#EE7733', 'ls': '--'
    },
    'Ramp-shaping (α=3, t_prep=10)': {
        'func': v_pulse_sta, 'kwargs': {'t_prep': 10.0, 'alpha': 3.0},
        'color': '#CC3311', 'ls': '--'
    },
    'Global sin² ramp': {
        'func': v_pulse_optimal_sta, 'kwargs': {},
        'color': '#009988', 'ls': '-.'
    },
    'Linear (t_prep=15)': {
        'func': v_pulse_linear, 'kwargs': {'t_prep': 15.0},
        'color': '#AA3377', 'ls': ':'
    },
}

fig5 = plt.figure(figsize=(FULL_W, 5.44))
gs5  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30)
ax5a = fig5.add_subplot(gs5[0, 0])
t_demo = np.linspace(0, 50, 500)
for name, cfg in pulse_configs.items():
    v_demo = [cfg['func'](t, v_tr, 50.0, **cfg['kwargs']) for t in t_demo]
    ax5a.plot(t_demo, v_demo, cfg['ls'], color=cfg['color'], lw=2, label=name)
ax5a.set_xlabel('Time $t$ ($\\hbar/J$)')
ax5a.set_ylabel('$v(t)$')
panel_label(ax5a, '(a)')

ax5b = fig5.add_subplot(gs5[0, 1])
t_scan_sta = np.arange(25, 120, 2.0)
sta_results = {}
for name, cfg in pulse_configs.items():
    print(f"  Scanning: {name}...")
    fids = scan_fidelity_vs_time_symmetric(
        N_dom, ell, cfg['func'], v_tr, w, t_scan_sta, dt, **cfg['kwargs'])
    best_i = np.argmax(fids)
    sta_results[name] = {
        'fidelities': fids, 'best_t': t_scan_sta[best_i],
        'best_f': fids[best_i], 'config': cfg
    }
    print(f"    Best: t_tr={t_scan_sta[best_i]:.1f}, f={fids[best_i]:.6f}")
    ax5b.plot(t_scan_sta, fids, cfg['ls'], color=cfg['color'], lw=1.5, alpha=0.85)
    ax5b.plot(t_scan_sta[best_i], fids[best_i], 'o', color=cfg['color'], ms=6, zorder=5)

ax5b.axhline(y=0.995, color='gray', ls='--', lw=1, alpha=0.7)
ax5b.set_xlabel('$t_{tr}$ ($\\hbar/J$)')
ax5b.set_ylabel('Fidelity $f$')
panel_label(ax5b, '(b)')
ax5b.set_ylim(-0.05, 1.1)

_h9, _l9 = ax5a.get_legend_handles_labels()
_h9.append(Line2D([0], [0], ls='--', color='gray', lw=1.0, alpha=0.7))
_l9.append(r'$f_0 = 0.995$')
fig5.legend(_h9, _l9, loc='upper center', bbox_to_anchor=(0.5, 1.01),
            ncol=4, fontsize=7, framealpha=0.9)

best_std_name = 'Standard (sin², t_prep=15)'
best_sta_name = 'Global sin² ramp'
std_res = sta_results[best_std_name]
sta_res = sta_results[best_sta_name]

print(f"  Running standard transfer: t_tr={std_res['best_t']:.1f}")
times_std, occ_std, v_std, L_check = time_evolve_symmetric(
    N_dom, ell, std_res['config']['func'], v_tr, w,
    std_res['best_t'], dt, **std_res['config']['kwargs'])

print(f"  Running ramp-shaping transfer: t_tr={sta_res['best_t']:.1f}")
times_sta, occ_sta, v_sta, _ = time_evolve_symmetric(
    N_dom, ell, sta_res['config']['func'], v_tr, w,
    sta_res['best_t'], dt, **sta_res['config']['kwargs'])

ax5c = fig5.add_subplot(gs5[1, 0])
plot_transfer_heatmap(ax5c, times_std, occ_std, L_sym)
panel_label(ax5c, '(c)', loc='inside')

ax5d = fig5.add_subplot(gs5[1, 1])
plot_transfer_heatmap(ax5d, times_sta, occ_sta, L_sym)
panel_label(ax5d, '(d)', loc='inside')

out9 = os.path.join(FIGURES_DIR, 'fig9_sta_pulse_comparison.pdf')
fig5.savefig(out9, bbox_inches='tight')
print(f"  -> {out9}")
plt.close(fig5)

# =========================================================================
# EXPERIMENT 6: fig10_tprep_dependence.pdf
# =========================================================================
print("=" * 72)
print("EXPERIMENT 6: Preparation Time Dependence -> fig10_tprep_dependence.pdf")
print("=" * 72)

t_tr_fixed   = 45.6
t_prep_range = np.linspace(2, 25, 15)
fid_vs_tprep_std  = np.zeros(len(t_prep_range))
fid_vs_tprep_sta2 = np.zeros(len(t_prep_range))
fid_vs_tprep_sta3 = np.zeros(len(t_prep_range))

for i, tp in enumerate(t_prep_range):
    if 2 * tp >= t_tr_fixed:
        fid_vs_tprep_std[i] = fid_vs_tprep_sta2[i] = fid_vs_tprep_sta3[i] = np.nan
        continue
    _, occ, _, _ = time_evolve_symmetric(N_dom, ell, v_pulse_standard, v_tr, w,
                                          t_tr_fixed, dt, t_prep=tp)
    fid_vs_tprep_std[i] = occ[-1, L_sym - 1]
    _, occ, _, _ = time_evolve_symmetric(N_dom, ell, v_pulse_sta, v_tr, w,
                                          t_tr_fixed, dt, t_prep=tp, alpha=2.0)
    fid_vs_tprep_sta2[i] = occ[-1, L_sym - 1]
    _, occ, _, _ = time_evolve_symmetric(N_dom, ell, v_pulse_sta, v_tr, w,
                                          t_tr_fixed, dt, t_prep=tp, alpha=3.0)
    fid_vs_tprep_sta3[i] = occ[-1, L_sym - 1]

fig6, axes6 = plt.subplots(1, 2, figsize=(FULL_W, 2.63))
ax6a = axes6[0]
ax6a.plot(t_prep_range, fid_vs_tprep_std,  '-',  color='#0077BB', lw=2, label='Standard (sin²)')
ax6a.plot(t_prep_range, fid_vs_tprep_sta2, '--', color='#EE7733', lw=2, label='Ramp-shaping ($\\alpha=2$)')
ax6a.plot(t_prep_range, fid_vs_tprep_sta3, '--', color='#CC3311', lw=2, label='Ramp-shaping ($\\alpha=3$)')
ax6a.axhline(y=0.995, color='gray', ls='--', lw=1, alpha=0.7)
ax6a.axvline(x=8, color='lightgray', ls=':', lw=1, alpha=0.7)
ax6a.text(8.5, 0.1, '$\\tau \\sim 8$ (adiabatic\nscale)', fontsize=8, color='gray')
ax6a.set_xlabel('Preparation time $t_{prep}$ ($\\hbar/J$)')
ax6a.set_ylabel('Fidelity $f$')
panel_label(ax6a, '(a)')
ax6a.legend(loc='upper left', fontsize=7.5)
ax6a.set_ylim(-0.05, 1.1)

# Panel (b): gap analysis
from domain_wall_asimetrico import build_ssh_multidomain
ax6b = axes6[1]
v_range = np.linspace(0.01, 0.99, 100)
gaps = np.zeros(len(v_range))
for i, v_val in enumerate(v_range):
    H, _ = build_ssh_multidomain(N_dom, ell, v_val, w)
    evals_gap = np.linalg.eigvalsh(H)
    sorted_e = np.sort(np.abs(evals_gap))
    if len(sorted_e) > 3:
        gaps[i] = sorted_e[3] - sorted_e[2]
tau_char = 2.0 / np.where(gaps > 0, gaps, np.nan)
ax6b.plot(v_range, gaps, '-', color='#0077BB', lw=2, label='Gap $\\Delta$')
ax6b_twin = ax6b.twinx()
ax6b_twin.plot(v_range, tau_char, '-', color='#CC3311', lw=2, label='$\\tau = 2/\\Delta$')
ax6b.axvline(x=v_tr, color='gray', ls='--', alpha=0.5)
ax6b.text(v_tr + 0.02, max(gaps) * 0.9, f'$v_{{tr}}={v_tr}$', fontsize=9, color='gray')
ax6b.set_xlabel('$v$')
ax6b.set_ylabel('Gap $\\Delta$', color='#0077BB')
ax6b_twin.set_ylabel('Adiabatic timescale $\\tau$ ($\\hbar/J$)', color='#CC3311')
panel_label(ax6b, '(b)')
ax6b.legend(loc='upper left')
ax6b_twin.legend(loc='upper right')

fig6.tight_layout()
out10 = os.path.join(FIGURES_DIR, 'fig10_tprep_dependence.pdf')
fig6.savefig(out10, bbox_inches='tight')
print(f"  -> {out10}")
plt.close(fig6)

# =========================================================================
# EXPERIMENT 7: fig11_combined_study.pdf
# =========================================================================
print("=" * 72)
print("EXPERIMENT 7: Combined Asymmetric + Ramp-shaping -> fig11_combined_study.pdf")
print("=" * 72)

protocols = {
    'Standard (sin², t_prep=15)': {
        'func': v_pulse_standard, 'kwargs': {'t_prep': 15.0},
        'color': '#0077BB', 'ls': '-', 'marker': 'o'
    },
    'Ramp-shaping (α=2, t_prep=10)': {
        'func': v_pulse_sta, 'kwargs': {'t_prep': 10.0, 'alpha': 2.0},
        'color': '#EE7733', 'ls': '--', 'marker': 's'
    },
    'Global sin² ramp': {
        'func': v_pulse_optimal_sta, 'kwargs': {},
        'color': '#009988', 'ls': '-.', 'marker': '^'
    },
}

fig7 = plt.figure(figsize=(FULL_W, 5.44))
gs7  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30)
ax7a = fig7.add_subplot(gs7[0, :])

t_scan_combined = np.arange(30, 600, 5.0)
summary_data = []
for label_w, wp in wall_positions.items():
    for label_p, proto in protocols.items():
        print(f"  {label_w} + {label_p}...")
        fids = scan_fidelity_vs_time(L_total, wp, proto['func'], v_tr, w,
                                     t_scan_combined, dt, **proto['kwargs'])
        best_i = np.argmax(fids)
        summary_data.append({
            'wall': label_w, 'wall_pos': wp, 'protocol': label_p,
            'best_t': t_scan_combined[best_i], 'best_f': fids[best_i]
        })
        print(f"    t_tr={t_scan_combined[best_i]:.1f}, f={fids[best_i]:.6f}")

ax7a.axis('off')
table_data = [[sd['wall'], f'DW={sd["wall_pos"]}', sd['protocol'],
               f'{sd["best_t"]:.1f}', f'{sd["best_f"]:.4f}'] for sd in summary_data]
table = ax7a.table(
    cellText=table_data,
    colLabels=['DW position', 'Site', 'Protocol', '$t_{tr}$ opt.', 'Fidelity'],
    loc='center', cellLoc='center',
    colWidths=[0.18, 0.1, 0.28, 0.12, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 1.6)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('0.7')
    cell.set_linewidth(0.5)
    if row == 0:
        cell.set_facecolor('0.92')
        cell.set_text_props(fontweight='bold')
    else:
        cell.set_facecolor('white')
panel_label(ax7a, '(a)')

ax7b = fig7.add_subplot(gs7[1, 0])
wp_center = wall_positions['Center (~1/2)']
for label_p, proto in protocols.items():
    fids = scan_fidelity_vs_time(L_total, wp_center, proto['func'], v_tr, w,
                                 t_scan_combined, dt, **proto['kwargs'])
    ax7b.plot(t_scan_combined, fids, proto['ls'], color=proto['color'],
              lw=1.5, label=label_p)
ax7b.axhline(y=0.995, color='gray', ls='--', lw=1, alpha=0.5)
ax7b.set_xlabel('$t_{tr}$ ($\\hbar/J$)')
ax7b.set_ylabel('Fidelity')
panel_label(ax7b, '(b)')
ax7b.legend(fontsize=7.5)
ax7b.set_ylim(-0.05, 1.1)

ax7c = fig7.add_subplot(gs7[1, 1])
wp_quarter = wall_positions['Quarter (~1/4)']
for label_p, proto in protocols.items():
    fids = scan_fidelity_vs_time(L_total, wp_quarter, proto['func'], v_tr, w,
                                 t_scan_combined, dt, **proto['kwargs'])
    ax7c.plot(t_scan_combined, fids, proto['ls'], color=proto['color'],
              lw=1.5, label=label_p)
ax7c.axhline(y=0.995, color='gray', ls='--', lw=1, alpha=0.5)
ax7c.set_xlabel('$t_{tr}$ ($\\hbar/J$)')
ax7c.set_ylabel('Fidelity')
panel_label(ax7c, '(c)')
ax7c.legend(fontsize=7.5)
ax7c.set_ylim(-0.05, 1.1)

out11 = os.path.join(FIGURES_DIR, 'fig11_combined_study.pdf')
fig7.savefig(out11, bbox_inches='tight')
print(f"  -> {out11}")
plt.close(fig7)

print("\nDone. Regenerated: fig9, fig10, fig11.")

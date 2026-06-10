#!/usr/bin/env python3
"""fig4_fullchain_ctap.py -- Fig 4: full-chain CTAP, 4-panel comparison.

Panels:
  (a) Counterintuitive pulse sequence v_R/v_L for j_DW=5, form (b).
  (b) Populations P_L, P_S, P_R vs t; inset log-scale leakage ~ 1e-7.
  (c) Heatmap |psi_j(t)|^2 site vs time -- L->R transfer visible.
  (d) Fidelity bars Rabi/CTAP_eff/CTAP_full x {j_DW=11,7,5} + QEC line.
"""
import sys, os, csv
_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _PARENT)

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from ctap_full_chain import build_hoppings, build_H, protected_triplet
from thesis_style import apply_thesis_style, FULL_W, COLORS, panel_label
apply_thesis_style()

# ── Physical parameters ─────────────────────────────────────────────────────
L, w, v_max = 21, 1.0, 0.5
J_DW = 5;  FORM = 'b';  TAU = 180

# ── Colour aliases ───────────────────────────────────────────────────────────
C_SR  = COLORS['blue'];   C_LS  = COLORS['red']
C_PL  = COLORS['blue'];   C_PS  = COLORS['orange'];  C_PR  = COLORS['teal']
C_RABI= COLORS['red'];    C_EFF = COLORS['blue'];     C_FULL= COLORS['teal']

# ── Pulse helpers ────────────────────────────────────────────────────────────
def pulse_params(T):
    sig = 0.18 * T
    return sig, 0.5*T - sig, 0.5*T + sig   # sig, tR, tL

def vL_val(t, sig, tL):
    return v_max * np.exp(-(t - tL)**2 / (2*sig**2))

def vR_val(t, sig, tR):
    return v_max * np.exp(-(t - tR)**2 / (2*sig**2))

# ── Isolated-domain J_eff ────────────────────────────────────────────────────
def left_jeff(j_dw, v):
    if v < 1e-15:
        return 0.0
    h = np.array([v if j % 2 == 0 else w for j in range(j_dw)])
    E = np.linalg.eigvalsh(build_H(h))
    return float(np.max(np.abs(E[np.argsort(np.abs(E))[:2]])))

def right_jeff(j_dw, v):
    n_R = L - 1 - j_dw
    h = np.array([v if j % 2 == 0 else w for j in range(n_R)])
    E = np.linalg.eigvalsh(build_H(h))
    return float(np.max(np.abs(E[np.argsort(np.abs(E))[:2]])))

# ── Calibrate T for j_DW=5 ──────────────────────────────────────────────────
J_LS_pk = left_jeff(J_DW, v_max)
J_SR_pk = right_jeff(J_DW, v_max)
Jbott   = min(J_LS_pk, J_SR_pk)
T       = TAU / Jbott
sig, tR, tL = pulse_params(T)
print(f"j_DW={J_DW}: J_LS={J_LS_pk:.4e}  J_SR={J_SR_pk:.4e}  Jbott={Jbott:.4e}  T={T:.1f}")

# ── Full-chain evolution j_DW=5 ──────────────────────────────────────────────
N_OUT = 400

def Ht(t):
    vL = max(0., vL_val(t, sig, tL))
    vR = max(0., vR_val(t, sig, tR))
    return build_H(build_hoppings(L, J_DW, vL, vR, w))

psi0 = np.zeros(L, complex); psi0[0] = 1.

def rhs(t, y):
    psi_c = y[:L] + 1j*y[L:]
    dp    = -1j*(Ht(t) @ psi_c)
    return np.concatenate([dp.real, dp.imag])

y0  = np.concatenate([psi0.real, psi0.imag])
sol = solve_ivp(rhs, [0, T], y0, method='RK45',
                t_eval=np.linspace(0, T, N_OUT + 1),
                rtol=1e-6, atol=1e-8)

psis_raw = sol.y[:L].T + 1j*sol.y[L:].T
norms    = np.linalg.norm(psis_raw, axis=1)
print(f"  norma_final = {norms[-1]:.10f}")
psis = psis_raw / norms[:, np.newaxis]

Nt   = len(sol.t)
dens = np.abs(psis)**2                     # (Nt, L)
PL   = np.zeros(Nt); PS = np.zeros(Nt)
PR   = np.zeros(Nt); lk = np.zeros(Nt)

for k, (t_k, psi) in enumerate(zip(sol.t, psis)):
    tri = protected_triplet(Ht(t_k))
    pL_ = tri['L'] / np.linalg.norm(tri['L'])
    pS_ = tri['S'] / np.linalg.norm(tri['S'])
    pR_ = tri['R'] / np.linalg.norm(tri['R'])
    PL[k] = float(np.abs(pL_ @ psi)**2)
    PS[k] = float(np.abs(pS_ @ psi)**2)
    PR[k] = float(np.abs(pR_ @ psi)**2)
    lk[k] = max(1e-12, 1. - (PL[k] + PS[k] + PR[k]))

t_n = sol.t / T    # normalised time [0,1]
f_final = float(np.abs(psis[-1, L-1])**2)
print(f"  f_final={f_final:.6f}  maxP_S={PS.max():.4e}  maxLeak={lk.max():.2e}")

# ── Read paso3d_final.csv ────────────────────────────────────────────────────
fbar   = {}    # fbar[(j, proto)] = fidelity
T_ctap = {}    # T_ctap[j] = T_transfer (shared by CTAP_eff / CTAP_full)

with open(os.path.join(_HERE, 'paso3d_final.csv'), newline='') as fp:
    for row in csv.DictReader(fp):
        j = int(row['j_DW']); proto = row['protocolo']
        fbar[(j, proto)] = float(row['f'])
        if proto == 'CTAP_full':
            T_ctap[j] = float(row['T_transfer'])

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FULL_W, 4.80))
gs  = gridspec.GridSpec(2, 2, figure=fig,
                        left=0.09, right=0.97, top=0.92, bottom=0.11,
                        hspace=0.52, wspace=0.40)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# ────────────────────────────────────────────────────────── Panel (a): pulses
t_pl  = np.linspace(0, 1, 3000)
vR_pl = v_max * np.exp(-(t_pl - tR/T)**2 / (2*(sig/T)**2))
vL_pl = v_max * np.exp(-(t_pl - tL/T)**2 / (2*(sig/T)**2))

ax_a.plot(t_pl, vR_pl, color=C_SR, lw=1.4, label=r'$v_R(t)$ — Stokes (1st)')
ax_a.plot(t_pl, vL_pl, color=C_LS, lw=1.4, label=r'$v_L(t)$ — pump (2nd)')
ax_a.fill_between(t_pl, vR_pl, alpha=0.13, color=C_SR)
ax_a.fill_between(t_pl, vL_pl, alpha=0.13, color=C_LS)
for xv, col in ((tR/T, C_SR), (tL/T, C_LS)):
    ax_a.axvline(xv, lw=0.65, ls='--', color=col, alpha=0.7)
ax_a.text(tR/T,  v_max*1.06, r'$t_R$', ha='center', va='bottom', color=C_SR, fontsize=8)
ax_a.text(tL/T,  v_max*1.06, r'$t_L$', ha='center', va='bottom', color=C_LS, fontsize=8)
ax_a.set_xlim(0, 1);  ax_a.set_ylim(0, v_max*1.30)
ax_a.set_xticks([0, tR/T, 0.5, tL/T, 1.0])
ax_a.set_xticklabels(['0', '0.32', '0.5', '0.68', '1'])
ax_a.set_xlabel(r'$t/T$');  ax_a.set_ylabel(r'$v(t)$')
ax_a.legend(loc='upper right', fontsize=7, handlelength=1.2)
panel_label(ax_a, '(a)')

# ─────────────────────────────── Panel (b): populations + leakage inset
ax_b.plot(t_n, PL, color=C_PL, lw=1.2, label=r'$P_L$')
ax_b.plot(t_n, PS, color=C_PS, lw=1.2, label=r'$P_S$')
ax_b.plot(t_n, PR, color=C_PR, lw=1.2, label=r'$P_R$')
ax_b.set_xlim(0, 1);  ax_b.set_ylim(-0.05, 1.12)
ax_b.set_xlabel(r'$t/T$');  ax_b.set_ylabel(r'$P_\alpha$')
ax_b.legend(loc='center right', fontsize=8, handlelength=1.2)

# inset: leakage on log scale
axi = ax_b.inset_axes([0.04, 0.38, 0.39, 0.37])
axi.semilogy(t_n, lk, color=COLORS['grey'], lw=0.9)
axi.set_xlim(0, 1);  axi.set_ylim(1e-9, 1e-5)
axi.set_xlabel(r'$t/T$', fontsize=6.5);  axi.set_ylabel('leak.', fontsize=6.5)
axi.tick_params(labelsize=6)
axi.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
axi.yaxis.set_minor_locator(ticker.NullLocator())
# reference line at max leakage
axi.axhline(lk.max(), lw=0.6, ls=':', color='k', alpha=0.6)
axi.text(0.97, lk.max()*2.2, f'{lk.max():.0e}',
         ha='right', va='bottom', fontsize=5.5, transform=axi.get_yaxis_transform())
panel_label(ax_b, '(b)')

# ─────────────────────────────── Panel (c): heatmap site vs time
# dens shape (Nt, L) -> transpose to (L, Nt) for imshow y=site, x=time
im = ax_c.imshow(dens.T, aspect='auto', origin='lower',
                 extent=[0, 1, -0.5, L - 0.5],
                 cmap='YlOrRd', vmin=0, vmax=0.55)
# Domain-wall bond at j=5 (between sites 5 and 6)
ax_c.axhline(J_DW + 0.5, lw=0.65, ls='--', color='white', alpha=0.75)
ax_c.text(0.02, J_DW + 0.65, 'DW', color='white', fontsize=6.5, va='bottom')
ax_c.set_xlabel(r'$t/T$');  ax_c.set_ylabel('site $j$')
ax_c.set_yticks([0, J_DW, L - 1])
ax_c.set_yticklabels([r'$0$', rf'${J_DW}$', r'$20$'])
cb = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.03)
cb.set_label(r'$|\psi_j|^2$', fontsize=8)
cb.ax.tick_params(labelsize=7)
panel_label(ax_c, '(c)', loc='inside')

# ─────────────────────────────── Panel (d): fidelity bar chart
J_DWs_d  = [11, 7, 5]
protos_d  = ['Rabi', 'CTAP_eff', 'CTAP_full']
colors_d  = [C_RABI, C_EFF, C_FULL]
labels_d  = ['Rabi', r'CTAP$_\mathrm{eff}$', r'CTAP$_\mathrm{full}$']
bw        = 0.24
x_ctr     = np.array([0., 1.05, 2.10])
offsets   = np.array([-bw, 0., bw])

for ip, (proto, col, lbl) in enumerate(zip(protos_d, colors_d, labels_d)):
    fv = [fbar.get((j, proto), 0.) for j in J_DWs_d]
    ax_d.bar(x_ctr + offsets[ip], fv, width=bw*0.90,
             color=col, label=lbl, edgecolor='k', linewidth=0.35, alpha=0.88)

# QEC threshold
ax_d.axhline(0.995, color='k', lw=0.85, ls='--', zorder=3, label=r'$f_0{=}0.995$')

# Annotate T_transfer above CTAP_full bars (between eff and full in x)
for i, j in enumerate(J_DWs_d):
    Tv  = T_ctap.get(j, 0.)
    lbl_T = f'T={Tv/1e3:.1f}k'
    x_ann = x_ctr[i] + (offsets[1] + offsets[2]) / 2   # midpoint between eff/full
    y_ann = fbar.get((j, 'CTAP_full'), 1.0) + 0.005
    ax_d.text(x_ann, y_ann, lbl_T, ha='center', va='bottom',
              fontsize=5.5, color=COLORS['teal'], rotation=90)

ax_d.set_xticks(x_ctr)
ax_d.set_xticklabels([r'$j_\mathrm{DW}{=}11$', r'$j_\mathrm{DW}{=}7$',
                      r'$j_\mathrm{DW}{=}5$'], fontsize=8)
ax_d.set_ylabel(r'fidelity $f$')
ax_d.set_ylim(0, 1.22)
ax_d.legend(loc='upper left', fontsize=7, ncol=2, columnspacing=0.6,
            handlelength=1.0, borderpad=0.4)
panel_label(ax_d, '(d)')

# ── Save ─────────────────────────────────────────────────────────────────────
out = os.path.join(_HERE, 'fig_fullchain_ctap.pdf')
fig.savefig(out)
print(f"\nSaved: {out}")
plt.close(fig)

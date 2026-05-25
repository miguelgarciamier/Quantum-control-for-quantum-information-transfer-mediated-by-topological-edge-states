#!/usr/bin/env python3
"""
ctap_asymmetric.py  --  Empirical reinforcement of the CTAP asymmetry claim.
Generates a *simulated* fig15 (3 protocols) and a new fig16 (physical limit).
"""
import os, sys
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thesis_style import apply_thesis_style, COL_W, COLORS, panel_label
apply_thesis_style()

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.normpath(os.path.join(_HERE, '..', 'tesis', 'figures'))
os.makedirs(FIGURES_DIR, exist_ok=True)
V, W = 0.5, 1.0
L = 21

def rabi_fmax(rho):
    return 4.0 * rho**2 / (1.0 + rho**2)**2

def _pulses(t, T, sig, JLm, JSm):
    tR, tL = T / 3.0, 2.0 * T / 3.0
    JSR = JSm * np.exp(-(t - tR)**2 / (2 * sig**2))
    JLS = JLm * np.exp(-(t - tL)**2 / (2 * sig**2))
    dJSR = JSR * (-(t - tR) / sig**2)
    dJLS = JLS * (-(t - tL) / sig**2)
    return JLS, JSR, dJLS, dJSR

def _rhs(t, y, T, sig, JLm, JSm, use_cd):
    JLS, JSR, dJLS, dJSR = _pulses(t, T, sig, JLm, JSm)
    H = np.array([[0, JLS, 0], [JLS, 0, JSR], [0, JSR, 0]], dtype=complex)
    if use_cd:
        den = JLS**2 + JSR**2
        thd = (JSR * dJLS - JLS * dJSR) / den if den > 1e-30 else 0.0
        H[0, 2] += 1j * thd
        H[2, 0] += -1j * thd
    psi = y[:3] + 1j * y[3:]
    d = -1j * (H @ psi)
    return np.concatenate([d.real, d.imag])

def ctap_fidelity(JLm, JSm, T=600.0, sig_frac=0.18, use_cd=False):
    sig = sig_frac * T
    y0 = np.concatenate([np.array([1.0, 0.0, 0.0]), np.zeros(3)])
    sol = solve_ivp(_rhs, (0.0, T), y0, args=(T, sig, JLm, JSm, use_cd),
                    method='DOP853', rtol=1e-11, atol=1e-13,
                    max_step=T / 4000.0, t_eval=[T])
    if not sol.success:
        raise RuntimeError(sol.message)
    psi = sol.y[:3, -1] + 1j * sol.y[3:, -1]
    return float(np.abs(psi[2])**2)

def peak_amplitudes(rho, l1, l2):
    big, small = 1.0, 1.0 / rho
    return (big, small) if l1 <= l2 else (small, big)

def plot_fig15():
    j_arr = np.arange(1, L - 1, 2)
    f_rabi, f_ctap, f_cd = [], [], []
    for j in j_arr:
        l1, l2 = j, (L - 1) - j
        rho = (W / V)**(abs(l1 - l2) / 2.0)
        f_rabi.append(rabi_fmax((V / W)**((l1 - l2) / 2.0)))
        JLm, JSm = peak_amplitudes(rho, l1, l2)
        f_ctap.append(ctap_fidelity(JLm, JSm, T=1500.0, use_cd=False))
        f_cd.append(ctap_fidelity(JLm, JSm, T=600.0, use_cd=True))
    f_rabi = np.array(f_rabi); f_ctap = np.array(f_ctap); f_cd = np.array(f_cd)
    fig, ax = plt.subplots(figsize=(COL_W, 0.80 * COL_W))
    ax.plot(j_arr, f_rabi, 'o-', color=COLORS['red'], label='Static Rabi (bound)')
    ax.plot(j_arr, f_ctap, 's-', color=COLORS['blue'], label='Bare CTAP')
    ax.plot(j_arr, f_cd, 'D--', color=COLORS['teal'], label='CD-assisted CTAP')
    ax.axhline(0.995, color=COLORS['grey'], lw=0.8, ls=':')
    ax.text(L - 2, 0.995, ' QEC', va='bottom', ha='right', fontsize=7, color='0.4')
    ax.set_xlabel(r'Domain-wall position $j_{\mathrm{DW}}$')
    ax.set_ylabel(r'Transfer fidelity $f$')
    ax.set_ylim(-0.04, 1.10); ax.set_xticks(j_arr)
    ax.legend(loc='lower center', fontsize=7.5); ax.grid(True)
    fig.savefig(os.path.join(FIGURES_DIR, 'fig15_ctap_comparison.pdf')); plt.close(fig)

def plot_fig16():
    rhos = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=float)
    fb300, fb1500, fcd = [], [], []
    for rho in rhos:
        JLm, JSm = 1.0, 1.0 / rho
        fb300.append(ctap_fidelity(JLm, JSm, T=300.0, use_cd=False))
        fb1500.append(ctap_fidelity(JLm, JSm, T=1500.0, use_cd=False))
        fcd.append(ctap_fidelity(JLm, JSm, T=300.0, use_cd=True))
    fig, ax = plt.subplots(figsize=(COL_W, 0.80 * COL_W))
    ax.semilogx(rhos, fb300, 'o-', color=COLORS['orange'], label=r'Bare CTAP, $T=300$')
    ax.semilogx(rhos, fb1500, 's-', color=COLORS['blue'], label=r'Bare CTAP, $T=1500$')
    ax.semilogx(rhos, fcd, 'D--', color=COLORS['teal'], label=r'CD-assisted, $T=300$')
    ax.axhline(0.995, color=COLORS['grey'], lw=0.8, ls=':')
    ax.set_xlabel(r'Peak-coupling ratio $\rho = J_{\max}^{L}/J_{\max}^{R}$')
    ax.set_ylabel(r'Transfer fidelity $f$')
    ax.set_ylim(-0.04, 1.10); ax.set_xticks(rhos)
    ax.set_xticklabels([f'{int(r)}' for r in rhos])
    ax.legend(loc='lower left'); ax.grid(True, which='both')
    fig.savefig(os.path.join(FIGURES_DIR, 'fig16_ctap_asymmetry.pdf')); plt.close(fig)

if __name__ == '__main__':
    print('Generating simulated CTAP asymmetry figures ...')
    plot_fig15(); plot_fig16(); print('Done.')

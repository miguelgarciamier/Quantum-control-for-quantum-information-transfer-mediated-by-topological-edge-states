#!/usr/bin/env python3
"""
fig15_wc_ctap.py  --  fig15 redesign with well-conditioned (WC) pulse.

f vs j_DW, L=21, j_DW in {1,3,...,19}.
  - Rabi: analytic 4*rho^2/(1+rho^2)^2
  - Bare CTAP: WC pulse (T/4, 3T/4, sig=0.12T), T=400*rho per position
  - CD-assisted CTAP: same WC pulse, T=600 (CD cancels non-adiabatic coupling)
  rho(j) = (W/V)^|j-10| = 2^|j-10|
Shading at |j-10|>=7 (rho>=128): 3-level model validity limit.
"""
import os, sys
_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _PARENT)

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from thesis_style import apply_thesis_style, COL_W, COLORS
from ctap_asymmetric import peak_amplitudes, rabi_fmax

apply_thesis_style()

FIGURES_DIR = os.path.normpath(os.path.join(_HERE, '..', 'tesis', 'figures'))
os.makedirs(FIGURES_DIR, exist_ok=True)

V, W, L = 0.5, 1.0, 21

# ── WC pulse ODE ──────────────────────────────────────────────────────────────
def _rhs_wc(t, y, T, sig, JLm, JSm, use_cd):
    tR, tL  = T / 4.0, 3.0 * T / 4.0
    JSR     = JSm * np.exp(-(t - tR)**2 / (2 * sig**2))
    JLS     = JLm * np.exp(-(t - tL)**2 / (2 * sig**2))
    dJSR    = JSR * (-(t - tR) / sig**2)
    dJLS    = JLS * (-(t - tL) / sig**2)
    H = np.array([[0, JLS, 0], [JLS, 0, JSR], [0, JSR, 0]], dtype=complex)
    if use_cd:
        den = JLS**2 + JSR**2
        thd = (JSR * dJLS - JLS * dJSR) / den if den > 1e-30 else 0.0
        H[0, 2] += 1j * thd
        H[2, 0] += -1j * thd
    psi = y[:3] + 1j * y[3:]
    d   = -1j * (H @ psi)
    return np.concatenate([d.real, d.imag])

def ctap_fidelity_wc(JLm, JSm, T, use_cd=False):
    sig = 0.12 * T
    y0  = np.concatenate([[1., 0., 0.], np.zeros(3)])
    sol = solve_ivp(_rhs_wc, (0., T), y0, args=(T, sig, JLm, JSm, use_cd),
                    method='DOP853', rtol=1e-11, atol=1e-13,
                    max_step=T / 4000., t_eval=[T])
    if not sol.success:
        raise RuntimeError(sol.message)
    psi = sol.y[:3, -1] + 1j * sol.y[3:, -1]
    return float(np.abs(psi[2])**2)

# ── Protocol timescales ───────────────────────────────────────────────────────
# WC pulse: T_ad(rho) = 17.36 / (sqrt(2)*J_c(rho)), J_c(rho) = crossing coupling.
# J_c does NOT scale as 1/rho: for rho=2 T_ad~154, for rho=128 T_ad~2407.
# T_0=800 gives T/T_ad >= 10 at rho=2 (j=9,11) and >> 10 for larger rho.
def T_bare(rho):
    return 800. * float(rho)

T_CD = 600.   # CD exact in 3-level model -> any T works; 600 comfortable

# ── Position array and rho ────────────────────────────────────────────────────
j_arr = np.arange(1, L - 1, 2, dtype=int)   # {1,3,5,7,9,11,13,15,17,19}

def rho_at_j(j):
    l1, l2 = int(j), (L - 1) - int(j)
    return (W / V) ** (abs(l1 - l2) / 2.0)

# ── Convergence check ─────────────────────────────────────────────────────────
print("=== Convergence check bare WC CTAP (T -> 2T) ===")
for jc in [9, 5, 3]:   # rho=2,32,128; rho=2 is most constraining for small T
    rho = rho_at_j(jc)
    T0  = T_bare(rho)
    l1, l2 = jc, (L - 1) - jc
    JLm, JSm = peak_amplitudes(rho, l1, l2)
    f1 = ctap_fidelity_wc(JLm, JSm, T0,     use_cd=False)
    f2 = ctap_fidelity_wc(JLm, JSm, 2.*T0,  use_cd=False)
    print(f"  j={jc}  rho={int(rho):3d}  T={int(T0):6d}  "
          f"f(T)={f1:.5f}  f(2T)={f2:.5f}  |df|={abs(f2-f1):.1e}  "
          f"{'OK' if abs(f2-f1)<0.005 else 'NOT CONV'}")
print()

# ── Full scan ─────────────────────────────────────────────────────────────────
f_rabi = []
f_bare = []
f_cd   = []

print(f"  {'j':>2}  {'rho':>6}  {'Rabi':>8}  {'BareWC':>8}  {'CD_WC':>8}")
print("  " + "-" * 42)
for j in j_arr:
    l1, l2   = int(j), (L - 1) - int(j)
    rho      = rho_at_j(j)
    JLm, JSm = peak_amplitudes(rho, l1, l2)

    fr = rabi_fmax(rho)
    fb = ctap_fidelity_wc(JLm, JSm, T_bare(rho), use_cd=False)
    fc = ctap_fidelity_wc(JLm, JSm, T_CD,         use_cd=True)

    f_rabi.append(fr)
    f_bare.append(fb)
    f_cd.append(fc)
    print(f"  {j:>2}  {rho:>6.0f}  {fr:>8.4f}  {fb:>8.4f}  {fc:>8.4f}")

f_rabi = np.array(f_rabi)
f_bare = np.array(f_bare)
f_cd   = np.array(f_cd)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(COL_W, 0.82 * COL_W))
C = COLORS

# Shading: 3-level model breaks for |j-10|>=7 (rho>=128)
ax.axvspan(0.,    3.5, alpha=0.09, color=C['grey'], zorder=0, lw=0)
ax.axvspan(16.5, 20.,  alpha=0.09, color=C['grey'], zorder=0, lw=0)
for xlab, ha in [(2.0, 'center'), (18.0, 'center')]:
    ax.text(xlab, 0.14, r'$\rho\!>\!100$' + '\n3-level\nbreaks',
            fontsize=5.5, color='0.50', ha=ha, va='bottom', linespacing=1.3)

# QEC threshold
ax.axhline(0.995, color='0.55', lw=0.7, ls=':', zorder=2)
ax.text(19., 0.998, r'$f_0\!=\!0.995$', fontsize=6, color='0.45',
        va='bottom', ha='right')

# Three curves
ax.plot(j_arr, f_rabi, 'o-',  color=C['red'],   ms=4.0, lw=0.9, zorder=4,
        markerfacecolor='white', markeredgewidth=1.1,
        label=r'Rabi  $4\rho^2/(1+\rho^2)^2$')
ax.plot(j_arr, f_bare, 's-',  color=C['blue'],  ms=4.0, lw=0.9, zorder=4,
        markerfacecolor='white', markeredgewidth=1.1,
        label=r'Bare CTAP WC  ($T/4$--$3T/4$, $\sigma\!=\!0.12T$, $T\!=\!800\rho$)')
ax.plot(j_arr, f_cd,   'D--', color=C['teal'],  ms=4.0, lw=0.9, zorder=4,
        markerfacecolor='white', markeredgewidth=1.1,
        label=r'CD-assisted WC  ($T\!=\!600$)')

ax.set_xlabel(r'Domain-wall position $j_{\mathrm{DW}}$')
ax.set_ylabel(r'Transfer fidelity $f$')
ax.set_xlim(0., 20.)
ax.set_ylim(-0.04, 1.10)
ax.set_xticks(j_arr)
ax.legend(loc='lower center', fontsize=6.5, borderpad=0.4,
          handlelength=1.5, handletextpad=0.4, labelspacing=0.3)

fig.tight_layout(pad=0.5)
out = os.path.join(FIGURES_DIR, 'fig15_ctap_comparison.pdf')
fig.savefig(out)
plt.close(fig)
print(f"\nSaved: {out}")

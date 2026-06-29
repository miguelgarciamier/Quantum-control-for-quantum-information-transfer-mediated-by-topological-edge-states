#!/usr/bin/env python3
"""
fig16_ctap_asymmetry_new.py  --  Redesign of fig16: proof of the f ≤ cos²Θ₀ law.

Three elements:
  1. Analytic ceiling  cos²Θ₀(ρ)  from original pulse evaluated at t=0.
  2. Bare CTAP, original pulse (σ=0.18T, tR=T/3 first, tL=2T/3), T_orig=400*rho.
     Should embrace the ceiling for all ρ.
  3. WC CTAP: tR=T/4, tL=3T/4, σ=0.12T, T_WC=50000.
     Should stay ≈1 for all ρ.

Shading at ρ > 100: 3-level model validity limit.
QEC line at f₀ = 0.995.
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
from ctap_asymmetric import ctap_fidelity    # reuses _rhs with tR=T/3, tL=2T/3

apply_thesis_style()

FIGURES_DIR = os.path.normpath(os.path.join(_HERE, '..', 'tesis', 'figures'))
os.makedirs(FIGURES_DIR, exist_ok=True)

rhos = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=float)

# ── 1. Analytic ceiling cos²Θ₀(ρ) ────────────────────────────────────────────
# Original pulse: tR/T = 1/3 (SR first), tL/T = 2/3, σ/T = 0.18
# JLm = 1, JSm = 1/ρ → at t=0:
#   JSR(0) = (1/ρ) · exp(−tR²/2σ²)   ≈ 0.1803 / ρ
#   JLS(0) = 1     · exp(−tL²/2σ²)   ≈ 0.001047
# Dark state at t=0: |D⟩ = (JSR|L⟩ − JLS|R⟩)/norm  →  ⟨L|D⟩ = JSR/norm
# Ceiling = cos²Θ₀ = JSR(0)² / (JSR(0)² + JLS(0)²)
_A_SR = np.exp(-(1./3)**2 / (2*0.18**2))    # ≈ 0.1803
_A_LS = np.exp(-(2./3)**2 / (2*0.18**2))    # ≈ 0.001047

def cos2theta0(rho):
    rho  = np.asarray(rho, float)
    JSR0 = _A_SR / rho
    JLS0 = _A_LS
    return JSR0**2 / (JSR0**2 + JLS0**2)

# ── 3. WC CTAP (well-conditioned): tR=T/4, tL=3T/4, σ=0.12T ─────────────────
# Larger separation → JLS(0) = exp(−(3/4)²/(2·0.12²)) ≈ 3.3e−9 → Θ₀ ≈ 0 ∀ρ

def _rhs_wc(t, y, T, JLm, JSm):
    sig = 0.12 * T;  tR = T / 4.;  tL = 3. * T / 4.
    JSR = JSm * np.exp(-(t - tR)**2 / (2*sig**2))
    JLS = JLm * np.exp(-(t - tL)**2 / (2*sig**2))
    H   = np.array([[0, JLS, 0], [JLS, 0, JSR], [0, JSR, 0]], dtype=complex)
    psi = y[:3] + 1j * y[3:]
    d   = -1j * (H @ psi)
    return np.concatenate([d.real, d.imag])

def ctap_fidelity_wc(JLm, JSm, T):
    y0  = np.concatenate([[1., 0., 0.], np.zeros(3)])
    sol = solve_ivp(_rhs_wc, (0., T), y0, args=(T, JLm, JSm),
                    method='DOP853', rtol=1e-11, atol=1e-13,
                    max_step=T / 4000., t_eval=[T])
    if not sol.success:
        raise RuntimeError(sol.message)
    psi = sol.y[:3, -1] + 1j * sol.y[3:, -1]
    return float(np.abs(psi[2])**2)

# ── Protocol timescales ───────────────────────────────────────────────────────
# Original pulse: T_ad ~ O(rho) because min gap at t=0 is O(1/rho) and
# dTheta/dt ~ (JLS/JSR)*(slope/T) ~ (rho * const)/T -> need T >> rho*const.
# T_orig(rho)=400*rho gives T/T_ad >> 10 for all rho<=128 (verified below).
def T_orig(rho):
    return 400. * float(rho)

T_WC = 50000.    # fixed — WC pulse gap at crossing ~0.007, T_ad~2400 for all rho

# ── Convergence check at rho = 32 and 128 ────────────────────────────────────
print("=== Convergence check (f stable under T -> 2T) ===")
for rho in [32, 128]:
    T0  = T_orig(rho)
    c   = float(cos2theta0(rho))
    f1  = ctap_fidelity(1., 1./rho, T=T0,   use_cd=False)
    f2  = ctap_fidelity(1., 1./rho, T=2*T0, use_cd=False)
    fw1 = ctap_fidelity_wc(1., 1./rho, T_WC)
    fw2 = ctap_fidelity_wc(1., 1./rho, 2*T_WC)
    print(f"  rho={int(rho):3d}  orig   T={int(T0)} f={f1:.5f}  2T f={f2:.5f}  "
          f"|df|={abs(f2-f1):.1e}   cos2Th0={c:.5f}")
    print(f"  rho={int(rho):3d}  wc   T={int(T_WC)} f={fw1:.5f}  2T f={fw2:.5f}  "
          f"|df|={abs(fw2-fw1):.1e}")
print()

# ── Full scan ─────────────────────────────────────────────────────────────────
print("Scanning original CTAP (T=400*rho) ...")
f_orig = np.array([ctap_fidelity(1., 1./r, T=T_orig(r), use_cd=False) for r in rhos])

print(f"Scanning WC CTAP (T={int(T_WC)}) ...")
f_wc   = np.array([ctap_fidelity_wc(1., 1./r, T_WC) for r in rhos])

ceil_vals = cos2theta0(rhos)

# ── Key verification ──────────────────────────────────────────────────────────
print("\n=== Verification ===")
for rho in [32, 128]:
    idx = int(np.where(rhos == rho)[0][0])
    fo  = f_orig[idx];  fw = f_wc[idx];  c = ceil_vals[idx]
    ok_orig = abs(fo - c) < 0.01
    ok_wc   = fw > 0.99
    print(f"  rho={int(rho):3d}  f_orig={fo:.4f}  cos2Th0={c:.4f}  "
          f"|diff|={abs(fo-c):.4f} {'OK' if ok_orig else 'FAIL'}   "
          f"f_wc={fw:.4f} {'~1 OK' if ok_wc else 'FAIL'}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(COL_W, 0.82 * COL_W))
C = COLORS

ax.set_xscale('log')

# Shading: 3-level model validity limit
ax.axvspan(100., 200., alpha=0.09, color=C['grey'], zorder=0, lw=0)
ax.axvline(100., color='0.55', lw=0.8, ls='--', zorder=1)
ax.text(108., 0.15, '3-level\nmodel\nbreaks', fontsize=5.5,
        color='0.50', va='bottom', ha='left', linespacing=1.3)

# QEC threshold
ax.axhline(0.995, color='0.55', lw=0.7, ls=':', zorder=2)
ax.text(128., 0.998, r'$f_0\!=\!0.995$', fontsize=6, color='0.45',
        va='bottom', ha='right')

# 1. Ceiling cos²Θ₀ — dashed over dense grid
rho_dense = np.logspace(0, np.log10(128), 400)
ax.plot(rho_dense, cos2theta0(rho_dense),
        '--', color=C['orange'], lw=1.3, zorder=3,
        label=r'$\cos^2\Theta_0(\rho)$  (ceiling)')

# 2. Bare CTAP, original pulse
ax.plot(rhos, f_orig, 'o-', color=C['blue'], ms=4.5, lw=0.9, zorder=4,
        markerfacecolor='white', markeredgewidth=1.1,
        label=r'Bare CTAP, $T\!=\!400\rho$ (orig.\ pulse)')

# 3. WC CTAP
ax.plot(rhos, f_wc, 's-', color=C['teal'], ms=4.5, lw=0.9, zorder=4,
        markerfacecolor='white', markeredgewidth=1.1,
        label=r'WC: $T/4$–$3T/4$,  $\sigma\!=\!0.12T$')

ax.set_xlabel(r'Coupling ratio $\rho = J_{\mathcal{LS}}^{\max}/J_{\mathcal{SR}}^{\max}$')
ax.set_ylabel(r'Transfer fidelity $f$')
ax.set_xlim(0.75, 165.)
ax.set_ylim(-0.04, 1.10)
ax.set_xticks(rhos)
ax.set_xticklabels([str(int(r)) for r in rhos])
ax.legend(loc='lower left', fontsize=7, borderpad=0.4,
          handlelength=1.6, handletextpad=0.4, labelspacing=0.3)

fig.tight_layout(pad=0.5)
out = os.path.join(FIGURES_DIR, 'fig16_ctap_asymmetry.pdf')
fig.savefig(out)
plt.close(fig)
print(f"\nSaved: {out}")

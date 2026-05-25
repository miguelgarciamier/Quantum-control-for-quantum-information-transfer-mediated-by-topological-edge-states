#!/usr/bin/env python3
"""
ctap_figures.py  —  Publication-quality figures for Section VIII (CTAP) of the SSH thesis.

Generates three PDFs saved to tesis/figures/:
  fig13_ctap_pulses.pdf    —  Figure 1: Counter-intuitive CTAP pulse sequence
  fig14_ctap_dynamics.pdf  —  Figure 2: 3-level CTAP population dynamics
  fig15_ctap_comparison.pdf —  Figure 3: Rabi vs CTAP fidelity vs DW position
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from thesis_style import apply_thesis_style, COL_W, COLORS
apply_thesis_style()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.normpath(os.path.join(_HERE, '..', 'tesis', 'figures'))
os.makedirs(FIGURES_DIR, exist_ok=True)

# Colour aliases (mapped to thesis_style palette)
C_SR = COLORS['blue']; C_LS = COLORS['red']; C_S = COLORS['orange']
C_RABI = COLORS['red']; C_CTAP = COLORS['blue']


# ===========================================================================
# FIGURE 1 — Counter-intuitive pulse sequence
# ===========================================================================

def make_ctap_pulses(T=300.0, sigma_frac=0.18, J_max=1.0):
    """Return t, J_SR(t), J_LS(t) for counter-intuitive Gaussian pulses."""
    sigma = sigma_frac * T        # σ  —  width of each Gaussian
    t_R   = T / 3.0               # right (SR) pulse peaks first
    t_L   = 2.0 * T / 3.0        # left  (LS) pulse peaks second
    t     = np.linspace(0.0, T, 3000)
    J_SR  = J_max * np.exp(-(t - t_R) ** 2 / (2.0 * sigma ** 2))
    J_LS  = J_max * np.exp(-(t - t_L) ** 2 / (2.0 * sigma ** 2))
    return t, J_SR, J_LS, t_R, t_L, sigma


def plot_ctap_pulses():
    T    = 300.0
    t, J_SR, J_LS, t_R, t_L, _ = make_ctap_pulses(T=T)

    fig, ax = plt.subplots(figsize=(COL_W, 2.19))

    ax.plot(t, J_SR, color=C_SR, lw=2.0,
            label=r'$J_{\mathcal{SR}}(t)$ — right (fires first)')
    ax.plot(t, J_LS, color=C_LS, lw=2.0,
            label=r'$J_{\mathcal{LS}}(t)$ — left (fires second)')
    ax.fill_between(t, J_SR, alpha=0.15, color=C_SR)
    ax.fill_between(t, J_LS, alpha=0.15, color=C_LS)

    # Dashed vertical lines at peak positions
    for x_peak, col in [(t_R, C_SR), (t_L, C_LS)]:
        ax.axvline(x_peak, lw=0.8, ls='--', color=col, alpha=0.65)

    # Labels directly on the x-axis ticks for T/3 and 2T/3
    ax.set_xticks([0, t_R, T / 2, t_L, T])
    ax.set_xticklabels([r'$0$', r'$T/3$', r'$T/2$', r'$2T/3$', r'$T$'])
    # Color the two special tick labels
    for tick, col in zip(ax.get_xticklabels(), ['k', C_SR, 'k', C_LS, 'k']):
        tick.set_color(col)

    # Counter-intuitive ordering: label each peak individually (avoids
    # spanning annotation that would fight with the legend)
    ax.text(t_R, 1.03, r'$\mathbf{1^{st}}$', ha='center', va='bottom',
            color=C_SR, fontsize=9, fontweight='bold')
    ax.text(t_L, 1.03, r'$\mathbf{2^{nd}}$', ha='center', va='bottom',
            color=C_LS, fontsize=9, fontweight='bold')
    ax.text(T / 2, 1.12, r'counter-intuitive ordering',
            ha='center', va='bottom', color='dimgray', fontsize=8)

    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.24)
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel('Coupling amplitude')
    # Legend at bottom centre — frees the top area for the annotation
    ax.legend(loc='lower center', ncol=2, framealpha=0.92,
              bbox_to_anchor=(0.5, 0.01))
    ax.grid(alpha=0.18, ls=':')

    fig.tight_layout(pad=0.6)
    out = os.path.join(FIGURES_DIR, 'fig13_ctap_pulses.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved  {out}')


# ===========================================================================
# FIGURE 2 — 3-level CTAP population dynamics
# ===========================================================================

def schrodinger_rhs(t, y, T, sigma_frac, J_LS_max, J_SR_max):
    """
    Schrödinger equation  d/dt ψ = −i H(t) ψ  for the effective 3-level system.
    State vector packed as real vector y = [Re(ψ), Im(ψ)].
    """
    sigma = sigma_frac * T
    t_R   = T / 3.0
    t_L   = 2.0 * T / 3.0

    JSR = J_SR_max * np.exp(-(t - t_R) ** 2 / (2.0 * sigma ** 2))
    JLS = J_LS_max * np.exp(-(t - t_L) ** 2 / (2.0 * sigma ** 2))

    # H in the {|L>, |S>, |R>} basis
    H = np.array([[0.0, JLS, 0.0],
                  [JLS, 0.0, JSR],
                  [0.0, JSR, 0.0]])

    psi       = y[:3] + 1j * y[3:]
    dpsi_dt   = -1j * (H @ psi)
    return np.concatenate([dpsi_dt.real, dpsi_dt.imag])


def simulate_ctap(T=300.0, sigma_frac=0.18, J_LS_max=1.0, J_SR_max=1.0):
    """
    Solve the 3-level Schrödinger equation with counter-intuitive CTAP pulses.
    Returns (t_arr, P_L, P_S, P_R) — populations of each level.
    """
    psi0    = np.array([1.0, 0.0, 0.0], dtype=complex)
    y0      = np.concatenate([psi0.real, psi0.imag])
    t_eval  = np.linspace(0.0, T, 4000)

    sol = solve_ivp(
        schrodinger_rhs,
        (0.0, T),
        y0,
        args=(T, sigma_frac, J_LS_max, J_SR_max),
        t_eval=t_eval,
        method='DOP853',
        max_step=T / 2000,
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f'ODE solver failed: {sol.message}')

    psi = sol.y[:3] + 1j * sol.y[3:]
    P   = np.abs(psi) ** 2
    return sol.t, P[0], P[1], P[2]          # t, P_L, P_S, P_R


def plot_ctap_dynamics():
    T          = 300.0
    sigma_frac = 0.18   # σ = 54  →  adiabaticity ratio ≈ 70
    t_arr, P_L, P_S, P_R = simulate_ctap(T=T, sigma_frac=sigma_frac)

    f_final = P_R[-1]
    pS_max  = P_S.max()
    print(f'  CTAP final: P_L={P_L[-1]:.6f}  P_S={P_S[-1]:.6f}  P_R={f_final:.6f}')
    print(f'  max P_S = {pS_max:.2e}')

    # --- main figure with two-panel layout: populations + P_S zoom ---
    fig = plt.figure(figsize=(COL_W, 3.17))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1.0], hspace=0.10)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    # ---- Top panel: all populations ----
    ax_top.plot(t_arr, P_L, color=C_LS, lw=2.0)
    ax_top.plot(t_arr, P_R, color=C_SR, lw=2.0)
    ax_top.plot(t_arr, P_S, color=C_S,  lw=1.6, ls='--')

    _wb = dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.88)

    # Direct curve labels — white background box prevents line-through overlap
    ax_top.text(0.015, 0.96, r'$P_{\mathcal{L}}$', transform=ax_top.transAxes,
                ha='left', va='top', color=C_LS, fontsize=11, fontweight='bold',
                bbox=_wb)
    ax_top.text(0.985, 0.96, r'$P_{\mathcal{R}}$', transform=ax_top.transAxes,
                ha='right', va='top', color=C_SR, fontsize=11, fontweight='bold',
                bbox=_wb)
    ax_top.text(0.985, 0.06, r'$P_{\mathcal{S}} \approx 0$  (dark state)',
                transform=ax_top.transAxes, ha='right', va='bottom',
                color=C_S, fontsize=8.5, bbox=_wb)

    # Final fidelity: use 5 decimal places to show it is NOT exactly 1
    ax_top.text(0.97, 0.50, rf'$f = {f_final:.5f}$',
                transform=ax_top.transAxes, ha='right', va='center',
                fontsize=9, color=C_SR,
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec=C_SR, lw=0.7, alpha=0.88))

    ax_top.set_xlim(0.0, T)
    ax_top.set_ylim(-0.03, 1.07)
    ax_top.set_ylabel('Population')
    ax_top.grid(alpha=0.18, ls=':')
    ax_top.tick_params(labelbottom=False)

    # ---- Bottom panel: P_S on log scale to show "stays near zero" ----
    ax_bot.semilogy(t_arr, np.maximum(P_S, 1e-14), color=C_S, lw=1.6)
    ax_bot.set_ylim(1e-7, 5e-2)
    ax_bot.set_xlabel(r'Time $t$')
    ax_bot.set_ylabel(r'$P_{\mathcal{S}}$  (log)', color=C_S, fontsize=9)
    ax_bot.tick_params(axis='y', labelcolor=C_S, labelsize=8)
    ax_bot.grid(alpha=0.18, ls=':')
    # Label above the P_S oscillations (max P_S ≈ 6e-4)
    ax_bot.text(T / 2, 2e-3,
                r'$P_{\mathcal{S}} \ll 1$  —  dark state never populated',
                ha='center', va='bottom', color=C_S, fontsize=7.5)

    fig.tight_layout(pad=0.5)
    out = os.path.join(FIGURES_DIR, 'fig14_ctap_dynamics.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved  {out}')


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    print('Generating CTAP figures …')
    print('Figure 1: counter-intuitive pulse sequence')
    plot_ctap_pulses()
    print('Figure 2: 3-level population dynamics')
    plot_ctap_dynamics()
    print('Done.')

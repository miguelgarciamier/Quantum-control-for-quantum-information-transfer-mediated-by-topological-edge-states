"""
SSH Dimer Chain: Topological phases, domain walls and transfer protocols
========================================================================

Experiments:
  1. Spectrum of a 10-atom (5-cell) SSH dimer chain vs v/w → phase transition
  2. Single domain wall: boundary states in a half-topo / half-trivial chain
  3. Transfer protocol (Fig 2 style) for N=2 domains with ℓ=4

Based on:
  "Fast quantum transfer mediated by topological domain walls"
  Zurita, Creffield, Platero — Quantum 7, 1043 (2023)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import expm
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thesis_style import apply_thesis_style, COL_W, FULL_W, COLORS, panel_label
apply_thesis_style()

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_SCRIPT_DIR, '..', 'tesis', 'figures')

# =============================================================================
# 1. SSH DIMER CHAIN — HAMILTONIAN
# =============================================================================

def build_ssh_dimer(L_sites, v, w):
    """
    Build the Hamiltonian of a finite SSH chain with L_sites sites.
    
    The SSH model is a 1D chain with alternating hopping amplitudes:
      site 0 —v— site 1 —w— site 2 —v— site 3 —w— ...
    
    For an EVEN number of sites (say 2M), there are M unit cells,
    each containing sublattice a (even sites) and b (odd sites).
    The intracell hopping is v, the intercell hopping is w.
    
    Topological phase: v < w  (winding number ν = 1)
    Trivial phase:     v > w  (winding number ν = 0)
    
    In the topological phase with even L, NO edge states appear because
    the chain has complete unit cells (both ends on sublattice a).
    With ODD L (incomplete last cell), an edge state appears at E=0.
    
    For even L in the topological phase: a gap opens at E=0 with no
    states inside it. The gap closes at v=w (phase transition).
    """
    H = np.zeros((L_sites, L_sites))
    for j in range(L_sites - 1):
        t = v if j % 2 == 0 else w
        H[j, j+1] = -t
        H[j+1, j] = -t
    return H


# =============================================================================
# 2. SPECTRUM VS v/w RATIO — TOPOLOGICAL PHASE DIAGRAM
# =============================================================================

def compute_spectrum_vs_ratio(L_sites, w=1.0, n_points=200):
    """
    Compute all eigenvalues of the SSH chain as a function of v/w.
    
    For a 10-site (5-cell) even chain:
    - When v/w < 1 (topological phase): gap open, no zero modes
      (even chain = complete unit cells, no edge states)
    - When v/w = 1: gap closes (phase transition point)
    - When v/w > 1 (trivial phase): gap opens again
    
    The bulk gap is Δ = 2|w - v| (minimum energy difference at k=π).
    """
    ratios = np.linspace(0.01, 2.5, n_points)
    all_spectra = np.zeros((n_points, L_sites))
    
    for i, ratio in enumerate(ratios):
        v = ratio * w
        H = build_ssh_dimer(L_sites, v, w)
        eigenvalues = np.linalg.eigvalsh(H)
        all_spectra[i] = eigenvalues
    
    return ratios, all_spectra


# =============================================================================
# 3. MULTIDOMAIN SSH CHAIN — HAMILTONIAN WITH DOMAIN WALL
# =============================================================================

def build_ssh_multidomain(N_domains, ell, v_values, w=1.0):
    """
    Build Hamiltonian for multidomain SSH chain.
    
    L = N*(ℓ+1)+1 sites. Bond j has hopping v if (j mod (ℓ+1)) is even,
    w if odd. Domain walls appear where two consecutive v-bonds meet.
    """
    L = N_domains * (ell + 1) + 1
    if np.isscalar(v_values):
        v_vals = np.full(N_domains, v_values)
    else:
        v_vals = np.array(v_values)
    
    H = np.zeros((L, L))
    for j in range(L - 1):
        d = min(j // (ell + 1), N_domains - 1)
        p = j % (ell + 1)
        t = v_vals[d] if p % 2 == 0 else w
        H[j, j+1] = -t
        H[j+1, j] = -t
    return H, L


def analytical_boundary_states(N_domains, ell, v, w=1.0):
    """
    Analytical boundary states (Eqs. 2-4 of paper).
    |L⟩ on sublattice a (even sites), |S_k⟩ on sublattice b (odd sites),
    |R⟩ on sublattice a (even sites). All decay with ratio r=v/w.
    """
    L = N_domains * (ell + 1) + 1
    r = v / w
    states = {}
    
    # |L⟩
    s = np.zeros(L)
    for n in range(ell // 2 + 1):
        s[2*n] = (-r)**n
    s /= np.linalg.norm(s)
    states['L'] = s
    
    # |R⟩
    s = np.zeros(L)
    last = L - 1
    for n in range(ell // 2 + 1):
        site = last - 2*n
        if site >= 0:
            s[site] = (-r)**n
    s /= np.linalg.norm(s)
    states['R'] = s
    
    # |S_k⟩
    for k in range(1, N_domains):
        s = np.zeros(L)
        wall = k * (ell + 1)
        parity = wall % 2
        s[wall] = 1.0
        for n in range(1, ell // 2 + 1):
            for site in [wall - 2*n, wall + 2*n]:
                if 0 <= site < L and site % 2 == parity:
                    s[site] = (-r)**n
        s /= np.linalg.norm(s)
        states[f'S{k}'] = s
    
    return states


# =============================================================================
# 4. TRANSFER PROTOCOL — TIME EVOLUTION (Fig 2 style)
# =============================================================================

def v_pulse(t, v_tr, t_tr, t_prep):
    """
    Control pulse v(t) from Eq. (11):
      - Ramp up:   v_tr * sin²(Ω*t)             for 0 ≤ t < t_prep
      - Constant:  v_tr                           for t_prep ≤ t < t_tr - t_prep
      - Ramp down: v_tr * sin²(Ω*(t - t_tr))     for t_tr - t_prep ≤ t ≤ t_tr
    where Ω = π/(2*t_prep).
    """
    omega = np.pi / (2 * t_prep)
    if t < t_prep:
        return v_tr * np.sin(omega * t)**2
    elif t < t_tr - t_prep:
        return v_tr
    else:
        return v_tr * np.sin(omega * (t - t_tr))**2


def time_evolve_transfer(N_domains, ell, v_tr, w, t_tr, t_prep, dt=0.1):
    """
    Simulate LR transfer using exact diagonalization + Trotter steps.
    
    Initial state: particle at site 0 (leftmost site, fully dimerized limit).
    At each time step, build H(t) with v(t) from the pulse, then apply
    exp(-i H dt) to the wavefunction.
    
    Returns occupation at each site as a function of time.
    """
    L = N_domains * (ell + 1) + 1
    n_steps = int(t_tr / dt) + 1
    times = np.linspace(0, t_tr, n_steps)
    
    # Initial state: particle at site 0
    psi = np.zeros(L, dtype=complex)
    psi[0] = 1.0
    
    occupation = np.zeros((n_steps, L))
    occupation[0] = np.abs(psi)**2
    v_values = np.zeros(n_steps)
    v_values[0] = v_pulse(0, v_tr, t_tr, t_prep)
    
    for i in range(1, n_steps):
        t = times[i]
        v_t = v_pulse(t, v_tr, t_tr, t_prep)
        v_values[i] = v_t
        
        H, _ = build_ssh_multidomain(N_domains, ell, v_t, w)
        
        # Time evolution: |ψ(t+dt)⟩ = exp(-i H dt) |ψ(t)⟩
        U = expm(-1j * H * dt)
        psi = U @ psi
        occupation[i] = np.abs(psi)**2
    
    return times, occupation, v_values


def compute_boundary_occupations(times, occupation, N_domains, ell, v_tr, w,
                                  t_tr, t_prep, dt=0.1):
    """
    Compute occupation of each analytical boundary state during the transfer.
    |⟨state|ψ(t)⟩|² for L, S_k, R states.
    
    The boundary states change with v(t), so we recompute them at each time step.
    """
    n_steps = len(times)
    keys = ['L'] + [f'S{k}' for k in range(1, N_domains)] + ['R']
    bound_occ = {k: np.zeros(n_steps) for k in keys}
    
    L = N_domains * (ell + 1) + 1
    psi = np.zeros(L, dtype=complex)
    psi[0] = 1.0
    
    for i in range(n_steps):
        t = times[i]
        v_t = v_pulse(t, v_tr, t_tr, t_prep)
        
        if v_t > 1e-10:
            states = analytical_boundary_states(N_domains, ell, v_t, w)
            for k in keys:
                bound_occ[k][i] = np.abs(np.dot(states[k], psi))**2
        else:
            # v=0: states are delta functions at isolated sites
            bound_occ['L'][i] = np.abs(psi[0])**2
            for kk in range(1, N_domains):
                wall = kk * (ell + 1)
                bound_occ[f'S{kk}'][i] = np.abs(psi[wall])**2
            bound_occ['R'][i] = np.abs(psi[L-1])**2
        
        if i < n_steps - 1:
            t_next = times[i+1]
            v_next = v_pulse(t_next, v_tr, t_tr, t_prep)
            H, _ = build_ssh_multidomain(N_domains, ell, v_next, w)
            U = expm(-1j * H * dt)
            psi = U @ psi
    
    return bound_occ


# =============================================================================
# 5. VISUALIZATION
# =============================================================================

def plot_dimer_spectrum(ax, ratios, spectra, L_sites):
    """Plot energy spectrum vs v/w ratio for the dimer chain."""
    for band in range(L_sites):
        ax.plot(ratios, spectra[:, band], '-', color='#0077BB', lw=0.8, alpha=0.7)
    
    ax.axvline(x=1.0, color='#CC3311', ls='--', lw=1.5, alpha=0.8,
               label='$v/w = 1$ (transition)')
    ax.axhline(y=0, color='gray', ls=':', lw=0.5)
    
    ax.set_xlabel('$v/w$')
    ax.set_ylabel('Energy $E/w$')
    ax.legend(loc='lower right', fontsize=7.5)
    ax.set_xlim(ratios[0], ratios[-1])

    # Annotate phases — use axes coordinates to avoid overlap with curves
    ax.text(0.18, 0.97, 'Topological\n($\\nu=1$)',
            ha='center', va='top', fontsize=8, color='#009988',
            fontweight='bold', transform=ax.transAxes)
    # Trivial label placed in the E≈0 spectral gap of the trivial phase
    ax.text(0.78, 0.50, 'Trivial\n($\\nu=0$)',
            ha='center', va='center', fontsize=8, color='#888',
            fontweight='bold', transform=ax.transAxes)
    
    # Annotate gap
    idx_05 = np.argmin(np.abs(ratios - 0.5))
    gap_05 = spectra[idx_05]
    pos_min = np.min(gap_05[gap_05 > 0])
    neg_max = np.max(gap_05[gap_05 < 0])
    # Clean bracket-style gap indicator (vertical bar + end caps)
    ax.vlines(0.5, neg_max, pos_min, color='#EE7733', lw=1.4)
    ax.hlines(pos_min, 0.46, 0.54, color='#EE7733', lw=1.4)
    ax.hlines(neg_max, 0.46, 0.54, color='#EE7733', lw=1.4)
    ax.text(0.57, 0, r'$\Delta$', fontsize=8, color='#EE7733',
            fontweight='bold', va='center')


def plot_dimer_gap(ax, L_sites, w=1.0, n_points=200):
    """Plot the bulk gap as a function of v/w with analytical comparison."""
    ratios = np.linspace(0.01, 2.5, n_points)
    gaps = np.zeros(n_points)
    gaps_large = np.zeros(n_points)  # larger chain for comparison
    L_large = 100
    
    for i, ratio in enumerate(ratios):
        v = ratio * w
        H = build_ssh_dimer(L_sites, v, w)
        evals = np.linalg.eigvalsh(H)
        pos = evals[evals >= -1e-12]
        neg = evals[evals <= 1e-12]
        if len(pos) > 0 and len(neg) > 0:
            gaps[i] = np.min(np.abs(pos)) + np.min(np.abs(neg))
        
        H2 = build_ssh_dimer(L_large, v, w)
        evals2 = np.linalg.eigvalsh(H2)
        pos2 = evals2[evals2 >= -1e-12]
        neg2 = evals2[evals2 <= 1e-12]
        if len(pos2) > 0 and len(neg2) > 0:
            gaps_large[i] = np.min(np.abs(pos2)) + np.min(np.abs(neg2))
    
    # Bulk formula: Δ = 2|w - v|
    bulk_gap = 2 * np.abs(w - ratios * w)
    
    ax.plot(ratios, bulk_gap, '--', color='#999999', lw=2,
            label=r'Bulk $\Delta = 2|w-v|$')
    ax.plot(ratios, gaps_large, '-', color='#AADDFF', lw=2,
            label=f'{L_large} sites')
    ax.plot(ratios, gaps, '-', color='#EE7733', lw=2.5,
            label=f'{L_sites} sites')
    ax.axvline(x=1.0, color='#CC3311', ls='--', lw=1.5, alpha=0.8)
    ax.set_xlabel('$v/w$')
    ax.set_ylabel('Gap $\\Delta/w$')
    ax.set_xlim(ratios[0], ratios[-1])
    ax.legend(loc='upper right')
    ax.text(0.92, max(bulk_gap)*0.85, '$v=w$\n(topological\ntransition)',
            ha='right', fontsize=8, color='#CC3311')


def plot_domain_wall_states(ax, N_domains, ell, v, w, states_dict, title):
    """Draw the SSH chain with boundary state wavefunctions."""
    L = N_domains * (ell + 1) + 1
    
    chain_y = -0.5
    wave_y = 0.3
    
    # Domain shading
    for k in range(N_domains):
        xs = k * (ell + 1) - 0.4
        xe = (k + 1) * (ell + 1) + 0.4
        xs, xe = max(xs, -0.4), min(xe, L - 0.6)
        c = '#DCEEFB' if k % 2 == 0 else '#FFF3CD'
        ax.axvspan(xs, xe, alpha=0.35, color=c, zorder=0)
        mid = (max(xs, 0) + min(xe, L-1)) / 2
        ax.text(mid, chain_y - 0.35, f'$D_{k+1}$', ha='center', fontsize=9,
                color='#666', style='italic')
    
    # Bonds
    for j in range(L - 1):
        d = min(j // (ell + 1), N_domains - 1)
        p = j % (ell + 1)
        if p % 2 == 0:
            lw, cc = 1.0, '#AAAAAA'
        else:
            lw, cc = 3.0, '#444444'
        ax.plot([j, j+1], [chain_y, chain_y], '-', color=cc, linewidth=lw,
                zorder=2, solid_capstyle='round')
    
    # Bond labels (v or w)
    for j in range(L - 1):
        p = j % (ell + 1)
        label = '$v$' if p % 2 == 0 else '$w$'
        ax.text(j + 0.5, chain_y + 0.12, label, ha='center', fontsize=6,
                color='#999')
    
    # Sites
    for j in range(L):
        mk = 'o' if j % 2 == 0 else 's'
        mc = '#4477AA' if j % 2 == 0 else '#CC6677'
        ax.plot(j, chain_y, mk, color=mc, markersize=5, zorder=5,
                markeredgecolor='black', markeredgewidth=0.4)
    
    # Domain wall lines
    for k in range(1, N_domains):
        ax.axvline(x=k*(ell+1), color='#FF8800', ls=':', alpha=0.5, lw=1.2)
    
    # Wavefunctions
    sc = {'L': '#CC3311', 'R': '#0077BB'}
    sl = {'L': r'$|\mathcal{L}\rangle$', 'R': r'$|\mathcal{R}\rangle$'}
    for k in range(1, N_domains):
        sc[f'S{k}'] = '#EE7733'
        sl[f'S{k}'] = rf'$|\mathcal{{S}}_{k}\rangle$'
    
    max_a = max(np.max(np.abs(s)) for s in states_dict.values())
    scale = 1.2 / max_a if max_a > 0 else 1.0
    
    for key, state in states_dict.items():
        col = sc.get(key, '#009988')
        for j in range(L):
            a = state[j] * scale
            if abs(a) > 0.005:
                ax.plot([j, j], [wave_y, wave_y + a], '-', color=col, lw=1.8,
                        alpha=0.85, zorder=4)
                ax.plot(j, wave_y + a, 'o', color=col, markersize=3, zorder=6)
        
        nz = np.where(np.abs(state) > 0.01)[0]
        if len(nz) > 0:
            pk = nz[np.argmax(np.abs(state[nz]))]
            ap = state[pk] * scale
            sg = np.sign(ap) if ap != 0 else 1
            ax.text(pk, wave_y + ap + 0.12*sg, sl.get(key, key),
                    ha='center', fontsize=8, color=col, fontweight='bold')
    ax.axhline(y=wave_y, color='lightgray', lw=0.5, zorder=0)
    ax.set_xlim(-1, L)
    ax.set_ylim(chain_y - 0.55, wave_y + 1.8)
    ax.set_xlabel('Site index $j$')
    ax.set_yticks([])
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)


def plot_transfer_heatmap(ax, times, occupation, L, title):
    """Plot site occupation vs time as a heatmap (like Fig 2 of paper)."""
    im = ax.pcolormesh(times, np.arange(L), occupation.T,
                       shading='auto', cmap='inferno', vmin=0, vmax=1,
                            rasterized=True)
    ax.set_xlabel('Time $t$ ($\\hbar/J$)')
    ax.set_ylabel('Site $j$')
    ax.set_ylim(-0.5, L - 0.5)
    plt.colorbar(im, ax=ax, label='$\\langle n_j \\rangle$', shrink=0.8)


def plot_pulse(ax, times, v_values, title='Control pulse'):
    """Plot the control pulse v(t)."""
    ax.plot(times, v_values, '-', color='#009988', lw=2)
    ax.fill_between(times, v_values, alpha=0.2, color='#009988')
    ax.set_xlabel('Time $t$ ($\\hbar/J$)')
    ax.set_ylabel('$v(t)$')
    ax.set_xlim(times[0], times[-1])


def plot_boundary_occupations(ax, times, bound_occ, title):
    """Plot occupation of boundary states vs time (like Fig 3 of paper)."""
    colors = {'L': '#CC3311', 'R': '#0077BB'}
    labels = {'L': r'$|\mathcal{L}\rangle$', 'R': r'$|\mathcal{R}\rangle$'}
    for k in bound_occ:
        if k.startswith('S'):
            colors[k] = '#EE7733'
            labels[k] = rf'$|\mathcal{{S}}_{k[1:]}\rangle$'
    
    for k, occ in bound_occ.items():
        ax.plot(times, occ, '-', color=colors.get(k, '#009988'), lw=2,
                label=labels.get(k, k))
    
    ax.set_xlabel('Time $t$ ($\\hbar/J$)')
    ax.set_ylabel('Occupation')
    ax.legend()
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-0.05, 1.05)


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SSH Dimer Chain: Topological phases, domain walls, transfer")
    print("=" * 70)
    
    # =====================================================================
    # FIGURE 1: Dimer chain spectrum (10 atoms = 5 unit cells, even chain)
    #           Also compare with 11 atoms (odd chain — has edge state!)
    # =====================================================================
    L_dimer = 10
    L_odd = 11
    w = 1.0
    
    print(f"\n--- Dimer chain spectrum: {L_dimer} atoms (even), "
          f"{L_odd} atoms (odd) ---")
    ratios, spectra = compute_spectrum_vs_ratio(L_dimer, w)
    ratios_odd, spectra_odd = compute_spectrum_vs_ratio(L_odd, w)
    
    fig1 = plt.figure(figsize=(FULL_W, 4.67))
    gs1 = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30)

    # (a) Even chain spectrum
    ax1a = fig1.add_subplot(gs1[0, 0])
    plot_dimer_spectrum(ax1a, ratios, spectra, L_dimer)
    panel_label(ax1a, '(a)')
    
    # (b) Odd chain spectrum — note the edge state at E=0 in topological phase!
    ax1b = fig1.add_subplot(gs1[0, 1])
    for band in range(L_odd):
        ax1b.plot(ratios_odd, spectra_odd[:, band], '-', color='#0077BB',
                  lw=0.8, alpha=0.7)
    ax1b.axvline(x=1.0, color='#CC3311', ls='--', lw=1.5, alpha=0.8,
                 label='$v/w = 1$ (transition)')
    ax1b.axhline(y=0, color='gray', ls=':', lw=0.5)
    ax1b.set_xlabel('$v/w$')
    ax1b.set_ylabel('Energy $E/w$')
    panel_label(ax1b, '(b)')
    ax1b.legend(loc='lower right', fontsize=7.5)
    ax1b.set_xlim(ratios_odd[0], ratios_odd[-1])
    ax1b.annotate('Edge\nstate\n$E=0$',
                  xy=(0.5, 0.0), xycoords='data',
                  xytext=(0.10, 0.87), textcoords='axes fraction',
                  fontsize=8, color='#CC3311', fontweight='bold',
                  ha='center', va='top',
                  arrowprops=dict(arrowstyle='->', color='#CC3311', lw=1.0))
    
    # (c) Gap comparison
    ax1c = fig1.add_subplot(gs1[1, 0])
    plot_dimer_gap(ax1c, L_dimer, w)
    panel_label(ax1c, '(c)')
    
    # (d) Sketch of the dimer chain
    ax1d = fig1.add_subplot(gs1[1, 1])
    ax1d.set_xlim(-1, 12)
    ax1d.set_ylim(-0.3, 2.2)
    ax1d.set_yticks([])
    for sp in ['top', 'right', 'left']:
        ax1d.spines[sp].set_visible(False)
    panel_label(ax1d, '(d)')
    
    # Draw topological phase (v < w)
    y_top = 1.5
    ax1d.text(5.5, 1.88, 'Topological phase: $v < w$', fontsize=8,
              ha='center', fontweight='bold', color='#009988')
    for j in range(10):
        mk = 'o' if j%2==0 else 's'
        mc = '#4477AA' if j%2==0 else '#CC6677'
        ax1d.plot(j+0.5, y_top, mk, color=mc, ms=7, zorder=5,
                  markeredgecolor='black', markeredgewidth=0.5)
        if j > 0:
            lw = 1.5 if j%2==1 else 4.0
            ax1d.plot([j-0.5, j+0.5], [y_top, y_top], '-', color='#333',
                      linewidth=lw, zorder=2)
    ax1d.text(0.5, y_top-0.22, '$a$', fontsize=8, ha='center', color='#4477AA')
    ax1d.text(1.5, y_top-0.22, '$b$', fontsize=8, ha='center', color='#CC6677')
    ax1d.annotate('$v$', xy=(1.0, y_top+0.08), fontsize=9, ha='center', color='#888')
    ax1d.annotate('$w$', xy=(2.0, y_top+0.08), fontsize=9, ha='center',
                  color='#333', fontweight='bold')
    
    # Draw trivial phase (v > w)
    y_triv = 0.5
    ax1d.text(5.5, 0.73, 'Trivial phase: $v > w$', fontsize=8,
              ha='center', fontweight='bold', color='#888')
    for j in range(10):
        mk = 'o' if j%2==0 else 's'
        mc = '#4477AA' if j%2==0 else '#CC6677'
        ax1d.plot(j+0.5, y_triv, mk, color=mc, ms=7, zorder=5,
                  markeredgecolor='black', markeredgewidth=0.5)
        if j > 0:
            lw = 4.0 if j%2==1 else 1.5
            ax1d.plot([j-0.5, j+0.5], [y_triv, y_triv], '-', color='#333',
                      linewidth=lw, zorder=2)
    ax1d.annotate('$v$', xy=(1.0, y_triv+0.08), fontsize=9, ha='center',
                  color='#333', fontweight='bold')
    ax1d.annotate('$w$', xy=(2.0, y_triv+0.08), fontsize=9, ha='center',
                  color='#888')
    
    # Unit cell boxes
    for i in range(5):
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch((2*i, y_top-0.15), 2, 0.3, boxstyle='round,pad=0.1',
                              ec='#009988', fc='none', lw=1.0, ls='--', alpha=0.5)
        ax1d.add_patch(rect)
    
    ax1d.set_xlabel('Site $j$')
    
    fig1.savefig(os.path.join(FIGURES_DIR, 'fig1_dimer_spectrum.pdf'),
                 bbox_inches='tight')
    print("  -> tesis/figures/fig1_dimer_spectrum.pdf")
    plt.close(fig1)
    
    # Print key values
    idx_half = np.argmin(np.abs(ratios - 0.5))
    evals_half = np.sort(np.linalg.eigvalsh(build_ssh_dimer(L_dimer, 0.5, 1.0)))
    print(f"  Spectrum at v/w=0.5: {np.round(evals_half, 4)}")
    print(f"  Gap at v/w=0.5: {evals_half[L_dimer//2] - evals_half[L_dimer//2 - 1]:.4f}")
    
    idx_one = np.argmin(np.abs(ratios - 1.0))
    evals_one = np.sort(np.linalg.eigvalsh(build_ssh_dimer(L_dimer, 1.0, 1.0)))
    print(f"  Spectrum at v/w=1.0: {np.round(evals_one, 4)}")
    print(f"  Gap at v/w=1.0: {evals_one[L_dimer//2] - evals_one[L_dimer//2 - 1]:.4f}")
    
    # =====================================================================
    # FIGURE 2: Domain wall — boundary states (N=2, ℓ=10)
    # =====================================================================
    N_dw, ell_dw, v_dw, w_dw = 2, 10, 0.5, 1.0
    L_dw = N_dw * (ell_dw + 1) + 1
    
    print(f"\n--- Domain wall: N={N_dw}, ℓ={ell_dw}, L={L_dw} ---")
    
    H_dw, _ = build_ssh_multidomain(N_dw, ell_dw, v_dw, w_dw)
    evals_dw, evecs_dw = np.linalg.eigh(H_dw)
    
    # Find protected states
    n_prot = N_dw + 1
    idx_prot = np.argsort(np.abs(evals_dw))[:n_prot]
    E_prot = evals_dw[idx_prot]
    print(f"  Protected E: {np.sort(E_prot)}")
    
    # Analytical states for the plot
    an_dw = analytical_boundary_states(N_dw, ell_dw, v_dw, w_dw)
    
    # Subspace validation
    prot_space = evecs_dw[:, idx_prot]
    keys_dw = ['L', 'S1', 'R']
    analyt_mat = np.column_stack([an_dw[k] for k in keys_dw])
    overlap = analyt_mat.T @ prot_space
    _, sv, _ = np.linalg.svd(overlap)
    print(f"  Subspace overlap (sing. vals): {sv}")
    print(f"  Product: {np.prod(sv):.6f}")
    
    # Bond pattern
    bp = []
    for j in range(L_dw - 1):
        p = j % (ell_dw + 1)
        bp.append('v' if p % 2 == 0 else 'w')
    print(f"  Pattern: {''.join(bp)}")
    print(f"  Wall at site: {ell_dw + 1}")
    
    # Full spectrum plot
    fig2 = plt.figure(figsize=(FULL_W, 5.25))
    gs2 = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.40,
                            wspace=0.30)

    ax_chain = fig2.add_subplot(gs2[0, :])
    plot_domain_wall_states(ax_chain, N_dw, ell_dw, v_dw, w_dw, an_dw, '')
    panel_label(ax_chain, '(a)')
    
    ax_spec = fig2.add_subplot(gs2[1, 0])
    ax_spec.plot(evals_dw, np.zeros_like(evals_dw), '|', color='#888', ms=12)
    ax_spec.plot(np.sort(E_prot), np.zeros(n_prot), 'o', color='#CC3311',
                 ms=10, zorder=5, label=f'{n_prot} protected')
    ax_spec.set_xlabel('Energy $E$')
    ax_spec.set_yticks([])
    ax_spec.legend(fontsize=7, markerscale=0.7, handlelength=1.0,
                   borderpad=0.25, handletextpad=0.5)
    panel_label(ax_spec, '(b)')
    for s in ['top', 'right', 'left']:
        ax_spec.spines[s].set_visible(False)
    
    # Plot |ψ|² for each analytical state
    ax_prob = fig2.add_subplot(gs2[1, 1])
    colors_p = {'L': '#CC3311', 'S1': '#EE7733', 'R': '#0077BB'}
    labels_p = {'L': r'$|\mathcal{L}\rangle$', 'S1': r'$|\mathcal{S}_1\rangle$',
                'R': r'$|\mathcal{R}\rangle$'}
    sites = np.arange(L_dw)
    for key in ['L', 'S1', 'R']:
        ax_prob.plot(sites, np.abs(an_dw[key])**2, 'o-',
                     color=colors_p[key], ms=4, lw=1.5, label=labels_p[key])
    ax_prob.set_xlabel('Site $j$')
    ax_prob.set_ylabel('$|\\psi_j|^2$')
    ax_prob.legend(loc='lower left', fontsize=7.5)
    panel_label(ax_prob, '(c)')
    
    fig2.savefig(os.path.join(FIGURES_DIR, 'fig2_domain_wall_states.pdf'),
                 bbox_inches='tight')
    print("  -> tesis/figures/fig2_domain_wall_states.pdf")
    plt.close(fig2)
    
    # =====================================================================
    # FIGURE 3: Transfer protocol for N=2, ℓ=4 (like Fig 2a of paper)
    # =====================================================================
    N_tr, ell_tr = 2, 4
    v_tr, w_tr = 0.5, 1.0
    t_prep = 15.0
    t_tr = 45.6
    dt = 0.1
    L_tr = N_tr * (ell_tr + 1) + 1
    
    print(f"\n--- Transfer: N={N_tr}, ℓ={ell_tr}, L={L_tr} ---")
    print(f"  v_tr={v_tr}, t_prep={t_prep}, t_tr={t_tr}, dt={dt}")
    
    times, occupation, v_values = time_evolve_transfer(
        N_tr, ell_tr, v_tr, w_tr, t_tr, t_prep, dt)
    
    # Final fidelity: probability at last site
    fidelity = occupation[-1, L_tr - 1]
    print(f"  Final fidelity |⟨R|ψ(t_tr)⟩|²: {fidelity:.6f}")
    print(f"  Final site 0 occupation: {occupation[-1, 0]:.6f}")
    print(f"  Wall (site {ell_tr+1}) final occupation: {occupation[-1, ell_tr+1]:.6f}")
    
    # Boundary state occupations (recompute with fresh psi evolution)
    bound_occ = compute_boundary_occupations(
        times, occupation, N_tr, ell_tr, v_tr, w_tr, t_tr, t_prep, dt)
    
    # Figure 3: Transfer heatmap + pulse + boundary occupations (shared x-axis)
    fig3 = plt.figure(figsize=(COL_W, 3.5))
    gs3 = gridspec.GridSpec(3, 1, height_ratios=[1.25, 0.5, 1.0], hspace=0.12)
    ax_heat = fig3.add_subplot(gs3[0])
    ax_pulse = fig3.add_subplot(gs3[1], sharex=ax_heat)
    ax_bound = fig3.add_subplot(gs3[2], sharex=ax_heat)

    im = ax_heat.pcolormesh(times, np.arange(L_tr), occupation.T,
                            shading='auto', cmap='inferno', vmin=0, vmax=1,
                            rasterized=True)
    ax_heat.set_ylabel('Site $j$')
    ax_heat.set_ylim(-0.5, L_tr - 0.5)
    plt.setp(ax_heat.get_xticklabels(), visible=False)
    fig3.colorbar(im, ax=[ax_heat, ax_pulse, ax_bound],
                  label='$\\langle n_j \\rangle$', shrink=0.5, pad=0.02, aspect=30)
    panel_label(ax_heat, '(a)', loc='inside')

    ax_pulse.plot(times, v_values, '-', color=COLORS['teal'])
    ax_pulse.fill_between(times, v_values, alpha=0.18, color=COLORS['teal'])
    ax_pulse.set_ylabel('$v(t)$')
    ax_pulse.set_ylim(0, max(v_values) * 1.25)
    plt.setp(ax_pulse.get_xticklabels(), visible=False)
    panel_label(ax_pulse, '(b)')

    cmap_b = {'L': COLORS['red'], 'R': COLORS['blue']}
    lab_b = {'L': r'$|\mathcal{L}\rangle$', 'R': r'$|\mathcal{R}\rangle$'}
    for k in bound_occ:
        if k.startswith('S'):
            cmap_b[k] = COLORS['orange']
            lab_b[k] = r'$|\mathcal{S}\rangle$'
    for k, occ in bound_occ.items():
        ax_bound.plot(times, occ, '-', color=cmap_b.get(k, COLORS['teal']),
                      label=lab_b.get(k, k))
    ax_bound.set_xlabel('Time $t$ ($\\hbar/J$)')
    ax_bound.set_ylabel('Occupation')
    ax_bound.set_ylim(-0.05, 1.05)
    ax_bound.legend(loc='lower right', fontsize=7.5)
    panel_label(ax_bound, '(c)')

    fig3.savefig(os.path.join(FIGURES_DIR, 'fig3_transfer_protocol.pdf'))
    print("  -> tesis/figures/fig3_transfer_protocol.pdf")
    plt.close(fig3)
    
    # =====================================================================
    # FIGURE 4: Scan transfer time to find optimal t_tr for high fidelity
    # =====================================================================
    print(f"\n--- Transfer time scan ---")
    t_scan = np.arange(30, 80, 0.5)
    fidelities = np.zeros(len(t_scan))
    
    for i, tt in enumerate(t_scan):
        ts, occ, _ = time_evolve_transfer(N_tr, ell_tr, v_tr, w_tr, tt,
                                           t_prep, dt)
        fidelities[i] = occ[-1, L_tr - 1]
    
    best_idx = np.argmax(fidelities)
    best_t = t_scan[best_idx]
    best_f = fidelities[best_idx]
    print(f"  Best t_tr = {best_t:.1f}, fidelity = {best_f:.6f}")
    
    fig4, ax4 = plt.subplots(figsize=(COL_W, 2.1))
    ax4.plot(t_scan, fidelities, '-', color='#0077BB', lw=2)
    ax4.axhline(y=0.995, color='#CC3311', ls='--', lw=1.5, alpha=0.7,
                label='$f_0 = 0.995$')
    ax4.plot(best_t, best_f, 'o', color='#CC3311', ms=10, zorder=5,
             label=f'Optimal: $t_{{tr}}={best_t:.1f}$, $f={best_f:.4f}$')
    ax4.set_xlabel('Transfer time $t_{tr}$')
    ax4.set_ylabel('Fidelity $f = |\\langle R|\\psi(t_{tr})\\rangle|^2$')
    ax4.legend(loc='lower left')
    ax4.set_ylim(-0.05, 1.05)
    
    fig4.savefig(os.path.join(FIGURES_DIR, 'fig4_fidelity_scan.pdf'),
                 bbox_inches='tight')
    print("  -> tesis/figures/fig4_fidelity_scan.pdf")
    plt.close(fig4)
    
    print("\n" + "=" * 70)
    print("All figures generated successfully.")
    print("=" * 70)


if __name__ == '__main__':
    main()
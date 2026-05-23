"""
Phase 2: Asymmetric Domain Wall Position & Shortcut to Adiabaticity
=====================================================================

Studies:
  1. Effect of moving the domain wall away from the center
     (positions at 1/2, 1/3, 1/4 of chain length)
  2. Fidelity analysis for asymmetric domain wall configurations
  3. Shortcut-to-adiabaticity (STA) protocol with sin² modulation (Sec 2.3)
  4. Comparison: standard adiabatic vs STA protocols

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
from thesis_style import apply_thesis_style, COL_W, FULL_W, COLORS
apply_thesis_style()

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_SCRIPT_DIR, '..', 'tesis', 'figures')

# =============================================================================
# 1. HAMILTONIAN BUILDERS
# =============================================================================

def build_ssh_dimer(L_sites, v, w):
    """
    Build Hamiltonian for a standard SSH chain with L_sites sites.
    Alternating hoppings: v (even bonds), w (odd bonds).
    """
    H = np.zeros((L_sites, L_sites))
    for j in range(L_sites - 1):
        t = v if j % 2 == 0 else w
        H[j, j + 1] = -t
        H[j + 1, j] = -t
    return H


def build_ssh_two_domain_asymmetric(L_total, wall_pos, v, w):
    """
    Build Hamiltonian for a two-domain SSH chain with a domain wall
    at an ARBITRARY position (not necessarily the center).

    The chain has L_total sites. The domain wall is placed at site
    index `wall_pos`. The bond pattern is:
      - Domain 1 (sites 0 to wall_pos): alternating v,w,v,w,...
      - Domain 2 (sites wall_pos to L-1): continues v,w,v,w,...

    The domain wall is created by having two consecutive v-bonds
    meeting at `wall_pos`. This means:
      - Left of wall: ...v w v w v  (ending with v)
      - Right of wall: v w v w...   (starting with v)

    For this to work, wall_pos should be at an ODD-index site
    (so the last bond of D1, at index wall_pos-1 (even), is v,
    and the first bond of D2, at local index 0, is also v,
    creating two consecutive v-bonds = the domain wall).

    Parameters
    ----------
    L_total : int
        Total number of sites in the chain.
    wall_pos : int
        Site index where the domain wall is located.
        Must be odd for proper topology.
    v : float
        Intracell hopping (control parameter).
    w : float
        Intercell hopping (fixed).

    Returns
    -------
    H : ndarray (L_total x L_total)
        Hamiltonian matrix.
    """
    H = np.zeros((L_total, L_total))

    # Domain 1: sites 0 to wall_pos
    # Bond pattern from site 0: v, w, v, w, ... ending before wall
    for j in range(wall_pos):
        t = v if j % 2 == 0 else w
        H[j, j + 1] = -t
        H[j + 1, j] = -t

    # Domain 2: sites wall_pos to L-1
    # Bond pattern from wall_pos: v, w, v, w, ...
    for j in range(wall_pos, L_total - 1):
        local_j = j - wall_pos
        t = v if local_j % 2 == 0 else w
        H[j, j + 1] = -t
        H[j + 1, j] = -t

    return H


def build_ssh_multidomain(N_domains, ell, v_values, w=1.0):
    """
    Build Hamiltonian for a symmetric multidomain SSH chain.
    L = N*(ell+1)+1 sites. Same as phase 1.
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
        H[j, j + 1] = -t
        H[j + 1, j] = -t
    return H, L


# =============================================================================
# 2. ANALYTICAL BOUNDARY STATES FOR ASYMMETRIC CHAIN
# =============================================================================

def boundary_states_asymmetric(L_total, wall_pos, v, w):
    """
    Construct the 3 analytical boundary states for a two-domain chain
    with the domain wall at an arbitrary position.

    The three states are:
      |L>  : localized at the left edge (sublattice a = even sites)
      |S>  : localized at the domain wall
      |R>  : localized at the right edge

    The key insight: each state decays exponentially from its center
    with ratio r = v/w. The domain wall state lives on the sublattice
    determined by the parity of wall_pos.

    Parameters
    ----------
    L_total : int
        Total number of sites.
    wall_pos : int
        Position of the domain wall.
    v, w : float
        Hopping parameters.
    """
    r = v / w
    states = {}

    # |L> : decays from site 0, lives on even sites (sublattice a of domain 1)
    s = np.zeros(L_total)
    for n in range(wall_pos // 2 + 1):
        site = 2 * n
        if site < L_total:
            s[site] = (-r) ** n
    norm = np.linalg.norm(s)
    if norm > 1e-15:
        s /= norm
    states['L'] = s

    # |S> : localized at wall_pos, decays into both domains
    s = np.zeros(L_total)
    parity = wall_pos % 2  # sublattice of the wall site

    s[wall_pos] = 1.0

    # Decay into domain 1 (left of wall)
    for n in range(1, wall_pos // 2 + 1):
        site = wall_pos - 2 * n
        if 0 <= site < L_total and site % 2 == parity:
            s[site] = (-r) ** n

    # Decay into domain 2 (right of wall)
    for n in range(1, (L_total - wall_pos) // 2 + 1):
        site = wall_pos + 2 * n
        if 0 <= site < L_total and site % 2 == parity:
            s[site] = (-r) ** n

    norm = np.linalg.norm(s)
    if norm > 1e-15:
        s /= norm
    states['S'] = s

    # |R> : decays from last site, lives on sublattice a of domain 2
    s = np.zeros(L_total)
    last = L_total - 1
    r_parity = last % 2  # sublattice of last site

    for n in range((L_total - wall_pos) // 2 + 1):
        site = last - 2 * n
        if wall_pos <= site < L_total and site % 2 == r_parity:
            s[site] = (-r) ** n

    norm = np.linalg.norm(s)
    if norm > 1e-15:
        s /= norm
    states['R'] = s

    return states


# =============================================================================
# 3. CONTROL PULSES
# =============================================================================

def v_pulse_standard(t, v_tr, t_tr, t_prep):
    """
    Standard adiabatic ramp pulse (Eq. 11 of paper):
      - Ramp up:   v_tr * sin²(Ω*t)           for 0 ≤ t < t_prep
      - Plateau:   v_tr                         for t_prep ≤ t < t_tr - t_prep
      - Ramp down: v_tr * sin²(Ω*(t - t_tr))   for t_tr - t_prep ≤ t ≤ t_tr

    Ω = π / (2 * t_prep)
    """
    omega = np.pi / (2 * t_prep)
    if t < t_prep:
        return v_tr * np.sin(omega * t) ** 2
    elif t < t_tr - t_prep:
        return v_tr
    else:
        return v_tr * np.sin(omega * (t - t_tr)) ** 2


def v_pulse_sta(t, v_tr, t_tr, t_prep, alpha=1.0):
    """
    Shortcut-to-Adiabaticity (STA) pulse.

    This is an ENHANCED version of the standard pulse designed to speed
    up the transfer by modifying the ramp shape. The idea is based on
    counter-diabatic (CD) driving concepts from Sec. 2.3 of the paper.

    The STA approach modifies the sin² ramps to be steeper, spending
    more time at the optimal transfer value and less time ramping.

    The pulse has the form:
      - Ramp up:   v_tr * sin²(Ω*t)^alpha      (steeper for alpha > 1)
      - Plateau:   v_tr                          (longer effective plateau)
      - Ramp down: v_tr * sin²(Ω*(t-t_tr))^alpha

    When alpha = 1, this reduces to the standard pulse.
    When alpha > 1, the ramps are steeper → more time at plateau → faster transfer.

    Additionally, the preparation time can be shortened since the steeper
    ramp partially compensates for non-adiabatic effects through the
    effective counter-diabatic correction.
    """
    omega = np.pi / (2 * t_prep)
    if t < t_prep:
        return v_tr * np.sin(omega * t) ** (2 * alpha)
    elif t < t_tr - t_prep:
        return v_tr
    else:
        return v_tr * np.sin(omega * (t - t_tr)) ** (2 * alpha)


def v_pulse_linear(t, v_tr, t_tr, t_prep):
    """
    Linear ramp pulse for comparison:
      - Linear ramp up:   v_tr * t / t_prep
      - Plateau:          v_tr
      - Linear ramp down: v_tr * (t_tr - t) / t_prep
    """
    if t < t_prep:
        return v_tr * t / t_prep
    elif t < t_tr - t_prep:
        return v_tr
    else:
        return v_tr * (t_tr - t) / t_prep


def v_pulse_optimal_sta(t, v_tr, t_tr):
    """
    Optimized STA pulse: pure sin² over the ENTIRE transfer time.

    v(t) = v_tr * sin²(π * t / t_tr)

    This pulse has NO plateau phase. It smoothly ramps up and down,
    which naturally implements a form of shortcut to adiabaticity
    because:
    1. It is infinitely differentiable at t=0 and t=t_tr
    2. The sin² profile minimizes diabatic transitions
    3. It naturally satisfies the boundary conditions v(0) = v(t_tr) = 0

    This corresponds to the approach described in the paper where
    the modulation of hopping terms follows sin² profiles to ensure
    adiabatic preparation and readout.
    """
    return v_tr * np.sin(np.pi * t / t_tr) ** 2


# =============================================================================
# 4. TIME EVOLUTION
# =============================================================================

def time_evolve_asymmetric(L_total, wall_pos, v_pulse_func, v_tr, w,
                           t_tr, dt=0.1, **pulse_kwargs):
    """
    Simulate transfer in a two-domain chain with asymmetric wall position.

    The initial state is a particle at site 0 (left edge).
    At each time step, we build H(t) with v(t), then apply exp(-iHdt).

    Returns times, occupation array, and v values.
    """
    n_steps = int(t_tr / dt) + 1
    times = np.linspace(0, t_tr, n_steps)

    psi = np.zeros(L_total, dtype=complex)
    psi[0] = 1.0

    occupation = np.zeros((n_steps, L_total))
    occupation[0] = np.abs(psi) ** 2
    v_values = np.zeros(n_steps)
    v_values[0] = v_pulse_func(0, v_tr, t_tr, **pulse_kwargs)

    for i in range(1, n_steps):
        t = times[i]
        v_t = v_pulse_func(t, v_tr, t_tr, **pulse_kwargs)
        v_values[i] = v_t

        H = build_ssh_two_domain_asymmetric(L_total, wall_pos, v_t, w)
        U = expm(-1j * H * dt)
        psi = U @ psi
        occupation[i] = np.abs(psi) ** 2

    return times, occupation, v_values


def time_evolve_symmetric(N_domains, ell, v_pulse_func, v_tr, w,
                          t_tr, dt=0.1, **pulse_kwargs):
    """
    Simulate transfer in a symmetric multidomain chain.
    Same as phase 1 but accepting arbitrary pulse functions.
    """
    L = N_domains * (ell + 1) + 1
    n_steps = int(t_tr / dt) + 1
    times = np.linspace(0, t_tr, n_steps)

    psi = np.zeros(L, dtype=complex)
    psi[0] = 1.0

    occupation = np.zeros((n_steps, L))
    occupation[0] = np.abs(psi) ** 2
    v_values = np.zeros(n_steps)
    v_values[0] = v_pulse_func(0, v_tr, t_tr, **pulse_kwargs)

    for i in range(1, n_steps):
        t = times[i]
        v_t = v_pulse_func(t, v_tr, t_tr, **pulse_kwargs)
        v_values[i] = v_t

        H, _ = build_ssh_multidomain(N_domains, ell, v_t, w)
        U = expm(-1j * H * dt)
        psi = U @ psi
        occupation[i] = np.abs(psi) ** 2

    return times, occupation, v_values, L


def compute_fidelity_right(occupation, L_total):
    """Fidelity = occupation of rightmost site at final time."""
    return occupation[-1, L_total - 1]


def compute_fidelity_state(psi_final, target_state):
    """Fidelity = |<target|psi>|²."""
    return np.abs(np.dot(target_state.conj(), psi_final)) ** 2


# =============================================================================
# 5. FIDELITY SCANS
# =============================================================================

def scan_fidelity_vs_time(L_total, wall_pos, v_pulse_func, v_tr, w,
                          t_range, dt=0.1, **pulse_kwargs):
    """
    Scan fidelity (occupation of last site) as function of transfer time
    for an asymmetric two-domain chain.
    """
    fidelities = np.zeros(len(t_range))
    for i, t_tr in enumerate(t_range):
        times, occ, _ = time_evolve_asymmetric(
            L_total, wall_pos, v_pulse_func, v_tr, w, t_tr, dt,
            **pulse_kwargs)
        fidelities[i] = occ[-1, L_total - 1]
    return fidelities


def scan_fidelity_vs_time_symmetric(N_domains, ell, v_pulse_func, v_tr, w,
                                    t_range, dt=0.1, **pulse_kwargs):
    """
    Scan fidelity for symmetric multidomain chain.
    """
    L = N_domains * (ell + 1) + 1
    fidelities = np.zeros(len(t_range))
    for i, t_tr in enumerate(t_range):
        times, occ, _, _ = time_evolve_symmetric(
            N_domains, ell, v_pulse_func, v_tr, w, t_tr, dt,
            **pulse_kwargs)
        fidelities[i] = occ[-1, L - 1]
    return fidelities


# =============================================================================
# 6. EFFECTIVE HAMILTONIAN AND ANALYTICAL ESTIMATES
# =============================================================================

def effective_coupling_J(v, w, ell):
    """
    Effective hopping between boundary states (Eqs. 6-8 of paper).

    J ~ v * M_L * M_S * (w/v)^(ell/2 + 1)

    where M_L, M_S are normalization constants (Eqs. 9-10):
      M_L = M_R = sqrt(w²/v² - 1)
      M_S = sqrt((w² - v²) / (w² + v²))
    """
    if v < 1e-10:
        return 0.0
    M_L = np.sqrt((w / v) ** 2 - 1)
    M_S = np.sqrt((w ** 2 - v ** 2) / (w ** 2 + v ** 2))
    J = v * M_L * M_S * (v / w) ** (ell / 2)
    return np.abs(J)


def transfer_time_estimate_N1(v, w, ell):
    """
    Analytical transfer time for single-domain chain (Eq. 13):
    t_tr = π * v / (2(w²-v²)) * (w/v)^(ell/2+2)
    """
    if v < 1e-10:
        return np.inf
    return np.pi * v / (2 * (w ** 2 - v ** 2)) * (w / v) ** (ell / 2 + 2)


def transfer_time_estimate_N2(v, w, ell):
    """
    Analytical transfer time for two-domain chain (Eq. 14):
    t_tr = π(w²+v²) / (2(w²-v²)) * (w/v)^(ell/2+1)
    """
    if v < 1e-10:
        return np.inf
    return (np.pi * (w ** 2 + v ** 2) /
            (2 * (w ** 2 - v ** 2)) * (w / v) ** (ell / 2 + 1))


# =============================================================================
# 7. VISUALIZATION FUNCTIONS
# =============================================================================

def plot_chain_diagram(ax, L_total, wall_pos, v, w, states_dict=None,
                       title=''):
    """Draw the SSH chain with domain wall at arbitrary position."""
    chain_y = -0.5
    wave_y = 0.3

    # Domain shading
    ax.axvspan(-0.4, wall_pos + 0.4, alpha=0.2, color='#DCEEFB', zorder=0)
    ax.axvspan(wall_pos - 0.4, L_total - 0.6, alpha=0.2, color='#FFF3CD',
               zorder=0)
    ax.text(wall_pos / 2, chain_y - 0.35, '$D_1$', ha='center', fontsize=9,
            color='#666', style='italic')
    ax.text((wall_pos + L_total - 1) / 2, chain_y - 0.35, '$D_2$',
            ha='center', fontsize=9, color='#666', style='italic')

    # Bonds
    for j in range(L_total - 1):
        if j < wall_pos:
            t_type = 'v' if j % 2 == 0 else 'w'
        else:
            local_j = j - wall_pos
            t_type = 'v' if local_j % 2 == 0 else 'w'
        lw = 1.0 if t_type == 'v' else 3.0
        cc = '#AAAAAA' if t_type == 'v' else '#444444'
        ax.plot([j, j + 1], [chain_y, chain_y], '-', color=cc, linewidth=lw,
                zorder=2)

    # Sites
    for j in range(L_total):
        mk = 'o' if j % 2 == 0 else 's'
        mc = '#4477AA' if j % 2 == 0 else '#CC6677'
        ax.plot(j, chain_y, mk, color=mc, markersize=5, zorder=5,
                markeredgecolor='black', markeredgewidth=0.4)

    # Domain wall marker
    ax.axvline(x=wall_pos, color='#FF8800', ls=':', alpha=0.7, lw=1.5)
    ax.text(wall_pos, chain_y + 0.18, 'DW', ha='center', fontsize=8,
            color='#FF8800', fontweight='bold')

    # Wavefunctions
    if states_dict is not None:
        sc = {'L': '#CC3311', 'S': '#EE7733', 'R': '#0077BB'}
        sl = {'L': r'$|\mathcal{L}\rangle$', 'S': r'$|\mathcal{S}\rangle$',
              'R': r'$|\mathcal{R}\rangle$'}

        max_a = max(np.max(np.abs(s)) for s in states_dict.values())
        scale = 1.2 / max_a if max_a > 0 else 1.0

        for key, state in states_dict.items():
            col = sc.get(key, '#009988')
            for j in range(L_total):
                a = state[j] * scale
                if abs(a) > 0.005:
                    ax.plot([j, j], [wave_y, wave_y + a], '-', color=col,
                            lw=1.8, alpha=0.85, zorder=4)
                    ax.plot(j, wave_y + a, 'o', color=col, markersize=3,
                            zorder=6)
            nz = np.where(np.abs(state) > 0.01)[0]
            if len(nz) > 0:
                pk = nz[np.argmax(np.abs(state[nz]))]
                ap = state[pk] * scale
                sg = np.sign(ap) if ap != 0 else 1
                ax.text(pk, wave_y + ap + 0.12 * sg, sl.get(key, key),
                        ha='center', fontsize=8, color=col, fontweight='bold')

    ax.axhline(y=wave_y, color='lightgray', lw=0.5, zorder=0)
    ax.set_xlim(-1, L_total)
    ax.set_ylim(chain_y - 0.55, wave_y + 1.8)
    ax.set_xlabel('Site index $j$')
    ax.set_title(title, pad=10)
    ax.set_yticks([])
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)


def plot_transfer_heatmap(ax, times, occupation, L, title):
    """Plot site occupation vs time as a heatmap."""
    im = ax.pcolormesh(times, np.arange(L), occupation.T,
                       shading='auto', cmap='inferno', vmin=0, vmax=1)
    ax.set_xlabel('Time $t$ ($\\hbar/J$)')
    ax.set_ylabel('Site $j$')
    ax.set_title(title)
    ax.set_ylim(-0.5, L - 0.5)
    plt.colorbar(im, ax=ax, label='$\\langle n_j \\rangle$', shrink=0.8)


def plot_pulse(ax, times, v_values, title='Control pulse'):
    """Plot the control pulse v(t)."""
    ax.plot(times, v_values, '-', color='#009988', lw=2)
    ax.fill_between(times, v_values, alpha=0.2, color='#009988')
    ax.set_xlabel('Time $t$ ($\\hbar/J$)')
    ax.set_ylabel('$v(t)$')
    ax.set_title(title)
    ax.set_xlim(times[0], times[-1])


# =============================================================================
# 8. MAIN — EXPERIMENTS
# =============================================================================

def main():
    print("=" * 72)
    print("PHASE 2: Asymmetric Domain Wall & Shortcut to Adiabaticity")
    print("=" * 72)

    w = 1.0
    v_tr = 0.5
    dt = 0.5  # Larger dt for efficiency; L=21 matrices are small enough

    # =====================================================================
    # EXPERIMENT 1: SPECTRUM AND STATES — ASYMMETRIC DOMAIN WALL
    # =====================================================================
    print("\n" + "=" * 72)
    print("EXPERIMENT 1: Asymmetric Domain Wall — Spectrum and States")
    print("=" * 72)

    L_total = 21  # 21-site chain → various wall positions possible

    # Define wall positions: center, 1/3, 1/4 from left
    # The wall must be at an ODD site for proper domain wall creation
    wall_positions = {
        'Center (~1/2)': 11,
        'Third (~1/3)': 7,
        'Quarter (~1/4)': 5,
    }

    fig1 = plt.figure(figsize=(FULL_W, 7.8))
    gs1 = gridspec.GridSpec(len(wall_positions), 2,
                           hspace=0.45, wspace=0.35,
                           width_ratios=[1.5, 1])

    for idx, (label, wp) in enumerate(wall_positions.items()):
        print(f"\n--- Domain wall at position {wp} ({label}), L={L_total} ---")

        # Build Hamiltonian and diagonalize
        H = build_ssh_two_domain_asymmetric(L_total, wp, v_tr, w)
        evals, evecs = np.linalg.eigh(H)

        # Find protected states (near E=0)
        n_prot = 3  # L, S, R
        idx_prot = np.argsort(np.abs(evals))[:n_prot]
        E_prot = evals[idx_prot]
        print(f"  Protected energies: {np.sort(E_prot)}")

        # Analytical states
        an_states = boundary_states_asymmetric(L_total, wp, v_tr, w)

        # Subspace overlap
        prot_space = evecs[:, idx_prot]
        keys = ['L', 'S', 'R']
        analyt_mat = np.column_stack([an_states[k] for k in keys])
        overlap = analyt_mat.T @ prot_space
        _, sv, _ = np.linalg.svd(overlap)
        print(f"  Subspace overlap (sing. vals): {sv}")

        # Bond pattern
        bp = []
        for j in range(L_total - 1):
            if j < wp:
                bp.append('v' if j % 2 == 0 else 'w')
            else:
                local_j = j - wp
                bp.append('v' if local_j % 2 == 0 else 'w')
        print(f"  Pattern: {''.join(bp)}")

        # Effective couplings
        ell_left = wp  # number of bonds in domain 1
        ell_right = L_total - 1 - wp  # number of bonds in domain 2
        print(f"  Domain 1: {ell_left} bonds, Domain 2: {ell_right} bonds")

        # Plot chain diagram
        ax_chain = fig1.add_subplot(gs1[idx, 0])
        plot_chain_diagram(ax_chain, L_total, wp, v_tr, w, an_states,
                          f'{label}: DW at site {wp}, L={L_total}')

        # Plot spectrum
        ax_spec = fig1.add_subplot(gs1[idx, 1])
        ax_spec.plot(evals, np.zeros_like(evals), '|', color='#888', ms=12)
        ax_spec.plot(np.sort(E_prot), np.zeros(n_prot), 'o', color='#CC3311',
                     ms=10, zorder=5, label=f'{n_prot} protected')
        ax_spec.set_xlabel('Energy $E$')
        ax_spec.set_title(f'Spectrum ({label})')
        ax_spec.set_yticks([])
        ax_spec.legend()
        for s in ['top', 'right', 'left']:
            ax_spec.spines[s].set_visible(False)

        # Print domain lengths
        print(f"  Eff. coupling L-S: J ~ (v/w)^({ell_left//2}) ~ "
              f"{(v_tr/w)**(ell_left//2):.6f}")
        print(f"  Eff. coupling S-R: J ~ (v/w)^({ell_right//2}) ~ "
              f"{(v_tr/w)**(ell_right//2):.6f}")

    fig1.suptitle('Effect of domain wall position\n'
                  'on protected states',
                  y=0.98)

    fig1.savefig(os.path.join(FIGURES_DIR, 'fig5_asymmetric_wall_states.pdf'),
                 bbox_inches='tight')
    print("\n  -> fig5_asymmetric_wall_states.pdf")
    plt.close(fig1)

    # =====================================================================
    # EXPERIMENT 2: TRANSFER WITH ASYMMETRIC WALL — HEATMAPS
    # =====================================================================
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: Transfer Protocol — Asymmetric Wall Positions")
    print("=" * 72)

    t_prep = 15.0

    fig2 = plt.figure(figsize=(FULL_W, 8.4))
    gs2 = gridspec.GridSpec(len(wall_positions), 2,
                           hspace=0.4, wspace=0.3)

    transfer_results = {}

    for idx, (label, wp) in enumerate(wall_positions.items()):
        # Estimate transfer time: use the larger domain to determine
        ell_left = wp
        ell_right = L_total - 1 - wp

        # The transfer time is dominated by the WEAKEST coupling,
        # which corresponds to the LONGER domain
        ell_max = max(ell_left, ell_right)

        # Scan to find optimal t_tr
        t_scan = np.arange(30, 600, 5.0)
        fidelities = scan_fidelity_vs_time(
            L_total, wp, v_pulse_standard, v_tr, w, t_scan, dt,
            t_prep=t_prep)

        best_idx = np.argmax(fidelities)
        best_t = t_scan[best_idx]
        best_f = fidelities[best_idx]
        print(f"\n--- {label}: DW={wp} ---")
        print(f"  Best t_tr = {best_t:.1f}, fidelity = {best_f:.6f}")

        transfer_results[label] = {
            'wall_pos': wp, 'best_t': best_t, 'best_f': best_f,
            't_scan': t_scan, 'fidelities': fidelities,
            'ell_left': ell_left, 'ell_right': ell_right
        }

        # Run transfer at optimal time
        times, occ, v_vals = time_evolve_asymmetric(
            L_total, wp, v_pulse_standard, v_tr, w, best_t, dt,
            t_prep=t_prep)

        # Heatmap
        ax_heat = fig2.add_subplot(gs2[idx, 0])
        plot_transfer_heatmap(ax_heat, times, occ, L_total,
                              f'{label}: DW={wp}, $t_{{tr}}$={best_t:.1f}')
        ax_heat.axhline(y=wp, color='#FF8800', ls='--', alpha=0.7, lw=1)
        ax_heat.text(best_t * 0.02, wp + 0.5, 'DW', fontsize=8,
                     color='#FF8800')

        # Pulse + final occupation
        ax_pulse = fig2.add_subplot(gs2[idx, 1])
        ax_pulse.plot(times, v_vals, '-', color='#009988', lw=2,
                      label='$v(t)$')
        ax_pulse.fill_between(times, v_vals, alpha=0.15, color='#009988')

        # Overlay occupation of key sites
        ax2 = ax_pulse.twinx()
        ax2.plot(times, occ[:, 0], '--', color='#CC3311', lw=1.5,
                 alpha=0.7, label='Site 0 (L)')
        ax2.plot(times, occ[:, wp], '--', color='#EE7733', lw=1.5,
                 alpha=0.7, label=f'Site {wp} (DW)')
        ax2.plot(times, occ[:, L_total - 1], '--', color='#0077BB', lw=1.5,
                 alpha=0.7, label=f'Site {L_total - 1} (R)')
        ax2.set_ylabel('Occupation')
        ax2.set_ylim(-0.05, 1.1)
        ax2.legend(fontsize=8, loc='center right')

        ax_pulse.set_xlabel('Time $t$ ($\\hbar/J$)')
        ax_pulse.set_ylabel('$v(t)$')
        ax_pulse.set_title(f'Pulse and occupation ({label})')
        ax_pulse.set_xlim(times[0], times[-1])

    fig2.suptitle('Transfer protocol with asymmetric domain wall\n'
                  f'$L={L_total}$, $v_{{tr}}={v_tr}$, $w={w}$',
                  y=0.99)

    fig2.savefig(os.path.join(FIGURES_DIR, 'fig6_asymmetric_transfer.pdf'),
                 bbox_inches='tight')
    print("\n  -> fig6_asymmetric_transfer.pdf")
    plt.close(fig2)

    # =====================================================================
    # EXPERIMENT 3: FIDELITY COMPARISON — WALL POSITION EFFECT
    # =====================================================================
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: Fidelity vs Transfer Time — Different Wall Positions")
    print("=" * 72)

    fig3, axes3 = plt.subplots(1, 2, figsize=(FULL_W, 2.63))

    colors = {'Center (~1/2)': '#0077BB', 'Third (~1/3)': '#EE7733',
              'Quarter (~1/4)': '#CC3311'}
    markers = {'Center (~1/2)': 'o', 'Third (~1/3)': 's',
               'Quarter (~1/4)': '^'}

    # (a) Fidelity vs t_tr
    ax3a = axes3[0]
    for label, res in transfer_results.items():
        ax3a.plot(res['t_scan'], res['fidelities'], '-',
                  color=colors[label], lw=1.5, label=label)
        ax3a.plot(res['best_t'], res['best_f'], markers[label],
                  color=colors[label], ms=8, zorder=5)

    ax3a.axhline(y=0.995, color='gray', ls='--', lw=1, alpha=0.7,
                 label='$f_0 = 0.995$')
    ax3a.set_xlabel('Transfer time $t_{tr}$ ($\\hbar/J$)')
    ax3a.set_ylabel('Fidelity $f$')
    ax3a.set_title('(a) Fidelity vs $t_{tr}$ for different DW positions')
    ax3a.legend()
    ax3a.set_ylim(-0.05, 1.1)

    # (b) Summary: optimal fidelity and time vs wall position
    ax3b = axes3[1]
    positions_frac = []
    opt_times = []
    opt_fids = []
    wall_idxs = []

    for label, res in transfer_results.items():
        frac = res['wall_pos'] / (L_total - 1)
        positions_frac.append(frac)
        opt_times.append(res['best_t'])
        opt_fids.append(res['best_f'])
        wall_idxs.append(res['wall_pos'])

    ax3b_twin = ax3b.twinx()
    bars = ax3b.bar(np.arange(len(positions_frac)) - 0.15, opt_fids,
                    width=0.3, color='#0077BB', alpha=0.7, label='Fidelity')
    bars2 = ax3b_twin.bar(np.arange(len(positions_frac)) + 0.15, opt_times,
                          width=0.3, color='#EE7733', alpha=0.7,
                          label='Optimal $t_{tr}$')

    labels_short = [f'DW={wi}\n({pf:.2f})' for wi, pf in
                    zip(wall_idxs, positions_frac)]
    ax3b.set_xticks(np.arange(len(positions_frac)))
    ax3b.set_xticklabels(labels_short)
    ax3b.set_ylabel('Maximum fidelity', color='#0077BB')
    ax3b_twin.set_ylabel('Optimal $t_{tr}$ ($\\hbar/J$)',
                         color='#EE7733')
    ax3b.set_title('(b) Summary: optimal fidelity and time')
    ax3b.set_ylim(0, 1.15)

    # Add value labels on bars
    for bar, val in zip(bars, opt_fids):
        ax3b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                  f'{val:.3f}', ha='center', fontsize=8, color='#0077BB')
    for bar, val in zip(bars2, opt_times):
        ax3b_twin.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 1,
                       f'{val:.1f}', ha='center', fontsize=8, color='#EE7733')

    fig3.suptitle('Effect of domain wall position on fidelity\n'
                  f'$L={L_total}$, $v_{{tr}}={v_tr}$, $w={w}$')
    fig3.tight_layout()
    fig3.savefig(os.path.join(FIGURES_DIR, 'fig7_fidelity_comparison.pdf'),
                 bbox_inches='tight')
    print("\n  -> fig7_fidelity_comparison.pdf")
    plt.close(fig3)

    # =====================================================================
    # EXPERIMENT 4: EFFECTIVE COUPLING ASYMMETRY ANALYSIS
    # =====================================================================
    print("\n" + "=" * 72)
    print("EXPERIMENT 4: Effective Coupling Asymmetry")
    print("=" * 72)

    fig4, axes4 = plt.subplots(1, 2, figsize=(FULL_W, 2.63))

    # (a) J_LS and J_SR vs wall position
    ax4a = axes4[0]
    wp_range = np.arange(1, L_total - 1, 2)  # odd positions only
    J_LS = np.zeros(len(wp_range))
    J_SR = np.zeros(len(wp_range))
    J_ratio = np.zeros(len(wp_range))

    for i, wp in enumerate(wp_range):
        ell_l = wp
        ell_r = L_total - 1 - wp
        J_LS[i] = effective_coupling_J(v_tr, w, ell_l)
        J_SR[i] = effective_coupling_J(v_tr, w, ell_r)
        J_ratio[i] = J_LS[i] / J_SR[i] if J_SR[i] > 1e-15 else np.inf

    ax4a.semilogy(wp_range, J_LS, 'o-', color='#CC3311', ms=6, lw=1.5,
                  label='$J_{LS}$ (left-wall)')
    ax4a.semilogy(wp_range, J_SR, 's-', color='#0077BB', ms=6, lw=1.5,
                  label='$J_{SR}$ (wall-right)')
    ax4a.set_xlabel('Wall position (site)')
    ax4a.set_ylabel('Effective coupling $|J|$')
    ax4a.set_title('(a) Effective couplings vs DW position')
    ax4a.legend()
    ax4a.axvline(x=(L_total - 1) / 2, color='gray', ls='--', alpha=0.5)
    ax4a.text((L_total - 1) / 2 + 0.3, ax4a.get_ylim()[1] * 0.5,
              'Center', fontsize=9, color='gray')
    ax4a.grid(True, alpha=0.3)

    # (b) J ratio
    ax4b = axes4[1]
    ax4b.plot(wp_range, J_ratio, 'D-', color='#009988', ms=6, lw=1.5)
    ax4b.axhline(y=1.0, color='gray', ls='--', alpha=0.5)
    ax4b.set_xlabel('Wall position (site)')
    ax4b.set_ylabel('$J_{LS} / J_{SR}$')
    ax4b.set_title('(b) Coupling asymmetry')
    ax4b.set_yscale('log')
    ax4b.grid(True, alpha=0.3)
    ax4b.text((L_total - 1) / 2 + 0.3, 1.3, 'Symmetric ($J_{LS}=J_{SR}$)',
              fontsize=9, color='gray')

    fig4.suptitle('Effective coupling analysis between protected states\n'
                  f'$L={L_total}$, $v={v_tr}$, $w={w}$')
    fig4.tight_layout()
    fig4.savefig(os.path.join(FIGURES_DIR, 'fig8_effective_coupling.pdf'),
                 bbox_inches='tight')
    print("\n  -> fig8_effective_coupling.pdf")
    plt.close(fig4)

    print("\n  Effective coupling for each config:")
    for label, res in transfer_results.items():
        J_ls = effective_coupling_J(v_tr, w, res['ell_left'])
        J_sr = effective_coupling_J(v_tr, w, res['ell_right'])
        ratio = J_ls / J_sr if J_sr > 1e-15 else np.inf
        print(f"  {label}: J_LS={J_ls:.6f}, J_SR={J_sr:.6f}, ratio={ratio:.4f}")

    # =====================================================================
    # EXPERIMENT 5: SHORTCUT TO ADIABATICITY — PULSE COMPARISON
    # =====================================================================
    print("\n" + "=" * 72)
    print("EXPERIMENT 5: Shortcut to Adiabaticity — Pulse Comparison")
    print("=" * 72)

    # Use symmetric N=2, ell=4 chain for clean comparison
    N_dom = 2
    ell = 4
    L_sym = N_dom * (ell + 1) + 1
    print(f"\n  Symmetric chain: N={N_dom}, ℓ={ell}, L={L_sym}")

    # Define pulse types to compare
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
        'STA α=2 (t_prep=10)': {
            'func': v_pulse_sta, 'kwargs': {'t_prep': 10.0, 'alpha': 2.0},
            'color': '#EE7733', 'ls': '--'
        },
        'STA α=3 (t_prep=10)': {
            'func': v_pulse_sta, 'kwargs': {'t_prep': 10.0, 'alpha': 3.0},
            'color': '#CC3311', 'ls': '--'
        },
        'STA global sin²': {
            'func': v_pulse_optimal_sta, 'kwargs': {},
            'color': '#009988', 'ls': '-.'
        },
        'Linear (t_prep=15)': {
            'func': v_pulse_linear, 'kwargs': {'t_prep': 15.0},
            'color': '#AA3377', 'ls': ':'
        },
    }

    # (a) Visualize pulse shapes
    fig5 = plt.figure(figsize=(FULL_W, 5.44))
    gs5 = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    ax5a = fig5.add_subplot(gs5[0, 0])
    t_demo = np.linspace(0, 50, 500)
    for name, cfg in pulse_configs.items():
        v_demo = [cfg['func'](t, v_tr, 50.0, **cfg['kwargs'])
                  for t in t_demo]
        ax5a.plot(t_demo, v_demo, cfg['ls'], color=cfg['color'], lw=2,
                  label=name)
    ax5a.set_xlabel('Time $t$ ($\\hbar/J$)')
    ax5a.set_ylabel('$v(t)$')
    ax5a.set_title('(a) Pulse profile comparison ($t_{tr}=50$)')
    ax5a.legend(fontsize=7, loc='upper right')

    # (b) Fidelity scan for each pulse type
    ax5b = fig5.add_subplot(gs5[0, 1])
    t_scan_sta = np.arange(25, 120, 2.0)

    sta_results = {}
    for name, cfg in pulse_configs.items():
        print(f"\n  Scanning: {name}...")
        fids = scan_fidelity_vs_time_symmetric(
            N_dom, ell, cfg['func'], v_tr, w, t_scan_sta, dt,
            **cfg['kwargs'])

        best_i = np.argmax(fids)
        best_t_sta = t_scan_sta[best_i]
        best_f_sta = fids[best_i]
        sta_results[name] = {
            'fidelities': fids, 'best_t': best_t_sta,
            'best_f': best_f_sta, 'config': cfg
        }
        print(f"    Best: t_tr={best_t_sta:.1f}, f={best_f_sta:.6f}")

        ax5b.plot(t_scan_sta, fids, cfg['ls'], color=cfg['color'], lw=1.5,
                  label=f'{name}: $f$={best_f_sta:.3f}', alpha=0.85)
        ax5b.plot(best_t_sta, best_f_sta, 'o', color=cfg['color'], ms=6,
                  zorder=5)

    ax5b.axhline(y=0.995, color='gray', ls='--', lw=1, alpha=0.7,
                 label='$f_0 = 0.995$')
    ax5b.set_xlabel('$t_{tr}$ ($\\hbar/J$)')
    ax5b.set_ylabel('Fidelity $f$')
    ax5b.set_title('(b) Fidelity vs $t_{tr}$ for different pulses')
    ax5b.legend(fontsize=6.5, loc='lower right')
    ax5b.set_ylim(-0.05, 1.1)

    # (c) and (d): Heatmaps for best standard and best STA
    # Pick best standard and best STA
    best_std_name = 'Standard (sin², t_prep=15)'
    best_sta_name = 'STA global sin²'

    std_res = sta_results[best_std_name]
    sta_res = sta_results[best_sta_name]

    # Run the two transfers
    print(f"\n  Running standard transfer: t_tr={std_res['best_t']:.1f}")
    times_std, occ_std, v_std, L_check = time_evolve_symmetric(
        N_dom, ell, std_res['config']['func'], v_tr, w,
        std_res['best_t'], dt, **std_res['config']['kwargs'])

    print(f"  Running STA transfer: t_tr={sta_res['best_t']:.1f}")
    times_sta, occ_sta, v_sta, _ = time_evolve_symmetric(
        N_dom, ell, sta_res['config']['func'], v_tr, w,
        sta_res['best_t'], dt, **sta_res['config']['kwargs'])

    ax5c = fig5.add_subplot(gs5[1, 0])
    plot_transfer_heatmap(ax5c, times_std, occ_std, L_sym,
        f'(c) Standard: $t_{{tr}}$={std_res["best_t"]:.1f}, '
        f'$f$={std_res["best_f"]:.4f}')

    ax5d = fig5.add_subplot(gs5[1, 1])
    plot_transfer_heatmap(ax5d, times_sta, occ_sta, L_sym,
        f'(d) STA global sin²: $t_{{tr}}$={sta_res["best_t"]:.1f}, '
        f'$f$={sta_res["best_f"]:.4f}')

    fig5.suptitle('Transfer protocol comparison\n'
                  f'Symmetric chain $N={N_dom}$, $\\ell={ell}$, '
                  f'$L={L_sym}$', y=1.01)

    fig5.savefig(os.path.join(FIGURES_DIR, 'fig9_sta_pulse_comparison.pdf'),
                 bbox_inches='tight')
    print("\n  -> fig9_sta_pulse_comparison.pdf")
    plt.close(fig5)

    # =====================================================================
    # EXPERIMENT 6: STA — PREPARATION TIME DEPENDENCE
    # =====================================================================
    print("\n" + "=" * 72)
    print("EXPERIMENT 6: Effect of Preparation Time on Fidelity")
    print("=" * 72)

    fig6, axes6 = plt.subplots(1, 2, figsize=(FULL_W, 2.63))

    # (a) Fidelity vs t_prep for standard pulse at fixed t_tr
    t_tr_fixed = 45.6  # from the paper
    t_prep_range = np.linspace(2, 25, 15)

    fid_vs_tprep_std = np.zeros(len(t_prep_range))
    fid_vs_tprep_sta2 = np.zeros(len(t_prep_range))
    fid_vs_tprep_sta3 = np.zeros(len(t_prep_range))

    for i, tp in enumerate(t_prep_range):
        if 2 * tp >= t_tr_fixed:
            fid_vs_tprep_std[i] = np.nan
            fid_vs_tprep_sta2[i] = np.nan
            fid_vs_tprep_sta3[i] = np.nan
            continue

        # Standard
        _, occ, _, _ = time_evolve_symmetric(
            N_dom, ell, v_pulse_standard, v_tr, w, t_tr_fixed, dt,
            t_prep=tp)
        fid_vs_tprep_std[i] = occ[-1, L_sym - 1]

        # STA alpha=2
        _, occ, _, _ = time_evolve_symmetric(
            N_dom, ell, v_pulse_sta, v_tr, w, t_tr_fixed, dt,
            t_prep=tp, alpha=2.0)
        fid_vs_tprep_sta2[i] = occ[-1, L_sym - 1]

        # STA alpha=3
        _, occ, _, _ = time_evolve_symmetric(
            N_dom, ell, v_pulse_sta, v_tr, w, t_tr_fixed, dt,
            t_prep=tp, alpha=3.0)
        fid_vs_tprep_sta3[i] = occ[-1, L_sym - 1]

    ax6a = axes6[0]
    ax6a.plot(t_prep_range, fid_vs_tprep_std, '-', color='#0077BB', lw=2,
              label='Standard (sin²)')
    ax6a.plot(t_prep_range, fid_vs_tprep_sta2, '--', color='#EE7733', lw=2,
              label='STA $\\alpha=2$')
    ax6a.plot(t_prep_range, fid_vs_tprep_sta3, '--', color='#CC3311', lw=2,
              label='STA $\\alpha=3$')
    ax6a.axhline(y=0.995, color='gray', ls='--', lw=1, alpha=0.7)
    ax6a.axvline(x=8, color='lightgray', ls=':', lw=1, alpha=0.7)
    ax6a.text(8.5, 0.1, '$\\tau \\sim 8$ (adiabatic\nscale)', fontsize=8,
              color='gray')
    ax6a.set_xlabel('Preparation time $t_{prep}$ ($\\hbar/J$)')
    ax6a.set_ylabel('Fidelity $f$')
    ax6a.set_title(f'(a) Fidelity vs $t_{{prep}}$ ($t_{{tr}}={t_tr_fixed}$)')
    ax6a.legend()
    ax6a.set_ylim(-0.05, 1.1)

    # (b) Gap analysis: why t_prep > tau is needed
    ax6b = axes6[1]
    v_range = np.linspace(0.01, 0.99, 100)
    gaps = np.zeros(len(v_range))

    for i, v_val in enumerate(v_range):
        H, L_gap = build_ssh_multidomain(N_dom, ell, v_val, w)
        evals_gap = np.linalg.eigvalsh(H)
        # Gap = distance from protected states to bulk
        sorted_e = np.sort(np.abs(evals_gap))
        if len(sorted_e) > 3:
            gaps[i] = sorted_e[3] - sorted_e[2]  # gap between protected and bulk

    tau_char = 2.0 / gaps  # characteristic time τ = 2/Δ

    ax6b.plot(v_range, gaps, '-', color='#0077BB', lw=2, label='Gap $\\Delta$')
    ax6b_twin = ax6b.twinx()
    ax6b_twin.plot(v_range, tau_char, '-', color='#CC3311', lw=2,
                   label='$\\tau = 2/\\Delta$')
    ax6b.axvline(x=v_tr, color='gray', ls='--', alpha=0.5)
    ax6b.text(v_tr + 0.02, ax6b.get_ylim()[1] * 0.9, f'$v_{{tr}}={v_tr}$',
              fontsize=9, color='gray')

    ax6b.set_xlabel('$v$')
    ax6b.set_ylabel('Gap $\\Delta$', color='#0077BB')
    ax6b_twin.set_ylabel('Adiabatic timescale $\\tau$ ($\\hbar/J$)',
                         color='#CC3311')
    ax6b.set_title('(b) Gap and adiabatic timescale')
    ax6b.legend(loc='upper left')
    ax6b_twin.legend(loc='upper right')

    fig6.suptitle('Fidelity dependence on preparation time\n'
                  f'$N={N_dom}$, $\\ell={ell}$, $L={L_sym}$')
    fig6.tight_layout()
    fig6.savefig(os.path.join(FIGURES_DIR, 'fig10_tprep_dependence.pdf'),
                 bbox_inches='tight')
    print("\n  -> fig10_tprep_dependence.pdf")
    plt.close(fig6)

    # =====================================================================
    # EXPERIMENT 7: COMBINED — ASYMMETRIC WALL + STA PROTOCOLS
    # =====================================================================
    print("\n" + "=" * 72)
    print("EXPERIMENT 7: Asymmetric Wall + STA Combined Study")
    print("=" * 72)

    fig7 = plt.figure(figsize=(FULL_W, 5.44))
    gs7 = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # For each wall position, compare standard vs STA
    protocols = {
        'Standard (sin², t_prep=15)': {
            'func': v_pulse_standard, 'kwargs': {'t_prep': 15.0},
            'color': '#0077BB', 'ls': '-', 'marker': 'o'
        },
        'STA α=2 (t_prep=10)': {
            'func': v_pulse_sta, 'kwargs': {'t_prep': 10.0, 'alpha': 2.0},
            'color': '#EE7733', 'ls': '--', 'marker': 's'
        },
        'STA global sin²': {
            'func': v_pulse_optimal_sta, 'kwargs': {},
            'color': '#009988', 'ls': '-.', 'marker': '^'
        },
    }

    # (a) Summary table: best fidelity and time for each combo
    ax7a = fig7.add_subplot(gs7[0, :])

    t_scan_combined = np.arange(30, 600, 5.0)
    summary_data = []

    for label_w, wp in wall_positions.items():
        for label_p, proto in protocols.items():
            print(f"\n  {label_w} + {label_p}...")
            fids = scan_fidelity_vs_time(
                L_total, wp, proto['func'], v_tr, w, t_scan_combined, dt,
                **proto['kwargs'])
            best_i = np.argmax(fids)
            best_t_c = t_scan_combined[best_i]
            best_f_c = fids[best_i]
            summary_data.append({
                'wall': label_w, 'wall_pos': wp,
                'protocol': label_p, 'best_t': best_t_c,
                'best_f': best_f_c
            })
            print(f"    t_tr={best_t_c:.1f}, f={best_f_c:.6f}")

    # Create summary table as text in axes
    ax7a.axis('off')
    table_data = []
    for sd in summary_data:
        table_data.append([
            sd['wall'], f'DW={sd["wall_pos"]}', sd['protocol'],
            f'{sd["best_t"]:.1f}', f'{sd["best_f"]:.4f}'
        ])

    table = ax7a.table(
        cellText=table_data,
        colLabels=['DW position', 'Site', 'Protocol',
                   '$t_{tr}$ opt.', 'Fidelity'],
        loc='center',
        cellLoc='center',
        colWidths=[0.18, 0.1, 0.28, 0.12, 0.12]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.6)

    # Color the cells by fidelity
    for i, sd in enumerate(summary_data):
        f_val = sd['best_f']
        if f_val > 0.99:
            color = '#D4EDDA'
        elif f_val > 0.95:
            color = '#FFF3CD'
        else:
            color = '#F8D7DA'
        for j in range(5):
            table[i + 1, j].set_facecolor(color)

    ax7a.set_title('(a) Results summary: DW position × protocol',
                   pad=20)

    # (b) Fidelity curves for center wall: std vs STA
    ax7b = fig7.add_subplot(gs7[1, 0])
    wp_center = wall_positions['Center (~1/2)']
    for label_p, proto in protocols.items():
        fids = scan_fidelity_vs_time(
            L_total, wp_center, proto['func'], v_tr, w,
            t_scan_combined, dt, **proto['kwargs'])
        ax7b.plot(t_scan_combined, fids, proto['ls'], color=proto['color'],
                  lw=1.5, label=label_p)
    ax7b.axhline(y=0.995, color='gray', ls='--', lw=1, alpha=0.5)
    ax7b.set_xlabel('$t_{tr}$ ($\\hbar/J$)')
    ax7b.set_ylabel('Fidelity')
    ax7b.set_title(f'(b) DW center (site {wp_center}): protocol comparison')
    ax7b.legend(fontsize=8)
    ax7b.set_ylim(-0.05, 1.1)

    # (c) Fidelity curves for 1/4 wall: std vs STA
    ax7c = fig7.add_subplot(gs7[1, 1])
    wp_quarter = wall_positions['Quarter (~1/4)']
    for label_p, proto in protocols.items():
        fids = scan_fidelity_vs_time(
            L_total, wp_quarter, proto['func'], v_tr, w,
            t_scan_combined, dt, **proto['kwargs'])
        ax7c.plot(t_scan_combined, fids, proto['ls'], color=proto['color'],
                  lw=1.5, label=label_p)
    ax7c.axhline(y=0.995, color='gray', ls='--', lw=1, alpha=0.5)
    ax7c.set_xlabel('$t_{tr}$ ($\\hbar/J$)')
    ax7c.set_ylabel('Fidelity')
    ax7c.set_title(f'(c) DW quarter (site {wp_quarter}): protocol comparison')
    ax7c.legend(fontsize=8)
    ax7c.set_ylim(-0.05, 1.1)

    fig7.suptitle('Combined study: asymmetric position + Shortcut to Adiabaticity\n'
                  f'$L={L_total}$, $v_{{tr}}={v_tr}$, $w={w}$',
                  y=1.01)

    fig7.savefig(os.path.join(FIGURES_DIR, 'fig11_combined_study.pdf'),
                 bbox_inches='tight')
    print("\n  -> fig11_combined_study.pdf")
    plt.close(fig7)

    # =====================================================================
    # EXPERIMENT 8: DOMAIN LENGTH EFFECT ON ASYMMETRY
    # =====================================================================
    print("\n" + "=" * 72)
    print("EXPERIMENT 8: Systematic Study — Fidelity vs Wall Position")
    print("=" * 72)

    fig8, axes8 = plt.subplots(1, 2, figsize=(FULL_W, 2.63))

    # Study for multiple chain lengths
    L_values = [11, 15]
    t_prep_sys = 15.0

    ax8a = axes8[0]
    ax8b = axes8[1]

    colors_L = {11: '#0077BB', 15: '#CC3311'}

    for L_val in L_values:
        wp_range_sys = np.arange(1, L_val - 1, 2)
        best_fids_sys = np.zeros(len(wp_range_sys))
        best_times_sys = np.zeros(len(wp_range_sys))

        print(f"\n  L = {L_val}, scanning {len(wp_range_sys)} positions...")
        for i, wp in enumerate(wp_range_sys):
            t_scan_sys = np.arange(30, 400, 10.0)
            fids = scan_fidelity_vs_time(
                L_val, wp, v_pulse_standard, v_tr, w, t_scan_sys, dt,
                t_prep=t_prep_sys)
            best_i = np.argmax(fids)
            best_fids_sys[i] = fids[best_i]
            best_times_sys[i] = t_scan_sys[best_i]
            print(f"    DW={wp}: f={fids[best_i]:.4f}, t={t_scan_sys[best_i]:.0f}")

        # Normalize wall position to fraction
        fracs = wp_range_sys / (L_val - 1)

        ax8a.plot(fracs, best_fids_sys, 'o-', color=colors_L[L_val],
                  ms=5, lw=1.5, label=f'$L={L_val}$')
        ax8b.plot(fracs, best_times_sys, 's-', color=colors_L[L_val],
                  ms=5, lw=1.5, label=f'$L={L_val}$')

    ax8a.axhline(y=0.995, color='gray', ls='--', lw=1, alpha=0.5)
    ax8a.set_xlabel('Relative DW position ($j_{DW}/(L-1)$)')
    ax8a.set_ylabel('Maximum fidelity')
    ax8a.set_title('(a) Fidelity vs relative DW position')
    ax8a.legend()
    ax8a.set_ylim(0, 1.1)

    ax8b.set_xlabel('Relative DW position ($j_{DW}/(L-1)$)')
    ax8b.set_ylabel('Optimal $t_{tr}$ ($\\hbar/J$)')
    ax8b.set_title('(b) Transfer time vs DW position')
    ax8b.legend()

    fig8.suptitle('Systematic study: effect of wall position and chain size\n'
                  f'$v_{{tr}}={v_tr}$, $w={w}$')
    fig8.tight_layout()
    fig8.savefig(os.path.join(FIGURES_DIR, 'fig12_systematic_wall_position.pdf'),
                 bbox_inches='tight')
    print("\n  -> fig12_systematic_wall_position.pdf")
    plt.close(fig8)

    print("\n" + "=" * 72)
    print("All figures generated successfully.")
    print("=" * 72)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
ctap_full_chain.py  --  CTAP on the full SSH chain, multi-domain generalization.

Core building blocks:
  build_hoppings   -- bond array for D domains (1 or more domain walls)
  build_H          -- tridiagonal Hamiltonian from hopping array
  protected_states -- D+1 near-zero eigenstates sorted by localization site
  protected_triplet -- convenience wrapper for D=2 (3 states)
  metrics          -- P_L, P_S, P_R, leakage, fidelity (2-domain / 3-state)
  evolve           -- exact-exponentiation Schrodinger time stepper
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.linalg import expm, eigh


# ---------------------------------------------------------------------------
# Hamiltonian builders
# ---------------------------------------------------------------------------

def build_hoppings(L, j_DW_list, v, w):
    """
    Return array t of L-1 bond strengths for a multi-domain SSH chain.

    Parameters
    ----------
    L          : int -- chain length (number of sites)
    j_DW_list  : int or list of ints -- domain-wall bond positions.
                 Wall k (k=0,1,2,...) must satisfy j[k] % 2 == (k+1) % 2,
                 i.e. walls alternate odd, even, odd, ... so that both bonds
                 adjacent to each wall are intracell (vv defect per interface).
    v          : float or list of D floats -- intracell hopping per domain
                 (D = len(walls)+1; scalar broadcasts to all domains)
    w          : float -- intercell hopping (fixed, same everywhere)

    Domain k (even k): bond j -> v[k] if j%2==0, w if j%2==1
    Domain k (odd  k): bond j -> v[k] if j%2==1, w if j%2==0
    """
    if isinstance(j_DW_list, (int, np.integer)):
        walls = [int(j_DW_list)]
    else:
        walls = sorted(int(x) for x in j_DW_list)
    for k, jw in enumerate(walls):
        expected = (k + 1) % 2
        if jw % 2 != expected:
            raise ValueError(
                f"Wall {k} at j={jw}: parity {jw%2}, expected {expected} "
                f"(walls must alternate odd, even, odd, ... for vv defects)"
            )
    D = len(walls) + 1
    if np.isscalar(v):
        v_arr = [float(v)] * D
    else:
        v_arr = [float(x) for x in v]
        if len(v_arr) != D:
            raise ValueError(f"Expected {D} v values for {D} domains, got {len(v_arr)}")

    t = np.empty(L - 1)
    for j in range(L - 1):
        domain = sum(1 for jw in walls if j >= jw)
        vk = v_arr[domain]
        t[j] = vk if j % 2 == (0 if domain % 2 == 0 else 1) else w
    return t


def build_H(hoppings):
    """Tridiagonal Hamiltonian H_ij = -t_j * (delta_{i,j+1} + delta_{i+1,j})."""
    L = len(hoppings) + 1
    H = np.zeros((L, L))
    for j, t in enumerate(hoppings):
        H[j, j + 1] = -t
        H[j + 1, j] = -t
    return H


def build_H_with_NNN(hoppings, kappa=0.0, onsite=None):
    """
    NN Hamiltonian plus optional NNN hoppings (kappa) and on-site energies.

    NNN: H[j,j+2] = -kappa for all j.  Connects same-sublattice sites
    (A-A and B-B) so it does NOT anticommute with the chiral operator Gamma
    and breaks chiral symmetry.  On-site energies (diagonal) also break it.
    With kappa=0 and onsite=None this is identical to build_H.
    """
    L = len(hoppings) + 1
    H = np.zeros((L, L))
    for j, t in enumerate(hoppings):
        H[j, j + 1] = -t
        H[j + 1, j] = -t
    if kappa != 0.0:
        for j in range(L - 2):
            H[j, j + 2] -= kappa
            H[j + 2, j] -= kappa
    if onsite is not None:
        np.fill_diagonal(H, H.diagonal() + np.asarray(onsite, dtype=float))
    return H


# ---------------------------------------------------------------------------
# Protected states (generalized)
# ---------------------------------------------------------------------------

def protected_states(H, n):
    """
    Return list of n eigenstates of H with smallest |E|,
    sorted by argmax(|psi|^2) (ascending site index).
    """
    vals, vecs = eigh(H)
    idx = np.argsort(np.abs(vals))[:n]
    states = [(np.argmax(np.abs(vecs[:, i])**2), vecs[:, i]) for i in idx]
    states.sort(key=lambda x: x[0])
    return [phi for _, phi in states]


def protected_triplet(H):
    """
    Return dict {'L': phi_L, 'S': phi_S, 'R': phi_R} -- the three
    eigenstates of H with smallest |E|, labeled by argmax(|psi|^2).
    """
    phis = protected_states(H, 3)
    return {'L': phis[0], 'S': phis[1], 'R': phis[2]}


# ---------------------------------------------------------------------------
# Metrics  (2-domain / 3-state interface)
# ---------------------------------------------------------------------------

def metrics(psi, H):
    """
    Project psi onto the instantaneous protected triplet of H.

    Returns
    -------
    P_L, P_S, P_R : float -- populations on each protected state
    leakage       : float -- 1 - (P_L + P_S + P_R)
    fid           : float -- |<L-1|psi>|^2  (bare site fidelity)
    """
    triplet = protected_triplet(H)
    P_L = float(np.abs(triplet['L'] @ psi)**2)
    P_S = float(np.abs(triplet['S'] @ psi)**2)
    P_R = float(np.abs(triplet['R'] @ psi)**2)
    fid = float(np.abs(psi[-1])**2)
    return P_L, P_S, P_R, 1.0 - (P_L + P_S + P_R), fid


# ---------------------------------------------------------------------------
# Time evolution
# ---------------------------------------------------------------------------

def evolve(psi0, H_of_t, T, dt=0.25):
    """
    Evolve psi0 under H(t) from t=0 to t=T using exact matrix exponentiation.

    Parameters
    ----------
    psi0   : (L,) complex array -- initial state
    H_of_t : callable t -> (L,L) real/complex array
    T      : float -- total time
    dt     : float -- time step (<= 0.25 recommended)

    Returns
    -------
    times : (N,) array
    psis  : (N, L) complex array -- wavefunction at each step
    """
    n_steps = max(int(np.ceil(T / dt)), 1)
    dt_actual = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)
    psis = np.empty((n_steps + 1, len(psi0)), dtype=complex)
    psis[0] = psi0.astype(complex)

    psi = psis[0].copy()
    for i in range(1, n_steps + 1):
        t_mid = times[i] - 0.5 * dt_actual
        H = H_of_t(t_mid)
        U = expm(-1j * H * dt_actual)
        psi = U @ psi
        psis[i] = psi

    return times, psis


# ---------------------------------------------------------------------------
# Helpers for sanity tests
# ---------------------------------------------------------------------------

def _segment_params(D, n_seg):
    """
    Equal-segment chain: D domains of n_seg bonds each.

    n_seg must be odd so walls alternate odd/even/odd parity correctly.
    Wall k is at bond position n_seg*(k+1), which is odd for k even and even
    for k odd -- satisfying the vv-defect parity rule automatically.

    Returns (L, walls).
    """
    if n_seg % 2 == 0:
        raise ValueError(f"n_seg must be odd for vv defects at all walls, got {n_seg}")
    L = n_seg * D + 1
    walls = [n_seg * (k + 1) for k in range(D - 1)]
    return L, walls


def _effective_couplings(H, n_states):
    """
    Nearest-neighbour couplings in the maximally localized (Wannier) basis.

    Projects the position operator onto the n_states near-zero eigenvectors and
    diagonalizes it to obtain states ordered by increasing <X>.  Off-diagonal
    elements of H in this localized basis give the physical couplings.
    """
    L = H.shape[0]
    vals, vecs = eigh(H)
    idx = np.argsort(np.abs(vals))[:n_states]
    Phi = vecs[:, idx]                         # (L, n_states) eigenstates

    X_diag = np.arange(L, dtype=float)
    X_sub = Phi.T @ (X_diag[:, None] * Phi)   # position matrix in subspace
    _, x_evecs = np.linalg.eigh(X_sub)         # sorted by ascending <X>
    Phi_loc = Phi @ x_evecs                    # (L, n_states) localized states

    H_sub = Phi_loc.T @ H @ Phi_loc            # effective H in localized basis
    return [float(abs(H_sub[k, k + 1])) for k in range(n_states - 1)]


# ---------------------------------------------------------------------------
# Sanity tests -- multi-DW generalization
# ---------------------------------------------------------------------------

def _run_tests():
    v, w = 0.5, 1.0
    n_seg = 11  # bonds per domain (odd, so walls alternate parity correctly)
    E_thresh = 0.1
    passed = 0
    failed = 0

    # ------------------------------------------------------------------
    # TEST 1: H hermitiana para varias configuraciones multi-DW
    # ------------------------------------------------------------------
    print("TEST 1  H hermitiana (D=2,3,4):")
    all_ok = True
    for D in [2, 3, 4]:
        L, walls = _segment_params(D, n_seg)
        t_bonds = build_hoppings(L, walls, v, w)
        H = build_H(t_bonds)
        err = float(np.max(np.abs(H - H.T)))
        ok_d = err < 1e-14
        if not ok_d:
            all_ok = False
        print(f"  D={D}  L={L}  walls={walls}  ||H-H^dag||={err:.2e}  {'PASS' if ok_d else 'FAIL'}")
    ok = all_ok
    print(f"TEST 1  {'PASS' if ok else 'FAIL'}")
    if ok:
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    # TEST 2: D dominios -> exactamente D+1 autovalores cerca de E=0
    # ------------------------------------------------------------------
    print("\nTEST 2  conteo D+1 estados near-zero (D=2,3,4):")
    all_ok = True
    for D in [2, 3, 4]:
        L, walls = _segment_params(D, n_seg)
        t_bonds = build_hoppings(L, walls, v, w)
        H = build_H(t_bonds)
        evals = np.abs(np.linalg.eigvalsh(H))
        n_near = int(np.sum(evals < E_thresh))
        ok_d = n_near == D + 1
        if not ok_d:
            all_ok = False
        five = np.sort(evals)[:5].round(6)
        print(f"  D={D}  L={L}  near-zero={n_near}  expected={D+1}  "
              f"smallest5={five}  {'PASS' if ok_d else 'FAIL'}")
    ok = all_ok
    print(f"TEST 2  {'PASS' if ok else 'FAIL'}")
    if ok:
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    # TEST 3: localizacion -- borde izq, cada DW, borde der
    # ------------------------------------------------------------------
    # For symmetric chains the eigenstates hybridize (e.g. two ±J√2 states both
    # peak at the DW site).  Use the collective density Σ_k|φ_k|² which sums
    # to exactly 1 at each localization site regardless of basis choice.
    print("\nTEST 3  localizacion de los D+1 estados (D=2,3,4):")
    all_ok = True
    for D in [2, 3, 4]:
        L, walls = _segment_params(D, n_seg)
        t_bonds = build_hoppings(L, walls, v, w)
        H = build_H(t_bonds)
        phis = protected_states(H, D + 1)
        density = sum(np.abs(phi)**2 for phi in phis)   # collective near-zero DOS
        peak_sites = sorted(int(x) for x in np.argsort(density)[::-1][:D + 1])
        expected = sorted([0] + list(walls) + [L - 1])
        tol = 3
        ok_d = all(abs(p - e) <= tol for p, e in zip(peak_sites, expected))
        if not ok_d:
            all_ok = False
        print(f"  D={D}  density_peaks={peak_sites}  expected={expected}  "
              f"{'PASS' if ok_d else 'FAIL'}")
    ok = all_ok
    print(f"TEST 3  {'PASS' if ok else 'FAIL'}")
    if ok:
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    # TEST 4: caso simetrico -- acoplos efectivos aproximadamente iguales
    # ------------------------------------------------------------------
    print("\nTEST 4  acoplos efectivos iguales en caso simetrico (D=2,3,4):")
    all_ok = True
    for D in [2, 3, 4]:
        L, walls = _segment_params(D, n_seg)
        t_bonds = build_hoppings(L, walls, v, w)
        H = build_H(t_bonds)
        Js = _effective_couplings(H, D + 1)
        ratio = max(Js) / min(Js) if min(Js) > 0 else float('inf')
        ok_d = ratio < 1.20  # edge-DW slightly > DW-DW (bilateral vs unilateral decay)
        if not ok_d:
            all_ok = False
        Js_str = [f"{j:.6e}" for j in Js]
        print(f"  D={D}  J={Js_str}  max/min={ratio:.5f}  {'PASS' if ok_d else 'FAIL'}")
    ok = all_ok
    print(f"TEST 4  {'PASS' if ok else 'FAIL'}")
    if ok:
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    # TEST 5: cross-check speedup -- J_per_seg(l) >> J_single(D*l)
    # ------------------------------------------------------------------
    print("\nTEST 5  speedup J_per_seg vs J_single (misma longitud total):")
    D_m = 3
    L_m, walls_m = _segment_params(D_m, n_seg)
    Js_m = _effective_couplings(build_H(build_hoppings(L_m, walls_m, v, w)), D_m + 1)
    J_per_seg = float(np.mean(Js_m))

    # Single domain spanning the same total length (no walls)
    H_s = build_H(build_hoppings(L_m, [], v, w))
    J_single = _effective_couplings(H_s, 2)[0]

    ok = J_per_seg > J_single
    ratio5 = J_per_seg / J_single if J_single > 0 else float('inf')
    print(f"  D=3  L={L_m}  J_per_seg={J_per_seg:.6e}  J_single={J_single:.6e}  "
          f"ratio={ratio5:.1f}x  {'PASS' if ok else 'FAIL'}")
    print(f"TEST 5  {'PASS' if ok else 'FAIL'}")
    if ok:
        passed += 1
    else:
        failed += 1

    print(f"\n{'='*56}")
    print(f"RESULTADO: {passed} PASS, {failed} FAIL")
    if failed > 0:
        print("STOP -- corrige los fallos antes de continuar.")
    else:
        print("Todos los tests superados. OK para el siguiente paso.")
    return failed == 0


if __name__ == '__main__':
    _run_tests()

#!/usr/bin/env python3
"""
paso4_chiral_spectral.py  --  Spectral demonstration of chiral protection.

Chain: L=21, j_DW=11, v=0.5, w=1.0.
Zero mode = eigenvalue closest to E=0.

A. Off-diagonal disorder t_j -> t_j*(1+delta_j):
     NN only      (kappa=0):   zero mode PINNED at E=0  (chiral preserved)
     NN + NNN (kappa=0.1w):   zero mode DETACHED        (NNN breaks chiral)
B. On-site disorder, NN only:  zero mode DETACHED        (onsite breaks chiral)
C. NNN only, no disorder:      zero mode DETACHED, grows with kappa

Spectral symmetry check: |E_k + E_{N-1-k}| -> 0 iff chiral symmetry holds.
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fase_2_domain_wall_asimetrico.ctap_full_chain import (
    build_hoppings, build_H_with_NNN
)

# ── Parameters ──────────────────────────────────────────────────────────────
L, j_DW  = 21, 11
v, w     = 0.5, 1.0
kappa_A  = 0.1 * w         # NNN magnitude used in case A comparison
N_real   = 200
sigmas   = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
kappas   = np.arange(0, 0.21, 0.02) * w

t0  = build_hoppings(L, j_DW, v, w)   # clean NN hoppings
rng = np.random.default_rng(0)


def E0(H):
    """Smallest |eigenvalue| = distance of zero mode from E=0."""
    return float(np.min(np.abs(np.linalg.eigvalsh(H))))


def spec_asym(H):
    """Spectral asymmetry: max_k |E_k + E_{N-1-k}|.  Zero iff chiral holds."""
    e = np.sort(np.linalg.eigvalsh(H))
    return float(np.max(np.abs(e + e[::-1])))


print(f"L={L}  j_DW={j_DW}  v={v}  w={w}  N_real={N_real}")
print(f"kappa_A={kappa_A:.3f}  (for case A NN+NNN comparison)")
print()

# ── Verify: 3 near-zero modes exist, check bipartite structure ──────────────
H0 = build_H_with_NNN(t0)
evals0 = np.sort(np.abs(np.linalg.eigvalsh(H0)))
print(f"Clean chain: 5 smallest |E| = {evals0[:5].round(8)}")
print(f"Exact zero mode count (|E|<1e-10): {int(np.sum(evals0 < 1e-10))}")
print(f"Near-zero count (|E|<0.01):        {int(np.sum(evals0 < 0.01))}")
print()

# ── Case A: off-diagonal disorder ───────────────────────────────────────────
print("Case A  off-diagonal disorder  (t_j -> t_j*(1+delta_j), delta_j~U[-s,s])")
print(f"  {'sigma':>6}  {'<|E0|>_NN':>14}  {'<|E0|>_NN+NNN':>16}")
rows_A = []
for sig in sigmas:
    e_nn, e_nnn = [], []
    for _ in range(N_real):
        d  = rng.uniform(-sig, sig, L - 1)
        td = t0 * (1.0 + d)
        e_nn.append(E0(build_H_with_NNN(td)))
        e_nnn.append(E0(build_H_with_NNN(td, kappa=kappa_A)))
    m_nn   = float(np.mean(e_nn))
    m_nnn  = float(np.mean(e_nnn))
    rows_A.append((sig, m_nn, m_nnn))
    tag_nn  = "PINNED" if m_nn  < 1e-10 else f"detached"
    tag_nnn = "PINNED" if m_nnn < 1e-10 else f"detached"
    print(f"  {sig:>6.2f}  {m_nn:>14.3e} ({tag_nn})  {m_nnn:>16.6f} ({tag_nnn})")

# ── Case B: on-site disorder ─────────────────────────────────────────────────
print()
print("Case B  on-site disorder  (eps_j~U[-s,s] on diagonal, NN only)")
print(f"  {'sigma':>6}  {'<|E0|>':>12}")
rows_B = []
for sig in sigmas:
    e = []
    for _ in range(N_real):
        eps = rng.uniform(-sig, sig, L)
        e.append(E0(build_H_with_NNN(t0, onsite=eps)))
    m = float(np.mean(e))
    rows_B.append((sig, m))
    tag = "PINNED" if m < 1e-10 else "detached"
    print(f"  {sig:>6.2f}  {m:>12.6f} ({tag})")

# ── Case C: NNN only, no disorder ────────────────────────────────────────────
print()
print("Case C  NNN only  no disorder  (clean chain + kappa)")
print(f"  {'kappa/w':>8}  {'|E0|':>12}")
rows_C = []
for k in kappas:
    e0 = E0(build_H_with_NNN(t0, kappa=k))
    rows_C.append((float(k), e0))
    tag = "PINNED" if e0 < 1e-10 else "detached"
    print(f"  {k/w:>8.2f}  {e0:>12.6f} ({tag})")

# ── Spectral symmetry check ──────────────────────────────────────────────────
print()
print("Spectral asymmetry  max_k |E_k + E_{N-1-k}|  (zero = chiral)")
rng2    = np.random.default_rng(7)
sig_t   = 0.2
d_t     = rng2.uniform(-sig_t, sig_t, L - 1)
eps_t   = rng2.uniform(-sig_t, sig_t, L)

cases_sym = [
    ("Clean NN",                    build_H_with_NNN(t0)),
    (f"NN + off-diag (s={sig_t})",  build_H_with_NNN(t0 * (1 + d_t))),
    (f"NN + onsite   (s={sig_t})",  build_H_with_NNN(t0, onsite=eps_t)),
    (f"NN + NNN (k={kappa_A:.2f})", build_H_with_NNN(t0, kappa=kappa_A)),
    ("NN+NNN+off-diag",             build_H_with_NNN(t0*(1+d_t), kappa=kappa_A)),
]
for label, H in cases_sym:
    a   = spec_asym(H)
    e0  = E0(H)
    tag = "chiral" if a < 1e-10 else "NOT chiral"
    print(f"  {label:30s}: asym={a:.2e}  |E0|={e0:.4f}  ({tag})")

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY: <|E0|> vs sigma / kappa")
print("=" * 60)
print(f"{'param':>8}  {'A_NN':>12}  {'A_NN+NNN':>12}  {'B_onsite':>12}  {'C_NNN':>10}")
print("-" * 60)
for i, sig in enumerate(sigmas):
    _, m_nn, m_nnn = rows_A[i]
    _, m_b         = rows_B[i]
    # Case C at kappa = sig * w (closest kappa value)
    c_idx = np.argmin(np.abs(kappas - sig * w))
    _, e_c = rows_C[c_idx]
    print(f"{sig:>8.2f}  {m_nn:>12.3e}  {m_nnn:>12.6f}  {m_b:>12.6f}  {e_c:>10.6f}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'paso4_chiral_spectral.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['case', 'param', 'value', 'mean_E0', 'note'])
    for sig, m_nn, m_nnn in rows_A:
        wr.writerow(['A_NN',  'sigma', f'{sig:.2f}', f'{m_nn:.6e}',  f'kappa=0'])
        wr.writerow(['A_NNN', 'sigma', f'{sig:.2f}', f'{m_nnn:.6e}', f'kappa={kappa_A:.3f}'])
    for sig, m in rows_B:
        wr.writerow(['B', 'sigma', f'{sig:.2f}', f'{m:.6e}', 'onsite_NN'])
    for k, e0 in rows_C:
        wr.writerow(['C', 'kappa', f'{k:.4f}', f'{e0:.6e}', 'no_disorder'])
print(f"\nSaved: {csv_path}")
print()
print("Message:")
print("  off-diag + NN  ->  zero mode PINNED  (chiral preserved: spec. symmetric)")
print("  off-diag + NNN ->  zero mode DETACHED (NNN breaks chiral)")
print("  onsite         ->  zero mode DETACHED (onsite breaks chiral)")
print("  NNN alone      ->  zero mode DETACHED, grows linearly with kappa")

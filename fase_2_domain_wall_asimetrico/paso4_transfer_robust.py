#!/usr/bin/env python3
"""
paso4_transfer_robust.py  --  Transfer robustness: Rabi vs CTAP under disorder.

Chain: L=11, j_DW=5 (centrado), v=0.5, w=1.0, n_seg=5.
N_real=80 realizations per condition.
sigmas = [0.00, 0.05, 0.10, 0.20, 0.30].

Perturbations:
  (i)  off-diagonal: t_j -> t_j*(1+delta_j), delta_j ~ U[-s,s]  [chiral, NN]
  (ii) on-site:      eps_j ~ U[-s,s] on diagonal                 [breaks chiral]
  (iii) position:    j_DW=3 (-2) or j_DW=7 (+2), protocol from j_DW=5

Table: protocol, type, sigma/error, <F>, std.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.linalg import expm, eigh
from scipy.integrate import solve_ivp
import csv

from fase_2_domain_wall_asimetrico.ctap_full_chain import (
    build_hoppings, build_H, build_H_with_NNN,
    _effective_couplings
)

# ── Chain & protocol parameters ───────────────────────────────────────────────
L, j_DW  = 11, 5
v_max, w  = 0.5, 1.0
n_seg     = 5
N_real    = 80
sigmas    = np.array([0.00, 0.05, 0.10, 0.20, 0.30])
rng       = np.random.default_rng(42)

t0     = build_hoppings(L, [j_DW], v_max, w)
Js     = _effective_couplings(build_H(t0), 3)
J_bott = float(min(Js))

T_Rabi = np.pi / (np.sqrt(2.0) * J_bott)
T_est  = np.pi / (2.0 * J_bott)
T_ctap = 5.0 * T_est
sig_c  = T_ctap / 4.0
t_Rpk  = T_ctap / 3.0
t_Lpk  = 2.0 * T_ctap / 3.0

print(f"L={L}  j_DW={j_DW}  n_seg={n_seg}  v={v_max}  w={w}")
print(f"Js=[{', '.join(f'{j:.4e}' for j in Js)}]  J_bott={J_bott:.4e}")
print(f"T_Rabi={T_Rabi:.4e}  T_ctap={T_ctap:.4e}  (5*T_est)")
print()


def gauss_c(t, t0_):
    return float(np.exp(-0.5 * ((t - t0_) / sig_c) ** 2))


def get_v(t):
    J_L = J_bott * gauss_c(t, t_Lpk)
    J_R = J_bott * gauss_c(t, t_Rpk)
    return (v_max * max(J_L / J_bott, 1e-30) ** (2.0 / n_seg),
            v_max * max(J_R / J_bott, 1e-30) ** (2.0 / n_seg))


def wannier3(H):
    """Position-localized (Wannier) left/DW/right states from 3 near-zero eigenvectors.

    For symmetric chains, energy eigenstates hybridize L and R. X-projection
    recovers the physical localized basis unambiguously.
    """
    Lsz = H.shape[0]
    vals, vecs = eigh(H)
    idx = np.argsort(np.abs(vals))[:3]
    Phi = vecs[:, idx]
    X_sub = Phi.T @ (np.arange(Lsz, dtype=float)[:, None] * Phi)
    _, x_evecs = np.linalg.eigh(X_sub)
    Phi_loc = Phi @ x_evecs          # cols sorted by ascending <X>: left, DW, right
    return Phi_loc[:, 0], Phi_loc[:, 2]  # (phi_L, phi_R)


def F_rabi(H_dis):
    """Rabi: evolve left-Wannier state under constant H_dis for T_Rabi."""
    phi_L, phi_R = wannier3(H_dis)
    psi_T = expm(-1j * H_dis * T_Rabi) @ phi_L.astype(complex)
    return float(np.abs(phi_R @ psi_T) ** 2)


def F_ctap(walls, delta=None, onsite=None):
    """CTAP: J-space Gaussian pulses. Uses Wannier states for init/final overlap."""
    def H_t(t):
        vl, vr = get_v(t)
        th = build_hoppings(L, walls, [vl, vr], w)
        if delta is not None:
            th = th * (1.0 + delta)
        return build_H_with_NNN(th, onsite=onsite)

    phi_L0, _ = wannier3(H_t(0.0))
    psi0  = phi_L0.astype(complex)
    sol   = solve_ivp(lambda t, y: -1j * H_t(t) @ y,
                      [0.0, T_ctap], psi0,
                      method='DOP853', rtol=1e-8, atol=1e-11)
    psiT  = sol.y[:, -1]
    psiT /= np.linalg.norm(psiT)
    _, phi_RT = wannier3(H_t(T_ctap))
    return float(np.abs(phi_RT @ psiT) ** 2)


# ── Clean verification ────────────────────────────────────────────────────────
F_r0 = F_rabi(build_H(t0))
F_c0 = F_ctap([j_DW])
print(f"Clean check:  F_Rabi={F_r0:.6f}  F_CTAP={F_c0:.6f}")
print()

rows = []

def report(proto, typ, param, Flist, single=False):
    if single:
        mF = float(Flist)
        print(f"{proto:>5}  {typ:>10}  {str(param):>6}  {mF:>8.4f}  {'N/A':>8}")
        rows.append([proto, typ, str(param), f'{mF:.6f}', ''])
    else:
        mF = float(np.mean(Flist))
        sF = float(np.std(Flist))
        print(f"{proto:>5}  {typ:>10}  {str(param):>6}  {mF:>8.4f}  {sF:>8.4f}")
        rows.append([proto, typ, str(param), f'{mF:.6f}', f'{sF:.6f}'])


print(f"{'proto':>5}  {'type':>10}  {'param':>6}  {'<F>':>8}  {'std':>8}")
print("-" * 50)

# ── (i) Off-diagonal disorder ─────────────────────────────────────────────────
print("  -- off-diagonal disorder --")
for sig in sigmas:
    Fr, Fc = [], []
    for _ in range(N_real):
        d   = rng.uniform(-sig, sig, L - 1)
        t_d = t0 * (1.0 + d)
        Fr.append(F_rabi(build_H_with_NNN(t_d)))
        Fc.append(F_ctap([j_DW], delta=d))
    report('Rabi', 'off-diag', f'{sig:.2f}', Fr)
    report('CTAP', 'off-diag', f'{sig:.2f}', Fc)

print()
# ── (ii) On-site disorder ─────────────────────────────────────────────────────
print("  -- on-site disorder --")
for sig in sigmas:
    Fr, Fc = [], []
    for _ in range(N_real):
        eps = rng.uniform(-sig, sig, L)
        Fr.append(F_rabi(build_H_with_NNN(t0, onsite=eps)))
        Fc.append(F_ctap([j_DW], onsite=eps))
    report('Rabi', 'onsite', f'{sig:.2f}', Fr)
    report('CTAP', 'onsite', f'{sig:.2f}', Fc)

print()
# ── (iii) Position error ──────────────────────────────────────────────────────
print("  -- position error (no disorder, single run) --")
for j_err, label in [(3, '-2'), (7, '+2')]:
    t_e   = build_hoppings(L, [j_err], v_max, w)
    H_err = build_H(t_e)
    report('Rabi', 'position', label, F_rabi(H_err),  single=True)
    report('CTAP', 'position', label, F_ctap([j_err]), single=True)

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'paso4_transfer_robust.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['protocol', 'type', 'sigma_or_error', 'mean_F', 'std_F'])
    for r in rows:
        wr.writerow(r)
print(f"\nSaved: {csv_path}")
print()
print("Message:")
print("  off-diag + NN (chiral): both Rabi and CTAP robust -- F stays high")
print("  onsite (breaks chiral): both degrade with sigma")
print("  position error: Rabi sensitive, CTAP adiabatically more robust")

#!/usr/bin/env python3
"""
paso4_fig_data.py  --  Clean robustness data for figure 4 (transfer panel).

Chain: L=11, j_DW=5 (symmetric, n_seg=5), v=0.5, w=1.0.
T = 10*T_est  (deep adiabatic margin).
Wannier X-projection for symmetric-chain initial/final states.
solve_ivp DOP853 rtol=1e-5.

Three conditions:
  (i)   CTAP kappa=0,         off-diagonal disorder  -> robusto (quiral)
  (ii)  CTAP kappa=0,         onsite disorder        -> cae con sigma
  (iii) CTAP kappa=J_bott/3,  off-diagonal disorder  -> cae mas que (i)

sigma in {0, 0.05, 0.10, 0.20}, N_real=40.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
import csv

from fase_2_domain_wall_asimetrico.ctap_full_chain import (
    build_hoppings, build_H, build_H_with_NNN, _effective_couplings
)

# ── Chain & protocol ──────────────────────────────────────────────────────────
L, j_DW  = 11, 5
v_max, w = 0.5, 1.0
n_seg    = 5
N_real   = 40
sigmas   = np.array([0.00, 0.05, 0.10, 0.20])
rng      = np.random.default_rng(42)

t0     = build_hoppings(L, [j_DW], v_max, w)
Js     = _effective_couplings(build_H(t0), 3)
J_bott = float(min(Js))
T_est   = np.pi / (2.0 * J_bott)
T_ctap  = 10.0 * T_est          # adiabatic core protocol time
Delta   = 0.5 * T_ctap          # buffer on each side: forces vL,vR->0 at endpoints
T       = T_ctap + 2.0 * Delta  # total simulation time = 2*T_ctap

sig_c  = T_ctap / 4.0           # Gaussian width based on core time
t_Rpk  = Delta + T_ctap / 3.0  # absolute peak of R (Stokes first)
t_Lpk  = Delta + 2.0 * T_ctap / 3.0  # absolute peak of L (pump second)
kap_NNN = J_bott / 3.0

print(f"L={L}  j_DW={j_DW}  n_seg={n_seg}  v={v_max}  w={w}")
print(f"Js={[f'{j:.4e}' for j in Js]}")
print(f"J_bott={J_bott:.6e}  T_est={T_est:.4e}  T_ctap=10*T_est={T_ctap:.4e}")
print(f"Delta={Delta:.4e}  T_sim={T:.4e}  (vR(0)~{v_max*float(np.exp(-0.5*((Delta+T_ctap/3)/sig_c)**2))**0.4:.4f})")
print(f"kap_NNN=J_bott/3={kap_NNN:.6e}")
print()

# ── Wannier X-projection (required for symmetric chain) ───────────────────────
def wannier3(H):
    """Left/right Wannier states via position-operator projection on 3 near-zero vecs."""
    Lsz = H.shape[0]
    vals, vecs = eigh(H)
    idx = np.argsort(np.abs(vals))[:3]
    Phi = vecs[:, idx]
    X_sub = Phi.T @ (np.arange(Lsz, dtype=float)[:, None] * Phi)
    _, x_evecs = np.linalg.eigh(X_sub)
    Phi_loc = Phi @ x_evecs             # ascending <X>: left, DW, right
    return Phi_loc[:, 0], Phi_loc[:, 2]

# ── Pulse (J-space Gaussians, counterintuitive: R first) ─────────────────────
def get_v(t):
    J_L = J_bott * np.exp(-0.5 * ((t - t_Lpk) / sig_c) ** 2)
    J_R = J_bott * np.exp(-0.5 * ((t - t_Rpk) / sig_c) ** 2)
    return (v_max * max(float(J_L / J_bott), 1e-30) ** (2.0 / n_seg),
            v_max * max(float(J_R / J_bott), 1e-30) ** (2.0 / n_seg))

# ── CTAP simulation ───────────────────────────────────────────────────────────
def F_ctap(kap=0.0, delta=None, onsite=None):
    """Fidelity |<phi_R(T)|psi(T)>|^2 with Wannier initial/final states."""
    kap = float(kap)

    def H_t(t):
        vl, vr = get_v(t)
        th = build_hoppings(L, [j_DW], [vl, vr], w)
        if delta is not None:
            th = th * (1.0 + delta)
        return build_H_with_NNN(th, kappa=kap, onsite=onsite)

    phi_L0, _ = wannier3(H_t(0.0))
    psi0 = phi_L0.astype(complex)
    sol = solve_ivp(lambda t, y: -1j * H_t(t) @ y,
                    [0.0, T], psi0,
                    method='DOP853', rtol=1e-5, atol=1e-8)
    psiT = sol.y[:, -1]
    psiT /= np.linalg.norm(psiT)
    _, phi_RT = wannier3(H_t(T))
    return float(np.abs(phi_RT @ psiT) ** 2)

# ── Clean F check ─────────────────────────────────────────────────────────────
F0_nn  = F_ctap(kap=0.0)
F0_nnn = F_ctap(kap=kap_NNN)
print(f"F_clean(kappa=0)            = {F0_nn:.6f}  {'OK' if F0_nn  >= 0.99 else 'FAIL (<0.99)'}")
print(f"F_clean(kappa=J/3={kap_NNN:.4f}) = {F0_nnn:.6f}  {'OK' if F0_nnn >= 0.99 else 'FAIL (<0.99)'}")
if F0_nn < 0.99 or F0_nnn < 0.99:
    print("Criterion not met. Stopping.")
    sys.exit(1)

# Zero-mode bridge
E0_nn  = float(np.min(np.abs(np.linalg.eigvalsh(build_H(t0)))))
E0_nnn = float(np.min(np.abs(np.linalg.eigvalsh(build_H_with_NNN(t0, kappa=kap_NNN)))))
print(f"|E0|(kappa=0)     = {E0_nn:.2e}  (pinned -- chiral symmetric)")
print(f"|E0|(kappa=J/3)   = {E0_nnn:.4f}  (detached -- NNN breaks chiral)")
print()

# ── Main scan ─────────────────────────────────────────────────────────────────
# (cond_key, label, kap, disorder_type)
conditions = [
    ('CTAP_NN_offdiag',  '(i)  kappa=0,   off-diag', 0.0,     'offdiag'),
    ('CTAP_NN_onsite',   '(ii) kappa=0,   onsite',   0.0,     'onsite'),
    ('CTAP_NNN_offdiag', '(iii)kappa=J/3, off-diag', kap_NNN, 'offdiag'),
]

print(f"{'condition':>22}  {'sigma':>5}  {'<F>':>8}  {'std':>8}")
print("-" * 52)

rows = []
for ckey, clabel, kap, dtype in conditions:
    print(f"  {clabel}")
    for sig in sigmas:
        sig_f = float(sig)
        if sig_f == 0.0:
            F0 = F0_nn if kap == 0.0 else F0_nnn
            print(f"  {'':>22}  {sig_f:>5.2f}  {F0:>8.4f}  {'---':>8}")
            rows.append([ckey, f'{sig_f:.2f}', f'{F0:.6f}', ''])
        else:
            flist = []
            for _i in range(N_real):
                if dtype == 'offdiag':
                    d = rng.uniform(-sig_f, sig_f, L - 1)
                    flist.append(F_ctap(kap=kap, delta=d))
                else:
                    eps = rng.uniform(-sig_f, sig_f, L)
                    flist.append(F_ctap(kap=kap, onsite=eps))
            mF = float(np.mean(flist))
            sF = float(np.std(flist))
            print(f"  {'':>22}  {sig_f:>5.2f}  {mF:>8.4f}  {sF:>8.4f}")
            rows.append([ckey, f'{sig_f:.2f}', f'{mF:.6f}', f'{sF:.6f}'])
    print()

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paso4_fig_data.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['condition', 'sigma', 'mean_F', 'std_F'])
    for r in rows:
        wr.writerow(r)

print(f"Saved: {csv_path}")
print()
print("Message:")
print("  (i)   CTAP kappa=0, off-diag:  <F> flat with sigma (chiral symmetry protects)")
print("  (ii)  CTAP kappa=0, onsite:    <F> degrades with sigma (onsite breaks chiral)")
print("  (iii) CTAP kappa=J/3, off-diag: <F> degrades more than (i) (NNN breaks chiral)")
print("  => off-diagonal disorder is only dangerous once chiral symmetry is broken by NNN")

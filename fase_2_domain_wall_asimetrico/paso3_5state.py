#!/usr/bin/env python3
"""
paso3_5state.py  --  CTAP D=4 / 5 protected states on SSH chain (n_seg=11).

Counterintuitive J-space Gaussian sequence:
  Group B (Stokes, early, t_B=T/3):  J2 (S1-S2), J4 (S3-R)
  Group A (pump,   late,  t_A=2T/3): J1 (L-S1),  J3 (S2-S3)

Dark state E=0: support only on {|L>,|S2>,|R>}; |S1> and |S3> always dark.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import solve_ivp
import csv

from fase_2_domain_wall_asimetrico.ctap_full_chain import (
    build_hoppings, build_H, protected_states, _effective_couplings
)

# ── Chain parameters ────────────────────────────────────────────────────────
v_max, w = 0.5, 1.0
L, walls, n_seg = 45, [11, 22, 33], 11

t_bonds  = build_hoppings(L, walls, v_max, w)
H_static = build_H(t_bonds)
Js = _effective_couplings(H_static, 5)   # [J1, J2, J3, J4]
J1_max, J2_max, J3_max, J4_max = Js
J_bott = min(Js)

print("D=4  L=45  walls=[11,22,33]  n_seg=11")
print(f"  Js = {[f'{j:.6e}' for j in Js]}")
print(f"  J_bott = {J_bott:.6e}  max/min = {max(Js)/J_bott:.4f}")

# ── Protocol ────────────────────────────────────────────────────────────────
T_est = 4.0 * np.pi / (2.0 * J_bott)
T_sim = 5.0 * T_est
sigma = T_sim / 4.0
t_B   = T_sim / 3.0
t_A   = 2.0 * T_sim / 3.0

J_cross    = J_bott * np.exp(-2.0 / 9.0)
gap_cross  = np.sqrt(2.0) * J_cross
adia_ratio = (4.0 / (3.0 * T_sim)) / gap_cross

print(f"\nT_est={T_est:.4e}  T_sim=5*T_est={T_sim:.4e}")
print(f"sigma=T/4={sigma:.4e}  t_B={t_B:.4e}  t_A={t_A:.4e}")
print(f"adiabaticity ratio = {adia_ratio:.4f}  (need << 1)")


def gauss(t, t0):
    return float(np.exp(-0.5 * ((t - t0) / sigma) ** 2))


def Jcouplings(t):
    GA, GB = gauss(t, t_A), gauss(t, t_B)
    return J1_max * GA, J2_max * GB, J3_max * GA, J4_max * GB


def v_from_J(Jt, Jmax):
    return v_max * max(Jt / Jmax, 1e-30) ** (2.0 / n_seg)


def get_v(t):
    J1, J2, J3, J4 = Jcouplings(t)
    return [v_from_J(J1, J1_max), v_from_J(J2, J2_max),
            v_from_J(J3, J3_max), v_from_J(J4, J4_max)]


# ── Verify dark state structure at t=0 ──────────────────────────────────────
J1_0, J2_0, J3_0, J4_0 = Jcouplings(0.0)
# a1 = -(J2/J1)*a3, a3=1, a5=-(J3/J4)*a3 (unnormalized)
a1_0 = -(J2_0 / J1_0) if J1_0 > 0 else float('inf')
a5_0 = -(J3_0 / J4_0) if J4_0 > 0 else float('inf')
print(f"\nDark state at t=0 (should approach |L>):")
print(f"  J_A(0)/J_B(0) = {J1_0/J2_0:.4f}  (<<1 -> a1>>a3 -> |L>)")
print(f"  a1/a3 = {a1_0:.4f}  a5/a3 = {a5_0:.4f}")

J1_T, J2_T, J3_T, J4_T = Jcouplings(T_sim)
a1_T = -(J2_T / J1_T) if J1_T > 0 else float('inf')
a5_T = -(J3_T / J4_T) if J4_T > 0 else float('inf')
print(f"Dark state at t=T (should approach |R>):")
print(f"  J_A(T)/J_B(T) = {J1_T/J2_T:.4f}  (>>1 -> a5>>a3 -> |R>)")
print(f"  a1/a3 = {a1_T:.4f}  a5/a3 = {a5_T:.4f}")

# ───────────────────────────────────────────────────────────────────────────
# Part 1: Effective 5-state model
# ───────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print("Part 1: Effective 5-state model")
print("=" * 58)


def H5(t):
    J1, J2, J3, J4 = Jcouplings(t)
    H = np.zeros((5, 5))
    H[0, 1] = H[1, 0] = -J1
    H[1, 2] = H[2, 1] = -J2
    H[2, 3] = H[3, 2] = -J3
    H[3, 4] = H[4, 3] = -J4
    return H


def ode5(t, y):
    return -1j * H5(t) @ y


from scipy.linalg import eigh as scipy_eigh

def dark_state_5(t):
    """E=0 eigenstate of H5(t); sign fixed so component 0 is positive."""
    evals, evecs = scipy_eigh(H5(t))
    idx = int(np.argmin(np.abs(evals)))
    v = evecs[:, idx].astype(complex)
    if v[0].real < 0:
        v = -v
    return v

# Start from instantaneous dark state at t=0 (matches what full chain does)
psi0_5 = dark_state_5(0.0)
print(f"  psi0_5 (dark state): |c0|²={abs(psi0_5[0])**2:.4f}  |c2|²={abs(psi0_5[2])**2:.4f}")

N_t    = 1001
t_eval = np.linspace(0.0, T_sim, N_t)

sol5 = solve_ivp(ode5, [0.0, T_sim], psi0_5, t_eval=t_eval,
                 method='DOP853', rtol=1e-9, atol=1e-12)

# Populations: project onto instantaneous 5-state basis
# Components 0..4 of sol5.y ARE the 5 bare-site amplitudes
P5       = np.abs(sol5.y) ** 2   # (5, N_t)
P_L5     = P5[0]
P_S15    = P5[1]   # |S1> -- dark
P_S25    = P5[2]   # |S2> -- transient
P_S35    = P5[3]   # |S3> -- dark
P_R5     = P5[4]

# Final fidelity: overlap with dark state at t=T (consistent with full chain)
phi_dark_T = dark_state_5(T_sim)
# flip sign so component 4 dominates
if phi_dark_T[4].real < 0:
    phi_dark_T = -phi_dark_T
F_eff      = float(abs(phi_dark_T @ sol5.y[:, -1]) ** 2)
maxPS1_eff = float(np.max(P_S15))
maxPS3_eff = float(np.max(P_S35))
maxPS2_eff = float(np.max(P_S25))
maxleak5   = float(np.max(np.abs(np.sum(P5, axis=0) - 1.0)))

print(f"  F_eff        = {F_eff:.6f}  {'PASS' if F_eff > 0.99 else 'FAIL'}")
print(f"  maxP(S1)     = {maxPS1_eff:.2e}  (dark, target < 0.02)")
print(f"  maxP(S3)     = {maxPS3_eff:.2e}  (dark, target < 0.02)")
print(f"  maxP(S2)     = {maxPS2_eff:.4f}  (transient)")
print(f"  maxLeak(eff) = {maxleak5:.2e}")

# ───────────────────────────────────────────────────────────────────────────
# Part 2: Full chain (L=45 sites, solve_ivp)
# ───────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print("Part 2: Full chain  L=45, walls=[11,22,33]")
print("=" * 58)


def H_full(t):
    return build_H(build_hoppings(L, walls, get_v(t), w))


def ode_full(t, y):
    return -1j * H_full(t) @ y


H0       = H_full(0.0)
psi0_f   = protected_states(H0, 5)[0].astype(complex)
print(f"  psi0: |psi[0]|^2 = {abs(psi0_f[0]) ** 2:.4f}")
print(f"  Solving ODE (DOP853, rtol=1e-7)...")

sol_f = solve_ivp(ode_full, [0.0, T_sim], psi0_f, t_eval=t_eval,
                   method='DOP853', rtol=1e-7, atol=1e-9)

y     = sol_f.y                         # (45, N_t) complex
norms = np.linalg.norm(y, axis=0)
yn    = y / norms                       # renormalized

# Site populations at localization sites (0, 11, 22, 33, 44)
P_L_s  = np.abs(yn[0,  :]) ** 2
P_S1_s = np.abs(yn[11, :]) ** 2
P_S2_s = np.abs(yn[22, :]) ** 2
P_S3_s = np.abs(yn[33, :]) ** 2
P_R_s  = np.abs(yn[44, :]) ** 2

# Final fidelity: overlap with rightmost protected state at t=T
H_fin  = H_full(T_sim)
phi_R  = protected_states(H_fin, 5)[4]
F_full = float(abs(phi_R @ yn[:, -1]) ** 2)

# Norm leakage (numerical)
max_norm_err = float(np.max(np.abs(norms - 1.0)))

# Subspace leakage at 11 checkpoints
ck_idx = np.linspace(0, N_t - 1, 11, dtype=int)
sub_leaks = []
for ci in ck_idx:
    psi_ci = yn[:, ci]
    phis_ci = protected_states(H_full(t_eval[ci]), 5)
    proj = sum(abs(phi @ psi_ci) ** 2 for phi in phis_ci)
    sub_leaks.append(1.0 - proj)
max_subleak = float(np.max(sub_leaks))

# T_transfer: first time site-44 population exceeds 0.9
idx_tr = np.where(P_R_s > 0.9)[0]
T_transfer = float(t_eval[idx_tr[0]]) if len(idx_tr) else float('nan')

maxPS1_f = float(np.max(P_S1_s))
maxPS3_f = float(np.max(P_S3_s))
maxPS2_f = float(np.max(P_S2_s))

# Reference: D=2 n_seg=21 T_sim = 5*T_est = 47956
T_sim_D2 = 47956.0

print(f"  F_full            = {F_full:.6f}  {'PASS' if F_full > 0.99 else 'FAIL'}")
print(f"  maxP_site(S1=11)  = {maxPS1_f:.2e}  (dark, target < 0.02)")
print(f"  maxP_site(S3=33)  = {maxPS3_f:.2e}  (dark, target < 0.02)")
print(f"  maxP_site(S2=22)  = {maxPS2_f:.4f}  (transient)")
print(f"  maxLeak_norm      = {max_norm_err:.2e}")
print(f"  maxLeak_subspace  = {max_subleak:.2e}")
print(f"  T_transfer        = {T_transfer:.4e}")
print(f"  T_sim(D=4,n11)    = {T_sim:.4e}")
print(f"  T_sim(D=2,n21)    = {T_sim_D2:.4e}  (reference)")
print(f"  Speedup T_sim     = {T_sim_D2 / T_sim:.2f}x")

# ── Save CSV ─────────────────────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paso3_5state.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['t', 'P_L', 'P_S1', 'P_S2', 'P_S3', 'P_R', 'norm'])
    for i in range(N_t):
        wr.writerow([f"{t_eval[i]:.4f}",
                     f"{P_L_s[i]:.6e}", f"{P_S1_s[i]:.6e}", f"{P_S2_s[i]:.6e}",
                     f"{P_S3_s[i]:.6e}", f"{P_R_s[i]:.6e}", f"{norms[i]:.10f}"])
print(f"\nSaved: {csv_path}")
print("Done.")

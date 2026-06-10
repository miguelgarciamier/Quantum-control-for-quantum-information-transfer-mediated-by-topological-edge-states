#!/usr/bin/env python3
"""
paso2_speedup.py  --  Multi-domain CTAP speedup at fixed L_total.

Demonstrates: more domain walls -> shorter segments -> larger J_bott -> faster transfer.
Part 1: speedup table for D=1,2,3,4 at approximately L_total~44 bonds.
Part 2: one real D=2 CTAP simulation to verify T_measured ~ T_est.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import csv
from fase_2_domain_wall_asimetrico.ctap_full_chain import (
    build_hoppings, build_H, protected_states, _effective_couplings, evolve
)

v_max, w = 0.5, 1.0

# D, L, walls, n_seg_bonds_per_domain
# All ~44 bonds total (+-2); n_seg odd so walls alternate parity automatically.
configs = [
    (1, 44, [],            43),   # D=1: even L=44, 43 bonds, 2 edge states
    (2, 43, [21],          21),   # D=2: wall at j=21 (odd✓), 21 bonds/domain
    (3, 46, [15, 30],      15),   # D=3: walls j=15(odd✓),30(even✓), 15 bonds/domain
    (4, 45, [11, 22, 33],  11),   # D=4: walls j=11✓,22✓,33✓, 11 bonds/domain
]

# ---------------------------------------------------------------------------
# Part 1: speedup table
# ---------------------------------------------------------------------------
print("Computing J_bott for each configuration...")

rows = []
T_ref = None

for D, L, walls, n_seg in configs:
    t_bonds = build_hoppings(L, walls, v_max, w)
    H = build_H(t_bonds)
    Js = _effective_couplings(H, D + 1)
    J_bott = float(np.min(Js))
    T_est = D * np.pi / (2.0 * J_bott)
    alpha = -np.log(J_bott) / (n_seg / 2.0)
    if T_ref is None:
        T_ref = T_est
    ratio = T_est / T_ref
    rows.append((D, n_seg, L, J_bott, T_est, ratio, alpha))

alpha_theory = np.log(w / v_max)

print()
header = (f"{'D':>3}  {'n_seg':>6}  {'L':>4}  {'J_bott':>12}  "
          f"{'T_est':>12}  {'T(D)/T(1)':>11}  {'alpha':>8}")
sep = '-' * len(header)
print(header)
print(sep)
for D, n_seg, L, J_bott, T_est, ratio, alpha in rows:
    print(f"{D:>3}  {n_seg:>6}  {L:>4}  {J_bott:>12.4e}  "
          f"{T_est:>12.4e}  {ratio:>11.6f}  {alpha:>8.4f}")
print()
print(f"  alpha_theory = ln(w/v) = ln({w}/{v_max}) = {alpha_theory:.4f}")

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paso2_speedup.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['D', 'n_seg', 'L', 'J_bott', 'T_est', 'T_ratio', 'alpha'])
    for D, n_seg, L, J_bott, T_est, ratio, alpha in rows:
        writer.writerow([D, n_seg, L,
                         f"{J_bott:.6e}", f"{T_est:.6e}",
                         f"{ratio:.6e}", f"{alpha:.6f}"])
print(f"Saved: {csv_path}")

# ---------------------------------------------------------------------------
# Part 2: D=2 CTAP simulation (J-space Gaussian pulses, counterintuitive)
#
# Pulses designed in J-space so that J_R(t) and J_L(t) follow Gaussians.
# This keeps J_R~J_L~0.8*J_bott at the crossover (not exponentially small),
# making the dark-state rotation adiabatic.
# ---------------------------------------------------------------------------
print()
print('=' * 58)
print('D=2 CTAP simulation (J-space Gaussian pulses, n_seg=21)')
print('=' * 58)

D2, D2_nseg, D2_L, D2_Jbott, D2_Test, _, _ = rows[1]  # D=2 entry

# T_sim = 5*T_est gives adiabaticity ratio dtheta/dt / gap ~ 0.075 << 1
T_sim  = 5.0 * D2_Test
sigma  = T_sim / 4.0    # wider sigma -> larger gap at crossover
t_R_pk = T_sim / 3.0    # Stokes (R) peaks first -- counterintuitive
t_L_pk = 2.0 * T_sim / 3.0

# Expected gap at crossover: sqrt(2)*exp(-2/9)*J_bott ~ 1.13*J_bott
J_cross = D2_Jbott * np.exp(-2.0/9.0)    # J_L = J_R at t=T/2
gap_cross = np.sqrt(2.0) * J_cross
dtheta_dt = (4.0 / (3.0 * T_sim))        # rate of dark-state rotation at T/2
adia_ratio = dtheta_dt / gap_cross

print(f"  D=2  L={D2_L}  n_seg={D2_nseg}  J_bott={D2_Jbott:.4e}  T_est={D2_Test:.4e}")
print(f"  T_sim=5*T_est={T_sim:.4e}  sigma=T/4={sigma:.4e}")
print(f"  adiabaticity ratio dtheta/dt / gap = {adia_ratio:.4f}  (<<1 required)")

walls2 = [21]

def gaussian(t, t0, sig):
    return np.exp(-0.5 * ((t - t0) / sig) ** 2)

def H_of_t(t):
    # J-space Gaussians: invert J = J_bott*(v/v_max)^(n_seg/2) -> v
    J_R_t = D2_Jbott * gaussian(t, t_R_pk, sigma)
    J_L_t = D2_Jbott * gaussian(t, t_L_pk, sigma)
    v_R = v_max * max(J_R_t / D2_Jbott, 1e-30) ** (2.0 / D2_nseg)
    v_L = v_max * max(J_L_t / D2_Jbott, 1e-30) ** (2.0 / D2_nseg)
    return build_H(build_hoppings(D2_L, walls2, [v_L, v_R], w))

# Initial state: instantaneous left-edge protected state at t=0
H0 = H_of_t(0.0)
phis0 = protected_states(H0, 3)
psi0 = phis0[0].astype(complex)
print(f"  psi0: |psi0[0]|^2 = {abs(psi0[0])**2:.4f}  (should be near 1.0)")

n_steps = int(round(T_sim / 5.0))
print(f"  Evolving {n_steps} steps (dt=5.0) ...")
times, psis = evolve(psi0, H_of_t, T_sim, dt=5.0)

# Intermediate populations to confirm adiabatic passage
print()
print(f"  {'t':>10}  {'P_L':>7}  {'P_S':>7}  {'P_R':>7}  {'leak':>7}")
for frac, label in [(0.0, '0'), (1/3, 'T/3'), (1/2, 'T/2'), (2/3, '2T/3'), (1.0, 'T')]:
    idx = min(int(round(frac * (len(times) - 1))), len(times) - 1)
    t_i  = times[idx]
    psi_i = psis[idx]
    H_i  = H_of_t(t_i)
    phi3 = protected_states(H_i, 3)
    PL = float(abs(phi3[0] @ psi_i) ** 2)
    PS = float(abs(phi3[1] @ psi_i) ** 2)
    PR = float(abs(phi3[2] @ psi_i) ** 2)
    lk = 1.0 - PL - PS - PR
    print(f"  {label:>10}  {PL:>7.3f}  {PS:>7.3f}  {PR:>7.3f}  {lk:>7.4f}")

# Final fidelity
H_fin = H_of_t(T_sim)
phi_R_fin = protected_states(H_fin, 3)[2]
F_final = float(abs(phi_R_fin @ psis[-1]) ** 2)
F_site  = float(abs(psis[-1, -1]) ** 2)

result = 'PASS' if F_final > 0.90 else 'FAIL'
print()
print(f"  F_final = |<phi_R(T)|psi(T)>|^2 = {F_final:.4f}  {result}  (target > 0.90)")
print(f"  F_site  = |psi(T)[L-1]|^2       = {F_site:.4f}")
print()
print('Done.')

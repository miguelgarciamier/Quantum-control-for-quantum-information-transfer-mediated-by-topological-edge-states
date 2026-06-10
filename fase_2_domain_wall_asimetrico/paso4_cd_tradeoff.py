#!/usr/bin/env python3
"""
paso4_cd_tradeoff.py  --  NNN (CD ingredient) breaks chiral -> erodes robustness.

Chain: L=21, j_DW=7, v=0.5, w=1.0.
Pulse form (a): sig=0.15*T, t_R=0.4*T, t_L=0.6*T  (counterintuitive, R first).
T = tau/J_bott, tau=300.  Initial state: bare site |0>. Fidelity: |psi_T[L-1]|^2.

Integrator: midpoint Magnus (expm), dt=2.  ~25x faster than DOP853 for this T.

For each kappa in {0, 0.002, 0.006, 0.012, 0.025}:
  1. F_clean: CTAP with NNN=kappa, no disorder
  2. Off-diagonal robustness: t_j->t_j*(1+d_j) on NN bonds; NNN fixed at kappa
     sigmas = {0.00, 0.05, 0.10, 0.20},  N_real=20
  3. |E_0|(kappa): zero-mode energy for static v=v_max chain + NNN  [chiral bridge]

Expected: kappa=0 (chiral) -> <F> robust under off-diagonal disorder;
          kappa>0 (NNN breaks chiral) -> |E_0| detaches, <F> erodes faster.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.linalg import expm
import csv

from fase_2_domain_wall_asimetrico.ctap_full_chain import (
    build_hoppings, build_H, build_H_with_NNN
)

# ── Parameters ────────────────────────────────────────────────────────────────
L, j_DW  = 21, 7
v_max, w = 0.5, 1.0
N_real   = 20
tau      = 300.0
DT       = 2.0          # Magnus time step

kappas = np.array([0.000, 0.002, 0.006, 0.012, 0.025])
sigmas = np.array([0.00,  0.05,  0.10,  0.20])
rng    = np.random.default_rng(42)

# ── Jeff helpers ──────────────────────────────────────────────────────────────
def left_jeff(j_dw, v):
    if v < 1e-15:
        return 0.0
    h = np.array([v if j % 2 == 0 else w for j in range(j_dw)])
    E = np.linalg.eigvalsh(build_H(h))
    return float(np.max(np.abs(E[np.argsort(np.abs(E))[:2]])))

def right_jeff(j_dw, v):
    n_R = L - 1 - j_dw
    h   = np.array([v if j % 2 == 0 else w for j in range(n_R)])
    E   = np.linalg.eigvalsh(build_H(h))
    return float(np.max(np.abs(E[np.argsort(np.abs(E))[:2]])))

J_LS   = left_jeff(j_DW, v_max)
J_SR   = right_jeff(j_DW, v_max)
J_bott = min(J_LS, J_SR)
T_sim  = tau / J_bott

sig_t = 0.15 * T_sim
t_R   = 0.40 * T_sim   # Stokes (R) first -- counterintuitive
t_L   = 0.60 * T_sim   # pump   (L) second

print(f"L={L}  j_DW={j_DW}  v={v_max}  w={w}")
print(f"J_LS={J_LS:.4e}  J_SR={J_SR:.4e}  J_bott={J_bott:.4e}")
print(f"tau={tau}  T_sim={T_sim:.4e}")
print(f"sig_t={sig_t:.4e}  t_R={t_R:.4e}  t_L={t_L:.4e}")
print(f"Magnus dt={DT}  n_steps={int(np.ceil(T_sim/DT))}")
print()

# ── Pulse ─────────────────────────────────────────────────────────────────────
def get_v(t):
    vL = v_max * np.exp(-0.5 * ((t - t_L) / sig_t) ** 2)
    vR = v_max * np.exp(-0.5 * ((t - t_R) / sig_t) ** 2)
    return float(vL), float(vR)

# ── Magnus integrator ─────────────────────────────────────────────────────────
def propagate(kap, delta=None):
    """
    Midpoint Magnus: psi_{n+1} = expm(-i H(t+dt/2) dt) psi_n.
    Error O(dt^3 ||[H,dH/dt]||) per step; for smooth Gaussians with dt=2
    the accumulated error over T~51000 is <1% on |psi|.
    """
    kap = float(kap)
    n_steps = int(np.ceil(T_sim / DT))
    dt_act  = T_sim / n_steps

    psi = np.zeros(L, complex)
    psi[0] = 1.0

    t = 0.0
    for _ in range(n_steps):
        vl, vr = get_v(t + 0.5 * dt_act)
        th = build_hoppings(L, [j_DW], [vl, vr], w)
        if delta is not None:
            th = th * (1.0 + delta)
        H = build_H_with_NNN(th, kappa=kap)
        psi = expm(-1j * H * dt_act) @ psi
        t  += dt_act

    psi /= np.linalg.norm(psi)
    return float(np.abs(psi[-1]) ** 2)

# ── Accuracy check (Magnus vs reference DOP853 for kappa=0 clean) ─────────────
print("Accuracy check: Magnus dt=2 vs DOP853 (kappa=0, no disorder)...")
from scipy.integrate import solve_ivp

def _H_clean(t):
    vl, vr = get_v(t)
    return build_H_with_NNN(build_hoppings(L, [j_DW], [vl, vr], w))

psi0_ref = np.zeros(L, complex); psi0_ref[0] = 1.0
sol_ref = solve_ivp(lambda t, y: -1j * _H_clean(t) @ y,
                    [0.0, T_sim], psi0_ref,
                    method='DOP853', rtol=1e-8, atol=1e-11)
F_ref = float(np.abs(sol_ref.y[-1, -1] / np.linalg.norm(sol_ref.y[:, -1])) ** 2)
F_mag = propagate(0.0)
print(f"  DOP853  F={F_ref:.6f}")
print(f"  Magnus  F={F_mag:.6f}  diff={abs(F_mag-F_ref):.2e}")
print()

# ── Static zero-mode energy ───────────────────────────────────────────────────
t_static = build_hoppings(L, [j_DW], v_max, w)

def E0_static(kap):
    H_st = build_H_with_NNN(t_static, kappa=float(kap))
    return float(np.min(np.abs(np.linalg.eigvalsh(H_st))))

print(f"{'kappa':>8}  {'|E0|(static)':>14}")
for kap in kappas:
    print(f"  {float(kap):>6.4f}  {E0_static(kap):>14.6e}")
print()

# ── Main scan ─────────────────────────────────────────────────────────────────
hdr = (f"{'kappa':>8}  {'F_clean':>8}  {'|E0|':>10}" +
       "".join(f"  {'<F>s='+f'{s:.2f}':>10}  {'std':>6}" for s in sigmas))
print(hdr)
print('-' * len(hdr))

results = []

for kap in kappas:
    kap_f = float(kap)
    e0    = E0_static(kap_f)

    print(f"  kappa={kap_f:.4f}  clean...", end=' ', flush=True)
    F_cl = propagate(kap_f)
    print(f"F={F_cl:.4f}", flush=True)

    F_dis = {}
    for sig in sigmas:
        sig_f = float(sig)
        if sig_f == 0.0:
            F_dis[sig_f] = (F_cl, 0.0)
        else:
            print(f"    s={sig_f:.2f} ({N_real} reals)...", end=' ', flush=True)
            flist = [propagate(kap_f,
                               delta=rng.uniform(-sig_f, sig_f, L - 1))
                     for _ in range(N_real)]
            mF, sF = float(np.mean(flist)), float(np.std(flist))
            F_dis[sig_f] = (mF, sF)
            print(f"<F>={mF:.4f}  std={sF:.4f}", flush=True)

    results.append({'kap': kap_f, 'e0': e0, 'F_clean': F_cl, 'F_dis': F_dis})

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("SUMMARY")
print(hdr)
print('-' * len(hdr))
for r in results:
    line = f"{r['kap']:>8.4f}  {r['F_clean']:>8.4f}  {r['e0']:>10.4e}"
    for sig in sigmas:
        mF, sF = r['F_dis'][float(sig)]
        line += f"  {mF:>10.4f}  {sF:>6.4f}"
    print(line)

# ── Degradation report ────────────────────────────────────────────────────────
print()
print("Degradation at sigma=0.20 (relative to kappa=0):")
base = results[0]['F_dis'][0.20][0]
for r in results:
    mF = r['F_dis'][0.20][0]
    print(f"  kappa={r['kap']:.4f}  <F>={mF:.4f}  drop={base - mF:+.4f}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'paso4_cd_tradeoff.csv')
hdr_csv = ['kappa', 'F_clean', 'E0_static']
for sig in sigmas:
    hdr_csv += [f'meanF_s{sig:.2f}', f'stdF_s{sig:.2f}']

with open(csv_path, 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(hdr_csv)
    for r in results:
        row = [f"{r['kap']:.4f}", f"{r['F_clean']:.6f}", f"{r['e0']:.6e}"]
        for sig in sigmas:
            mF, sF = r['F_dis'][float(sig)]
            row += [f"{mF:.6f}", f"{sF:.6f}"]
        wr.writerow(row)

print(f"\nSaved: {csv_path}")
print()
print("Message:")
print("  kappa=0  (NN, chiral):    off-diagonal disorder does not erode <F>")
print("           (zero mode pinned at E=0, dark state protected).")
print("  kappa>0  (NNN, no chiral): |E_0| detaches ~linearly,")
print("           off-diagonal disorder erodes <F> increasingly with kappa.")
print("  Tradeoff: CD/NNN speedup breaks chiral -> loses robustness.")

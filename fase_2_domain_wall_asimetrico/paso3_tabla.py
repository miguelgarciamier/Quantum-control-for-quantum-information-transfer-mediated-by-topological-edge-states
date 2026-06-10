#!/usr/bin/env python3
"""paso3_tabla.py -- Tabla comparativa: 3 protocolos × j_DW in {11,7,5}."""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.integrate import solve_ivp
from ctap_full_chain import build_hoppings, build_H, protected_triplet

L, w, v_max = 21, 1.0, 0.5

# ── J^eff por splitting de dominios aislados (base localizada) ────────────
def domain_jeff(j_dw):
    """Devuelve J_LS_eff, J_SR_eff del splitting eigenvalores de dominios aislados."""
    h_L = np.array([v_max if j%2==0 else w for j in range(j_dw)])
    E_L = np.linalg.eigvalsh(build_H(h_L))
    J_LS = float(np.max(np.abs(E_L[np.argsort(np.abs(E_L))[:2]])))

    n_R = L - 1 - j_dw
    h_R = np.array([v_max if j%2==0 else w for j in range(n_R)])
    E_R = np.linalg.eigvalsh(build_H(h_R))
    J_SR = float(np.max(np.abs(E_R[np.argsort(np.abs(E_R))[:2]])))
    return J_LS, J_SR

# ── A: Rabi estático (fórmula rabi_fmax del repo ctap_asymmetric) ─────────
def rabi_fmax(rho):
    return 4.0 * rho**2 / (1.0 + rho**2)**2

def proto_rabi(J_LS, J_SR):
    rho   = J_LS / J_SR
    f     = rabi_fmax(rho)
    Omega = np.sqrt(J_LS**2 + J_SR**2)
    t_tr  = np.pi / Omega
    maxPS = (J_LS / Omega)**2   # pico de poblacion en |S> durante la oscilacion
    return f, t_tr, maxPS

# ── B: CTAP efectivo 3×3 ─────────────────────────────────────────────────
def proto_ctap_eff(J_LS_eff, J_SR_eff, T, n_out=200):
    sig = 0.15*T; tR = 0.4*T; tL = 0.6*T

    def rhs(t, y):
        JSR = J_SR_eff * np.exp(-(t-tR)**2 / (2*sig**2))
        JLS = J_LS_eff * np.exp(-(t-tL)**2 / (2*sig**2))
        H   = np.array([[0,JLS,0],[JLS,0,JSR],[0,JSR,0]], dtype=complex)
        psi = y[:3] + 1j*y[3:]
        dp  = -1j*(H @ psi)
        return np.concatenate([dp.real, dp.imag])

    y0  = np.concatenate([[1., 0., 0.], np.zeros(3)])
    sol = solve_ivp(rhs, [0, T], y0, method='RK45',
                    t_eval=np.linspace(0, T, n_out+1),
                    rtol=1e-10, atol=1e-12)
    psis  = sol.y[:3].T + 1j*sol.y[3:].T
    f_fin = float(np.abs(psis[-1, 2])**2)
    maxPS = float(np.max(np.abs(psis[:, 1])**2))
    return f_fin, maxPS

# ── C: CTAP cadena completa ───────────────────────────────────────────────
def proto_ctap_full(j_dw, T, n_out=80):
    sig = 0.15*T; tR = 0.4*T; tL = 0.6*T
    vR_fn = lambda t: max(0., v_max*np.exp(-(t-tR)**2/(2*sig**2)))
    vL_fn = lambda t: max(0., v_max*np.exp(-(t-tL)**2/(2*sig**2)))

    def Ht(t):
        return build_H(build_hoppings(L, j_dw, vL_fn(t), vR_fn(t), w))

    psi0 = np.zeros(L, complex); psi0[0] = 1.

    def rhs(t, y):
        psi_c = y[:L] + 1j*y[L:]
        dp    = -1j*(Ht(t) @ psi_c)
        return np.concatenate([dp.real, dp.imag])

    y0   = np.concatenate([psi0.real, psi0.imag])
    sol  = solve_ivp(rhs, [0, T], y0, method='RK45',
                     t_eval=np.linspace(0, T, n_out+1),
                     rtol=1e-8, atol=1e-10)
    psis = sol.y[:L].T + 1j*sol.y[L:].T
    psis /= np.linalg.norm(psis, axis=1, keepdims=True)

    N  = len(sol.t)
    PS = np.zeros(N); PL = np.zeros(N); PR = np.zeros(N); leak = np.zeros(N)
    for k, (t, psi) in enumerate(zip(sol.t, psis)):
        tri = protected_triplet(Ht(t))
        pL_ = tri['L']/np.linalg.norm(tri['L'])
        pS_ = tri['S']/np.linalg.norm(tri['S'])
        pR_ = tri['R']/np.linalg.norm(tri['R'])
        PL[k]   = abs(pL_@psi)**2
        PS[k]   = abs(pS_@psi)**2
        PR[k]   = abs(pR_@psi)**2
        leak[k] = 1. - (PL[k]+PS[k]+PR[k])

    f_fin   = float(np.abs(psis[-1, L-1])**2)
    maxPS   = float(PS.max())
    maxLeak = float(leak.max())
    return f_fin, maxPS, maxLeak

# ── Loop principal ────────────────────────────────────────────────────────
TAU_REF = 130
J_DWs   = [11, 7, 5]
rows    = []

for j_dw in J_DWs:
    J_LS, J_SR = domain_jeff(j_dw)
    Jbott  = min(J_LS, J_SR)
    T_ctap = TAU_REF / Jbott

    print(f"j_DW={j_dw}: J_LS={J_LS:.4e}  J_SR={J_SR:.4e}  Jbott={Jbott:.4e}  T={T_ctap:.1f}")

    # A
    f_r, t_r, mPS_r = proto_rabi(J_LS, J_SR)
    rows.append((j_dw, 'Rabi',      f_r,   t_r,    mPS_r,      float('nan')))

    # B
    f_e, mPS_e = proto_ctap_eff(J_LS, J_SR, T_ctap)
    rows.append((j_dw, 'CTAP_eff',  f_e,   T_ctap, mPS_e,      0.0))

    # C
    f_c, mPS_c, mL_c = proto_ctap_full(j_dw, T_ctap)
    rows.append((j_dw, 'CTAP_full', f_c,   T_ctap, mPS_c,      mL_c))

# ── Tabla ─────────────────────────────────────────────────────────────────
hdr = f"\n{'j_DW':>5} {'protocolo':>10} {'f_final':>9} {'T_transfer':>12} {'maxP_S':>9} {'maxLeak':>12}"
sep = "-" * 65
print(hdr); print(sep)
for (j, proto, f, T, mPS, mL) in rows:
    mPS_s = f"{mPS:.4f}"
    mL_s  = f"{mL:.2e}" if not np.isnan(mL) else "     n/a   "
    print(f"{j:>5} {proto:>10} {f:>9.6f} {T:>12.1f} {mPS_s:>9} {mL_s:>12}")
print(sep)

# ── CSV ───────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paso3_tabla.csv")
with open(out, 'w', newline='') as fp:
    cw = csv.writer(fp)
    cw.writerow(['j_DW','protocolo','f_final','T_transfer','maxP_S','maxLeak'])
    for row in rows:
        cw.writerow(row)
print(f"CSV: {out}")

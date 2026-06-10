#!/usr/bin/env python3
"""paso3b_tabla.py -- Paso 3b: B y C con el mismo drive efectivo.

Drive físico: v_L(t), v_R(t) gaussianas contraintuitivas (v_R primero).
B: extrae J_LS(t), J_SR(t) del splitting del dominio aislado a v(t) actual.
   Esto es la reducción exacta del drive de C al modelo 3×3.
C: cadena 21-dim con esos mismos v_L(t), v_R(t), solve_ivp rtol=1e-7.
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.integrate import solve_ivp
from ctap_full_chain import build_hoppings, build_H, protected_triplet

L, w, v_max = 21, 1.0, 0.5
TAU_REF = 180

# ── Drive físico ──────────────────────────────────────────────────────────────
def make_vfns(T):
    sig = 0.15 * T; tR = 0.4 * T; tL = 0.6 * T
    vR = lambda t: max(0., v_max * np.exp(-(t - tR)**2 / (2 * sig**2)))
    vL = lambda t: max(0., v_max * np.exp(-(t - tL)**2 / (2 * sig**2)))
    return vL, vR

# ── Splitting de dominio aislado a v dado ─────────────────────────────────────
def left_jeff(j_dw, v):
    """Splitting energético del dominio izquierdo (j_dw bonds) a v_L=v."""
    h = np.array([v if j % 2 == 0 else w for j in range(j_dw)])
    E = np.linalg.eigvalsh(build_H(h))
    return float(np.max(np.abs(E[np.argsort(np.abs(E))[:2]])))

def right_jeff(j_dw, v):
    """Splitting energético del dominio derecho ((L-1-j_dw) bonds) a v_R=v."""
    n_R = L - 1 - j_dw
    h = np.array([v if j % 2 == 0 else w for j in range(n_R)])
    E = np.linalg.eigvalsh(build_H(h))
    return float(np.max(np.abs(E[np.argsort(np.abs(E))[:2]])))

def domain_jeff_peak(j_dw):
    """J_LS, J_SR al pico v=v_max (para calibrar T)."""
    return left_jeff(j_dw, v_max), right_jeff(j_dw, v_max)

# ── A: Rabi estático -- scan numérico ────────────────────────────────────────
def proto_rabi(J_LS, J_SR):
    from scipy.linalg import expm
    Omega = np.sqrt(J_LS**2 + J_SR**2)
    H_st = np.array([[0, J_LS, 0], [J_LS, 0, J_SR], [0, J_SR, 0]], dtype=complex)
    psi0 = np.array([1., 0., 0.], dtype=complex)
    N = 2000; T_sc = 3.0 * np.pi / Omega; dt = T_sc / N
    U = expm(-1j * H_st * dt)
    psi = psi0.copy()
    f_max = 0.; t_tr = 0.; mPS = 0.
    for k in range(N + 1):
        t_k = k * dt
        fk = float(np.abs(psi[2])**2)
        PSk = float(np.abs(psi[1])**2)
        if fk > f_max:
            f_max = fk; t_tr = t_k
        if PSk > mPS:
            mPS = PSk
        if k < N:
            psi = U @ psi
    return f_max, t_tr, mPS

# ── B: J(t) extraído del splitting de dominio aislado al v(t) actual ─────────
def extract_Jeffs_isolated(j_dw, T, N_grid=600):
    vL_fn, vR_fn = make_vfns(T)
    t_arr = np.linspace(0, T, N_grid)
    JLS_arr = np.zeros(N_grid)
    JSR_arr = np.zeros(N_grid)
    for k, t in enumerate(t_arr):
        JLS_arr[k] = left_jeff(j_dw, vL_fn(t))
        JSR_arr[k] = right_jeff(j_dw, vR_fn(t))
    return t_arr, JLS_arr, JSR_arr

def proto_ctap_eff(t_arr, JLS_arr, JSR_arr, T):
    def rhs(t, y):
        JLS = float(np.interp(t, t_arr, JLS_arr))
        JSR = float(np.interp(t, t_arr, JSR_arr))
        H = np.array([[0, JLS, 0], [JLS, 0, JSR], [0, JSR, 0]], dtype=complex)
        psi = y[:3] + 1j * y[3:]
        dp = -1j * (H @ psi)
        return np.concatenate([dp.real, dp.imag])

    y0 = np.concatenate([[1., 0., 0.], np.zeros(3)])
    sol = solve_ivp(rhs, [0, T], y0, method='RK45',
                    t_eval=np.linspace(0, T, 401),
                    rtol=1e-10, atol=1e-12)
    psis = sol.y[:3].T + 1j * sol.y[3:].T
    f_fin = float(np.abs(psis[-1, 2])**2)
    maxPS = float(np.max(np.abs(psis[:, 1])**2))
    return f_fin, maxPS

# ── C: cadena completa 21-dim ─────────────────────────────────────────────────
def proto_ctap_full(j_dw, T, n_out=300):
    vL_fn, vR_fn = make_vfns(T)

    def Ht(t):
        return build_H(build_hoppings(L, j_dw, vL_fn(t), vR_fn(t), w))

    psi0 = np.zeros(L, complex); psi0[0] = 1.

    def rhs(t, y):
        psi_c = y[:L] + 1j * y[L:]
        dp = -1j * (Ht(t) @ psi_c)
        return np.concatenate([dp.real, dp.imag])

    y0 = np.concatenate([psi0.real, psi0.imag])
    sol = solve_ivp(rhs, [0, T], y0, method='RK45',
                    t_eval=np.linspace(0, T, n_out + 1),
                    rtol=1e-7, atol=1e-9)
    psis_raw = sol.y[:L].T + 1j * sol.y[L:].T
    norms = np.linalg.norm(psis_raw, axis=1)
    print(f"    norma_final (antes renorm) = {norms[-1]:.10f}")
    psis = psis_raw / norms[:, np.newaxis]

    N = len(sol.t)
    PS = np.zeros(N); PL = np.zeros(N); PR = np.zeros(N); leak = np.zeros(N)
    for k, (t, psi) in enumerate(zip(sol.t, psis)):
        tri = protected_triplet(Ht(t))
        pL_ = tri['L'] / np.linalg.norm(tri['L'])
        pS_ = tri['S'] / np.linalg.norm(tri['S'])
        pR_ = tri['R'] / np.linalg.norm(tri['R'])
        PL[k] = float(np.abs(pL_ @ psi)**2)
        PS[k] = float(np.abs(pS_ @ psi)**2)
        PR[k] = float(np.abs(pR_ @ psi)**2)
        leak[k] = 1. - (PL[k] + PS[k] + PR[k])

    f_fin = float(np.abs(psis[-1, L - 1])**2)
    maxPS = float(PS.max())
    maxLeak = float(leak.max())
    return f_fin, maxPS, maxLeak

# ── Loop principal ────────────────────────────────────────────────────────────
J_DWs = [11, 7, 5]
rows = []

for j_dw in J_DWs:
    J_LS, J_SR = domain_jeff_peak(j_dw)
    Jbott = min(J_LS, J_SR)
    T = TAU_REF / Jbott

    print(f"\nj_DW={j_dw}  J_LS={J_LS:.4e}  J_SR={J_SR:.4e}  Jbott={Jbott:.4e}  T={T:.1f}")

    # A: Rabi estático
    f_r, t_r, mPS_r = proto_rabi(J_LS, J_SR)
    rows.append((j_dw, 'Rabi', f_r, t_r, mPS_r, float('nan')))
    print(f"  A Rabi:     f={f_r:.6e}  t_tr={t_r:.4e}  maxP_S={mPS_r:.4e}")

    # B: extraer J(t) del splitting de dominio aislado a v(t) actual
    print(f"  Extrayendo J_LS(t), J_SR(t) del drive fisico (dominio aislado)...")
    t_arr, JLS_arr, JSR_arr = extract_Jeffs_isolated(j_dw, T)
    peak_JLS = JLS_arr.max(); peak_JSR = JSR_arr.max()
    print(f"    peak|J_LS(t)|={peak_JLS:.4e}  peak|J_SR(t)|={peak_JSR:.4e}")
    # check orden contraintuitivo
    t_pkL = t_arr[np.argmax(JLS_arr)]; t_pkR = t_arr[np.argmax(JSR_arr)]
    print(f"    t_peak_J_SR={t_pkR:.1f}  t_peak_J_LS={t_pkL:.1f}  "
          f"orden_ok={t_pkR < t_pkL}")

    f_e, mPS_e = proto_ctap_eff(t_arr, JLS_arr, JSR_arr, T)
    rows.append((j_dw, 'CTAP_eff', f_e, T, mPS_e, 0.0))
    print(f"  B CTAP_eff: f={f_e:.6e}  maxP_S={mPS_e:.4e}")

    # C: cadena completa (mismo v_L, v_R que B)
    print(f"  Evolucionando cadena 21-dim (C)...")
    f_c, mPS_c, mL_c = proto_ctap_full(j_dw, T)
    rows.append((j_dw, 'CTAP_full', f_c, T, mPS_c, mL_c))
    print(f"  C CTAP_full: f={f_c:.6e}  maxP_S={mPS_c:.4e}  maxLeak={mL_c:.2e}")

    # Check B ≈ C
    diff_BC = abs(f_e - f_c)
    if diff_BC > 0.01 and mL_c < 0.01:
        print(f"  AVISO: |f_B - f_C| = {diff_BC:.4e} > 1% con leakage~0 -> desajuste de drive")
    else:
        print(f"  |f_B - f_C| = {diff_BC:.4e}  OK (leakage={mL_c:.2e})")

# ── Tabla ────────────────────────────────────────────────────────────────────
hdr = (f"\n{'j_DW':>5} {'protocolo':>10} {'f_final':>12} {'T_transfer':>12}"
       f" {'maxP_S':>12} {'maxLeak':>12}")
sep = "-" * 70
print(hdr)
print(sep)
for (j, proto, f, T_val, mPS, mL) in rows:
    mPS_s = f"{mPS:.4e}"
    mL_s  = f"{mL:.2e}" if not np.isnan(mL) else "         n/a"
    print(f"{j:>5} {proto:>10} {f:>12.6e} {T_val:>12.1f} {mPS_s:>12} {mL_s:>12}")
print(sep)

# ── CSV ──────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paso3b_tabla.csv")
with open(out, 'w', newline='') as fp:
    cw = csv.writer(fp)
    cw.writerow(['j_DW', 'protocolo', 'f_final', 'T_transfer', 'maxP_S', 'maxLeak'])
    for row in rows:
        cw.writerow(row)
print(f"\nCSV: {out}")

#!/usr/bin/env python3
"""paso3d_final.py -- Paso 3d-final: tabla A/B/C x {11,7,5}, pulso por config.

j_DW 11,7 -> forma (a): sigma=0.15T, tR=0.4T, tL=0.6T
j_DW  5   -> forma (b): sigma=0.18T, tR=0.32T, tL=0.68T
Scan tau in {180,300,450}: tau* minimo con f_C>0.995 y maxP_S<0.02 (rtol=1e-6).
B: reduccion 3-nivel del MISMO drive de C (splitting dominio aislado, no normalizado).
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from ctap_full_chain import build_hoppings, build_H, protected_triplet

L, w, v_max = 21, 1.0, 0.5
FORM   = {11: 'a', 7: 'a', 5: 'b'}
TAU_SC = [180, 300, 450]

def pulse_params(T, form):
    if form == 'a':
        sig = 0.15 * T; tR = 0.4 * T; tL = 0.6 * T
    else:
        sig = 0.18 * T; tR = 0.5 * T - sig; tL = 0.5 * T + sig
    return sig, tR, tL

def vL_val(t, sig, tL):
    return v_max * np.exp(-(t - tL)**2 / (2 * sig**2))

def vR_val(t, sig, tR):
    return v_max * np.exp(-(t - tR)**2 / (2 * sig**2))

def left_jeff(j_dw, v):
    if v < 1e-15:
        return 0.0
    h = np.array([v if j % 2 == 0 else w for j in range(j_dw)])
    E = np.linalg.eigvalsh(build_H(h))
    return float(np.max(np.abs(E[np.argsort(np.abs(E))[:2]])))

def right_jeff(j_dw, v):
    n_R = L - 1 - j_dw
    h = np.array([v if j % 2 == 0 else w for j in range(n_R)])
    E = np.linalg.eigvalsh(build_H(h))
    return float(np.max(np.abs(E[np.argsort(np.abs(E))[:2]])))

def domain_jpeaks(j_dw):
    return left_jeff(j_dw, v_max), right_jeff(j_dw, v_max)

# -- A: Rabi estatico (scan numerico) ----------------------------------------
def proto_A(J_LS, J_SR):
    Omega = np.sqrt(J_LS**2 + J_SR**2)
    H_st  = np.array([[0, J_LS, 0], [J_LS, 0, J_SR], [0, J_SR, 0]], dtype=complex)
    psi0  = np.array([1., 0., 0.], dtype=complex)
    N_sc  = 2000; T_sc = 3.0 * np.pi / Omega; dt = T_sc / N_sc
    U     = expm(-1j * H_st * dt)
    psi   = psi0.copy()
    f_max = 0.; t_tr = 0.; mPS = 0.
    for k in range(N_sc + 1):
        t_k = k * dt
        fk  = float(np.abs(psi[2])**2)
        PSk = float(np.abs(psi[1])**2)
        if fk  > f_max: f_max = fk; t_tr = t_k
        if PSk > mPS:   mPS   = PSk
        if k < N_sc:    psi   = U @ psi
    return f_max, t_tr, mPS

# -- B: 3x3 con J(t) del dominio aislado al v(t) actual ----------------------
def proto_B(j_dw, T, form, N_grid=600, n_out=400):
    sig, tR, tL = pulse_params(T, form)
    t_arr   = np.linspace(0, T, N_grid)
    JLS_arr = np.array([left_jeff(j_dw,  max(0., vL_val(t, sig, tL))) for t in t_arr])
    JSR_arr = np.array([right_jeff(j_dw, max(0., vR_val(t, sig, tR))) for t in t_arr])

    def rhs(t, y):
        JLS = float(np.interp(t, t_arr, JLS_arr))
        JSR = float(np.interp(t, t_arr, JSR_arr))
        H3  = np.array([[0, JLS, 0], [JLS, 0, JSR], [0, JSR, 0]], dtype=complex)
        psi = y[:3] + 1j * y[3:]
        dp  = -1j * (H3 @ psi)
        return np.concatenate([dp.real, dp.imag])

    y0  = np.concatenate([[1., 0., 0.], np.zeros(3)])
    sol = solve_ivp(rhs, [0, T], y0, method='RK45',
                    t_eval=np.linspace(0, T, n_out + 1),
                    rtol=1e-10, atol=1e-12)
    psis  = sol.y[:3].T + 1j * sol.y[3:].T
    f_fin = float(np.abs(psis[-1, 2])**2)
    maxPS = float(np.max(np.abs(psis[:, 1])**2))
    return f_fin, maxPS

# -- C: cadena completa 21-dim (rtol=1e-6 para el scan) ----------------------
def proto_C(j_dw, T, form, n_out=120):
    sig, tR, tL = pulse_params(T, form)

    def Ht(t):
        vL = max(0., vL_val(t, sig, tL))
        vR = max(0., vR_val(t, sig, tR))
        return build_H(build_hoppings(L, j_dw, vL, vR, w))

    psi0 = np.zeros(L, complex); psi0[0] = 1.

    def rhs(t, y):
        psi_c = y[:L] + 1j * y[L:]
        dp    = -1j * (Ht(t) @ psi_c)
        return np.concatenate([dp.real, dp.imag])

    y0  = np.concatenate([psi0.real, psi0.imag])
    sol = solve_ivp(rhs, [0, T], y0, method='RK45',
                    t_eval=np.linspace(0, T, n_out + 1),
                    rtol=1e-6, atol=1e-8)
    psis_raw = sol.y[:L].T + 1j * sol.y[L:].T
    norms    = np.linalg.norm(psis_raw, axis=1)
    print(f"    norma_final(antes renorm)={norms[-1]:.10f}")
    psis = psis_raw / norms[:, np.newaxis]

    N  = len(sol.t)
    PS = np.zeros(N); PL = np.zeros(N); PR = np.zeros(N); leak = np.zeros(N)
    for k, (t, psi) in enumerate(zip(sol.t, psis)):
        tri  = protected_triplet(Ht(t))
        pL_  = tri['L'] / np.linalg.norm(tri['L'])
        pS_  = tri['S'] / np.linalg.norm(tri['S'])
        pR_  = tri['R'] / np.linalg.norm(tri['R'])
        PL[k]   = float(np.abs(pL_ @ psi)**2)
        PS[k]   = float(np.abs(pS_ @ psi)**2)
        PR[k]   = float(np.abs(pR_ @ psi)**2)
        leak[k] = 1. - (PL[k] + PS[k] + PR[k])

    f_fin   = float(np.abs(psis[-1, L - 1])**2)
    maxPS   = float(PS.max())
    maxLeak = float(leak.max())
    return f_fin, maxPS, maxLeak

# ============================================================ Loop principal
J_DWs = [11, 7, 5]
rows  = []
check_lines = []

for j_dw in J_DWs:
    form = FORM[j_dw]
    J_LS_pk, J_SR_pk = domain_jpeaks(j_dw)
    Jbott = min(J_LS_pk, J_SR_pk)

    print(f"\n{'='*60}")
    print(f"j_DW={j_dw}  forma=({form})  J_LS={J_LS_pk:.4e}  J_SR={J_SR_pk:.4e}  Jbott={Jbott:.4e}")

    # Theta0 y ceiling: independientes de tau (ratio v_L(0)/v_R(0) fijo por forma)
    # Usar tau=1 normalizado: T_tmp = 1/Jbott -> parametros escalan con T_tmp, pero
    # los angulos dependen solo de las fracciones (sig/T, tL/T), que son constantes.
    T_tmp = 1.0 / Jbott
    sig_t, tR_t, tL_t = pulse_params(T_tmp, form)
    vL0  = vL_val(0., sig_t, tL_t)
    vR0  = vR_val(0., sig_t, tR_t)
    JLS0 = left_jeff(j_dw,  vL0)
    JSR0 = right_jeff(j_dw, vR0)
    if JSR0 > 0. and JLS0 > 0.:
        Theta0_deg = float(np.degrees(np.arctan2(JLS0, JSR0)))
    elif JSR0 > 0.:
        Theta0_deg = 0.0
    else:
        Theta0_deg = 90.0
    ceiling = float(np.cos(np.radians(Theta0_deg))**2)
    print(f"  v_L(0)={vL0:.3e}  v_R(0)={vR0:.3e}  J_LS(0)={JLS0:.3e}  J_SR(0)={JSR0:.3e}")
    print(f"  Theta0={Theta0_deg:.6f} deg  ceiling={ceiling:.8f}")

    # -- A: Rabi estatico (una sola vez con picos) ----------------------------
    f_A, t_A, mPS_A = proto_A(J_LS_pk, J_SR_pk)
    print(f"  A Rabi: f={f_A:.6e}  t_tr={t_A:.4e}  maxP_S={mPS_A:.4e}")
    row_A = (j_dw, 'Rabi', f_A, t_A, mPS_A, float('nan'), Theta0_deg, ceiling, 0)

    # -- Scan C para encontrar tau* -------------------------------------------
    tau_sel = TAU_SC[-1]
    f_C_sel = None; mPS_C_sel = None; mL_C_sel = None; T_sel = None
    crit_met = False

    for tau in TAU_SC:
        T = tau / Jbott
        print(f"  [C scan] tau={tau}  T={T:.0f}", flush=True)
        f_C, mPS_C, mL_C = proto_C(j_dw, T, form)
        print(f"    -> f_C={f_C:.6e}  maxP_S={mPS_C:.4e}  maxLeak={mL_C:.2e}")
        tau_sel = tau; T_sel = T
        f_C_sel = f_C; mPS_C_sel = mPS_C; mL_C_sel = mL_C
        if f_C > 0.995 and mPS_C < 0.02:
            print(f"    -> CRITERIO OK: tau*={tau}")
            crit_met = True
            break

    if not crit_met:
        print(f"  AVISO: criterio no alcanzado en ningun tau. Se usa tau={tau_sel}.")

    row_C = (j_dw, 'CTAP_full', f_C_sel, T_sel, mPS_C_sel, mL_C_sel,
             Theta0_deg, ceiling, tau_sel)

    # -- B: reduccion 3-nivel al tau* seleccionado ----------------------------
    print(f"  [B] tau={tau_sel}  T={T_sel:.0f}", flush=True)
    f_B, mPS_B = proto_B(j_dw, T_sel, form)
    print(f"    -> f_B={f_B:.6e}  maxP_S={mPS_B:.4e}")
    row_B = (j_dw, 'CTAP_eff', f_B, T_sel, mPS_B, 0.0,
             Theta0_deg, ceiling, tau_sel)

    # Orden de tabla: A, B, C
    rows.extend([row_A, row_B, row_C])

    # -- 4 checks -------------------------------------------------------------
    diff_BC   = abs(f_B - f_C_sel)
    diff_ceil = abs(f_C_sel - ceiling)
    ok1 = mPS_C_sel < 0.02
    ok2 = f_C_sel   > 0.995
    ok3 = diff_BC   < 0.01 and mL_C_sel < 1e-5
    ok4 = diff_ceil < 0.005
    lbl1 = "OK" if ok1 else f"FALLO(mPS={mPS_C_sel:.4f})"
    lbl2 = "OK" if ok2 else f"FALLO(f={f_C_sel:.6f})"
    lbl3 = "OK" if ok3 else f"FALLO(|f_B-f_C|={diff_BC:.4e} leak={mL_C_sel:.2e})"
    lbl4 = "OK" if ok4 else f"FALLO(gap={diff_ceil:.4e})"
    check_lines.append(
        f"  j_DW={j_dw} tau*={tau_sel}:\n"
        f"    CHECK1 maxP_S_C={mPS_C_sel:.4e} < 0.02         -> {lbl1}\n"
        f"    CHECK2 f_C={f_C_sel:.6f} > 0.995            -> {lbl2}\n"
        f"    CHECK3 |f_B-f_C|={diff_BC:.4e} <1%, leak~1e-7 -> {lbl3}\n"
        f"    CHECK4 |f_C-ceil|={diff_ceil:.4e}               -> {lbl4}"
    )

# -- Tabla -------------------------------------------------------------------
hdr = (f"\n{'j_DW':>5} {'proto':>10} {'f':>12} {'T_transf':>12} "
       f"{'maxP_S':>10} {'maxLeak':>10} {'Theta0_deg':>11} {'ceiling':>10} {'tau*':>5}")
sep = "-" * 97
print(hdr)
print(sep)
for (j, proto, f, T_v, mPS, mL, th, ceil, tau) in rows:
    mL_s  = f"{mL:.2e}" if not np.isnan(mL) else "       n/a"
    tau_s = str(tau) if tau > 0 else "n/a"
    print(f"{j:>5} {proto:>10} {f:>12.6e} {T_v:>12.1f} "
          f"{mPS:>10.4e} {mL_s:>10} {th:>11.6f} {ceil:>10.8f} {tau_s:>5}")
print(sep)

print("\nComprobaciones:")
for line in check_lines:
    print(line)

# -- CSV ---------------------------------------------------------------------
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paso3d_final.csv")
with open(out, 'w', newline='') as fp:
    cw = csv.writer(fp)
    cw.writerow(['j_DW', 'protocolo', 'f', 'T_transfer', 'maxP_S', 'maxLeak',
                 'Theta0_deg', 'ceiling', 'tau_usado'])
    for row in rows:
        cw.writerow(row)
print(f"\nCSV: {out}")

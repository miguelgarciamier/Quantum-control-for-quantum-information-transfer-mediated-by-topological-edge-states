#!/usr/bin/env python3
"""paso3c_jdw5.py -- Paso 3c: caracterizar j_DW=5, reconciliar B con Fig.15.

Forma (a): sigma=0.15T, sep=0.2T  -> centros en 0.4T, 0.6T
Forma (b): sigma=0.18T, half-sep=sigma -> centros en T/2-sigma, T/2+sigma  (sep=2sigma=0.36T)
  Interpretación: cada centroide desplazado de T/2 en un sigma (convención STIRAP habitual).
  Con rho=32, la forma (a) tiene techo geometrico ~0.876 (limite adiabatico);
  la forma (b) lo eleva a ~0.985. El maximo visible en f(tau) puede superar el techo
  adiabatico por interferencia constructiva del estado brillante.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.integrate import solve_ivp
from ctap_full_chain import build_hoppings, build_H, protected_triplet

L, w, v_max = 21, 1.0, 0.5
j_dw = 5

# -- Acoplos reales j_DW=5 ----------------------------------------------------
def left_jeff(v):
    h = np.array([v if j % 2 == 0 else w for j in range(j_dw)])
    E = np.linalg.eigvalsh(build_H(h))
    return float(np.max(np.abs(E[np.argsort(np.abs(E))[:2]])))

def right_jeff(v):
    n_R = L - 1 - j_dw
    h = np.array([v if j % 2 == 0 else w for j in range(n_R)])
    E = np.linalg.eigvalsh(build_H(h))
    return float(np.max(np.abs(E[np.argsort(np.abs(E))[:2]])))

J_LS_peak = left_jeff(v_max)
J_SR_peak = right_jeff(v_max)
Jbott = min(J_LS_peak, J_SR_peak)
rho_real = J_LS_peak / J_SR_peak

print(f"j_DW=5: J_LS={J_LS_peak:.4e}  J_SR={J_SR_peak:.4e}  "
      f"Jbott={Jbott:.4e}  rho={rho_real:.2f}")

# -- Parámetros de pulso -------------------------------------------------------
def pulse_params(T, form):
    if form == 'a':
        sig = 0.15 * T; tR = 0.4 * T; tL = 0.6 * T
    else:  # 'b': each center displaced from T/2 by one sigma
        sig = 0.18 * T; tR = 0.5 * T - sig; tL = 0.5 * T + sig
    return sig, tR, tL

# Techo geometrico del estado oscuro (limite adiabatico perfecto)
def dark_ceiling(form, rho=32.0):
    sig, tR, tL = pulse_params(1.0, form)  # T=1 normalizado
    eps_SR = np.exp(-tR**2 / (2 * sig**2))
    eps_LS = np.exp(-tL**2 / (2 * sig**2))
    JSR0 = (1.0 / rho) * eps_SR
    JLS0 = 1.0 * eps_LS
    ratio = JSR0 / JLS0  # debe ser >1 para orden contraintuitivo
    cos2 = JSR0**2 / (JSR0**2 + JLS0**2)
    return ratio, cos2

print("\n-- Techo geometrico del estado oscuro (rho=32) --")
for form in ['a', 'b']:
    ratio, ceil = dark_ceiling(form)
    order_ok = ratio > 1.0
    sig_label = "0.15T sep=0.2T" if form == 'a' else "0.18T sep=2sigma=0.36T"
    print(f"  Forma ({form}) sigma={sig_label}: "
          f"J_SR(0)/J_LS(0)={ratio:.3f}  "
          f"{'orden_ok' if order_ok else 'ORDEN_MAL (pump antes que Stokes)'}  "
          f"techo_adiabatico={ceil:.4f}")

# -- B: modelo 3-nivel normalizado (rho=32) -----------------------------------
J_strong = 1.0
J_weak   = J_strong / 32.0

def run_B(tau, form, n_out=400):
    T = tau / J_weak  # T = tau * 32
    sig, tR, tL = pulse_params(T, form)

    def rhs(t, y):
        JLS = J_strong * np.exp(-(t - tL)**2 / (2 * sig**2))
        JSR = J_weak   * np.exp(-(t - tR)**2 / (2 * sig**2))
        H3 = np.array([[0, JLS, 0], [JLS, 0, JSR], [0, JSR, 0]], dtype=complex)
        psi = y[:3] + 1j * y[3:]
        dp = -1j * (H3 @ psi)
        return np.concatenate([dp.real, dp.imag])

    y0 = np.concatenate([[1., 0., 0.], np.zeros(3)])
    sol = solve_ivp(rhs, [0, T], y0, method='RK45',
                    t_eval=np.linspace(0, T, n_out + 1),
                    rtol=1e-10, atol=1e-12)
    psis = sol.y[:3].T + 1j * sol.y[3:].T
    f_fin  = float(np.abs(psis[-1, 2])**2)
    maxPS  = float(np.max(np.abs(psis[:, 1])**2))
    return f_fin, maxPS

# -- Barrido B ----------------------------------------------------------------
TAUS  = [180, 360, 720, 1200]
FORMS = ['a', 'b']

print("\n-- Barrido B: J_strong=1, J_weak=1/32, rho=32 --")
print(f"{'tau':>6} {'forma':>6} {'T_3lv':>10} {'f':>10} {'maxP_S':>10}")
print("-" * 50)

results_B = {}
for form in FORMS:
    for tau in TAUS:
        f, mPS = run_B(tau, form)
        print(f"{tau:>6} {form:>6} {tau * 32:>10.0f} {f:>10.6f} {mPS:>10.4e}")
        results_B[(tau, form)] = (f, mPS)

print("-" * 50)

# -- Identificar tau* ---------------------------------------------------------
# tau* mínimo donde f > 0.88 AND (maxP_S < 0.20 OR f está próximo a saturar)
print("\n-- Análisis de saturación B --")
best_form = None; best_tau = None; best_f = 0.
for form in FORMS:
    fs = [results_B[(t, form)][0] for t in TAUS]
    mps_arr = [results_B[(t, form)][1] for t in TAUS]
    print(f"  Forma ({form}):  f = " + "  ".join(f"{v:.4f}" for v in fs))
    print(f"           mPS = " + "  ".join(f"{v:.3e}" for v in mps_arr))
    # primer tau con f>0.88 y mejora <0.05 respecto al siguiente (o último tau)
    for i, tau in enumerate(TAUS):
        f_i = fs[i]
        f_next = fs[i + 1] if i + 1 < len(TAUS) else f_i
        delta = f_next - f_i
        if f_i > 0.88 and abs(delta) < 0.05:
            print(f"  -> tau*={tau} forma={form}: f={f_i:.6f} mPS={mps_arr[i]:.3e}  delta_f_next={delta:+.4f}")
            if best_tau is None or tau < best_tau or (tau == best_tau and f_i > best_f):
                best_tau = tau; best_form = form; best_f = f_i
            break
    else:
        # no saturation found; use max f
        imax = int(np.argmax(fs))
        print(f"  -> sin saturacion clara. Max f={fs[imax]:.6f} en tau={TAUS[imax]}")
        if best_tau is None:
            best_tau = TAUS[imax]; best_form = form; best_f = fs[imax]

# fallback: use the combination with best f overall
if best_tau is None:
    best_key = max(results_B, key=lambda k: results_B[k][0])
    best_tau, best_form = best_key
    best_f, _ = results_B[best_key]

best_mPS = results_B[(best_tau, best_form)][1]
print(f"\nSelección: tau*={best_tau}  forma='{best_form}'  "
      f"f_B={best_f:.6f}  maxP_S_B={best_mPS:.4e}")

cerca_091 = abs(best_f - 0.91) < 0.04
print(f"B cerca de ~0.91: {'SI' if cerca_091 else 'NO -- ver discrepancia con Fig.15'}")

# -- C: cadena completa j_DW=5 -------------------------------------------------
print(f"\n-- Cadena completa C: j_DW=5, tau={best_tau}, forma='{best_form}' --")
T_C = best_tau / Jbott
sig_C, tR_C, tL_C = pulse_params(T_C, best_form)

print(f"  T_C={T_C:.1f}  sigma={sig_C:.1f}  tR={tR_C:.1f}  tL={tL_C:.1f}")

vL_fn = lambda t: max(0., v_max * np.exp(-(t - tL_C)**2 / (2 * sig_C**2)))
vR_fn = lambda t: max(0., v_max * np.exp(-(t - tR_C)**2 / (2 * sig_C**2)))

def Ht(t):
    return build_H(build_hoppings(L, j_dw, vL_fn(t), vR_fn(t), w))

psi0 = np.zeros(L, complex); psi0[0] = 1.

def rhs_C(t, y):
    psi_c = y[:L] + 1j * y[L:]
    dp = -1j * (Ht(t) @ psi_c)
    return np.concatenate([dp.real, dp.imag])

y0 = np.concatenate([psi0.real, psi0.imag])
n_out_C = 120
sol_C = solve_ivp(rhs_C, [0, T_C], y0, method='RK45',
                  t_eval=np.linspace(0, T_C, n_out_C + 1),
                  rtol=1e-6, atol=1e-8)
psis_raw = sol_C.y[:L].T + 1j * sol_C.y[L:].T
norms = np.linalg.norm(psis_raw, axis=1)
print(f"  norma_final (antes renorm) = {norms[-1]:.10f}")
psis_C = psis_raw / norms[:, np.newaxis]

N = len(sol_C.t)
PS = np.zeros(N); PL_a = np.zeros(N); PR_a = np.zeros(N); leak = np.zeros(N)
for k, (t, psi) in enumerate(zip(sol_C.t, psis_C)):
    tri = protected_triplet(Ht(t))
    pL_ = tri['L'] / np.linalg.norm(tri['L'])
    pS_ = tri['S'] / np.linalg.norm(tri['S'])
    pR_ = tri['R'] / np.linalg.norm(tri['R'])
    PL_a[k] = float(np.abs(pL_ @ psi)**2)
    PS[k]   = float(np.abs(pS_ @ psi)**2)
    PR_a[k] = float(np.abs(pR_ @ psi)**2)
    leak[k] = 1. - (PL_a[k] + PS[k] + PR_a[k])

f_C     = float(np.abs(psis_C[-1, L - 1])**2)
maxPS_C = float(PS.max())
maxLk_C = float(leak.max())

print(f"  C: f={f_C:.6e}  maxP_S={maxPS_C:.4e}  maxLeak={maxLk_C:.2e}")

# -- Resumen -------------------------------------------------------------------
print("\n-- RESUMEN --")
print(f"  B (3-nivel rho=32): f={best_f:.6f}  maxP_S={best_mPS:.4e}")
print(f"  C (cadena real):    f={f_C:.6f}  maxP_S={maxPS_C:.4e}  maxLeak={maxLk_C:.2e}")
diff_BC = abs(f_C - best_f)
print(f"  |f_C - f_B| = {diff_BC:.4e}")

if maxPS_C > 0.15:
    print(f"  AVISO: maxP_S_C={maxPS_C:.4f} > 0.15 -> C aun no adiabatico a tau={best_tau}")
else:
    if f_C < best_f - 0.01 and maxLk_C < 0.01:
        print(f"  C < B con leak~0: correccion genuina de cadena completa")
    elif diff_BC < 0.02:
        print(f"  C ~= B: buen acuerdo")
    else:
        print(f"  Discrepancia B-C sin leakage significativo -> investigar")

print(f"\nConclusion tau*={best_tau} forma='{best_form}': "
      f"B={'converge a ~0.91' if cerca_091 else 'NO converge a ~0.91 en este rango'}")

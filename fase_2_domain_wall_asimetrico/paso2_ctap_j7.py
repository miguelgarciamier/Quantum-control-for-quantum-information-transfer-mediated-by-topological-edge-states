#!/usr/bin/env python3
"""paso2_ctap_j7.py -- CTAP paso 2: j_DW=7, sin figuras."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.integrate import solve_ivp
from ctap_full_chain import build_hoppings, build_H, protected_triplet

L, j_DW, w, v_max = 21, 7, 1.0, 0.5

def make_pulses(T):
    sig, tR, tL = 0.15*T, 0.4*T, 0.6*T
    return (lambda t: max(0., v_max*np.exp(-(t-tL)**2/(2*sig**2))),
            lambda t: max(0., v_max*np.exp(-(t-tR)**2/(2*sig**2))))

def Ht(t, vL_fn, vR_fn):
    return build_H(build_hoppings(L, j_DW, vL_fn(t), vR_fn(t), w))

# ── GATE DE SANIDAD  T=2000 ──────────────────────────────────────────────
T_san = 2000.
vL_s, vR_s = make_pulses(T_san)
print(f"v_L(0)={vL_s(0.):.3e}  v_R(T)={vR_s(T_san):.3e}")

N_g = 400
tg = np.linspace(0, T_san, N_g)
JLS = np.zeros(N_g); JSR = np.zeros(N_g)
for k, t in enumerate(tg):
    H = Ht(t, vL_s, vR_s)
    tri = protected_triplet(H)
    pL = tri['L']/np.linalg.norm(tri['L'])
    pS = tri['S']/np.linalg.norm(tri['S'])
    pR = tri['R']/np.linalg.norm(tri['R'])
    JLS[k] = abs(pL@H@pS)
    JSR[k] = abs(pS@H@pR)

pk_JLS, t_pkJLS = JLS.max(), tg[JLS.argmax()]
pk_JSR, t_pkJSR = JSR.max(), tg[JSR.argmax()]
Jmax_raw = max(pk_JLS, pk_JSR)
print(f"peak|J_LS|={pk_JLS:.4e}  t={t_pkJLS:.1f}")
print(f"peak|J_SR|={pk_JSR:.4e}  t={t_pkJSR:.1f}")
assert Jmax_raw < 0.1, f"STOP bug: Jmax={Jmax_raw:.4f}"
print("Assert |J|<0.1: OK")

if Jmax_raw > 1e-8:
    Jmax_raw_used = Jmax_raw
    assert t_pkJSR < t_pkJLS, "STOP orden"
    print(f"Orden OK: t_JSR={t_pkJSR:.1f} < t_JLS={t_pkJLS:.1f}")
else:
    print("J~0 (eigenstates ortogonales, dark state desacoplado -- correcto)")
    print(f"Orden OK: t_pico_vR=0.4*T < t_pico_vL=0.6*T")

# Jmax desde splitting de dominios aislados en pico (v=v_max)
# dominio izquierdo: j=0..j_DW-1, patron v_L,w,v_L,...
h_left  = np.array([v_max if j%2==0 else w for j in range(j_DW)])
E_left  = np.linalg.eigvalsh(build_H(h_left))
J_LS_eff = float(np.max(np.abs(E_left[np.argsort(np.abs(E_left))[:2]])))

# dominio derecho: j=j_DW..L-2, patron v_R,w,v_R,...
n_right = L - 1 - j_DW
h_right = np.array([v_max if j%2==0 else w for j in range(n_right)])
E_right = np.linalg.eigvalsh(build_H(h_right))
J_SR_eff = float(np.max(np.abs(E_right[np.argsort(np.abs(E_right))[:2]])))

Jmax = J_SR_eff   # cuello de botella
print(f"J_LS_eff={J_LS_eff:.6e}  J_SR_eff={J_SR_eff:.6e}")
print(f"Jmax={Jmax:.6e}  (cuello de botella = dominio derecho, {n_right} bonds)")

# ── BARRIDO ──────────────────────────────────────────────────────────────
taus = np.logspace(np.log10(5), np.log10(150), 8)
Ts   = taus / Jmax
psi0 = np.zeros(L, complex); psi0[0] = 1.

def ivp_evolve(T_val, n_out=60):
    vL_fn, vR_fn = make_pulses(T_val)
    def rhs(t, y):
        psi_c = y[:L] + 1j*y[L:]
        dp = -1j*(Ht(t, vL_fn, vR_fn)@psi_c)
        return np.concatenate([dp.real, dp.imag])
    y0 = np.concatenate([psi0.real, psi0.imag])
    sol = solve_ivp(rhs, [0, T_val], y0, method='RK45',
                    t_eval=np.linspace(0, T_val, n_out+1),
                    rtol=1e-8, atol=1e-10)
    psis = sol.y[:L].T + 1j*sol.y[L:].T
    psis /= np.linalg.norm(psis, axis=1, keepdims=True)
    return sol.t, psis

def get_metrics(times, psis, T_val):
    vL_fn, vR_fn = make_pulses(T_val)
    N = len(times)
    PL=np.zeros(N); PS=np.zeros(N); PR=np.zeros(N)
    leak=np.zeros(N); fid=np.zeros(N)
    for k, (t, psi) in enumerate(zip(times, psis)):
        H_ = Ht(t, vL_fn, vR_fn)
        tri = protected_triplet(H_)
        pL_=tri['L']/np.linalg.norm(tri['L'])
        pS_=tri['S']/np.linalg.norm(tri['S'])
        pR_=tri['R']/np.linalg.norm(tri['R'])
        PL[k]=abs(pL_@psi)**2; PS[k]=abs(pS_@psi)**2; PR[k]=abs(pR_@psi)**2
        leak[k]=1.-(PL[k]+PS[k]+PR[k]); fid[k]=abs(psi[L-1])**2
    return PL, PS, PR, leak, fid

print(f"\n{'tau':>8} {'T':>10} {'f_fin':>8} {'maxP_S':>8} {'maxLeak':>8}")
print("-"*52)
results=[]; best_f=-1.; best_idx=-1; best_data=None

for i, (tau, T_val) in enumerate(zip(taus, Ts)):
    times, psis = ivp_evolve(T_val)
    PL, PS, PR, leak, fid = get_metrics(times, psis, T_val)
    f_f=fid[-1]; mPS=PS.max(); mL=leak.max()
    results.append((tau, T_val, f_f, mPS, mL))
    if f_f > best_f:
        best_f=f_f; best_idx=i
        best_data=(times, psis, PL, PS, PR, leak, fid, T_val)
    print(f"{tau:8.2f} {T_val:10.1f} {f_f:8.4f} {mPS:8.4f} {mL:8.4f}")

print("-"*52)
print(f"\n{'tau':>8} {'T':>10} {'f_fin':>8} {'maxP_S':>8} {'maxLeak':>8}")
print("-"*52)
for i, (tau, T_val, f, mPS, mL) in enumerate(results):
    m=" <-- BEST" if i==best_idx else ""
    print(f"{tau:8.2f} {T_val:10.1f} {f:8.4f} {mPS:8.4f} {mL:8.4f}{m}")

# guardar mejor
times_b,psis_b,PL_b,PS_b,PR_b,leak_b,fid_b,T_best = best_data
vL_b, vR_b = make_pulses(T_best)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"ctap_paso2_jDW{j_DW}_best.npz")
np.savez(out, times=times_b, P_L=PL_b, P_S=PS_b, P_R=PR_b,
         leakage=leak_b, fidelity=fid_b,
         v_L=np.array([vL_b(t) for t in times_b]),
         v_R=np.array([vR_b(t) for t in times_b]),
         T_best=T_best, Jmax=Jmax)
print(f"\nGuardado: {out}")
print(f"Mejor: T={T_best:.1f}  f={best_f:.6f}")
if best_f < 0.99:
    print("f<0.99. PARA.")
else:
    print("f>0.99. OK para paso 3.")

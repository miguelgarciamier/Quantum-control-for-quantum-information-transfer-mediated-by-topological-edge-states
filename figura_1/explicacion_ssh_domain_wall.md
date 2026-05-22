# SSH: Cadena dimérica, topología y muros de dominio

## Índice
1. [El modelo SSH y la cadena dimérica](#1-el-modelo-ssh-y-la-cadena-dimérica)
2. [Espectro de energía y transición de fase](#2-espectro-de-energía-y-transición-de-fase)
3. [¿Por qué es topológica?](#3-por-qué-es-topológica)
4. [Muros de dominio](#4-muros-de-dominio)
5. [Protocolo de transferencia](#5-protocolo-de-transferencia)
6. [Estructura del código](#6-estructura-del-código)

---

## 1. El modelo SSH y la cadena dimérica

### El Hamiltoniano

El modelo de **Su-Schrieffer-Heeger (SSH)** es la cadena 1D más simple con topología no trivial. Consiste en una cadena de sitios con **dos hoppings alternantes**:

$$H = -\sum_j \left[ v \, c^\dagger_{j,a} c_{j,b} + w \, c^\dagger_{j+1,a} c_{j,b} + \text{h.c.} \right]$$

Equivalentemente, para una cadena de $L$ sitios numerados $0, 1, 2, \ldots, L-1$:

$$H = -\sum_{j=0}^{L-2} t_j \left( |j\rangle\langle j+1| + |j+1\rangle\langle j| \right)$$

donde:
- $t_j = v$ si $j$ es **par** (enlace **intraceldilla**)
- $t_j = w$ si $j$ es **impar** (enlace **interceldilla**)

### ¿Qué es un dímero?

Un **dímero** es un par de sitios unidos por un enlace fuerte. En el modelo SSH:
- Cada celdilla unidad contiene **dos sitios** (subred $a$ en posiciones pares, subred $b$ en posiciones impares)
- Si $v > w$: los dímeros están **dentro** de la celdilla → fase **trivial**
- Si $v < w$: los dímeros conectan celdillas **adyacentes** → fase **topológica**

### Cadena par vs impar

Para **$L$ par** (ej: 10 sitios = 5 celdillas completas):
- Todas las celdillas están completas
- **No hay** estados de borde (ni en topológica ni en trivial)
- El gap se cierra solo en $v = w$

Para **$L$ impar** (ej: 11 sitios = 5 celdillas + 1 sitio extra):
- La última celdilla está incompleta → queda un sitio "colgando"
- Aparece un **estado de borde a E=0** en la fase topológica ($v < w$)
- Este estado está localizado exponencialmente en el borde

**En nuestro código**: usamos $L = 10$ (par) para estudiar la cadena dimérica pura, y $L = 11$ (impar) para mostrar el contraste con el estado de borde.

---

## 2. Espectro de energía y transición de fase

### Espectro en función de $v/w$

El espectro del SSH se obtiene diagonalizando $H$ para cada valor de $v/w$.

**Resultados clave** (Figura `01_dimer_spectrum.png`):

| $v/w$ | Fase | Gap ($\Delta$) | Observación |
|-------|------|----------------|-------------|
| $< 1$ | Topológica ($\nu = 1$) | Abierto: $\sim 2|w-v|$ | Sin estados de borde (cadena par) |
| $= 1$ | Transición | $\Delta = 0$ (bulk) | El gap se cierra → punto crítico |
| $> 1$ | Trivial ($\nu = 0$) | Abierto: $\sim 2|v-w|$ | Sin estados de borde |

### El gap

En el **límite termodinámico** ($L \to \infty$), la relación de dispersión del SSH es:

$$E(k) = \pm \sqrt{v^2 + w^2 + 2vw\cos k}$$

El gap mínimo ocurre en $k = \pi$:

$$\Delta = 2|w - v|$$

Este gap se anula **exactamente** en $v = w$ (transición de fase topológica).

Para cadenas **finitas** ($L = 10$), el gap no se cierra exactamente pero la tendencia es clara. El efecto de tamaño finito es $\sim \pi / L$.

### Interpretación física

- En la fase **trivial** ($v > w$), los sitios $a$ y $b$ dentro de cada celdilla están fuertemente acoplados → los electrones quedan atrapados en dímeros intraceldilla.
- En la fase **topológica** ($v < w$), los sitios $b$ de una celdilla y $a$ de la siguiente están fuertemente acoplados → los dímeros "cruzan" las fronteras de celdilla. Los sitios de los extremos "sobran" si la cadena no tiene celdillas completas.

---

## 3. ¿Por qué es topológica?

### Simetría quiral

El SSH tiene **simetría quiral** (también llamada de sub-red): existe un operador $\Gamma$ tal que

$$\Gamma H \Gamma = -H, \qquad \Gamma = \sum_j \left( |j,a\rangle\langle j,a| - |j,b\rangle\langle j,b| \right)$$

Esto significa que $\Gamma$ asigna +1 a la subred $a$ (pares) y −1 a la subred $b$ (impares). Si $|\psi\rangle$ es autoestado con energía $E$, entonces $\Gamma|\psi\rangle$ es autoestado con energía $-E$.

**Consecuencias**:
1. El espectro es simétrico respecto a $E = 0$
2. Los estados a $E = 0$ (si existen) pueden elegirse para vivir en **una sola subred**
3. Los estados de borde topológicos están protegidos por esta simetría

### Número de enrollamiento (winding number)

En el espacio de momentos, el Hamiltoniano se escribe:

$$h(k) = \begin{pmatrix} 0 & v + w e^{-ik} \\ v + w e^{ik} & 0 \end{pmatrix}$$

El elemento fuera de la diagonal, $d(k) = v + w e^{-ik}$, traza una curva en el plano complejo cuando $k$ va de $0$ a $2\pi$.

El **número de enrollamiento** (winding number) es:

$$\nu = \frac{1}{2\pi i} \oint \frac{d'(k)}{d(k)} dk$$

- Si $v < w$: la curva $d(k)$ **encierra el origen** → $\nu = 1$ (topológica)
- Si $v > w$: la curva **no encierra** el origen → $\nu = 0$ (trivial)
- Si $v = w$: la curva **pasa por** el origen → transición (gap = 0)

### Correspondencia bulk-borde

El **teorema bulk-borde** establece que:

> El número de estados de borde a $E = 0$ en cada extremo de la cadena = $|\nu|$

Para $\nu = 1$: hay 1 estado protegido en cada borde (si existen bordes apropiados, i.e., cadena semi-infinita o impar finita). Para cadenas pares finitas, los dos estados de borde se hibridan exponencialmente → splitting $\sim e^{-L/\xi}$.

---

## 4. Muros de dominio

### ¿Qué es un muro de dominio?

Un **muro de dominio** (domain wall) es la interfaz entre una región **topológica** y una **trivial** dentro de la misma cadena. Se crea haciendo que la secuencia de hoppings forme un patrón como:

$$\underbrace{v\, w\, v\, w\, v}_{\text{dominio 1}} \underbrace{v\, w\, v\, w\, v}_{\text{dominio 2}}$$

En la notación del paper: un bond pattern `vwvwv|vwvwv` donde `|` marca el muro. En el muro, dos enlaces $v$ consecutivos crean una **dislocación** topológica.

### Parámetros

- **$N$**: número de dominios
- **$\ell$**: longitud de cada dominio (número de enlaces dentro del dominio, excluyendo los bordes)
- **$L = N(\ell+1) + 1$**: número total de sitios

Para $N = 2$, $\ell = 10$: $L = 23$ sitios, con 1 muro de dominio en el sitio 11.

### Estados protegidos

El sistema con $N$ dominios tiene **$N+1$ estados protegidos** cerca de $E = 0$:
- $|\mathcal{L}\rangle$: localizado en el borde **izquierdo** (subred $a$, sitios pares)
- $|\mathcal{S}_k\rangle$: localizado en el **muro de dominio $k$** ($k = 1, \ldots, N-1$, subred $b$, sitios impares)
- $|\mathcal{R}\rangle$: localizado en el borde **derecho** (subred $a$, sitios pares)

Todos decaen exponencialmente con ratio $r = v/w$:

**Estado izquierdo** (Eq. 2 del paper):
$$|\mathcal{L}\rangle \propto \sum_{n=0}^{\ell/2} (-r)^n |2n, a\rangle$$

**Estado del muro** (Eq. 3):
$$|\mathcal{S}_k\rangle \propto |j_k, b\rangle + \sum_{n=1}^{\ell/2} (-r)^n \left( |j_k - 2n, b\rangle + |j_k + 2n, b\rangle \right)$$

**Estado derecho** (Eq. 4):
$$|\mathcal{R}\rangle \propto \sum_{n=0}^{\ell/2} (-r)^n |L-1-2n, a\rangle$$

### Validación numérica

- Diagonalizamos $H$ → encontramos $N+1$ autovalores cerca de $E = 0$
- Construimos los estados analíticos y calculamos el **overlap de subespacio** (SVD)
- Resultado: valores singulares $\approx 1$ → los estados analíticos describen correctamente el subespacio protegido

**Para $N=2$, $\ell=10$**:
- Energías protegidas: $E \approx \{-0.015, 0, +0.015\}$
- Overlap de subespacio: $\{1.000, 0.9999, 0.9999\}$

---

## 5. Protocolo de transferencia

### Idea

Los $N+1$ estados protegidos forman una cadena efectiva 1D con hoppings $J_k$. Si controlamos $v(t)$, podemos transferir un estado cuántico de $|\mathcal{L}\rangle$ a $|\mathcal{R}\rangle$ pasando por los estados del muro.

### Pulso de control (Eq. 11)

El hopping intraceldilla se modula con un pulso suave:

$$v(t) = \begin{cases}
v_{\text{tr}} \sin^2(\Omega t) & 0 \leq t < t_{\text{prep}} \\
v_{\text{tr}} & t_{\text{prep}} \leq t < t_{\text{tr}} - t_{\text{prep}} \\
v_{\text{tr}} \sin^2(\Omega(t - t_{\text{tr}})) & t_{\text{tr}} - t_{\text{prep}} \leq t \leq t_{\text{tr}}
\end{cases}$$

donde $\Omega = \pi / (2 t_{\text{prep}})$.

- **Rampa de subida** ($t < t_{\text{prep}}$): lleva $v$ de 0 a $v_{\text{tr}}$ suavemente (adiabáticamente)
- **Meseta** ($t_{\text{prep}} \leq t < t_{\text{tr}} - t_{\text{prep}}$): $v = v_{\text{tr}}$ constante → transferencia
- **Rampa de bajada** ($t > t_{\text{tr}} - t_{\text{prep}}$): devuelve $v$ a 0

### Parámetros (Tabla 1 del paper)

Para $N = 2$, $\ell = 4$:
- $v_{\text{tr}} = 0.5$, $w = 1.0$
- $t_{\text{prep}} = 15$ (tiempo de rampa)
- $t_{\text{tr}} = 45.6$ (tiempo total de transferencia)
- $\Delta t = 0.1$ (paso de tiempo)
- $L = 11$ sitios

### Evolución temporal

A cada paso $\Delta t$:
1. Se calcula $v(t)$ con el pulso
2. Se construye $H(t)$ con el nuevo $v(t)$
3. Se aplica $U = e^{-i H \Delta t}$ al estado $|\psi\rangle$
4. Se calcula $\langle n_j \rangle = |\langle j|\psi(t)\rangle|^2$

### Resultados

- **Fidelidad final** $f = |\langle R|\psi(t_{\text{tr}})\rangle|^2 = 0.996$ → transferencia con >99.5%
- **Tiempo óptimo**: $t_{\text{tr}} = 45.0$ da $f = 0.999$
- La ocupación fluye de sitio 0 → sitio 5 (muro) → sitio 10, confirmando la transferencia mediada por el muro de dominio

---

## 6. Estructura del código

### Archivos generados

| Archivo | Contenido |
|---------|-----------|
| `01_dimer_spectrum.png/.pdf` | Espectro SSH par/impar + gap + esquema |
| `02_domain_wall_states.png/.pdf` | Cadena con muro, estados protegidos, espectro |
| `03_transfer_protocol.png/.pdf` | Heatmap ocupación, pulso, estados topológicos |
| `04_fidelity_scan.png/.pdf` | Fidelidad vs $t_{\text{tr}}$ (búsqueda del óptimo) |

### Funciones principales

```python
# Hamiltonianos
build_ssh_dimer(L, v, w)          # Cadena dimérica simple
build_ssh_multidomain(N, ℓ, v, w) # Cadena con N dominios

# Espectro
compute_spectrum_vs_ratio(L, w)    # Autovalores vs v/w

# Estados analíticos
analytical_boundary_states(N, ℓ, v, w)  # |L⟩, |S_k⟩, |R⟩

# Transferencia
v_pulse(t, v_tr, t_tr, t_prep)    # Pulso de control Eq.(11)
time_evolve_transfer(...)          # Evolución temporal ⟨n_j⟩(t)
compute_boundary_occupations(...)  # ⟨state|ψ(t)⟩² para cada estado

# Visualización
plot_dimer_spectrum(...)           # Espectro vs v/w
plot_dimer_gap(...)                # Gap vs v/w (finito vs bulk)
plot_domain_wall_states(...)       # Cadena + funciones de onda
plot_transfer_heatmap(...)         # ⟨n_j⟩(t) heatmap
plot_pulse(...)                    # v(t)
plot_boundary_occupations(...)     # Ocupación estados topol.
```

### Ejecución

```bash
cd "SSH dimer and domain wall"
.\.venv\Scripts\Activate.ps1
python ssh_dimer_domain_wall.py
```

---

## Referencia

Zurita, Creffield, Platero — *"Fast quantum transfer mediated by topological domain walls"* — Quantum **7**, 1043 (2023).

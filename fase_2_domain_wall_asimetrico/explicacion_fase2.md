# Fase 2: Muro de dominio asimétrico y Shortcut to Adiabaticity

## Índice
1. [Motivación y objetivos](#1-motivación-y-objetivos)
2. [Muro de dominio en posición arbitraria](#2-muro-de-dominio-en-posición-arbitraria)
3. [Hamiltoniano asimétrico](#3-hamiltoniano-asimétrico)
4. [Estados protegidos en configuración asimétrica](#4-estados-protegidos-en-configuración-asimétrica)
5. [Acoplamientos efectivos y asimetría](#5-acoplamientos-efectivos-y-asimetría)
6. [Shortcut to Adiabaticity (STA)](#6-shortcut-to-adiabaticity-sta)
7. [Protocolo de transferencia con pulsos optimizados](#7-protocolo-de-transferencia-con-pulsos-optimizados)
8. [Experimentos realizados](#8-experimentos-realizados)
9. [Análisis de resultados](#9-análisis-de-resultados)
10. [Estructura del código](#10-estructura-del-código)

---

## 1. Motivación y objetivos

### Contexto

En la **fase 1** reprodujimos los resultados básicos del paper de Zurita et al.: el espectro SSH, los estados de borde y del muro de dominio, y el protocolo de transferencia para una cadena *simétrica* con $N=2$ dominios de igual longitud ($\ell=4$), donde el muro de dominio se encuentra exactamente en el **centro** de la cadena.

### Preguntas que abordamos ahora

1. **¿Qué ocurre si movemos el muro de dominio fuera del centro?** En una configuración real, puede ser necesario o deseable tener dominios de distinta longitud. ¿Cómo afecta esto a la fidelidad de la transferencia?

2. **¿Podemos acelerar la transferencia usando técnicas de Shortcut to Adiabaticity (STA)?** El paper utiliza pulsos $\sin^2$ para las rampas de preparación. ¿Qué ocurre si modificamos la forma del pulso para hacer la evolución más rápida?

### Relevancia

Estas preguntas son fundamentales para la aplicabilidad práctica de los protocolos topológicos:
- En sistemas experimentales reales, la posición del muro de dominio puede no ser exactamente central.
- La velocidad de transferencia es crítica para sistemas con tiempos de coherencia limitados.
- Comprender la robustez ante asimetrías es esencial para el diseño de redes cuánticas.

---

## 2. Muro de dominio en posición arbitraria

### Concepto

Recordemos que en una cadena SSH multidomain con $N=2$ dominios, el **muro de dominio** es el punto donde la dimerización cambia de patrón. En el caso simétrico ($\ell_1 = \ell_2$), el muro está exactamente en el centro.

Ahora consideramos una cadena de longitud fija $L=21$ sitios, con el muro en distintas posiciones. **Nota importante**: el muro debe estar en un sitio de índice **impar** para que el último enlace del dominio 1 (índice par → tipo $v$) y el primer enlace del dominio 2 (índice local 0 → tipo $v$) formen dos enlaces $v$ consecutivos, creando la dislocación topológica.

| Configuración | Sitio del muro | $\ell_1$ (enlaces D1) | $\ell_2$ (enlaces D2) | Fracción |
|---------------|:--------------:|:--------------------:|:--------------------:|:--------:|
| Centro (~1/2) | 11             | 11                   | 9                    | 0.55     |
| Tercio (~1/3) | 7              | 7                    | 13                   | 0.35     |
| Cuarto (~1/4) | 5              | 5                    | 15                   | 0.25     |

### Patrón de enlaces

Para una cadena con $L=21$ y muro en el sitio $j_{DW}$:

**Dominio 1** (sitios $0$ a $j_{DW}$):
$$t_j = \begin{cases} v & \text{si } j \text{ par} \\ w & \text{si } j \text{ impar} \end{cases}, \quad j = 0, 1, \ldots, j_{DW}-1$$

**Dominio 2** (sitios $j_{DW}$ a $L-1$):
$$t_j = \begin{cases} v & \text{si } (j-j_{DW}) \text{ par} \\ w & \text{si } (j-j_{DW}) \text{ impar} \end{cases}, \quad j = j_{DW}, \ldots, L-2$$

El muro de dominio se reconoce porque en la frontera aparecen **dos enlaces $v$ consecutivos**, creando la dislocación topológica.

### Ejemplo: L=21, muro en sitio 7 (~1/3)

```
Dominio 1:            Dominio 2:
v  w  v  w  v  w  v | v  w  v  w  v  w  v  w  v  w  v  w  v
0  1  2  3  4  5  6   7  8  9 10 11 12 13 14 15 16 17 18 19 20
                      ^
                   DW aquí
```

El patrón de enlaces es `vwvwvwvvwvwvwvwvwvwv`: los dos enlaces $v$ consecutivos (posiciones 6-7) crean el muro de dominio.

**Nota crítica sobre paridad**: Para que se formen dos enlaces $v$ consecutivos, el muro debe estar en un sitio de índice **impar**. Si $j_{DW}$ es impar, el último enlace de D1 (índice $j_{DW}-1$, par) es $v$, y el primer enlace de D2 (índice local 0) es $v$. Si $j_{DW}$ fuera par, el último enlace de D1 sería $w$, y no habría dislocación.

---

## 3. Hamiltoniano asimétrico

### Formulación

El Hamiltoniano para la cadena con muro asimétrico es:

$$\mathcal{H} = -\sum_{j=0}^{L-2} t_j \left( |j\rangle\langle j+1| + |j+1\rangle\langle j| \right)$$

donde $t_j$ depende de a qué dominio pertenece el enlace $j$ y de su posición relativa dentro del dominio.

### Implementación

En el código, `build_ssh_two_domain_asymmetric(L_total, wall_pos, v, w)` construye este Hamiltoniano:

1. Para los enlaces $j < j_{DW}$: se usa el patrón $v, w, v, w, \ldots$ partiendo de $j=0$.
2. Para los enlaces $j \geq j_{DW}$: se reinicia el patrón $v, w, v, w, \ldots$ partiendo de $j_{DW}$.

Esto garantiza que el enlace que cruza el muro ($j = j_{DW}-1$ y $j = j_{DW}$) son ambos de tipo $v$, creando la dislocación.

### Diferencia con el caso simétrico

En `build_ssh_multidomain(N, ell, v, w)` del código de la fase 1, la longitud del dominio $\ell$ es la misma para todos. Aquí, los dos dominios pueden tener longitudes arbitrarias: $\ell_1 = j_{DW}$ y $\ell_2 = L - 1 - j_{DW}$.

---

## 4. Estados protegidos en configuración asimétrica

### Los tres estados

El sistema con $N=2$ dominios siempre tiene **3 estados protegidos** cerca de $E=0$:

1. **$|\mathcal{L}\rangle$** — Estado localizado en el borde izquierdo
2. **$|\mathcal{S}\rangle$** — Estado localizado en el muro de dominio
3. **$|\mathcal{R}\rangle$** — Estado localizado en el borde derecho

### Forma analítica

**Estado izquierdo** (subred $a$, sitios pares del dominio 1):
$$|\mathcal{L}\rangle \propto \sum_{n=0}^{\ell_1/2} (-r)^n |2n\rangle, \quad r = v/w$$

**Estado del muro** ($j_{DW}$, se extiende a ambos dominios):
$$|\mathcal{S}\rangle \propto |j_{DW}\rangle + \sum_{n=1}^{\ell_1/2} (-r)^n |j_{DW}-2n\rangle + \sum_{n=1}^{\ell_2/2} (-r)^n |j_{DW}+2n\rangle$$

**Estado derecho** (subred $a$ del dominio 2, partiendo del extremo):
$$|\mathcal{R}\rangle \propto \sum_{n=0}^{\ell_2/2} (-r)^n |L-1-2n\rangle$$

### Efecto de la asimetría en los estados

Cuando el muro se desplaza del centro:

- $|\mathcal{L}\rangle$ tiene **menos sitios** para extenderse → está más localizado → su solapamiento con $|\mathcal{S}\rangle$ cambia.
- $|\mathcal{R}\rangle$ tiene **más sitios** para extenderse → está más deslocalizado.
- $|\mathcal{S}\rangle$ se extiende asimétricamente: más hacia el dominio largo que hacia el corto (limitado por el tamaño de cada dominio).

### Validación numérica

Comprobamos que los estados analíticos reproducen correctamente el subespacio protegido calculando:
- Los 3 autovalores más cercanos a $E=0$ del Hamiltoniano exacto.
- El **overlap de subespacio**: valores singulares de $\langle \text{analítico}|\text{numérico}\rangle$ deben ser $\approx 1$.

---

## 5. Acoplamientos efectivos y asimetría

### Hamiltoniano efectivo

Los 3 estados protegidos forman una **cadena efectiva de 3 sitios**:

$$\mathcal{H}_{\text{eff}} = J_{LS} |\mathcal{S}\rangle\langle\mathcal{L}| + J_{SR} |\mathcal{R}\rangle\langle\mathcal{S}| + \text{h.c.}$$

donde $J_{LS}$ y $J_{SR}$ son los acoplamientos efectivos entre estados vecinos.

### Dependencia con la longitud del dominio

De las ecuaciones (6-8) del paper, el acoplamiento efectivo escala como:

$$J \sim v \cdot \mathcal{M}_i \cdot \mathcal{M}_j \cdot \left(\frac{v}{w}\right)^{\ell/2}$$

donde $\ell$ es la longitud del dominio que separa los dos estados, y $\mathcal{M}_i$ son las constantes de normalización:
$$\mathcal{M}_{\mathcal{L}} = \mathcal{M}_{\mathcal{R}} = \sqrt{\frac{w^2}{v^2} - 1}, \quad \mathcal{M}_{\mathcal{S}} = \sqrt{\frac{w^2 - v^2}{w^2 + v^2}}$$

### Asimetría de acoplamientos

Cuando $\ell_1 \neq \ell_2$:

$$\frac{J_{LS}}{J_{SR}} \sim \left(\frac{w}{v}\right)^{(\ell_2 - \ell_1)/2}$$

Para $w = 2v$ (nuestro caso):
- Si $\ell_1 = \ell_2 = 10$: $J_{LS}/J_{SR} = 1$ (simétrico)
- Si $\ell_1 = 6, \ell_2 = 14$: $J_{LS}/J_{SR} = 2^4 = 16$ (muy asimétrico)
- Si $\ell_1 = 4, \ell_2 = 16$: $J_{LS}/J_{SR} = 2^6 = 64$ (extremadamente asimétrico)

### Consecuencias para la transferencia

La **asimetría de acoplamientos es el factor clave** que degrada la fidelidad:

1. **Cadena efectiva desbalanceada**: La cadena de 3 sitios tiene hoppings muy diferentes. La transferencia $|\mathcal{L}\rangle \to |\mathcal{R}\rangle$ requiere que el estado pase por $|\mathcal{S}\rangle$, pero si $J_{LS} \gg J_{SR}$ (o viceversa), el tránsito de $|\mathcal{S}\rangle$ a $|\mathcal{R}\rangle$ es **exponencialmente más lento** que de $|\mathcal{L}\rangle$ a $|\mathcal{S}\rangle$.

2. **Tiempo de transferencia mayor**: El tiempo óptimo viene determinado por el acoplamiento más débil:
$$t_{tr} \sim \frac{\pi}{2 J_{\min}} = \frac{\pi}{2} \cdot \left(\frac{w}{v}\right)^{\ell_{\max}/2}$$

3. **Velocidad de oscilación desigual**: El patrón de oscilaciones de Rabi entre los tres estados pierde la simetría, haciendo que nunca toda la probabilidad llegue simultáneamente al extremo derecho.

---

## 6. Shortcut to Adiabaticity (STA)

### Fundamento teórico

Los protocolos de **Shortcut to Adiabaticity** son técnicas para acelerar procesos que normalmente requieren evolución adiabática (infinitamente lenta). La idea central es:

> Añadir correcciones al Hamiltoniano (o modificar los pulsos de control) para compensar las **transiciones diabáticas** que aparecen cuando se opera más rápido que el límite adiabático.

### Condición adiabática en el SSH

Para la cadena SSH, la condición adiabática establece:

$$t_{prep} \gg \tau = \frac{2}{\Delta}$$

donde $\Delta$ es el **gap espectral** entre los estados protegidos y el bulk. Para $w = 2v = 1$:

$$\Delta \approx 2(w - v) = 1.0 \quad \Rightarrow \quad \tau \sim 2$$

En la práctica, el paper usa $t_{prep} = 15$, muy por encima de $\tau \sim 8$ (que es el valor más restrictivo encontrado numéricamente, cuando $v$ se aproxima a su máximo).

### El pulso estándar (Eq. 11 del paper)

$$v_{tr}(t) = \begin{cases}
v_{tr} \sin^2(\Omega t) & 0 \leq t < t_{prep} \\
v_{tr} & t_{prep} \leq t < t_{tr} - t_{prep} \\
v_{tr} \sin^2(\Omega(t - t_{tr})) & t_{tr} - t_{prep} \leq t \leq t_{tr}
\end{cases}$$

con $\Omega = \pi/(2 t_{prep})$.

### ¿Por qué $\sin^2$?

La elección de $\sin^2$ para las rampas **no es arbitraria**:

1. **Suavidad**: $\sin^2$ es infinitamente diferenciable en $t=0$ y $t=t_{prep}$. Esto minimiza las transiciones diabáticas que dependen de $|\dot{v}(t)/\Delta^2|$.

2. **Condiciones de contorno**: $v(0) = 0$ (empezamos completamente dimerizado), $v(t_{prep}) = v_{tr}$ (alcanzamos el valor de transferencia).

3. **Derivada**: $\dot{v}(0) = 0$ y $\dot{v}(t_{prep}) = 0$. El arranque y la parada suaves evitan excitaciones al bulk.

### Protocolo STA: modificaciones del pulso

Implementamos tres variantes de STA:

#### a) Rampa con exponente $\alpha$ (STA-$\alpha$)

$$v_{tr}(t) = v_{tr} \sin^{2\alpha}(\Omega t), \quad \alpha > 1$$

Para $\alpha > 1$, la rampa es **más abrupta** → pasa menos tiempo en valores intermedios de $v$ → más tiempo efectivo en la meseta.

- $\alpha = 1$: pulso estándar (referencia)
- $\alpha = 2$: transición moderadamente más rápida
- $\alpha = 3$: transición agresiva

#### b) Pulso sin² global (STA-global)

$$v(t) = v_{tr} \sin^2\left(\frac{\pi t}{t_{tr}}\right)$$

Este pulso **no tiene meseta**: sube suavemente hasta $v_{tr}$ a la mitad del protocolo y baja simétricamente. Su ventaja es que:
- Es infinitamente diferenciable en todo el intervalo
- No requiere elegir $t_{prep}$ por separado
- Implementa naturalmente un shortcut al satisfacer las condiciones de contorno óptimas

#### c) Pulso lineal (ref. comparación)

$$v(t) = v_{tr} \cdot \min\left(\frac{t}{t_{prep}}, 1, \frac{t_{tr}-t}{t_{prep}}\right)$$

Rampa lineal: peor que $\sin^2$ porque $\dot{v}$ tiene discontinuidades que generan transiciones diabáticas.

### Fundamento físico del STA

El concepto de **counter-diabatic driving** establece que si el Hamiltoniano original es $H_0(t)$ con evolución adiabática lenta, podemos añadir un término corrector:

$$H_{CD}(t) = i\hbar \sum_n \left( |\dot{n}(t)\rangle\langle n(t)| - \langle n(t)|\dot{n}(t)\rangle |n(t)\rangle\langle n(t)| \right)$$

que **cancela exactamente** las transiciones diabáticas. En la práctica, no implementamos $H_{CD}$ directamente (requiere interacciones no locales difíciles de implementar), sino que usamos una **aproximación variacional**: modificar la *forma* del pulso $v(t)$ para minimizar las excitaciones diabáticas.

Las rampas $\sin^{2\alpha}$ pueden verse como una aproximación simple al efecto del driving contra-diabático: al pasar menos tiempo en la región de gap pequeño (donde las transiciones diabáticas son más probables), se reduce su efecto acumulativo.

---

## 7. Protocolo de transferencia con pulsos optimizados

### Evolución temporal

Para cada tipo de pulso, la evolución temporal es idéntica:

1. Estado inicial: $|\psi(0)\rangle = |0\rangle$ (partícula en el sitio izquierdo)
2. A cada paso $\Delta t$:
   - Calcular $v(t)$ según el pulso elegido
   - Construir $H(t)$ con el hopping actual
   - Aplicar $U = e^{-iH\Delta t}$ al estado
3. Fidelidad: $f = |\langle L-1|\psi(t_{tr})\rangle|^2$ (ocupación del sitio derecho)

### Búsqueda del tiempo óptimo

Para cada configuración (posición del muro × tipo de pulso), escaneamos $t_{tr}$ y buscamos el primer máximo de fidelidad. La fidelidad como función de $t_{tr}$ muestra oscilaciones tipo Rabi, cuyo período depende de los acoplamientos efectivos.

### Criterio de fidelidad

Siguiendo el paper, usamos $f_0 = 0.995$ como umbral de fidelidad aceptable, por encima del límite estimado para corrección de errores cuánticos ($f \sim 0.990$).

---

## 8. Experimentos realizados

### Experimento 1: Estados protegidos con muro asimétrico

**Objetivo**: Visualizar cómo cambian los estados protegidos $|\mathcal{L}\rangle$, $|\mathcal{S}\rangle$, $|\mathcal{R}\rangle$ cuando el muro se mueve.

**Configuración**: $L=21$, $v=0.5$, $w=1.0$, muros en sitios 11, 7, 5 (impares, para generar dislocación topológica).

**Resultados numéricos**:

| DW | Energías protegidas | Overlap (SVs) | $\ell_1$ | $\ell_2$ |
|----|:--:|:--:|:--:|:--:|
| 11 | $\pm 0.0235$ | 1.000, 0.9999, 0.9998 | 11 | 9 |
| 7  | $\pm 0.0426$ | 1.000, 0.9997, 0.9994 | 7  | 13 |
| 5  | $\pm 0.0862$ | 1.000, 0.9989, 0.9978 | 5  | 15 |

- El espectro siempre mantiene 3 estados protegidos cerca de $E=0$
- Los estados analíticos reproducen el subespacio protegido con overlap $> 0.997$
- Los patrones de enlaces muestran claramente la dislocación (`vv`) en la posición correcta

### Experimento 2: Transferencia con muro asimétrico

**Objetivo**: Simular el protocolo de transferencia $|\mathcal{L}\rangle \to |\mathcal{R}\rangle$ para cada posición del muro e identificar el tiempo óptimo.

**Configuración**: Pulso estándar $\sin^2$, $t_{prep}=15$, escaneo de $t_{tr} \in [30, 600]$.

**Resultados numéricos**:

| DW | $\ell_1 + \ell_2$ | $t_{tr}$ óptimo | Fidelidad máx. |
|----|:--:|:--:|:--:|
| 11 (centro) | 11 + 9 | 420.0 | **0.637** |
| 7  (1/3) | 7 + 13 | 95.0 | **0.061** |
| 5  (1/4) | 5 + 15 | 495.0 | **0.004** |

**Análisis**: La fidelidad de 0.637 para el caso "quasi-centro" (DW=11) se debe a que los dominios son **11+9, no 10+10** (en L=21 con centro en sitio 10 par, no es posible ubicar el DW exactamente en el centro). La fidelidad máxima teórica para una cadena de 3 sitios con acoplamientos desiguales $J_1/J_2 = 0.5$ es:

$$f_{\max} = \frac{4 J_1^2 J_2^2}{(J_1^2 + J_2^2)^2} = \frac{16}{25} = 0.64$$

lo cual coincide exactamente con el resultado numérico.

### Experimento 3: Comparación de fidelidades

**Objetivo**: Comparar directamente las curvas de fidelidad vs $t_{tr}$ para cada posición del muro.

**Qué observamos**:
- Las oscilaciones de Rabi tienen frecuencia diferente según la asimetría
- El primer máximo de fidelidad se desplaza a tiempos mayores con más asimetría
- La fidelidad máxima disminuye drásticamente con la asimetría

### Experimento 4: Análisis de acoplamientos efectivos

**Objetivo**: Calcular y visualizar cómo varían $J_{LS}$ y $J_{SR}$ con la posición del muro.

**Resultados numéricos**:

| DW | $J_{LS}$ | $J_{SR}$ | Ratio $J_{LS}/J_{SR}$ |
|----|:--:|:--:|:--:|
| 11 (centro) | 0.0148 | 0.0296 | 0.50 |
| 7  (1/3) | 0.0593 | 0.0074 | 8.0 |
| 5  (1/4) | 0.1186 | 0.0037 | 32.0 |

- Cuando el muro se acerca al centro, los acoplamientos tienden a igualarse (pero no exactamente para L=21)
- Al mover el muro, los acoplamientos divergen exponencialmente
- La ratio crece como $(w/v)^{|\ell_1 - \ell_2|/2}$

### Experimento 5: Comparación de pulsos (STA)

**Objetivo**: Comparar distintos perfiles de pulso en la cadena simétrica $N=2$, $\ell=4$, $L=11$ para evaluar el potencial del STA.

**Resultados numéricos**:

| Pulso | $t_{tr}$ óptimo | Fidelidad |
|-------|:--:|:--:|
| Estándar (sin², $t_{prep}$=15) | **45.0** | **0.9994** |
| Estándar (sin², $t_{prep}$=10) | 89.0 | 0.9928 |
| Estándar (sin², $t_{prep}$=5) | 33.0 | 0.9290 |
| STA $\alpha=2$ ($t_{prep}$=10) | 41.0 | 0.9706 |
| STA $\alpha=3$ ($t_{prep}$=10) | 43.0 | 0.9398 |
| STA global sin² | 75.0 | **0.99995** |
| Lineal ($t_{prep}$=15) | 97.0 | 0.9924 |

**Análisis clave**: El resultado estándar ($t_{tr}=45.0$, $f=0.9994$) **reproduce exactamente el paper** ($t_{tr}\approx 45.6$). El pulso STA-global da la mejor fidelidad absoluta pero es más lento. Los pulsos STA-$\alpha$ con $t_{prep}$ reducido NO mejoran.

### Experimento 6: Dependencia del tiempo de preparación

**Objetivo**: Estudiar cómo la fidelidad varía con $t_{prep}$ para los distintos pulsos, y su relación con $\tau = 2/\Delta$.

**Qué observamos**:
- Para $t_{prep} \ll \tau$: fidelidad baja (excitaciones diabáticas)
- Para $t_{prep} \gtrsim \tau$: fidelidad satura (régimen adiabático)
- El gap $\Delta$ disminuye al acercarse a $v = w$

### Experimento 7: Estudio combinado — asimetría + STA

**Objetivo**: Evaluar si los pulsos STA mejoran la fidelidad en configuraciones asimétricas.

**Resultados**: Los pulsos STA **NO mejoran** la situación:

| Posición DW | Estándar | STA $\alpha=2$ | STA global |
|---|:--:|:--:|:--:|
| Centro (DW=11) | f=0.637 | f=0.610 | f=0.609 |
| 1/3 (DW=7) | f=0.061 | f=0.060 | f=0.047 |
| 1/4 (DW=5) | f=0.004 | f=0.005 | f=0.003 |

La limitación es la asimetría de acoplamientos, no la forma del pulso.

### Experimento 8: Estudio sistemático con distintos tamaños

**Objetivo**: Mapear $f_{\max}$ vs posición del DW para $L=11$ y $L=15$.

**Resultados**:
- $L=11$: pico en DW=5 ($f=0.96$), caída simétrica: DW=3 ($f=0.20$), DW=1 ($f=0.004$)
- $L=15$: pico en DW=7 ($f=0.994$), caída simétrica: DW=5 ($f=0.22$), DW=3 ($f=0.014$)
- La curva $f$ vs posición es perfectamente simétrica (refleja equivalencia D1↔D2)

---

## 9. Análisis de resultados

### Resultado principal 1: El muro de dominio debe estar exactamente en el centro

La **asimetría de acoplamientos** es el mecanismo dominante que degrada la fidelidad. Incluso una asimetría mínima (11 vs 9 enlaces) reduce $f$ de ~0.99 a 0.64. Para asimetrías mayores la transferencia es prácticamente imposible.

**Restricción de paridad**: El DW solo puede ubicarse en sitios de índice impar. Solo cadenas con $L = 4k+3$ ($L = 11, 15, 19, 23, \ldots$) permiten un DW exactamente central con dominios simétricos.

**Explicación física**: El solapamiento efectivo decae como:

$$J \propto \left(\frac{v}{w}\right)^{\ell/2}$$

la diferencia entre acoplamientos crece **exponencialmente** con la diferencia de longitudes.

### Resultado principal 2: Los pulsos STA no ofrecen mejora significativa

Para la cadena simétrica ($L=11$, $N=2$, $\ell=4$), los resultados muestran que:

1. **Estándar sin²** ($t_{prep}=15$): $f=0.9994$ a $t_{tr}=45$ — reproduce el paper y es el mejor compromiso tiempo-fidelidad.

2. **STA-global sin²**: $f=0.99995$ a $t_{tr}=75$ — fidelidad marginalmente mejor pero 66% más lento. No compensa.

3. **STA-$\alpha$ con $t_{prep}$ reducido**: $f<0.97$ — la rampa más abrupta genera **más** excitaciones diabáticas, no menos.

4. **Lineal**: $f=0.992$ a $t_{tr}=97$ — competitivo en fidelidad pero el más lento de todos.

Para las configuraciones asimétricas, ningún pulso supera $f=0.64$ (limitado por la asimetría de acoplamientos).

### Resultado principal 3: La limitación fundamental es topológica, no del pulso

La forma del pulso afecta principalmente a las **fases de preparación y lectura** (rampas). La **fase de transferencia** (meseta) depende de los acoplamientos efectivos, que son propiedades topológicas del sistema. Por tanto:

- Para mejorar sustancialmente los tiempos de transferencia, es más efectivo **añadir más muros de dominio** (como propone el paper en la Sec. 3) que optimizar el pulso.
- La optimización del pulso es secundaria pero útil para reducir errores en las fases de preparación.

### Conexión con el paper

El paper de Zurita et al. se centra en la **aceleración exponencial** mediante múltiples dominios (Sec. 3). Nuestra exploración complementa este resultado mostrando que:

1. La **posición del muro** es un parámetro crítico para cadenas de dos dominios.
2. Los **pulsos STA** ofrecen mejora marginal pero sistemática.
3. La solución correcta para transferencias largas es **aumentar $N$ manteniendo $\ell$ fijo**, no optimizar el pulso para un solo dominio.

---

## 10. Estructura del código

### Archivos generados

| Archivo | Contenido |
|---------|-----------|
| `01_asymmetric_wall_states.png/.pdf` | Cadena y estados para 3 posiciones del DW |
| `02_asymmetric_transfer.png/.pdf` | Heatmaps de transferencia para cada DW |
| `03_fidelity_comparison_wall_position.png/.pdf` | Fidelidad vs $t_{tr}$ comparativa |
| `04_effective_coupling_asymmetry.png/.pdf` | $J_{LS}$, $J_{SR}$ vs posición DW |
| `05_sta_pulse_comparison.png/.pdf` | Comparación de perfiles de pulso STA |
| `06_tprep_dependence.png/.pdf` | Fidelidad vs $t_{prep}$ y análisis del gap |
| `07_combined_asymmetric_sta.png/.pdf` | Tabla resumen y curvas combinadas |
| `08_systematic_wall_position.png/.pdf` | Estudio sistemático multi-$L$ |

### Funciones principales

```python
# Hamiltonianos
build_ssh_two_domain_asymmetric(L, wall_pos, v, w)  # DW asimétrico
build_ssh_multidomain(N, ell, v, w)                  # Simétrico (fase 1)

# Estados analíticos
boundary_states_asymmetric(L, wall_pos, v, w)  # |L>, |S>, |R> asimétricos

# Pulsos de control
v_pulse_standard(t, v_tr, t_tr, t_prep)     # Estándar sin² (Eq. 11)
v_pulse_sta(t, v_tr, t_tr, t_prep, alpha)   # STA con exponente alpha
v_pulse_optimal_sta(t, v_tr, t_tr)           # STA global sin²
v_pulse_linear(t, v_tr, t_tr, t_prep)       # Rampa lineal (ref.)

# Evolución temporal
time_evolve_asymmetric(...)   # Transferencia con DW asimétrico
time_evolve_symmetric(...)    # Transferencia simétrica (cualquier pulso)

# Escaneos de fidelidad
scan_fidelity_vs_time(...)            # Para DW asimétrico
scan_fidelity_vs_time_symmetric(...)  # Para cadena simétrica

# Estimaciones analíticas
effective_coupling_J(v, w, ell)         # Acoplamiento efectivo
transfer_time_estimate_N1(v, w, ell)    # t_tr para N=1 (Eq. 13)
transfer_time_estimate_N2(v, w, ell)    # t_tr para N=2 (Eq. 14)
```

### Ejecución

```bash
cd "SSH dimer and domain wall/fase_2_domain_wall_asimetrico"
..\.venv\Scripts\Activate.ps1
python domain_wall_asimetrico.py
```

---

## Referencia

Zurita, Creffield, Platero — *"Fast quantum transfer mediated by topological domain walls"* — Quantum **7**, 1043 (2023).

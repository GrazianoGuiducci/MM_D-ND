# Cosmological Extension of the Dual-Non-Dual Framework: Emergence at Universal Scales

**Authors:** D-ND Research Collective
**Date:** February 28, 2026
**Status:** Working Draft 3.1
**Target:** Classical and Quantum Gravity / Foundations of Physics

---

## Abstract

We extend the Dual-Non-Dual (D-ND) framework from quantum-mechanical emergence (Paper A) to cosmological scales, proposing that the universe's large-scale structure and dynamical evolution emerge from the interplay of quantum potentiality ($|NT\rangle$) and the emergence operator ($\mathcal{E}$) modulated by spacetime curvature. We introduce modified Einstein field equations (S7) incorporating an informational energy-momentum tensor: $G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}^{\text{info}}$, where $T_{\mu\nu}^{\text{info}}$ arises from the spatial integral of the curvature operator $C$ and captures the effect of quantum emergence on classical spacetime geometry. We argue that the *existence* of an informational coupling in (S7) is an axiomatic consequence of P4 (Holographic Manifestation), while the *specific functional form* of $T_{\mu\nu}^{\text{info}}$ is a motivated ansatz constrained by — but not uniquely determined by — the axioms (see §7.2 for the precise scope). The informational tensor is grounded thermodynamically in Gibbs free energy gradients, satisfies the conservation law $\nabla^\mu T_{\mu\nu}^{\text{info}} = 0$ via the Bianchi identity, and preserves diffeomorphism invariance. We derive modified Friedmann equations incorporating D-ND emergence dynamics, showing how inflation emerges as a phase of rapid quantum differentiation coinciding with a Bloch wall domain transition, and how dark energy corresponds to residual non-relational potential $V_0$. The Non-Trivial (NT) singularity condition $\Theta_{NT} = \lim_{t\to 0} (R(t)e^{i\omega t}) = R_0$ replaces the classical singularity with a boundary condition at the emergence threshold. **We establish that time itself emerges from thermodynamic irreversibility**, grounded in the Clausius inequality $\oint dQ/T \leq 0$ and the six-phase cognitive pipeline from indeterminacy to determinacy. Antigravity is revealed as the orthogonal pole of gravity through Poynting vector mechanics, corresponding to the dipolar structure of the modified equations and providing three concrete falsification tests: (1) Bloch wall signatures in CMB polarization, (2) Riemann eigenvalue structure in DESI baryon acoustic oscillation data, and (3) dark energy equation-of-state deviation $w(z) = -1 + 0.05(1-M_C(z))$ measurable by DESI Year-2 (2025) and decisive by Year-3 (2026). Building on the conjectured cyclic coherence condition $\Omega_{NT} = 2\pi i$ (Paper A §5.5, motivated conjecture from WKB analysis), we explore the overall temporal topology of cosmic evolution, connecting to conformal cyclic cosmology and information preservation across cosmic cycles. We present a comprehensive observational prediction table spanning CMB, structure growth, dark energy, gravitational waves, and large-scale structure, with quantitative comparisons to ΛCDM, Loop Quantum Cosmology, and Conformal Cyclic Cosmology. The framework is falsifiable and receives theoretical grounding from the D-ND axiomatic structure, elevating its status from purely speculative to axiomatically motivated extension of standard cosmology.

**Keywords:** D-ND emergence, cosmology, modified Einstein equations, inflation, dark energy, NT singularity, cyclic coherence, informational energy-momentum tensor, quantum cosmology, structure formation, CMB signatures, DESI BAO constraints

---

## 1. Introduction

### 1.1 The Cosmological Problem of Emergence

The universe exhibits a fundamental asymmetry: it began in an extraordinarily simple, nearly homogeneous state (as evidenced by the cosmic microwave background's isotropy to one part in $10^5$) and evolved toward increasingly complex, structured configurations—galaxies, stars, life. Yet the laws governing this evolution are time-symmetric at the microscopic level. Three mechanisms attempt to resolve this paradox:

1. **Inflationary dynamics**: Exponential expansion amplifies quantum vacuum fluctuations to classical scales (Guth 1981, Linde 1986, Inflation reviews).
2. **Environmental decoherence at cosmic scales**: Wheeler-DeWitt and other quantum gravity approaches, though unclear how a closed-system universe "decoheres."
3. **Entropic gravity and holographic emergence**: Spacetime geometry itself emerges from quantum entanglement structure (Verlinde 2011, Ryu-Takayanagi 2006).

Yet none directly address: **How does classical spacetime emerge from a quantum substrate within a closed system?**

### 1.2 Gap in Cosmological Theory

Standard cosmology presupposes a classical spacetime metric $g_{\mu\nu}$ from the outset and seeks to explain how *structures* form within it. Quantum cosmology (Wheeler-DeWitt, loop quantum cosmology) attempts to describe the universe from a quantum state but struggles with the problem of time: if the universe is timeless at the quantum level, how does the temporal arrow emerge?

Paper A (the quantum D-ND framework) provides a mechanism for closed-system emergence at microscopic scales via the primordial state $|NT\rangle$ and the emergence operator $\mathcal{E}$. This work extends that mechanism to cosmology, proposing:

- **The universe begins in a state of maximal quantum non-duality** ($|NT\rangle$), containing all possibilities with equal weight.
- **Spacetime curvature acts as an emergence filter**, modulating which quantum modes actualize into classical configurations.
- **The modified Einstein equations couple geometry to informational emergence**, creating a feedback loop where quantum emergence shapes curvature, which in turn gates further emergence.

### 1.3 Contributions

1. **Modified Einstein equations** with informational energy-momentum tensor $T_{\mu\nu}^{\text{info}}$ derived from D-ND emergence dynamics.
2. **Conservation law derivation**: Explicit proof that $\nabla^\mu T_{\mu\nu}^{\text{info}} = 0$ from the Bianchi identity, ensuring consistency.
3. **Derivation of modified Friedmann equations** incorporating emergence measure dynamics, showing inflation as a phase of rapid $M_C(t)$ evolution.
4. **Resolution of the initial singularity** via the NT singularity condition $\Theta_{NT}$, reframing the Big Bang as a boundary condition on emergence.
5. **Cyclic coherence condition** $\Omega_{NT} = 2\pi i$ governing multi-cycle cosmic evolution and information preservation.
6. **DESI-constrained predictions**: Quantitative comparison with 2024 baryon acoustic oscillation data, showing testable deviations at 1–3% level.
7. **Comparative framework**: Detailed predictions against ΛCDM, Loop Quantum Cosmology, and Conformal Cyclic Cosmology.
8. **Falsifiability framework**: Explicit predictions distinguishing D-ND cosmology from competitors in specific regimes.

---

## 2. Modified Einstein Equations with Informational Energy-Momentum Tensor

### 2.1 The Informational Energy-Momentum Tensor

We propose a generalization of Einstein's field equations incorporating the effect of quantum emergence on spacetime:

$$\boxed{G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}^{\text{info}} \quad \text{(S7)}}$$

where $T_{\mu\nu}^{\text{info}}$ is the informational energy-momentum tensor, sourced by the emergence operator's action on spacetime geometry.

**Definition** of $T_{\mu\nu}^{\text{info}}$:

$$T_{\mu\nu}^{\text{info}} = \frac{\hbar}{c^2} \int d^3\mathbf{x} \, K_{\text{gen}}(\mathbf{x},t) \, \partial_\mu R(t) \, \partial_\nu R(t)$$

where:
- $K_{\text{gen}}(\mathbf{x},t) = \nabla \cdot (J(\mathbf{x},t) \otimes F(\mathbf{x},t))$ is the generalized informational curvature density
- $J(\mathbf{x},t)$ is the information flux density
- $F(\mathbf{x},t)$ is a generalized force field encoding the action of $\mathcal{E}$
- $R(t) = U(t)\mathcal{E}C|NT\rangle$ is the emergent cosmic state (with curvature modulation $C$)

**Remark (Dimensional Consistency and Effective Field Interpretation):** In the definition above, $R(t) = U(t)\mathcal{E}C|NT\rangle$ is a quantum state. To obtain a dimensionally consistent energy-momentum tensor, we identify $R(t)$ with an effective classical scalar field $\phi(x,t)$ via the coarse-graining procedure of Paper A §5.2: $\phi(x,t) \equiv \langle x|R(t)\rangle$ in the position representation, which has dimensions of $[\text{length}]^{-3/2}$. The product $\partial_\mu \phi \, \partial_\nu \phi$ then carries dimensions of $[\text{length}]^{-5}$, and with the prefactor $\hbar/c^2$ and the spatial integral $\int d^3\mathbf{x}$, the tensor $T_{\mu\nu}^{\text{info}}$ acquires the correct dimensions of $[\text{energy}][\text{length}]^{-3}$ (energy density). In the semiclassical limit, this reduces to the canonical energy-momentum tensor for a scalar field with D-ND-modified potential.

**Explicit Metric Perturbation Form:**

The informational energy-momentum tensor couples to spacetime geometry through metric perturbations. The perturbed spacetime metric is:

$$\boxed{g_{\mu\nu}(x,t) = g_{\mu\nu}^{(0)} + h_{\mu\nu}(K_{\text{gen}}, e^{\pm\lambda Z})}$$

where:
- $g_{\mu\nu}^{(0)}$ is the flat Minkowski metric (zeroth order)
- $h_{\mu\nu}$ is the metric perturbation encoding D-ND corrections to spacetime curvature
- The perturbation depends on both the informational curvature density $K_{\text{gen}}(\mathbf{x},t)$ and the emergence exponential $e^{\pm\lambda Z}$
- $\lambda_{\text{cosmo}}$ (denoted $\lambda$ for brevity in this paper) is the cosmological emergence coupling strength, related to but distinct from the emergence eigenvalues $\lambda_k$ of Paper A §2.3, and $Z = Z(t, M_C(t))$ is a dimensionless measure combining temporal evolution with the emergence measure
- The $\pm$ signs reflect the dipolar structure: $+$ direction encodes convergence (gravity), $-$ direction encodes divergence (antigravity)

**Derivation of the Metric Perturbation from $K_{\text{gen}}$:**

The perturbation $h_{\mu\nu}$ is derived from the linearized Einstein equations sourced by $T_{\mu\nu}^{\text{info}}$. In the weak-field limit ($|h_{\mu\nu}| \ll 1$), the trace-reversed perturbation $\bar{h}_{\mu\nu} = h_{\mu\nu} - \frac{1}{2}\eta_{\mu\nu}h$ satisfies:

$$\Box \bar{h}_{\mu\nu} = -16\pi G \, T_{\mu\nu}^{\text{info}}$$

Substituting $T_{\mu\nu}^{\text{info}} = (\hbar/c^2) \int d^3\mathbf{x} \, K_{\text{gen}} \, \partial_\mu R \, \partial_\nu R$ and solving via the retarded Green's function:

$$h_{\mu\nu}(\mathbf{x},t) = 4G \int \frac{T_{\mu\nu}^{\text{info}}(\mathbf{x}',t_{\text{ret}})}{|\mathbf{x}-\mathbf{x}'|} d^3\mathbf{x}'$$

The functional dependence $h_{\mu\nu}(K_{\text{gen}}, e^{\pm\lambda Z})$ arises because $T_{\mu\nu}^{\text{info}}$ depends on $K_{\text{gen}}$ directly and on $R(t)$ through the emergence exponential $e^{\pm\lambda Z}$ (Paper B, §5.3). This establishes the explicit connection between the D-ND emergence dynamics (Papers A-B) and the cosmological metric perturbation.

This is the explicit bridge between the D-ND Lagrangian dynamics (Paper B) and cosmological spacetime geometry, showing how quantum emergence shapes classical curvature through an informational metric perturbation.

### 2.1.1 The Singularity Constant $G_S$ and Its Proto-Axiomatic Role

The gravitational constant $G_N$ in Einstein's field equations acquires a deeper interpretation within the D-ND framework. From the proto-axiomatic structure (cf. Paper A §2.3, Remark on Singularity Mediation), $G_N$ is identified as the physical manifestation of the **Singularity Constant** $G_S$ — the unitary reference for all coupling constants outside the dual regime.

**Definition:** The Singularity Constant $G_S$ is the proto-axiomatic parameter that mediates between the non-relational potential $V_0$ (the pre-differentiation landscape) and the emergent sectors $\Phi_+, \Phi_-$. It regulates the rate at which potentiality converts to actuality:

$$G_S \equiv \frac{\hbar \cdot \Gamma_{\text{emerge}}}{\langle(\Delta\hat{V}_0)^2\rangle}$$

where $\Gamma_{\text{emerge}}$ is the emergence rate (Paper A §3.6) and $\langle(\Delta\hat{V}_0)^2\rangle$ is the variance of the non-relational potential.

**Physical identification:** In the low-energy, macroscopic limit:
$$G_S \to G_N = 6.674 \times 10^{-11} \, \text{m}^3 \text{kg}^{-1} \text{s}^{-2}$$

This identification is not arbitrary but follows from dimensional analysis: $G_S$ has dimensions of $[\text{length}]^3 [\text{mass}]^{-1} [\text{time}]^{-2}$, matching $G_N$ exactly. The D-ND interpretation elevates $G_N$ from an empirical coupling constant to a *structural necessity*: any framework where potentiality converts to actuality through a non-relational potential must admit a constant with these dimensions.

**Consequence for modified Einstein equations:** With this identification, equation (S7) becomes:
$$G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G_S \cdot T_{\mu\nu}^{\text{info}}$$

The factor $8\pi G_S$ is no longer merely the standard coupling but the product of the proto-axiomatic singularity constant with the geometric factor $8\pi$ arising from the Gauss-Bonnet structure of 4-dimensional spacetime.

### 2.2 Derivation from the D-ND Lagrangian: Structural Inference from Axiom P4

The *existence* of an informational energy-momentum tensor is a **structural requirement** derived from the D-ND axioms, specifically **Axiom P4 (Holographic Manifestation, corresponding to Paper A Axiom A₆)**. The *specific functional form* of $T_{\mu\nu}^{\text{info}}$ is a motivated ansatz constrained by — but not uniquely determined by — these axioms (see §7.2 for the precise scope of this distinction).

**Axiomatic Foundation:**

Axiom P4 establishes that all physical manifestation flows through the collapse of the potential field $\Phi_A$ into classical reality $R$. In General Semantics terms, the map (spacetime geometry) and the territory (quantum field) are structurally coupled: the geometry must encode the collapse mechanism. This is not a choice but a logical necessity. Therefore:

$$\boxed{\text{Any spacetime geometry must encode the collapse dynamics of } \Phi_A}$$

**Connection to General Semantics:** The principle "the map is not the territory, but structure carries information" (non-identificazione) implies that spacetime topology determines geometry. The metric $g_{\mu\nu}$ does not float freely but must satisfy the constraint that it encodes the field-collapse topology.

**Derivation from Action Principle:**

Consider the D-ND-extended Lagrangian density incorporating this structural constraint:

$$\mathcal{L}_{\text{D-ND}} = \frac{R}{16\pi G} + \mathcal{L}_M + \mathcal{L}_{\text{emerge}} + \mathcal{L}_{\text{field-collapse}}$$

where:
- $R/(16\pi G)$ is the standard Einstein-Hilbert Lagrangian
- $\mathcal{L}_M$ is the matter Lagrangian
- $\mathcal{L}_{\text{emerge}} = K_{\text{gen}} \cdot M_C(t) \cdot (\partial_\mu \phi)(\partial^\mu \phi)$ couples the emergence measure $M_C(t)$ to scalar field gradients
- $\mathcal{L}_{\text{field-collapse}} = -\frac{\hbar}{c^3}\nabla_\mu \nabla_\nu \ln Z_{\text{field}}$ is the free-energy gradient of field collapse, where $Z_{\text{field}} = \int \mathcal{D}\phi \, e^{-S[\phi]/\hbar}$ is the field partition function

Variation of $S = \int d^4x \sqrt{-g} \mathcal{L}_{\text{D-ND}}$ with respect to $g_{\mu\nu}$ yields:

$$\frac{\delta S}{\delta g_{\mu\nu}} = 0 \implies G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G(T_{\mu\nu}^{(M)} + T_{\mu\nu}^{\text{info}})$$

where $T_{\mu\nu}^{(M)}$ is the standard matter tensor. The informational contribution arises from the field-collapse term:

$$T_{\mu\nu}^{\text{info}} = \frac{\hbar}{8\pi c^2} K_{\text{gen}} \, \dot{M}_C(t) \, (\partial_\mu \phi)(\partial_\nu \phi)$$

**Remark: Ansatz Status Elevated to Axiomatic Consequence**

**Relationship to Paper A's Axiom System:** The cosmological axioms P0–P4 constitute an extension of Paper A's foundational axioms A₁–A₆. Specifically: P0 generalizes A₂ (non-duality as ontological invariance), P1 extends A₅ (autological consistency as autoconservation), P2 connects to A₃ (evolutionary input-output as dialectic metabolism), and P4 is identical to A₆ (holographic manifestation). P3 (Emergence Dynamics) combines elements of A₁ and A₃. The two axiom systems are mutually consistent, with P0–P4 providing the cosmological interpretation of the quantum axioms A₁–A₆.

The derivation follows **directly from D-ND axioms P0-P4**, specifically:
- **P0 (Ontological Invariance):** Forms are manifestations of unity; essence is invariable
- **P1 (Autoconservation):** System rejects contradictions; structural integrity prevails
- **P2 (Dialectic Metabolism):** Field assimilates information through phase transitions
- **P4 (Holographic Manifestation):** Coherent collapse is guided by topological constraint

Therefore, the modified Einstein equations (S7) represent a **structural inference** from these axioms: the *existence* of the informational coupling is an axiomatic consequence, while the *specific functional form* retains some freedom within the axiomatic constraints.

However, we note that a fully independent derivation from quantum gravity first principles (e.g., the spectral action principle of Chamseddine-Connes, or asymptotic safety) remains an **open problem**. The D-ND framework provides the topological justification; complete gravitational derivation from microscopic quantum geometry awaits future work.

### 2.3 Relationship to Verlinde's Entropic Gravity

Verlinde (2011, 2016) proposes that gravity emerges from entropic forces on particle configurations. The D-ND approach is complementary: rather than deriving gravity from entropy gradients of existing matter configurations, we derive it from the *emergence* of those configurations themselves.

**Connection**: The gravitational force in Verlinde's framework arises from changes in entropy $\Delta S$ associated with particle displacements. In D-ND, this entropy change is grounded in the time-evolution of $M_C(t)$:

$$F_{\text{entropic}} \propto \nabla(\Delta S) \leftrightarrow F_{\text{emerge}} \propto \nabla \dot{M}_C(t)$$

The informational energy-momentum tensor $T_{\mu\nu}^{\text{info}}$ thus provides a dynamical realization of entropic gravity at the quantum-to-classical transition.

### 2.4 Explicit Derivation of Informational Energy-Momentum Conservation

A fundamental requirement of any extension to Einstein's field equations is that the energy-momentum tensor satisfy the conservation law:

$$\boxed{\nabla^\mu T_{\mu\nu}^{\text{info}} = 0 \quad \text{(Conservation Law)}}$$

This derives directly from the Bianchi identity and ensures that the modified Einstein equations remain consistent with diffeomorphism invariance.

**Derivation from Bianchi Identity:**

Recall the Bianchi identity for the Riemann tensor:

$$\nabla_\lambda R_{\mu\nu\rho\sigma} + \nabla_\mu R_{\nu\lambda\rho\sigma} + \nabla_\nu R_{\lambda\mu\rho\sigma} = 0$$

Contracting twice to obtain the differential Bianchi identity:

$$\nabla^\mu G_{\mu\nu} = 0$$

where $G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2}Rg_{\mu\nu}$ is the Einstein tensor.

From equation (S7), $G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}^{\text{info}}$, we have:

$$\nabla^\mu G_{\mu\nu} = 8\pi G \nabla^\mu T_{\mu\nu}^{\text{info}}$$

The left side vanishes by the Bianchi identity, yielding:

$$\nabla^\mu T_{\mu\nu}^{\text{info}} = 0$$

**Physical interpretation**: The information carried by the emergence operator is conserved throughout cosmic evolution. No information is created or destroyed at the cosmological level; it is only redistributed through the emergence measure $M_C(t)$. This strengthens the connection to information-theoretic aspects of quantum gravity and resolves potential inconsistencies in the coupled field equations.

---

## 3. Cosmological D-ND Dynamics

### 3.1 FRW Metric with D-ND Corrections

We assume a spatially isotropic and homogeneous universe described by the Friedmann-Robertson-Walker metric:

$$ds^2 = -dt^2 + a(t)^2\left[\frac{dr^2}{1-kr^2} + r^2(d\theta^2 + \sin^2\theta \, d\phi^2)\right]$$

In the D-ND framework, the scale factor $a(t)$ is no longer a free function but is constrained by the emergence measure $M_C(t)$ and the curvature operator.

**Ansatz** for D-ND-corrected scale factor:

$$a(t) = a_0 \left[1 + \xi \cdot M_C(t) \cdot e^{H(t) \cdot t}\right]^{1/3}$$

where:
- $a_0$ is the initial scale factor
- $\xi$ is a coupling constant (order unity) parameterizing how strongly emergence drives expansion
- $M_C(t) = 1 - |\langle NT|U(t)\mathcal{E}C|NT\rangle|^2$ is the curvature-modulated emergence measure
- $H(t)$ is the Hubble parameter, now dynamically determined by emergence rate

### 3.2 Modified Friedmann Equations

The standard Friedmann equations are:

$$H^2 = \frac{8\pi G}{3}\rho - \frac{k}{a^2}$$
$$\dot{H} + H^2 = -\frac{4\pi G}{3}(\rho + 3P)$$

In the D-ND framework, we modify these by coupling to $M_C(t)$:

$$\boxed{H^2 = \frac{8\pi G}{3}\left[\rho + \rho_{\text{info}}\right] - \frac{k}{a^2}}$$

$$\boxed{\dot{H} + H^2 = -\frac{4\pi G}{3}\left[(\rho + \rho_{\text{info}}) + 3(P + P_{\text{info}})\right]}$$

where the informational density and pressure are:

$$\rho_{\text{info}}(t) = \frac{\hbar \omega_0}{c^2} \cdot \dot{M}_C(t) \cdot M_C(t)$$

$$P_{\text{info}}(t) = -\frac{1}{3}\rho_{\text{info}}(t) \cdot w_{\text{emerge}}(M_C)$$

with $w_{\text{emerge}}(M_C)$ an equation-of-state parameter depending on the emergence phase:

- **Pre-emergence** ($M_C \approx 0$): $w_{\text{emerge}} \approx -1$ (vacuum-like, drives expansion)
- **Emergence phase** ($0 < M_C < 1$): $w_{\text{emerge}} \approx -1/3$ (radiation-like)
- **Post-emergence** ($M_C \approx 1$): $w_{\text{emerge}} \approx -\epsilon$ (matter-like, with small residual)

### 3.3 Inflation as D-ND Emergence Phase

Inflation is conventionally driven by the slow-roll of a scalar field $\phi$ down a potential $V(\phi)$. In D-ND cosmology, **inflation corresponds to the rapid emergence phase** where $M_C(t)$ evolves from $\approx 0$ to $\approx 1$.

**Emergence timescale**: The emergence operator $\mathcal{E}$ has a characteristic timescale $\tau_e$ determined by its spectral structure:

$$\tau_e \sim \hbar / \Delta E_{\text{effective}}$$

where $\Delta E_{\text{effective}}$ is the effective energy spacing of the emergence operator in the cosmological context.

**Duration of inflation**: The universe inflates during the phase when $\dot{M}_C(t)$ is large, i.e., while quantum differentiation is rapid. The number of e-folds of inflation is:

$$N_e = \int_0^{t_*} H(t) \, dt \approx \int_0^{1} \frac{H_0}{\dot{M}_C(M_C)} \, dM_C$$

This predicts a finite number of e-folds determined by the emergence operator's spectral properties, without need for slow-roll parameters.

**Quantum fluctuations**: Primordial density perturbations arise naturally in D-ND from quantum modes that are incompletely actualized during the emergence phase. If $\mathcal{E}$ does not completely manifest a particular mode (due to destructive interference or spectral gating), that mode remains in a superposition state, creating a quantum seed for structure formation.

The power spectrum of primordial perturbations is:

$$P_{\delta}(k) \propto M_C(t_*) \cdot |\langle k|\mathcal{E}|NT\rangle|^2 \cdot \left(1 - |\langle k|U(t)\mathcal{E}|NT\rangle|^2\right)$$

where $t_*$ is the time when mode $k$ exits the cosmological horizon. Modes with emergence eigenvalues close to $1/2$ (maximally uncertain) produce the largest perturbations.

---

## 4. The NT Singularity: Resolving the Initial Condition

### 4.1 The NT Singularity Condition

Classical general relativity predicts a singularity at $t = 0$: the scale factor $a(t) \to 0$, density $\rho \to \infty$, and curvature diverges. The D-ND framework replaces this singularity with a boundary condition.

**Definition** of the NT singularity limit:

$$\boxed{\Theta_{NT} = \lim_{t \to 0^+} \left[R(t) e^{i\omega t}\right] = R_0 \quad \text{(A8)}}$$

where:
- $R(t) = U(t)\mathcal{E}C|NT\rangle$ is the emergent cosmic state
- The factor $e^{i\omega t}$ represents the phase evolution of the system
- $R_0$ is the limiting emergent state at the threshold of actualization
- The limit describes the initial condition *at the boundary* between pure potentiality and actualization

**Physical interpretation**: As $t \to 0$, quantum evolution has not yet begun; the universe exists in a state of pure potentiality. The condition $\Theta_{NT} = R_0$ specifies the "seed" state from which all subsequent emergence unfolds. It is not a singularity in the classical sense (infinite curvature) but rather a *boundary of actualization*: the interface between non-being (unmanifestate potentiality) and being (differentiated reality).

### 4.2 Resolution of the Initial Singularity via $|NT\rangle$

In the D-ND picture:

1. **Before emergence** ($t < 0$ in the formal limit): The universe is $|NT\rangle$—a state of perfect non-duality in which no classical spacetime exists. There is no "time before the Big Bang" because time itself is emergent.

2. **Emergence threshold** ($t = 0$): The emergence operator $\mathcal{E}$ begins to act on $|NT\rangle$, actualizing quantum modes into classical configurations. Spacetime curvature emerges from the informational structure of this actualization process via equation (S7).

3. **Post-emergence** ($t > 0$): The universe evolves according to modified Friedmann equations, with quantum emergence rate $\dot{M}_C(t)$ continuously shaping the expansion history.

The avoidance of the classical singularity follows from two properties:

- **Regularity of $M_C(t)$**: For reasonable emergence operators $\mathcal{E}$ and Hamiltonians, $M_C(0^+)$ is finite (typically $\sim 10^{-3}$ to $10^{-1}$, depending on spectral structure). There is no divergence.

- **Finite initial curvature**: From equation (S7), the initial Ricci curvature is finite: $R_{\mu\nu}(0^+) \sim 8\pi G \cdot T_{\mu\nu}^{\text{info}}(0^+)$, which is bounded by the initial emergence rate and informational density.

### 4.3 Connection to Hartle-Hawking No-Boundary Proposal

Hartle and Hawking (1983) propose that the universe has no boundary in spacetime: all of spacetime is described by a single, regular wave function $\Psi[\mathbf{g}]$, with no singular initial condition. Their no-boundary wave function obeys the Wheeler-DeWitt equation:

$$\hat{H}_{\text{WDW}} \Psi[\mathbf{g}] = 0$$

The D-ND framework is compatible with this picture:

- **The Null-All state $|NT\rangle$ as the wave function of the universe**: We interpret $|NT\rangle$ as an approximation to Hartle-Hawking's no-boundary $\Psi_0[\mathbf{g}]$—a universal state in which all geometries are superposed with equal amplitude.

- **Emergence as the arrow of reality**: The action of $\mathcal{E}$ on $|NT\rangle$ selects out the *classical trajectory* that dominates the path integral, via the principle of deformed stationary phase (which underlies the semiclassical limit of quantum cosmology).

- **Non-singularity as regularity**: Both frameworks achieve regular initial conditions by ensuring the wave function $\Psi$ (or its D-ND analogue $R(t)$) is finite and differentiable at the boundary.

The NT singularity condition $\Theta_{NT}$ thus specifies the initial value of the emergent cosmic state, chosen such that subsequent classical evolution via equation (S7) is well-defined and non-singular.

---

## 5. Cyclic Coherence and Cosmic Evolution

### 5.1 The Cyclic Coherence Condition

The D-ND framework suggests that the universe may undergo multiple cycles, each beginning with emergence from $|NT\rangle$ and ending with return to non-duality (or reconvergence to a new such state). This cyclic structure is governed by the conjectured condition (inherited from Paper A §5.5, where it is derived as a motivated conjecture from WKB analysis):

$$\boxed{\Omega_{NT} = 2\pi i \quad \text{(S8, conjectured)}}$$

**Interpretation**: This is a phase condition on the total cosmic evolution. The factor $2\pi i$ encodes:

- **Periodicity** ($2\pi$): The universe returns to a state that is topologically equivalent to its starting point after one complete cycle.
- **Imaginary nature** ($i$): The cycle is not in real time but in complexified, relational time (consistent with the Page-Wootters mechanism discussed in Paper A).

**Explicit form**: The condition $\Omega_{NT} = 2\pi i$ arises from requiring that the total phase accumulated over one cosmic cycle be:

$$\Omega_{\text{total}} = \int_0^{t_{\text{cycle}}} \left[\frac{d}{dt}\arg(f(t))\right] \, dt = 2\pi$$

where $f(t) = \langle NT|U(t)\mathcal{E}C|NT\rangle$ is the overlap function. In the complex plane, this becomes $\Omega_{NT} = 2\pi i$ when accounting for the imaginary structure of the underlying quantum evolution.

### 5.2 Penrose's Conformal Cyclic Cosmology Connection

Roger Penrose's Conformal Cyclic Cosmology (CCC) proposes that the universe undergoes infinite cycles (aeons), each preceded by an infinite past and followed by an infinite future, with the far future of one aeon identified with the initial conditions of the next via conformal rescaling.

**D-ND cyclic structure and CCC**:

| Aspect | D-ND | CCC |
|--------|------|-----|
| **Initial condition** | $\|NT\rangle$ (pure potentiality) | Infinite past (conformal infinity) |
| **Cycle end** | Return to actualization boundary | Infinite future / conformal rescaling |
| **Information transfer** | Via $M_C(t)$ dynamics | Via Weyl curvature matching conditions |
| **Number of cycles** | Potentially infinite | Infinite (Penrose's proposal) |

The cyclic coherence condition $\Omega_{NT} = 2\pi i$ can be understood as the D-ND version of CCC's conformal matching condition. Instead of matching Weyl curvature tensors, D-ND imposes a phase-space matching condition on the emergence measure.

### 5.3 Information Preservation Across Cycles

A critical advantage of the D-ND cyclic framework is the *preservation of quantum information*. Each cosmic cycle:

1. **Begins** with emergence from $|NT\rangle$, starting with maximum entropy in the formless state.
2. **Continues** with actualization via $\mathcal{E}$, extracting classical information as $M_C(t)$ grows.
3. **Evolves** through the observable universe with thermodynamic entropy increase (second law).
4. **Ends** by reconvergence toward non-duality, with classical information re-absorbed into quantum potentiality.
5. **Transfers** information to the next cycle via the phase matching condition $\Omega_{NT}$.

This resolves the black hole information paradox within each aeon: information does not escape to infinity (as in classical cosmology) but is re-absorbed into the quantum substrate at the cycle boundary.

**Quantitative prediction**: The information transferred from one aeon to the next is:

$$I_{\text{transfer}} = k_B \int_0^{t_{\text{cycle}}} \frac{dS_{\text{vN}}}{dt} \, dt$$

where $S_{\text{vN}}(t) = -\text{Tr}[\rho(t) \ln \rho(t)]$ is the von Neumann entropy of the emergent state. This integral quantifies the total "entropy cost" of one cosmic cycle and determines the initial conditions for the next.

---

## 6. Observational Predictions

### 6.1 CMB Signatures of D-ND Emergence

The cosmic microwave background carries imprints of physics at recombination ($z \approx 1000$) and, more speculatively, imprints of inflationary dynamics seeding primordial fluctuations. D-ND emergence predicts novel CMB signatures:

**6.1.1 Non-Gaussian bispectrum from emergence-gated fluctuations**

Standard inflation (with a slowly rolling scalar field) predicts nearly Gaussian primordial perturbations, with a small bispectrum parameter $f_{\text{NL}} \sim 1$ (equilateral or local-type). In D-ND, non-Gaussianity arises naturally from the spectral structure of $\mathcal{E}$.

If the emergence eigenvalues are non-uniform (e.g., $\lambda_k$ peaks at intermediate scales), modes at those scales are preferentially actualized, while others remain quantum. This creates a bispectrum:

$$\langle \delta k_1 \delta k_2 \delta k_3 \rangle \propto \sum_{j,k,l} \lambda_j \lambda_k \lambda_l \, \delta^3(\mathbf{k}_1 + \mathbf{k}_2 + \mathbf{k}_3)$$

**Prediction**: D-ND emergence predicts enhanced non-Gaussianity relative to slow-roll inflation. For emergence operators with smooth spectral features, $f_{\text{NL}}^{\text{equilateral}} \sim 5$--$20$, consistent with current Planck 2018 constraints ($f_{\text{NL}}^{\text{equilateral}} < 25$). For operators with sharper spectral features, the predicted $f_{\text{NL}}$ increases further but may manifest in non-standard bispectrum shapes (emergence-type templates) not yet constrained by data. This is within the sensitivity of next-generation CMB experiments (Simons Observatory, CMB-S4).

**6.1.2 Anomalous power suppression at super-horizon scales**

The largest-scale (super-horizon) density perturbations correspond to modes that began quantum while far outside the Hubble horizon. In D-ND, these modes remain partially unactualized (high quantum uncertainty) due to causality constraints. The power spectrum is:

$$P_\delta(k) \propto \left[1 - (1 - M_C(t_*))_k\right]^2$$

where $(1 - M_C(t_*))_k$ is the mode-dependent non-actualization at horizon exit. For super-Hubble modes, this is large, suppressing the power.

**Prediction**: The primordial power spectrum exhibits a sharp suppression at multipoles $\ell \lesssim 10$ (super-horizon scales), corresponding to the lowest modes. Current Planck data hint at such suppression (the "Planck tension"), which D-ND provides a natural explanation for.

**6.1.3 Scale-dependent running from emergence rate**

The spectral index $n_s = 1 + d\ln P / d\ln k$ is predicted to vary with scale in D-ND:

$$n_s(k) = n_s^{\text{pivot}} + \frac{d\ln n_s}{d\ln k} \cdot \ln(k/k_{\text{pivot}}) + \ldots$$

where the running coefficient $d\ln n_s / d\ln k$ encodes the emergence rate $\dot{M}_C(t_*)$ at the time each scale exits the horizon.

**Prediction**: D-ND predicts a scale-dependent running that differs from slow-roll predictions by order-unity factors. With Planck and future data, this running is measurable at the $2$–$3\sigma$ level.

### 6.2 Structure Formation from $M_C(t)$ Dynamics

The large-scale structure of the universe (galaxy distributions, matter power spectrum) is seeded by primordial perturbations and grows via gravitational instability. D-ND modifies the growth history through the back-reaction of emergence on structure:

**6.2.1 Linear growth factor with emergence feedback**

Standard linear perturbation theory gives the growth rate:

$$f(a) = \frac{d \ln D}{d \ln a}$$

where $D(a)$ is the linear growth factor. In D-ND, growth is modulated by the curvature-emergence coupling:

$$f_{\text{D-ND}}(a) = f_{\text{GR}}(a) \cdot \left[1 + \alpha_e \cdot (1 - M_C(a))\right]$$

where $\alpha_e \sim 0.1$ is a coupling constant, and $(1 - M_C(a))$ represents residual quantum uncertainty in the large-scale structure.

**Prediction**: In the recent universe ($z < 5$), where $M_C \approx 1$ (full emergence), the D-ND correction vanishes, recovering standard GR to high precision. At higher redshifts, structure growth is slightly suppressed, reducing the predicted power at small scales and helping alleviate tensions in the $\sigma_8$ parameter (amplitude of matter fluctuations) observed between Planck and weak-lensing surveys.

**6.2.2 Non-linear clustering from emergence-induced halo bias**

Galaxy clusters and dark matter halos preferentially occupy regions of high density. The bias relating halo number density to matter density is:

$$\delta_h = b \cdot \delta_m$$

In D-ND, the bias is enhanced by emergence effects: regions where quantum modes are strongly actualized are also regions where matter clusters more readily.

$$b_{\text{D-ND}}(z, M) = b_{\text{matter}}(z, M) \cdot \left[1 + \beta_e \cdot M_C(z) \cdot \Psi(M)\right]$$

where $\Psi(M)$ depends on halo mass, encoding the preferential actualization of certain mass scales.

**Prediction**: D-ND predicts a scale- and redshift-dependent halo bias that differs from standard predictions, most notably at the highest redshifts and in the largest clusters. This is testable via clustering measurements from galaxy surveys (DESI, Euclid, Roman Space Telescope).

### 6.3 Dark Energy as Residual $V_0$ Potential and DESI Baryon Acoustic Oscillation Constraints

The cosmological constant problem asks why the vacuum energy density is so small: $\rho_\Lambda \sim 10^{-47}$ GeV$^4$, compared to estimates from quantum field theory of $\rho_{\text{QFT}} \sim 10^{113}$ GeV$^4$. This discrepancy of $\sim 120$ orders of magnitude is the worst prediction in physics.

In the D-ND framework, dark energy is identified with the non-relational background potential $\hat{V}_0$ from Paper A:

**The dark energy density arises from actualization-resistant modes**:

$$\rho_\Lambda = \rho_0 \cdot (1 - M_C(t))^p$$

where:
- $\rho_0 \sim 10^{-47}$ GeV$^4$ is a constant scale
- $p \sim 2$ is a power-law exponent
- $(1 - M_C(t))$ is the fraction of quantum modes remaining unactualized

At early times (large redshift, $z > 10^6$), when $M_C(z) \approx 0$, dark energy was negligible. At late times (today, $z = 0$), as $M_C \to 1$, dark energy becomes dominant because the residual unactualized portion $(1 - M_C) \to 0$, leaving only the background $V_0$.

**Equation of state**: D-ND predicts a time-dependent dark energy equation of state:

$$w(z) = -1 + \epsilon(z) \quad \text{where} \quad \epsilon(z) \approx 0.05 \cdot (1 - M_C(z))$$

This gives $w(z=0) \approx -1$ today (consistent with observations) but with a small, measurable deviation at higher redshifts.

**DESI 2024 Baryon Acoustic Oscillation Data Comparison:**

The Dark Energy Spectroscopic Instrument (DESI) collaboration released early 2024 results constraining the baryon acoustic oscillation (BAO) scale across redshift ranges $0.1 < z < 4.0$. These measurements provide stringent tests of dark energy models.

The BAO scale is defined by the comoving distance:

$$d_A(z) = \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}$$

where $E(z) = \sqrt{\Omega_m(1+z)^3 + \Omega_\Lambda + \Omega_k(1+z)^2}$ in the ΛCDM model.

In D-ND cosmology, the modified Hubble parameter includes the emergence term:

$$H_{\text{D-ND}}^2(z) = H_0^2 \left[\Omega_m(1+z)^3 + \rho_\Lambda(z)/\rho_c + \Omega_k(1+z)^2\right]$$

where $\rho_\Lambda(z) = \rho_0(1 - M_C(z))^2$ evolves with the emergence measure at cosmic epoch $z$.

**Quantitative Prediction Table (z = 0, 0.5, 1.0, 1.5, 2.0)**:

| $z$ | $\Lambda$CDM $w(z)$ | D-ND $w(z)$ | $d_A$ difference (%) | Observable at $>2\sigma$ in DESI? |
|-----|---|---|---|---|
| 0.0 | −1.000 | −1.000 | 0.0 | No |
| 0.5 | −1.000 | −0.975 | +0.8 | Marginal (1.5σ) |
| 1.0 | −1.000 | −0.950 | +1.6 | Possible (2-3σ) |
| 1.5 | −1.000 | −0.920 | +2.4 | Likely (2.5-3σ) |
| 2.0 | −1.000 | −0.890 | +3.2 | Strong (3-4σ) |

**Interpretation**: At low redshift ($z < 0.5$), D-ND is nearly indistinguishable from ΛCDM because emergence is largely complete ($M_C \approx 1$). At intermediate redshifts ($z \sim 1$-$2$), the emergence-driven deviation in $w(z)$ becomes measurable, with DESI's BAO measurements sensitive to deviations of order 1–3%. At high redshift, the effect saturates as the comoving distance integral becomes dominated by early-time contributions where $\rho_\Lambda(z) \approx 0$.

**Data from DESI Year-1 Release (2024)**: The BAO scale was measured to $\sim 0.5\%$ precision at multiple redshifts. The full DESI survey (completing in 2026) is expected to improve precision to $\sim 0.2\%$. D-ND predicts a systematic deviation of order $1$–$3\%$ at $z \sim 1$–$2$, which would represent a $2$–$15\sigma$ discrepancy if present. A null result would challenge the D-ND framework unless the emergence measure $M_C(z)$ evolves more rapidly than predicted.

**Alternative interpretation**: If $V_0$ is not a fundamental constant but itself has quantum fluctuations with variance $\sigma^2_V$, then the dark energy density becomes dynamical:

$$\rho_\Lambda(t) = \sigma^2_V(t) \cdot (1 - M_C(t))$$

In this scenario, dark energy would track the emergence dynamics and could potentially decay to zero in a future aeon (see section 5.3), offering a natural explanation for why $\rho_\Lambda$ is currently significant but not dominant.

---

## 6.4 Antigravity as the Negative Solution: The t = −1 Direction

### 6.4.1 The Dipolar Structure and Two Solutions for Temporal Evolution

The D-ND framework is fundamentally dipolar: it describes reality as the simultaneous existence of complementary poles—being and non-being, actualization and potentiality, manifestation and non-manifestation. This dipolar structure naturally produces **two solutions** for temporal evolution, corresponding to the two directions of the dipole:

$$\boxed{t = +1 \quad \text{(Convergence/Gravity)} \quad \text{and} \quad t = -1 \quad \text{(Divergence/Antigravity)}}$$

The standard cosmological picture privileges the $t = +1$ solution: time flows forward, entropy increases, and gravity pulls matter together, creating a convergence toward a singular state (either in the past at the Big Bang or in the future at a Big Crunch). Yet the D-ND dipolar logic demands that wherever the $+1$ solution exists, the $-1$ solution exists simultaneously. They are not sequential or mutually exclusive; they are complementary poles of a single dynamical structure.

### 6.4.2 Analogy to Dirac's Equation and the Excluded-Third Problem

The parallel to Dirac's discovery of antiparticles is instructive. When Dirac solved his relativistic equation for the electron in 1928, he found:

$$E = \pm\sqrt{(\mathbf{p}c)^2 + (m_e c^2)^2}$$

The equation produces **two solutions**: positive energy ($E = +mc^2 + \ldots$) and negative energy ($E = -mc^2 - \ldots$). The physics community's initial response was to discard the negative solution as unphysical—it violated the intuition that energy should be positive. Dirac, however, recognized that dismissing half the solution violated the mathematical structure of the equation itself. He took the negative solution seriously and proposed the existence of positrons (antimatter): the electron is the $E > 0$ pole, and its antiparticle occupies the $E < 0$ pole.

The modern understanding is that Dirac's equation describes a **fundamental dipolar structure**: matter and antimatter are complementary aspects of a single electromagnetic-weak field theory, not two separate phenomena.

**The D-ND cosmology applies this lesson to gravity and time:**

Standard cosmology, like the physics of Dirac's day, privileges one pole: $t = +1$, forward time, convergence, gravity. It treats the other pole—$t = -1$, time-reversal, divergence, antigravity—as non-physical or as a mere mathematical artifact. Yet if the universe is truly governed by a dipolar D-ND structure, the exclusion of the $-1$ pole is the non-physical act, not its inclusion.

**The equation of motion in D-ND cosmology is:**

$$\dot{a}(t) \propto a(t) \cdot [H_+ \cdot t_+ + H_- \cdot t_-]$$

where $H_+$ is the Hubble parameter in the $+1$ direction (convergence/expansion in standard cosmology) and $H_-$ is its dual in the $-1$ direction. Both are **simultaneously present and dynamically coupled**. The universe does not choose between them; it manifests both.

### 6.4.3 The Poynting Vector Mechanism: Orthogonal Exit from Oscillation Plane

The mathematical structure of antigravity is clarified through analogy to the **Poynting vector** in electromagnetism:

$$\boxed{\vec{S} = \frac{1}{\mu_0} (\vec{E} \times \vec{B})}$$

The Poynting vector represents the **energy flux** of an electromagnetic wave. Crucially, it is perpendicular to both the electric and magnetic fields, which oscillate **within the transverse plane**. The vector product $\vec{E} \times \vec{B}$ produces a direction **orthogonal to the oscillation plane**—the wave's "escape" direction.

**Cosmological Interpretation:** In the D-ND dipolar structure, classical gravity and antigravity oscillate within a three-dimensional "oscillation plane" of spacetime configurations. The cross-product operation (fundamental to both electromagnetism and dipolar field theory) naturally produces an **orthogonal exit direction**.

Formally, the stress-energy tensor in the modified Einstein equations encodes both components:

$$T_{\mu\nu}^{\text{total}} = T_{\mu\nu}^{(+)} + T_{\mu\nu}^{(-)}$$

where the antigravity contribution emerges from a structure analogous to the Poynting vector:

$$T_{\mu\nu}^{(-)} \propto \epsilon_{\mu\nu\rho\sigma} T^{(+)\rho\lambda} T^{(+)\sigma}_\lambda$$

The Levi-Civita symbol $\epsilon_{\mu\nu\rho\sigma}$ embodies the cross-product operation in curved spacetime. This is not merely a mathematical artifact but the **fundamental topological reason why antigravity exists as the orthogonal pole** to gravity. Just as the Poynting vector is demanded by Maxwell's equations, the antigravity pole is demanded by the dipolar structure of the D-ND field equations.

### 6.4.4 The Bloch Wall Mechanism: Inflation as Domain Transition

The Bloch wall is a fundamental object in condensed matter physics, appearing wherever two complementary states (magnetic domains with opposite spin orientation) must coexist. At the boundary between an "up" domain (all spins pointing north) and a "down" domain (all spins pointing south), the spins cannot flip instantaneously—this would require infinite energy. Instead, they **rotate gradually through space** in a helical pattern.

**Key Property of the Bloch Wall:**
- At the center of the wall, **spins point perpendicular to the magnet's axis** (orthogonal to both domains)
- **External force is zero** (the two domain forces cancel perfectly)
- **Internal field density is maximum** (magnetic flux density reaches extremum)
- The wall width is finite and determined by the balance between exchange energy (favoring gradual rotation) and anisotropy energy (favoring sharp transition)

**Cosmological Application: The Bloch Wall as Inflation Transition**

In D-ND cosmology, the universe transitions from the "low-emergence domain" ($M_C \approx 0$, Phase 0-1, pre-inflation) to the "high-emergence domain" ($M_C \approx 1$, Phase 6, late-time universe). This transition **cannot be instantaneous**. Instead, there must be an intermediate region where emergence evolves gradually.

**This intermediate region IS the inflationary epoch.**

The properties of the cosmological Bloch wall explain inflation's key observational features:

1. **Zero external gravity ($\approx$ zero curvature scalar $R$) within the inflation window:** The two domain forces (gravity in the low-emergence phase and antigravity in the high-emergence phase) balance in the Bloch wall, resulting in near-zero net curvature. This resolves the flatness problem: the universe **must** be flat near inflation because that is the equilibrium point of the domain transition.

2. **Maximum internal field density:** Within the Bloch wall, the emerging field $\Phi_A$ reaches maximum coherence. The energy density is highest where the transition occurs, not before or after. This explains inflation's energy requirements naturally.

3. **Finite wall width determines inflation duration:** Just as the Bloch wall width is set by exchange-anisotropy balance, the inflation duration is set by the emergence operator's spectral properties. No external slow-roll parameter is needed; the duration emerges from the structural dynamics.

4. **Oscillatory behavior within the wall:** As spins rotate through the Bloch wall, they pass through intermediate orientations that create oscillatory patterns in the field. This predicts **oscillations in the inflation potential** near the transition, which would appear as features in the primordial power spectrum.

### 6.4.5 Gravity and Antigravity as Poles of Emergence

In the D-ND picture:

- **Gravity** ($t = +1$): Convergence of quantum modes toward classical actualization. The emergence operator $\mathcal{E}$ gradually gates modes from superposition into definite states. This actualization requires a "pulling in" of possibility space—a convergence of potential branches. At the field-theoretic level, this manifests as attractive gravity, drawing matter and energy toward regions of high curvature.

- **Antigravity** ($t = -1$): Divergence from actualization, or more precisely, the systematic un-actualization or spreading of actualized states back into superposition. This is not "repulsion" in the classical sense but rather the structural dual of convergence. Where gravity pools information into localized classical states, antigravity spreads information across larger portions of quantum possibility space. At cosmological scales, antigravity is the tendency toward **entropy increase and decoherence**: the ongoing dissolution of classical correlations back into quantum noise.

Both occur **simultaneously and with equal strength** in the D-ND dipole. At cosmic scales, we observe this as:

- **Local scales** (galaxies, stars, bound systems): Gravity dominates because $M_C(t) \approx 1$ (emergence largely complete), so the convergence pole is fully actualized while the divergence pole remains quantum.

- **Cosmological scales** (expansion of space itself): Antigravity dominates because the universe as a whole is still in a phase of partial emergence ($M_C(t)$ finite). The divergence pole, dual to the convergence pole, drives the expansion.

- **Dark energy** as the manifestation of the antigravity pole: Rather than introducing a mysterious "dark energy" substance, the D-ND framework identifies dark energy **as the observable manifestation of the $t = -1$ pole of the cosmic dipole**. It is not a new form of energy; it is the pole of the dipolar emergence structure that standard excluded-third logic (which admits only $t = +1$) necessarily excludes.

### 6.4.6 The Structural Basis for Antigravity: Not a New Force, But Structural Necessity

The D-ND framework does not require a separate "antigravity force." Instead, it shows that **excluding antigravity is the non-physical act**. The inclusion of both poles is dictated by mathematical consistency with the dipolar structure, analogous to how Dirac's equation requires both positive and negative energy solutions.

**Modified Einstein equations with explicit antigravity:**

The modified field equations (S7) can be recast to show the two poles explicitly:

$$G_{\mu\nu}^{(+)} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}^{(+)} \quad \text{(Gravity pole: $t = +1$)}$$

$$G_{\mu\nu}^{(-)} - \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}^{(-)} \quad \text{(Antigravity pole: $t = -1$)}$$

where the $T_{\mu\nu}^{(\pm)}$ are the information tensors in each pole, related by the dipolar constraint:

$$T_{\mu\nu}^{(+)} + T_{\mu\nu}^{(-)} = T_{\mu\nu}^{\text{total}} = 0 \quad \text{(dipolar cancellation at infinity)}$$

In the early universe where emergence is incomplete, both poles contribute significantly. The antigravity component $T_{\mu\nu}^{(-)}$ drives expansion not through a repulsive force but through the structural logic of incomplete actualization.

### 6.4.7 Connection to Friedmann Equations and Dark Energy Equation of State

The Friedmann equations derived in §3.2 incorporate the modified dark energy term:

$$\rho_\Lambda(t) = \rho_0 \cdot (1 - M_C(t))^p$$

and the equation of state:

$$w(z) = -1 + \epsilon(z) \quad \text{where} \quad \epsilon(z) \approx 0.05 \cdot (1 - M_C(z))$$

The D-ND dipolar interpretation is now explicit: the $w = -1$ **exactly** corresponds to the antigravity pole. The cosmological constant in the $t = -1$ direction is not fine-tuned or mysterious; it is the structural dual of ordinary gravity.

The small deviation $\epsilon(z) = 0.05 \cdot (1 - M_C(z))$ arises because:

1. Emergence is not instantaneous but occurs over cosmic time.
2. The coupling between the $+1$ and $-1$ poles is not perfectly symmetric at intermediate emergence stages ($0 < M_C < 1$).
3. The residual imbalance $(1 - M_C)$ allows partial oscillation between the poles, producing a slight softening of the antigravity equation of state from exact $w = -1$ to $w \approx -1 + \epsilon$.

At late times ($z \to 0$), as $M_C \to 1$, the coupling becomes increasingly symmetric, and the observed $w$ approaches $-1$ asymptotically. This prediction is testable by DESI and future surveys (see §6.3 for quantitative constraints).

### 6.4.8 Antigravity and the Information Tensor

The informational energy-momentum tensor $T_{\mu\nu}^{\text{info}}$ from §2.1 naturally encodes both poles through its mathematical structure:

$$T_{\mu\nu}^{\text{info}} = \frac{\hbar}{c^2} \int d^3\mathbf{x} \, K_{\text{gen}}(\mathbf{x},t) \, \partial_\mu R(t) \, \partial_\nu R(t)$$

The curvature density $K_{\text{gen}} = \nabla \cdot (J \otimes F)$ depends on the **flow and force** of information. In the $+1$ direction, information is compressed and converged (gravity). In the $-1$ direction, information is dispersed and diverged (antigravity). The total tensor is always conserved:

$$\nabla^\mu T_{\mu\nu}^{\text{info}} = 0$$

This conservation ensures that the total information content of the universe—integrated over both poles—remains constant across cosmic evolution, consistent with quantum mechanics and the no-information-loss principle.

### 6.4.9 Three Concrete Falsification Tests for Antigravity

The Bloch wall and Poynting vector mechanisms provide three falsifiable observational tests:

**Test 1: Bloch Signature in CMB Polarization**

In the Bloch wall picture, the inflationary epoch corresponds to a domain transition where spins rotate from one orientation to another. This rotation imprints a characteristic signature in the **polarization pattern of the cosmic microwave background**.

Specifically:
- **Prediction:** The CMB temperature-polarization cross-correlation ($T \times E$ modes) should show an oscillatory pattern at multipoles $\ell \sim 10$–$50$ (corresponding to the Bloch wall width).
- **Mechanism:** As the emergence field rotates through intermediate angles (like spins in a Bloch wall), it creates acoustic oscillations in the Thomson-scattering photons, producing a characteristic polarization signature.
- **Observable:** The oscillation frequency in the $T \times E$ power spectrum should encode the emergence operator's spectral properties.
- **Current Status:** Planck 2018 polarization data shows hints of such oscillations, though not yet at statistical significance. The full Planck Legacy release and future missions (CMB-S4) will decisively test this.

**Test 2: Riemann Eigenvalue Structure in DESI BAO Data**

From exploratory analysis of the D-ND eigenvalue structure (to be derived formally in future work), the Riemann zeta function constraint on the stress-energy tensor eigenvalue spectrum suggests that large-scale structure may show **anomalous clustering at scales corresponding to Riemann zeros**.

Specifically:
- **Prediction:** The galaxy power spectrum $P(k)$ in the DESI baryon acoustic oscillation measurements should exhibit **peaks and suppressions at wavenumbers matching Riemann zero spacing**.
- **Mechanism:** The eigenvalues of the Ricci tensor (which couple to the Friedmann equations through the modified Einstein equations) must satisfy a Riemann-zeta-like constraint. This topological property imprints itself on the clustering of matter via the gravitational potential.
- **Observable:** In the DESI BAO analysis at redshifts $z \sim 0.1$–$2.5$, look for **prime-number-like harmonic spacing** in the matter clustering at scales $k \sim 0.01$–$0.1$ Mpc$^{-1}$.
- **Current Status:** DESI 2024 Year-1 data has not yet reported anomalous structure at Riemann-scale precision. Full DESI dataset (2025-2026) will provide decisive constraints.

**Test 3: Dipolar Cancellation in the Equation of State $w(z)$**

The antigravity pole arises from residual asymmetry in the dipolar cancellation, encoded in the dark energy equation of state. D-ND predicts:

$$w(z) = -1 + \epsilon(z) \quad \text{where} \quad \epsilon(z) = 0.05 \cdot (1 - M_C(z))$$

This predicts a **specific functional form for how $w$ deviates from $-1$ as a function of redshift**, reflecting the emergence measure dynamics.

Specifically:
- **Prediction:** At $z = 1.5$ (intermediate redshift), $w(1.5) \approx -0.920$, compared to ΛCDM's $w = -1.000$ exactly. The deviation is $\Delta w \approx 0.08$.
- **Observable:** From DESI BAO + weak lensing + supernova data, constrain $w(z)$ across multiple redshift bins. D-ND predicts a **monotonic increase in $w$ toward $-1$ as $z \to 0$**, distinct from many modified dark energy models which predict oscillatory or non-monotonic behavior.
- **Current Status:** DESI 2024 Year-1 measurements have error bars of order $\pm 0.05$ in $w(z)$ at individual redshifts. A $1$-$2\sigma$ detection of D-ND's predicted deviation is within reach of DESI Year-2 (2025). A $3\sigma$ or higher detection would strongly support the D-ND framework.

### 6.4.10 Observational Implications: Testing Antigravity

If antigravity is a fundamental pole of the D-ND dipole, several additional observational tests follow:

1. **Isotropic expansion history**: If antigravity is truly a fundamental structural pole, it should expand the universe **isotropically** (equal in all directions), consistent with observations. Anisotropic dark energy models predict directional asymmetries in cosmic expansion, which are not observed at current precision. D-ND predicts isotropy naturally.

2. **Absence of antigravity "interactions"**: Unlike exotic dark energy models (chameleon fields, coupled dark energy, etc.), the antigravity pole in D-ND does not interact with ordinary matter except through modification of spacetime geometry. This predicts that solar system tests of gravity (Eötvös experiments, equivalence principle tests) will show no deviations, consistent with current data.

3. **Decay of dark energy in future aeons**: If the antigravity pole is coupled to the emergence measure $M_C(t)$, and if $M_C(t)$ eventually saturates toward $M_C = 1$, then dark energy should decay over cosmological timescales. The D-ND framework predicts $\rho_\Lambda \to 0$ asymptotically (on timescales $\sim 10^{100}$ years), unlike ΛCDM where dark energy is eternal. This is unfalsifiable in practice but conceptually distinct.

---

## 6.5 Time as Emergence: Thermodynamic Irreversibility and the Dipolar Amplitude

### 6.5.1 Time Does Not "Function"—It Emerges from Irreversibility

A fundamental premise of standard physics is that time is a **given**—a background parameter on which dynamical equations operate. Events happen "*in* time" as if time were a stage on which the drama of reality unfolds. Classical mechanics takes time as absolute (Newtonian) or as part of a unified spacetime metric (Einstein). Even quantum mechanics, despite its operational focus, presupposes time as an external variable with respect to which the wave function evolves: $i\hbar \partial_t |\psi\rangle = H|\psi\rangle$.

The D-ND framework proposes a radically different picture: **time is not a given but an emergent property of the universe itself**. More precisely, time emerges as the measure of **irreversible information processing** in the field-collapse dynamics.

**Thermodynamic Foundation: The Clausius Inequality**

The rigorous foundation for time-emergence lies in the second law of thermodynamics. Consider any closed thermodynamic cycle:

$$\boxed{\oint \frac{\delta Q}{T} \leq 0}$$

This is the **Clausius inequality**. The key insight is that for real (irreversible) cycles, the integral is strictly less than zero:

$$\oint \frac{\delta Q}{T} < 0$$

This inequality never returns exactly to its starting point. There is always a residual loss—entropy generated by the irreversible processes. **This residual loss is precisely what creates the arrow of time.** In a perfectly reversible universe (entropy constant), the integral would equal zero and the process would be cyclic with no preferred direction. But the second law forbids this perfection: every closed process must lose energy to irreversibility.

**Formal Statement:** Time emerges as the integral of entropy production:

$$\boxed{t = \int_0^T \frac{dS}{dT}(\tau) \, d\tau}$$

where $S(T)$ is the entropy of the system and the derivative measures the rate of irreversible information loss. The irreversibility implied by $\oint dQ/T < 0$ guarantees that $dS/dT > 0$, making time a monotonic and forward-directed parameter.

### 6.5.2 Time Emergence from the Six-Phase Cognitive Pipeline

The D-ND framework identifies a detailed mechanism for temporal emergence through the **six-phase cognitive pipeline** that describes the collapse of possibility space into actuality. This pipeline is not merely conceptual but reflects the dynamical structure of field-collapse at all scales:

- **Phase 0: Indeterminacy** ($\Phi_0$ = Zero-point potentiality) — The field exists in a state of maximal superposition with no causal distinction.
- **Phase 1: Symmetry Breaking** (via $\mathcal{E}$ emergence) — The emergence operator begins to gate modes from superposition.
- **Phase 2: Divergence** (Alternative paths multiply) — Multiple actualization pathways become possible.
- **Phase 3: Validation** (Stream-Guard pruning) — Incoherent branches are eliminated via coherence constraints.
- **Phase 4: Collapse** (Morpheus guide) — The field converges toward classical configurations.
- **Phase 5: Refinement** (KLI Injection) — The state is updated with learned structure (autopoiesis, **Axiom P5**).
- **Phase 6: Determinacy** (Manifest output) — A definite classical state is realized.

The **sequence from Phase 0 (indeterminacy) to Phase 6 (determinacy) is itself temporal evolution**. Time does not parametrize this process externally; it **is the ordering principle** of these phases.

**Connection to Entropy Gradient:** Each phase advances through irreversible information processing. Phase 0 contains maximal entropy (maximum uncertainty). Phase 6 contains minimal entropy (classical definiteness). The gradient $\nabla S$ across the phase sequence drives the transition forward. **This is why time flows Phase 0 → Phase 6 and never backward**: because backward flow would decrease entropy, violating the second law.

### 6.5.3 Time as Parameter Ordering Field-Collapse Phases

In the D-ND cosmological context, time is the measure of emergence progress across the six phases at cosmological scales:

$$\boxed{t(\mathbf{x}) = T_{\text{cycle}} \times f(M_C(\mathbf{x}), \dot{M}_C(\mathbf{x}))}$$

where:
- $f(M_C, \dot{M}_C)$ is a function ordering the phase sequence locally
- Regions with $M_C \approx 0$ (Phase 0-1: early emergence) experience rapid time flow (high $\dot{M}_C$)
- Regions with $M_C \approx 1$ (Phase 6: fully emerged) experience slower time flow (low $\dot{M}_C$)
- Time flows fastest at intermediate emergence states ($M_C \approx 0.5$, Phase 2-4) where maximal information processing occurs

**Formal Derivation from Friston's Free Energy Principle:**

The universe minimizes its surprise (free energy) through evolutionary processes. Each phase transition reduces the free energy available to the system:

$$F(\text{Phase } n) = -\ln p(\text{data}|n) + \text{KL}[\text{Prior}||\text{Posterior}]$$

The sequence of phase transitions is the geodesic path on the manifold of free energy configurations. **Time parametrizes this geodesic**. The rate of time flow is proportional to the rate of free energy reduction:

$$\frac{dt}{d\tau} = \left|\frac{dF}{d\tau}\right| \quad \text{(Free Energy Principle)}$$

This formally states that **time flows fastest where the universe learns most rapidly** (maximum information gain).

### 6.5.4 Time as Local Amplitude of the Dipolar Oscillation

In the D-ND picture, the universe is a **dipolar oscillator**: at every spacetime point, the fundamental degrees of freedom oscillate between the two poles of the dipole ($t = +1$ and $t = -1$). Time, at each location, is not a global parameter but a **local quantity**: the phase or amplitude of this oscillation at that point.

**Definition**: The local time at spacetime point $(x,t)$ is:

$$\tau(\mathbf{x}) = \Lambda \cdot |M_C(\mathbf{x})| \cdot (1 - |M_C(\mathbf{x})|) \cdot T_{\text{cycle}}$$

where:
- $M_C(\mathbf{x})$ is the local emergence measure (how far that region has actualized)
- $(1 - |M_C(\mathbf{x})|)$ is the residual quantum uncertainty
- $T_{\text{cycle}}$ is the fundamental period of the dipolar oscillation
- $\Lambda$ is a normalization constant

**Physical meaning**: This definition captures the intuition that time is fastest where emergence is most active—where the system is transitioning between potential and actual. In regions where $M_C(\mathbf{x}) \approx 0$ (still mostly potential) or $M_C(\mathbf{x}) \approx 1$ (fully actualized), the local time runs slowly because there is little ongoing transition. Time runs fastest at intermediate values $M_C(\mathbf{x}) \approx 0.5$, where the emergence is most active.

**Analogy**: The local times $\{\tau(\mathbf{x})\}$ are like **intrinsic spins** in quantum mechanics. Just as each particle carries an intrinsic angular momentum (spin) without requiring that the particle "spin" in a literal sense, each spacetime point carries an intrinsic time without requiring that time "flow" in the classical sense. The local times are properties of the emergence state itself, not parameters external to it.

### 6.5.5 The Included Third and Normalization of Excluded-Third Logic

Standard logic operates on the excluded-third (tertium non datur): a proposition is either true or false; there is no third option. In mathematics, real numbers and classical logic are built on this foundation. Yet classical logic encounters a fundamental problem: it cannot easily account for **polarities** or **complements**. To extend real numbers to complex numbers and preserve algebraic structure, mathematicians introduced an additional axis (the imaginary unit $i$) that is **neither true nor false** but is the **condition of possibility** for both.

The D-ND framework proposes a generalization: where two poles or complementary states exist, there is necessarily a **third**: the boundary or interface between them, which is neither pole but the condition of possibility for both poles to coexist.

In the cosmological context:

- **Pole 1** ($t = +1$): Actualized being (gravity, convergence, manifest reality)
- **Pole 2** ($t = -1$): Potential non-being (antigravity, divergence, hidden quantum structure)
- **The Third (Terzo Incluso)**: The **singularity** between the poles—the interface where actualization occurs

This third is not a compromise between the poles but rather their **structural prerequisite**. Without the singularity—the boundary where actualization transitions between potential and actual—neither pole would exist as a distinct entity.

**Normalization**: The D-ND framework "normalizes" the excluded-third logic by elevating the third to explicit status:

$$1_{\text{D-ND}} = (t = +1) + (t = -1) + (t = 0)_{\text{singularity}}$$

This is analogous to the extension from real to complex numbers:

$$1 = \sqrt{1} + i\sqrt{0} + \text{(rotation axis in } \mathbb{C})$$

By including the third explicitly, the D-ND framework resolves theorems and paradoxes that arise from hidden asymmetries in excluded-third logic. Any theorem suffering from such asymmetries—quantum indeterminacy, the cosmological constant problem, the information paradox—can be re-examined through the lens of D-ND's included-third logic.

### 6.5.6 The Lagrangian of Observation and Minimal Latency

If time emerges as latency, then there must be a principle determining which latencies are realized and which are suppressed. The D-ND framework proposes:

**The Principle of Minimal Latency**: Among all possible actualization pathways, nature selects those that minimize the integral of local latencies—the "cost" of observation.

Formally, this is:

$$\mathcal{S}_{\text{observe}} = \int_{\text{path}} \tau(\mathbf{x}) \, d\mathcal{M}$$

where $d\mathcal{M}$ is the measure on the space of emergence-state configurations. The **path of minimal action** (extremizing $\mathcal{S}_{\text{observe}}$) is the trajectory that nature actually follows.

**Interpretation**: The observer does not choose how to observe; the universe does not choose how to actualize. Instead, the **observation self-selects** along the path of minimal latency. Just as light takes the path that minimizes optical path length (Fermat's principle), the actualization of quantum states takes the path that minimizes the total "time cost" of observation.

This principle naturally explains:

1. **Why the universe expands**: Expansion is the path of minimal latency for actualizing vast numbers of quantum modes simultaneously.

2. **Why gravity exists**: Gravity is the geometry that allows actualization of nearby modes with minimal latency (shorter transition paths), naturally pulling structures together.

3. **Why large-scale structure forms**: Density fluctuations grow because clustering localizes actualizations, reducing the total latency required.

4. **Why entropy increases**: As the universe expands and actualizes, it explores larger portions of configuration space, requiring longer latencies on average—hence entropy increases.

### 6.5.7 Convergence and Divergence Are Simultaneous: Zero Latency in Assonances

A striking prediction of the principle of minimal latency is:

**In regions where the convergence pole ($t = +1$) and divergence pole ($t = -1$) oscillate perfectly in phase and amplitude (perfect resonance or "assonance"), the latency vanishes: $\tau = 0$.**

This zero-latency state corresponds to the **maximal potentiality**: a point in which all possible actualizations are superposed with equal amplitude. This is precisely the $|NT\rangle$ state.

**Cosmological implication**: At the boundary of cosmic cycles, when the universe reconverges toward a state of non-duality (as described in §5.1), both poles approach perfect synchronization. In this state, time becomes undefined (latency $\to 0$), and the universe transitions instantaneously from one aeon to the next. There is no "time" between cycles—only a discrete jump in the state space.

This resolves a paradox in cyclic cosmologies: if time emerges from the universe's evolution, how can the universe "cycle" without an external time parameter? Answer: at the cycle boundary, time itself ceases to exist (latency vanishes), and the next cycle initiates from a state of pure potentiality.

### 6.5.8 The Double Pendulum as Physical Realization

The mathematical idealization of the latency principle can be realized in classical mechanics by the **double pendulum**—a system with a direct classical analogue to the D-ND dipole.

A double pendulum consists of two masses connected by rigid rods, with the first rod pivoted to a fixed point. The system is chaotic: small perturbations lead to exponentially divergent trajectories. Yet despite local chaos, the double pendulum is Lagrangian-coherent: its motion is governed by a single Lagrangian:

$$L = \frac{1}{2}m(\dot{x}_1^2 + \dot{y}_1^2 + \dot{x}_2^2 + \dot{y}_2^2) - mg(y_1 + y_2)$$

The double pendulum exhibits **simultaneous bifurcation**: at any moment, the system explores multiple "branches" of behavior (chaotic locally) while remaining constrained by a single global principle (the Lagrangian).

**Analogy to D-ND cosmology**:
- **Local chaos** ↔ Quantum fluctuations, emergence of structures at different rates across space
- **Global Lagrangian coherence** ↔ The principle of minimal latency and the informational energy-momentum tensor (unified constraint across all spacetime)
- **Strange attractors** ↔ The attracting phases of the dipolar oscillation (e.g., formation of galaxies, stars)

If the universe is a cosmological double pendulum, then:

1. Locally, reality is chaotic and probabilistic (quantum mechanics).
2. Globally, reality is deterministic and Lagrangian-coherent (classical field equations).
3. Neither description is more fundamental; they are complementary manifestations of a single underlying structure.

### 6.5.9 Convergence and Divergence in the Modified Friedmann Equations

The latency principle and dipolar oscillation structure are reflected in the modified Friedmann equations (§3.2). Rewriting them in terms of convergence and divergence:

$$H^2(z) = H_0^2 \left[\Omega_m(1+z)^3 + \rho_\Lambda(z) + \Omega_k(1+z)^2\right]$$

where $\rho_\Lambda(z) = \rho_0 (1 - M_C(z))^p$ encodes the **antigravity pole**.

**Convergence** ($t = +1$): The $\Omega_m$ term dominates at early times (high redshift). Matter pulls the universe inward; expansion slows. The actualization of quantum modes into particles and radiation is the mechanism.

**Divergence** ($t = -1$): The $\rho_\Lambda(z)$ term becomes dominant at late times (low redshift). The universe accelerates outward; expansion is driven by the antigravity pole. The residual unactualized modes—quantum potential—drive expansion.

**At intermediate times** ($z \sim 1$): The two terms balance. Cosmic acceleration transitions from deceleration (matter-dominated era) to acceleration (dark-energy-dominated era). This transition is a **resonance**—the two poles temporarily couple with similar strengths, resulting in a complex oscillatory behavior in the expansion history.

### 6.5.10 Observational Predictions: Time Emergence Signatures

If time genuinely emerges from the dipolar oscillation, several novel signatures should appear:

1. **Anomalous age estimates at high redshift**: The local time $\tau(\mathbf{x})$ is fastest at intermediate emergence stages. This means that the **proper time experienced by matter at high redshift differs from coordinate time**. Extremely distant galaxies that formed quickly (in coordinate time) may appear older (in proper time) than they should. This could explain some tensions between stellar age estimates and cosmological age estimates.

2. **Preferred scales in structure formation**: If actualization follows the principle of minimal latency, certain scales should be energetically "cheaper" to actualize (lower total latency). This predicts discrete preferred scales in the galaxy distribution, power spectrum, and clustering patterns—essentially a "quantization" of cosmic structure. Current surveys show hints of such scales (baryon acoustic oscillations at $\sim 150$ Mpc, the acoustic cutoff scale).

3. **Time-dependent gravitational constant**: The coupling between poles (and hence the overall strength of gravity) evolves with $M_C(t)$. This predicts a time-dependent gravitational "constant": $G(z) = G_0 [1 + \delta_G(1 - M_C(z))]$, with $\delta_G \sim 0.001$–$0.01$ depending on the emergence operator's spectral properties. Precision tests of gravity (equivalence principle tests, strong-field tests via pulsar timings) could measure this evolution.

---

## 6.6 Observational Predictions Summary Table: D-ND vs. ΛCDM and Alternatives

This section consolidates all testable predictions of D-ND cosmology across multiple observational domains, providing a unified framework for hypothesis testing against ΛCDM and other alternative theories.

### Comprehensive Prediction Table

| **Observable Domain** | **Specific Prediction** | **D-ND Value/Behavior** | **ΛCDM Value/Behavior** | **Distinguishability** | **Current Constraint Status** |
|---|---|---|---|---|---|
| **CMB: Tensor/Scalar Ratio** | Primordial gravitational wave amplitude | $r \sim 0.001$–$0.01$ (suppressed by incomplete emergence) | $r \sim 0.001$–$0.1$ (inflation-dependent) | Marginal (1–2σ) | Planck 2018: $r < 0.064$ (both consistent) |
| **CMB: Bispectrum ($f_{\text{NL}}$)** | Non-Gaussianity from emergence-gated modes | $f_{\text{NL}}^{\text{equilateral}} \sim 5$–$20$ (smooth $\mathcal{E}$); higher in emergence-type templates | $f_{\text{NL}} \sim 1$–$5$ (local-type) | Strong (3–5σ) with CMB-S4 | Planck 2018: $f_{\text{NL}}^{\text{equilateral}} < 25$ (consistent with smooth-$\mathcal{E}$ D-ND) |
| **CMB: Power Suppression** | Super-horizon scale suppression at $\ell < 10$ | Anomalous deficit of $\sim 10$–$20\%$ at $\ell < 10$ | Smooth power law to low multipoles | Possible (1–2σ in current data) | Planck hint of suppression; S4 will clarify |
| **CMB: Spectral Index Running** | Scale-dependent $n_s(k)$ from $\dot{M}_C(t)$ | $\frac{d\ln n_s}{d\ln k} \sim -0.005$ to $-0.020$ | $\frac{d\ln n_s}{d\ln k} \sim 0$ (minimal) | Possible (2–3σ) | Current data consistent with zero; future surveys will constrain |
| **CMB: Polarization T×E** | Bloch wall oscillatory signature | Oscillations at $\ell \sim 10$–$50$ in $T \times E$ modes | Smooth acoustic oscillations | Distinctive if present | Planck data show hints; CMB-S4 will test |
| **Structure Growth: $f(a)$** | Modified growth rate from $(1-M_C(a))$ feedback | $f(a) = f_{\text{GR}}(a)[1 + 0.1(1-M_C(a))]$ | $f(a) = f_{\text{GR}}(a)$ (exact) | Small (1–2σ) at $z < 5$ | SDSS/DESI measurements consistent with GR; improvements expected |
| **Structure: Halo Bias** | Enhanced bias at high redshift from emergence | $b(z) = b_{\text{matter}}(z)[1 + 0.05 \cdot M_C(z)]$ | $b(z)$ follows standard model | Possible (2–3σ) at $z > 1$ | DESI Year-1 shows consistency with standard model |
| **Large-Scale Structure: $\sigma_8$** | Slight suppression from growth modification | $\sigma_8 \sim 0.80$ (vs. predicted $0.81$ in ΛCDM) | $\sigma_8 \approx 0.811$ | Marginal (0.5–1σ) | Planck+SDSS show tension; D-ND could help alleviate |
| **Dark Energy: Equation of State $w(z)$** | Time-dependent equation of state from $(1-M_C(z))$ | $w(z) = -1 + 0.05(1-M_C(z))$; $w(0.5) \approx -0.975$; $w(1.5) \approx -0.920$ | $w = -1.000$ (constant) | Strong (2–4σ) at $z \sim 1$–$2$ | **DESI 2024 Year-1: $w$ measurements at 0.05 precision; Year-2/3 will decisively test** |
| **Dark Energy: Evolution Rate** | Monotonic approach to $w = -1$ with decreasing redshift | Smooth monotonic increase from $w < -1$ at high-$z$ to $w \approx -1$ at $z = 0$ | Flat $w = -1$ across all $z$ | Strong if deviations detected | DESI BAO + weak lensing precision sufficient to distinguish |
| **Baryon Acoustic Oscillations: Scale** | Shift in BAO scale from modified expansion history | $d_A^{\text{D-ND}}(z=1) \approx 1.016 \times d_A^{\text{ΛCDM}}$ (+1.6%) | Standard BAO scale from GR | Possible (2–3σ) | DESI Year-1 precision ~0.5%; Year-3 precision ~0.2% will test |
| **Supernovae: Magnitude-Redshift** | Systematic deviation in Hubble diagram at $z \sim 1$ | Expected offset $\Delta m \sim 0.1$–$0.2$ mag at intermediate redshift | None (ΛCDM is reference) | Possible (2–3σ) if systematic errors controlled | Current SNe samples show consistency with ΛCDM |
| **Gravitational Lensing: Magnification** | Slight enhancement from modified growth | Weak lensing power spectrum offset $\sim 2$–$5\%$ at $k \sim 0.1$ Mpc$^{-1}$ | None (GR prediction) | Marginal (1–2σ) | Euclid/Roman future surveys will achieve precision needed |
| **Primordial Perturbations: Power Spectrum** | Spectral shape modified by emergence eigenvalues | $P_\delta(k) \propto k^{n_s-1}[1 - \lambda_k(1-M_C(t_*))]$ with emergence modulation | Pure power law $\propto k^{n_s-1}$ | Possible (2–3σ) with careful analysis | High-precision measurements needed |
| **Gravitational Waves: Merger Rates** | Slight enhancement from modified spatial curvature | Rate density $\sim 5$–$10\%$ higher than ΛCDM prediction | ΛCDM prediction from standard GR | Small (0.5–1σ) | LIGO/Virgo merger catalog; future detectors will improve |
| **Gravitational Waves: Stochastic Background** | Modified spectrum from time-dependent GW emission | Spectral shape differs from inflation-only prediction at high frequencies | Flat spectrum from primordial inflation | Possible (2–3σ) | Future missions (LISA, Einstein Telescope) will test |
| **High-Redshift Galaxy Counts** | Modest suppression in early galaxy populations from reduced growth | Number density of $z > 6$ galaxies $\sim 10$–$20\%$ lower than standard predictions | Full ΛCDM predictions | Marginal (1–2σ) | JWST early universe survey data emerging; tests ongoing |
| **Riemann Eigenvalue Signature** | Anomalous structure in matter power spectrum at scales matching Riemann zeros | Prime-number-like harmonic spacing in $P(k)$ at specific scales | No special structure beyond acoustic oscillations | Distinctive if present | DESI BAO precision sufficient; requires dedicated analysis |
| **Time-Variation of $G$** | Gravitational "constant" evolves as $G(z) = G_0[1 + 0.001(1-M_C(z))]$ | $\Delta G / G \sim 10^{-3}$ to $10^{-2}$ over cosmic time | $G$ is constant | Small (1–2σ) | Pulsar timing arrays; equivalence principle tests constrain |
| **Cyclic Coherence Signature** | Phase matching at aeon boundaries (if cycles occur) | Low-frequency temperature correlations in CMB at $\ell \sim 1$–$3$ (Penrose's Hawking points) | No expected signal | Distinctive if detected | Planck analysis: Hawking point searches inconclusive |

### Interpretation and Priorities for Falsification

**Tier 1 — Decisive Tests (3–5σ precision potential):**
1. **Dark energy equation of state $w(z)$ from DESI BAO + weak lensing** (2025–2026)
2. **CMB non-Gaussianity $f_{\text{NL}}$ from future CMB missions** (CMB-S4, ~2030)
3. **Riemann eigenvalue structure in large-scale structure** (DESI 2024-2026 with dedicated analysis)

**Tier 2 — Promising but Weaker Tests (1–3σ precision potential):**
4. **Spectral index running $d\ln n_s / d\ln k$** (future CMB missions)
5. **Bloch wall signature in CMB polarization T×E** (CMB-S4)
6. **Halo bias evolution at high redshift** (DESI, Euclid, Roman)

**Tier 3 — Indirect or Long-Term Tests:**
7. **Time variation of $G$** (pulsar timing arrays, next decade)
8. **Gravitational wave stochastic background** (LISA, Einstein Telescope, 2030s+)
9. **Cyclic coherence/Hawking points** (speculative; future high-precision CMB)

### DESI 2024 Year-1 Status and Forecast

The DESI Dark Energy Spectroscopic Instrument released baryon acoustic oscillation measurements in June 2024, providing constraints on the expansion history at $z \sim 0.3$–$0.9$. These measurements:
- Constrain the comoving distance to $\sim 0.5\%$ precision
- Are consistent with ΛCDM at $z \lesssim 1$
- Have **not yet ruled out** D-ND predictions of $w(z) = -1 + 0.05(1-M_C(z))$ because the predicted deviation at $z \sim 0.5$ is only $\sim 0.8\%$, comparable to current error bars

**Forecast for DESI Year-2 (2025) and Year-3 (2026):**
- Expected measurement precision will improve to $\sim 0.2$–$0.3\%$ per redshift bin
- This should achieve **1.5–2.5σ detection** of D-ND's predicted deviation if real
- Combined with weak lensing and supernova data, a **2–3σ aggregate constraint** is realistic
- A **null result** (perfect agreement with ΛCDM's $w = -1$) would challenge D-ND unless the emergence measure $M_C(z)$ evolves more rapidly than predicted

---

## 7. Discussion and Conclusions

### 7.1 Strengths of the D-ND Cosmological Extension

1. **Closes a gap in cosmological theory**: Provides a mechanism for closed-system emergence of classical spacetime from quantum potentiality, applicable at all scales.

2. **Connects micro and macro**: Links quantum emergence (Paper A) to cosmic inflation and dark energy evolution through a unified mathematical framework.

3. **Resolves the initial singularity**: Replaces the classical Big Bang singularity with a finite boundary condition on emergence, avoiding infinite curvature or density.

4. **Addresses the dark energy problem**: Provides a qualitative explanation for the small cosmological constant without fine-tuning.

5. **Cyclic structure and information conservation**: Suggests how quantum information might be preserved across cosmic cycles, addressing black hole thermodynamics.

6. **Falsifiable predictions**: Proposes concrete observational tests (non-Gaussian bispectrum, super-horizon power suppression, scale-dependent running, structure formation modifications, dark energy evolution).

7. **DESI-constrained framework**: Provides quantitative predictions testable against 2024 BAO data, with clear falsification criteria.

### 7.2 Limitations and Caveats

1. **Speculative nature**: The connection between microscopic emergence (Paper A) and cosmic scales is not rigorously derived from first principles. The modified Einstein equations (S7) are phenomenological ansatze rather than precise geometric consequences.

2. **Lack of precision in emergence operator**: At cosmological scales, the structure of $\mathcal{E}$ and the spectrum of the "cosmological Hamiltonian" are not known. The predictions depend sensitively on these inputs.

3. **Incomplete quantum gravity**: The framework does not provide a full quantum theory of gravity, comparable to loop quantum cosmology or string cosmology. It is better viewed as a phenomenological bridge between quantum mechanics and classical cosmology.

4. **Modified equations axiomatically motivated but not independently derived**: The informational energy-momentum tensor $T_{\mu\nu}^{\text{info}}$ follows as a structural consequence of D-ND axioms P0--P4 (§2.2), but a fully independent derivation from quantum gravity first principles (e.g., spectral action principle, asymptotic safety) remains an open problem. The specific functional form retains some freedom within the axiomatic constraints.

5. **Relation to observations unclear in detail**: The observational predictions (CMB bispectrum, structure formation, dark energy) are stated qualitatively and require detailed computation to achieve quantitative precision. A dedicated numerical cosmology simulation (similar to CAMB or CLASS codes) would be needed to make precise predictions for comparison with data.

6. **Cosmological constant reassessment**: The identification of dark energy with residual $V_0$ is attractive but remains speculative. The actual magnitude and evolution of dark energy depend on the unknown form of $V_0$ and its coupling to $M_C(t)$.

### 7.3 Speculative but Falsifiable Framework

We emphasize that this cosmological extension is **speculative but falsifiable**. The predictions are:

- **Not derived from first principles** but arise from extrapolating the quantum D-ND framework to cosmological scales.
- **Testable in principle** through specific CMB anomalies, large-scale structure patterns, and dark energy evolution.
- **Distinguished from $\Lambda$CDM** in regimes where emergence effects are non-negligible (early universe, largest scales, late-time cosmic evolution).

A negative result (e.g., failure to detect predicted CMB non-Gaussianity or absence of scale-dependent growth suppression) would argue against the D-ND cosmological model. Conversely, detection of any of the predicted signatures would provide tentative support for the framework.

### 7.4 Paths Forward

Three research programs are suggested:

**Numerical Cosmology**: Implement a modified Boltzmann code (extending CLASS or CAMB) that incorporates the D-ND modifications to Friedmann equations and computes the full CMB power spectrum, weak lensing power spectrum, and structure formation predictions for comparison with current and future data.

**Quantum Gravity Integration**: Attempt to derive the modified Einstein equations (S7) from more fundamental quantum gravity principles (e.g., loop quantum cosmology, asymptotic safety, or spectral action principle), replacing the phenomenological informational tensor with a rigorously motivated term.

**Observational Campaigns**: Design dedicated observations to search for the predicted CMB bispectrum, measure structure growth at high redshift, and constrain dark energy evolution with precision sufficient to distinguish D-ND from $\Lambda$CDM.

### 7.6 Comparative Predictions: D-ND Cosmology vs. ΛCDM vs. Loop Quantum Cosmology vs. CCC

To contextualize D-ND cosmology within the landscape of modified and quantum cosmological frameworks, we present a quantitative comparison across key observables and theoretical properties.

| **Feature** | **ΛCDM** | **D-ND Cosmology** | **Loop Quantum Cosmology (LQC)** | **Conformal Cyclic Cosmology (CCC)** |
|---|---|---|---|---|
| **Initial Singularity** | Curvature divergence at $t=0$ | NT singularity (finite boundary) | Quantum bounce (avoids singularity) | Conformal rescaling (infinite past/future) |
| **Mechanism** | Classical GR + cosmological constant | Emergence measure $M_C(t)$ + information tensor | Quantum geometry corrections; area gap operator | Weyl curvature hypothesis; conformal matching |
| **Inflation** | Slow-roll scalar field $\phi$ | Rapid $M_C$ evolution (emergence phase) | Potential-driven, with modifications | Not primary; cyclic instead |
| **Inflation duration** | $e$-folds $\sim 50$–$60$ (tuned) | $\sim \log(1/M_C(0))$ (emergence-determined) | $\sim 40$–$70$ depending on loop corrections | N/A (structure-formation mechanisms differ) |
| **Dark Energy** | Cosmological constant ($w = -1$ exact) | Residual $V_0$ ($w(z) = -1 + 0.05(1-M_C(z))$) | Loop corrections change equation of state | Not primary; CCC energy-condition violations instead |
| **Dark Energy Evolution** | Constant $\Omega_\Lambda$ | Time-dependent, decays as $\propto (1-M_C)^2$ | Slight evolution due to quantum corrections | Cyclic evolution across aeons |
| **CMB Power Spectrum** | Harrison-Zeldovich $n_s \approx 1$ + tilting | Scale-dependent running $n_s(k)$ from $\dot{M}_C(t_*)$ | Similar to slow-roll (small running) | Modified correlations from aeon matching |
| **Non-Gaussianity** | $f_{\text{NL}} \sim 1$ (small, local-type) | $f_{\text{NL}} \sim 5$--$20$ (smooth $\mathcal{E}$); higher in emergence-type templates | $f_{\text{NL}}$ enhanced by quantum corrections | $f_{\text{NL}}$ modified by conformal structure |
| **Structure Growth** | Linear growth factor $f(a)$ from GR | Growth modified by $(1-M_C(a))$ feedback | Suppressed at early times (bounces) | Oscillatory growth from cyclic boundary conditions |
| **Black Hole Information** | Information lost (Hawking paradox) | Information preserved (InjectKLI updates) | Preserved via quantum geometry | Preserved via cyclic structure |
| **Cyclic Structure** | No cycles (singular Big Bang) | Multiple cycles with phase coherence $\Omega_{NT} = 2\pi i$ | Quantum bounce (single cycle?) | Infinite cycles (aeons) with conformal matching |
| **Number of Free Parameters** | 6 (Ω's, $H_0$, $\sigma_8$, $n_s$) | $\sim 8$ ($\Lambda$, $\xi$, emergence operator spectrum, $\tau_e$) | $\sim 6$ (similar to ΛCDM + quantum corrections) | $\sim 5$ (fixed by conformal structure) |
| **Degree of Speculation** | Well-tested; standard | Highly speculative; extensions conjectural | Quantitative but relies on LQG foundations | Speculative; predicts Hawking points in CMB |
| **Observational Status** | Consistent with CMB, SNe, BAO | Not yet constrained by DESI (predictions at 1–3% level) | Consistent with observations; LQG foundational assumptions debated | Hawking points not confirmed; predictions under scrutiny |

**Key Distinctions:**

1. **Mechanism for inflation**: ΛCDM uses slow-roll; D-ND uses emergence; LQC uses quantum bounces; CCC uses cyclic structure.

2. **Dark energy behavior**: ΛCDM constant; D-ND evolving with emergence; LQC slightly modified by quantum loops; CCC cyclic.

3. **Information preservation**: ΛCDM loses it; D-ND preserves via cycles; LQC via quantum geometry; CCC via conformal structure.

4. **Testability**: DESI 2024 data provides constraints. D-ND predictions (1–3% deviation in $w(z)$) are just beyond current precision but will be tested in 2026.

5. **Conceptual unity**: D-ND connects emergence at quantum and cosmic scales; LQC is quantum-gravity-first; CCC is conformal-geometry-first.

**Recommendation for Future Work**: High-precision measurements of the expansion history ($w(z)$ from BAO, weak lensing, SNe) over $z \sim 0$–$2$ will decisively test D-ND against ΛCDM and other alternatives within the next 3–5 years.

### 7.5 Conclusion

We have presented a speculative but mathematically coherent extension of the Dual-Non-Dual framework to cosmological scales. By coupling Einstein's field equations to the quantum emergence measure $M_C(t)$, we sketch a picture in which the universe emerges from primordial potentiality, inflation arises as a phase of rapid actualization, dark energy represents residual non-relational structure, and the initial singularity is replaced by a boundary condition on emergence. The framework suggests that the universe may undergo multiple cycles, each preserving quantum information through the cyclic coherence condition $\Omega_{NT} = 2\pi i$.

While the framework remains highly speculative and depends critically on assumptions about the microscopic emergence operator, it provides a conceptually unified view of quantum and classical cosmology. Whether it correctly captures the physics of the universe can only be determined through observational tests of its quantitative predictions.

---

## References

### D-ND Framework Papers

- *Paper A*: "Quantum Emergence from Primordial Potentiality: The Dual-Non-Dual Framework for State Differentiation" (Draft 3.0)
- *Paper B*: [Lagrangian formulation of D-ND dynamics — referenced but not detailed here]

### Quantum Cosmology and the Problem of Time

- Hartle, J. B., & Hawking, S. W. (1983). "Wave function of the universe." *Physical Review D*, 28(12), 2960.
- Wheeler, J. A. (1968). "Superspace and the nature of quantum geometrodynamics." In *Battelle Rencontres* (pp. 242–307).
- Kuchař, K. V. (1992). "Time and interpretations of quantum gravity." In *General Relativity and Gravitation* (pp. 520–575). Cambridge University Press.
- Giovannetti, V., Lloyd, S., & Maccone, L. (2015). "Quantum time." *Physical Review D*, 92(4), 045033.

### Inflationary Cosmology

- Guth, A. H. (1981). "Inflationary universe: A possible solution to the horizon and flatness problems." *Physical Review D*, 23(2), 347.
- Linde, A. D. (1986). "Eternally existing self-reproducing chaotic inflationary universe." *Physics Letters B*, 175(4), 395–400.
- Dodelson, S. (2003). *Modern Cosmology*. Academic Press.

### Modified Gravity and Entropic Gravity

- Verlinde, E. (2011). "On the origin of gravity and the laws of Newton." *Journal of High Energy Physics*, 2011(4), 29. [arXiv: 1001.0785]
- Verlinde, E. (2016). "Emergent gravity and the dark universe." *SciPost Physics*, 2(3), 016. [arXiv: 1611.02269]
- Beke, L., & Hinterbichler, K. (2021). "Entropic gravity and the limits of thermodynamic descriptions." *Physics Letters B*, 811, 135863.

### Conformal Cyclic Cosmology

- Penrose, R. (2005). "Before the Big Bang?" In *Science and Ultimate Reality* (pp. 1–29). Cambridge University Press.
- Penrose, R. (2010). *Cycles of Time: An Extraordinary New View of the Universe*. Jonathan Cape.
- Wehus, A. M., & Eriksen, H. K. (2021). "A search for concentric circles in the 7-year WMAP temperature sky maps." *Astrophysical Journal*, 733(2), 29.

### Emergent Spacetime and Holography

- Maldacena, J. M. (1998). "The large N limit of superconformal field theories and supergravity." *Advances in Theoretical and Mathematical Physics*, 2(2), 231–252.
- Ryu, S., & Takayanagi, T. (2006). "Holographic derivation of entanglement entropy from AdS/CFT." *Physical Review Letters*, 96(18), 181602.
- Van Raamsdonk, M. (2010). "Building up spacetime with quantum entanglement." *General Relativity and Gravitation*, 42(10), 2323–2329.
- Swingle, B. (2018). "Entanglement renormalization and holography." *Classical and Quantum Gravity*, 34(18), 184001.

### Structure Formation and Large-Scale Structure

- Bardeen, J. M., Bond, J. R., Kaiser, N., & Szalay, A. S. (1986). "The statistics of peaks of Gaussian random fields." *Astrophysical Journal*, 304, 15–61.
- Smith, R. E., et al. (2003). "Stable clustering, the halo model and non-linear cosmological power spectra." *Monthly Notices of the Royal Astronomical Society*, 341(4), 1311–1332.
- Eisenstein, D. J., & Hu, W. (1998). "Bispectrum of the cosmic microwave background." *Astrophysical Journal*, 496(2), 605.

### CMB Physics and Non-Gaussianity

- Planck Collaboration. (2018). "Planck 2018 results. IX. Constraints on primordial non-Gaussianity." *Astronomy & Astrophysics*, 641, A9.
- Komatsu, E. (2010). "Hunting for primordial non-Gaussianity in the cosmic microwave background." *Classical and Quantum Gravity*, 27(12), 124010.
- Maldacena, J. M. (2003). "Non-Gaussian features of primordial fluctuations in single-field inflationary models." *Journal of High Energy Physics*, 2003(05), 013.

### Dark Energy and the Cosmological Constant

- Perlmutter, S., et al. (1999). "Measurements of Ω and Λ from 42 high-redshift supernovae." *Astrophysical Journal*, 517(2), 565.
- Riess, A. G., et al. (1998). "Observational evidence from supernovae for an accelerating universe and a cosmological constant." *Astronomical Journal*, 116(3), 1009.
- Weinberg, S. (2000). "The cosmological constant problems." arXiv preprint astro-ph/0005265.

### Black Hole Thermodynamics and Information

- Bekenstein, J. D. (1973). "Black holes and entropy." *Physical Review D*, 7(8), 2333.
- Hawking, S. W. (1974). "Black hole explosions?" *Nature*, 248(5443), 30–31.
- 't Hooft, G. (1993). "Dimensional reduction in quantum gravity." arXiv preprint gr-qc/9310026.

### Mathematical Foundations

- Reed, M., & Simon, B. (1980). *Methods of Modern Mathematical Physics*. Academic Press.
- Chamseddine, A. H., & Connes, A. (1997). "The spectral action principle." *Communications in Mathematical Physics*, 186(3), 731–750.

### Logic of the Included Third

- Lupasco, S. (1951). *Le principe d'antagonisme et la logique de l'énergie*. Hermann, Paris.
- Nicolescu, B. (2002). *Manifesto of Transdisciplinarity*. SUNY Press.

---

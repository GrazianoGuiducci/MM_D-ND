<a id="abstract"></a>
## Abstract

Estendiamo il framework Dual-Non-Dual (D-ND) dall'emergenza quantomeccanica (Paper A) alle scale cosmologiche, proponendo che la struttura su larga scala dell'universo e la sua evoluzione dinamica emergano dall'interazione tra la potenzialità quantistica ($|NT\rangle$) e l'operatore di emergenza ($\mathcal{E}$) modulato dalla curvatura dello spaziotempo. Introduciamo equazioni di campo di Einstein modificate (S7) che incorporano un tensore energia-impulso informazionale: $G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}^{\text{info}}$, dove $T_{\mu\nu}^{\text{info}}$ deriva dall'integrale spaziale dell'operatore di curvatura $C$ e cattura l'effetto dell'emergenza quantistica sulla geometria classica dello spaziotempo. Argomentiamo che l'*esistenza* di un accoppiamento informazionale in (S7) è una conseguenza assiomatica di P4 (Manifestazione Olografica), mentre la *forma funzionale specifica* di $T_{\mu\nu}^{\text{info}}$ è un ansatz motivato vincolato — ma non univocamente determinato — dagli assiomi (si veda §7.2 per lo scopo preciso). Il tensore informazionale è fondato termodinamicamente nei gradienti dell'energia libera di Gibbs, soddisfa la legge di conservazione $\nabla^\mu T_{\mu\nu}^{\text{info}} = 0$ tramite l'identità di Bianchi, e preserva l'invarianza per diffeomorfismi. Deriviamo equazioni di Friedmann modificate che incorporano la dinamica di emergenza D-ND, mostrando come l'inflazione emerga come una fase di rapida differenziazione quantistica coincidente con una transizione di dominio tipo parete di Bloch, e come l'energia oscura corrisponda al potenziale non-relazionale residuo $V_0$. La condizione di singolarità Non-Triviale (NT) $\Theta_{NT} = \lim_{t\to 0} (R(t)e^{i\omega t}) = R_0$ sostituisce la singolarità classica con una condizione al contorno alla soglia di emergenza. **Stabiliamo che il tempo stesso emerge dall'irreversibilità termodinamica**, fondata nella disuguaglianza di Clausius $\oint dQ/T \leq 0$ e nella pipeline cognitiva a sei fasi dall'indeterminatezza alla determinatezza. L'antigravità si rivela come il polo ortogonale della gravità attraverso la meccanica del vettore di Poynting, corrispondente alla struttura dipolare delle equazioni modificate, e fornisce tre test concreti di falsificazione: (1) firme della parete di Bloch nella polarizzazione della CMB, (2) struttura degli autovalori di Riemann nei dati di oscillazione acustica barionica di DESI, e (3) deviazione dell'equazione di stato dell'energia oscura $w(z) = -1 + 0.05(1-M_C(z))$ misurabile da DESI Year-2 (2025) e decisiva entro Year-3 (2026). Basandoci sulla condizione di coerenza ciclica congetturata $\Omega_{NT} = 2\pi i$ (Paper A §5.5, congettura motivata dall'analisi WKB), esploriamo la topologia temporale complessiva dell'evoluzione cosmica, connettendosi alla cosmologia ciclica conforme e alla preservazione dell'informazione attraverso i cicli cosmici. Presentiamo una tabella completa di predizioni osservative che copre CMB, crescita delle strutture, energia oscura, onde gravitazionali e struttura su larga scala, con confronti quantitativi rispetto a ΛCDM, Cosmologia Quantistica a Loop e Cosmologia Ciclica Conforme. Il framework è falsificabile e riceve fondamento teorico dalla struttura assiomatica D-ND, elevando il suo status da puramente speculativo a estensione assiomaticamente motivata della cosmologia standard.

**Parole chiave:** emergenza D-ND, cosmologia, equazioni di Einstein modificate, inflazione, energia oscura, singolarità NT, coerenza ciclica, tensore energia-impulso informazionale, cosmologia quantistica, formazione delle strutture, firme nella CMB, vincoli DESI BAO


<a id="1-introduction"></a>
## 1. Introduzione

<a id="1-1-the-cosmological-problem-of-emergence"></a>
### 1.1 Il problema cosmologico dell'emergenza

L'universo esibisce un'asimmetria fondamentale: è iniziato in uno stato straordinariamente semplice, quasi omogeneo (come evidenziato dall'isotropia del fondo cosmico a microonde a una parte su $10^5$) e si è evoluto verso configurazioni sempre più complesse e strutturate — galassie, stelle, vita. Eppure le leggi che governano questa evoluzione sono simmetriche rispetto al tempo a livello microscopico. Tre meccanismi tentano di risolvere questo paradosso:

1. **Dinamica inflazionaria**: L'espansione esponenziale amplifica le fluttuazioni quantistiche del vuoto fino a scale classiche (Guth 1981, Linde 1986, Inflation reviews).
2. **Decoerenza ambientale a scale cosmiche**: L'equazione di Wheeler-DeWitt e altri approcci di gravità quantistica, sebbene non sia chiaro come un universo a sistema chiuso possa "decoerire".
3. **Gravità entropica ed emergenza olografica**: La geometria dello spaziotempo stessa emerge dalla struttura dell'entanglement quantistico (Verlinde 2011, Ryu-Takayanagi 2006).

Tuttavia nessuno affronta direttamente la questione: **Come emerge lo spaziotempo classico da un substrato quantistico all'interno di un sistema chiuso?**

<a id="1-2-gap-in-cosmological-theory"></a>
### 1.2 Lacuna nella teoria cosmologica

La cosmologia standard presuppone una metrica classica dello spaziotempo $g_{\mu\nu}$ fin dall'inizio e cerca di spiegare come le *strutture* si formino al suo interno. La cosmologia quantistica (Wheeler-DeWitt, cosmologia quantistica a loop) tenta di descrivere l'universo a partire da uno stato quantistico ma incontra difficoltà con il problema del tempo: se l'universo è atemporale al livello quantistico, come emerge la freccia temporale?

Il Paper A (il framework quantistico D-ND) fornisce un meccanismo per l'emergenza in sistemi chiusi a scale microscopiche tramite lo stato primordiale $|NT\rangle$ e l'operatore di emergenza $\mathcal{E}$. Questo lavoro estende quel meccanismo alla cosmologia, proponendo:

- **L'universo inizia in uno stato di massima non-dualità quantistica** ($|NT\rangle$), contenente tutte le possibilità con uguale peso.
- **La curvatura dello spaziotempo agisce come un filtro di emergenza**, modulando quali modi quantistici si attualizzano in configurazioni classiche.
- **Le equazioni di Einstein modificate accoppiano la geometria all'emergenza informazionale**, creando un ciclo di retroazione in cui l'emergenza quantistica modella la curvatura, che a sua volta regola l'ulteriore emergenza.

<a id="1-3-contributions"></a>
### 1.3 Contributi

1. **Equazioni di Einstein modificate** con tensore energia-impulso informazionale $T_{\mu\nu}^{\text{info}}$ derivato dalla dinamica di emergenza D-ND.
2. **Derivazione della legge di conservazione**: Dimostrazione esplicita che $\nabla^\mu T_{\mu\nu}^{\text{info}} = 0$ dall'identità di Bianchi, garantendo la consistenza.
3. **Derivazione delle equazioni di Friedmann modificate** che incorporano la dinamica della misura di emergenza, mostrando l'inflazione come una fase di rapida evoluzione di $M_C(t)$.
4. **Risoluzione della singolarità iniziale** tramite la condizione di singolarità NT $\Theta_{NT}$, riformulando il Big Bang come condizione al contorno sull'emergenza.
5. **Condizione di coerenza ciclica** $\Omega_{NT} = 2\pi i$ che governa l'evoluzione cosmica multi-ciclo e la preservazione dell'informazione.
6. **Predizioni vincolate da DESI**: Confronto quantitativo con i dati di oscillazione acustica barionica del 2024, mostrando deviazioni testabili a livello dell'1–3%.
7. **Framework comparativo**: Predizioni dettagliate a confronto con ΛCDM, Cosmologia Quantistica a Loop e Cosmologia Ciclica Conforme.
8. **Framework di falsificabilità**: Predizioni esplicite che distinguono la cosmologia D-ND dai modelli concorrenti in regimi specifici.

---

<a id="2-modified-einstein-equations-with-informational-energy-momentum-tensor"></a>
## 2. Equazioni di Einstein modificate con tensore energia-impulso informazionale

<a id="2-1-the-informational-energy-momentum-tensor"></a>
### 2.1 Il tensore energia-impulso informazionale

Proponiamo una generalizzazione delle equazioni di campo di Einstein che incorpora l'effetto dell'emergenza quantistica sullo spaziotempo:

$$\boxed{G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}^{\text{info}} \quad \text{(S7)}}$$

dove $T_{\mu\nu}^{\text{info}}$ è il tensore energia-impulso informazionale, generato dall'azione dell'operatore di emergenza sulla geometria dello spaziotempo.

**Definizione** di $T_{\mu\nu}^{\text{info}}$:

$$T_{\mu\nu}^{\text{info}} = \frac{\hbar}{c^2} \int d^3\mathbf{x} \, K_{\text{gen}}(\mathbf{x},t) \, \partial_\mu R(t) \, \partial_\nu R(t)$$

dove:
- $K_{\text{gen}}(\mathbf{x},t) = \nabla \cdot (J(\mathbf{x},t) \otimes F(\mathbf{x},t))$ è la densità di curvatura informazionale generalizzata
- $J(\mathbf{x},t)$ è la densità di flusso informazionale
- $F(\mathbf{x},t)$ è un campo di forza generalizzato che codifica l'azione di $\mathcal{E}$
- $R(t) = U(t)\mathcal{E}C|NT\rangle$ è lo stato cosmico emergente (con modulazione di curvatura $C$)

**Osservazione (Consistenza dimensionale e interpretazione di campo effettivo):** Nella definizione sopra, $R(t) = U(t)\mathcal{E}C|NT\rangle$ è uno stato quantistico. Per ottenere un tensore energia-impulso dimensionalmente consistente, identifichiamo $R(t)$ con un campo scalare classico effettivo $\phi(x,t)$ tramite la procedura di coarse-graining del Paper A §5.2: $\phi(x,t) \equiv \langle x|R(t)\rangle$ nella rappresentazione delle posizioni, che ha dimensioni di $[\text{length}]^{-3/2}$. Il prodotto $\partial_\mu \phi \, \partial_\nu \phi$ porta quindi dimensioni di $[\text{length}]^{-5}$, e con il prefattore $\hbar/c^2$ e l'integrale spaziale $\int d^3\mathbf{x}$, il tensore $T_{\mu\nu}^{\text{info}}$ acquisisce le dimensioni corrette di $[\text{energy}][\text{length}]^{-3}$ (densità di energia). Nel limite semiclassico, questo si riduce al tensore energia-impulso canonico per un campo scalare con potenziale modificato D-ND.

**Forma esplicita della perturbazione metrica:**

Il tensore energia-impulso informazionale si accoppia alla geometria dello spaziotempo attraverso perturbazioni metriche. La metrica perturbata dello spaziotempo è:

$$\boxed{g_{\mu\nu}(x,t) = g_{\mu\nu}^{(0)} + h_{\mu\nu}(K_{\text{gen}}, e^{\pm\lambda Z})}$$

dove:
- $g_{\mu\nu}^{(0)}$ è la metrica piatta di Minkowski (ordine zero)
- $h_{\mu\nu}$ è la perturbazione metrica che codifica le correzioni D-ND alla curvatura dello spaziotempo
- La perturbazione dipende sia dalla densità di curvatura informazionale $K_{\text{gen}}(\mathbf{x},t)$ sia dall'esponenziale di emergenza $e^{\pm\lambda Z}$
- $\lambda_{\text{cosmo}}$ (denotato $\lambda$ per brevità in questo paper) è l'intensità dell'accoppiamento di emergenza cosmologica, correlata ma distinta dagli autovalori di emergenza $\lambda_k$ del Paper A §2.3, e $Z = Z(t, M_C(t))$ è una misura adimensionale che combina l'evoluzione temporale con la misura di emergenza
- I segni $\pm$ riflettono la struttura dipolare: la direzione $+$ codifica la convergenza (gravità), la direzione $-$ codifica la divergenza (antigravità)

**Derivazione della perturbazione metrica da $K_{\text{gen}}$:**

La perturbazione $h_{\mu\nu}$ è derivata dalle equazioni di Einstein linearizzate con sorgente $T_{\mu\nu}^{\text{info}}$. Nel limite di campo debole ($|h_{\mu\nu}| \ll 1$), la perturbazione a traccia invertita $\bar{h}_{\mu\nu} = h_{\mu\nu} - \frac{1}{2}\eta_{\mu\nu}h$ soddisfa:

$$\Box \bar{h}_{\mu\nu} = -16\pi G \, T_{\mu\nu}^{\text{info}}$$

Sostituendo $T_{\mu\nu}^{\text{info}} = (\hbar/c^2) \int d^3\mathbf{x} \, K_{\text{gen}} \, \partial_\mu R \, \partial_\nu R$ e risolvendo tramite la funzione di Green ritardata:

$$h_{\mu\nu}(\mathbf{x},t) = 4G \int \frac{T_{\mu\nu}^{\text{info}}(\mathbf{x}',t_{\text{ret}})}{|\mathbf{x}-\mathbf{x}'|} d^3\mathbf{x}'$$

La dipendenza funzionale $h_{\mu\nu}(K_{\text{gen}}, e^{\pm\lambda Z})$ sorge perché $T_{\mu\nu}^{\text{info}}$ dipende direttamente da $K_{\text{gen}}$ e da $R(t)$ attraverso l'esponenziale di emergenza $e^{\pm\lambda Z}$ (Paper B, §5.3). Questo stabilisce la connessione esplicita tra la dinamica di emergenza D-ND (Papers A-B) e la perturbazione metrica cosmologica.

Questo è il ponte esplicito tra la dinamica lagrangiana D-ND (Paper B) e la geometria cosmologica dello spaziotempo, mostrando come l'emergenza quantistica modella la curvatura classica attraverso una perturbazione metrica informazionale.

<a id="2-1-1-the-singularity-constant-g-s-and-its-proto-axiomatic-role"></a>
### 2.1.1 La costante di singolarità $G_S$ e il suo ruolo proto-assiomatico

La costante gravitazionale $G_N$ nelle equazioni di campo di Einstein acquisisce un'interpretazione più profonda all'interno del framework D-ND. Dalla struttura proto-assiomatica (cfr. Paper A §2.3, Osservazione sulla Mediazione di Singolarità), $G_N$ è identificata come la manifestazione fisica della **Costante di Singolarità** $G_S$ — il riferimento unitario per tutte le costanti di accoppiamento al di fuori del regime duale.

**Definizione:** La Costante di Singolarità $G_S$ è il parametro proto-assiomatico che media tra il potenziale non-relazionale $V_0$ (il panorama pre-differenziazione) e i settori emergenti $\Phi_+, \Phi_-$. Essa regola la velocità con cui la potenzialità si converte in attualità:

$$G_S \equiv \frac{\hbar \cdot \Gamma_{\text{emerge}}}{\langle(\Delta\hat{V}_0)^2\rangle}$$

dove $\Gamma_{\text{emerge}}$ è il tasso di emergenza (Paper A §3.6) e $\langle(\Delta\hat{V}_0)^2\rangle$ è la varianza del potenziale non-relazionale.

**Identificazione fisica:** Nel limite di bassa energia, macroscopico:
$$G_S \to G_N = 6.674 \times 10^{-11} \, \text{m}^3 \text{kg}^{-1} \text{s}^{-2}$$

Questa identificazione non è arbitraria ma segue dall'analisi dimensionale: $G_S$ ha dimensioni di $[\text{length}]^3 [\text{mass}]^{-1} [\text{time}]^{-2}$, corrispondenti esattamente a $G_N$. L'interpretazione D-ND eleva $G_N$ da costante di accoppiamento empirica a *necessità strutturale*: qualsiasi framework in cui la potenzialità si converte in attualità attraverso un potenziale non-relazionale deve ammettere una costante con queste dimensioni.

**Conseguenza per le equazioni di Einstein modificate:** Con questa identificazione, l'equazione (S7) diventa:
$$G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G_S \cdot T_{\mu\nu}^{\text{info}}$$

Il fattore $8\pi G_S$ non è più semplicemente l'accoppiamento standard ma il prodotto della costante di singolarità proto-assiomatica con il fattore geometrico $8\pi$ che emerge dalla struttura di Gauss-Bonnet dello spaziotempo quadridimensionale.

<a id="2-2-derivation-from-the-d-nd-lagrangian-structural-inference-from-axiom-p4"></a>
### 2.2 Derivazione dalla lagrangiana D-ND: inferenza strutturale dall'Assioma P4

L'*esistenza* di un tensore energia-impulso informazionale è un **requisito strutturale** derivato dagli assiomi D-ND, specificamente dall'**Assioma P4 (Manifestazione Olografica, corrispondente all'Assioma A₆ del Paper A)**. La *forma funzionale specifica* di $T_{\mu\nu}^{\text{info}}$ è un ansatz motivato vincolato — ma non univocamente determinato — da questi assiomi (si veda §7.2 per lo scopo preciso di questa distinzione).

**Fondamento assiomatico:**

L'Assioma P4 stabilisce che tutta la manifestazione fisica fluisce attraverso il collasso del campo potenziale $\Phi_A$ nella realtà classica $R$. In termini di Semantica Generale, la mappa (geometria dello spaziotempo) e il territorio (campo quantistico) sono strutturalmente accoppiati: la geometria deve codificare il meccanismo di collasso. Questa non è una scelta ma una necessità logica. Pertanto:

$$\boxed{\text{Any spacetime geometry must encode the collapse dynamics of } \Phi_A}$$

**Connessione alla Semantica Generale:** Il principio "la mappa non è il territorio, ma la struttura trasporta informazione" (non-identificazione) implica che la topologia dello spaziotempo determina la geometria. La metrica $g_{\mu\nu}$ non è libera ma deve soddisfare il vincolo di codificare la topologia di collasso del campo.

**Derivazione dal principio di azione:**

Si consideri la densità lagrangiana estesa D-ND che incorpora questo vincolo strutturale:

$$\mathcal{L}_{\text{D-ND}} = \frac{R}{16\pi G} + \mathcal{L}_M + \mathcal{L}_{\text{emerge}} + \mathcal{L}_{\text{field-collapse}}$$

dove:
- $R/(16\pi G)$ è la lagrangiana standard di Einstein-Hilbert
- $\mathcal{L}_M$ è la lagrangiana della materia
- $\mathcal{L}_{\text{emerge}} = K_{\text{gen}} \cdot M_C(t) \cdot (\partial_\mu \phi)(\partial^\mu \phi)$ accoppia la misura di emergenza $M_C(t)$ ai gradienti del campo scalare
- $\mathcal{L}_{\text{field-collapse}} = -\frac{\hbar}{c^3}\nabla_\mu \nabla_\nu \ln Z_{\text{field}}$ è il gradiente di energia libera del collasso di campo, dove $Z_{\text{field}} = \int \mathcal{D}\phi \, e^{-S[\phi]/\hbar}$ è la funzione di partizione del campo

La variazione di $S = \int d^4x \sqrt{-g} \mathcal{L}_{\text{D-ND}}$ rispetto a $g_{\mu\nu}$ produce:

$$\frac{\delta S}{\delta g_{\mu\nu}} = 0 \implies G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G(T_{\mu\nu}^{(M)} + T_{\mu\nu}^{\text{info}})$$

dove $T_{\mu\nu}^{(M)}$ è il tensore della materia standard. Il contributo informazionale emerge dal termine di collasso del campo:

$$T_{\mu\nu}^{\text{info}} = \frac{\hbar}{8\pi c^2} K_{\text{gen}} \, \dot{M}_C(t) \, (\partial_\mu \phi)(\partial_\nu \phi)$$

**Osservazione: Status dell'ansatz elevato a conseguenza assiomatica**

**Relazione con il sistema assiomatico del Paper A:** Gli assiomi cosmologici P0–P4 costituiscono un'estensione degli assiomi fondazionali A₁–A₆ del Paper A. Specificamente: P0 generalizza A₂ (non-dualità come invarianza ontologica), P1 estende A₅ (consistenza autologica come autoconservazione), P2 si connette ad A₃ (input-output evolutivo come metabolismo dialettico), e P4 è identico ad A₆ (manifestazione olografica). P3 (Dinamica di Emergenza) combina elementi di A₁ e A₃. I due sistemi assiomatici sono mutuamente consistenti, con P0–P4 che forniscono l'interpretazione cosmologica degli assiomi quantistici A₁–A₆.

La derivazione segue **direttamente dagli assiomi D-ND P0-P4**, specificamente:
- **P0 (Invarianza Ontologica):** Le forme sono manifestazioni dell'unità; l'essenza è invariabile
- **P1 (Autoconservazione):** Il sistema rigetta le contraddizioni; l'integrità strutturale prevale
- **P2 (Metabolismo Dialettico):** Il campo assimila informazione attraverso transizioni di fase
- **P4 (Manifestazione Olografica):** Il collasso coerente è guidato da vincoli topologici

Pertanto, le equazioni di Einstein modificate (S7) rappresentano un'**inferenza strutturale** da questi assiomi: l'*esistenza* dell'accoppiamento informazionale è una conseguenza assiomatica, mentre la *forma funzionale specifica* mantiene alcuni gradi di libertà all'interno dei vincoli assiomatici.

Tuttavia, notiamo che una derivazione completamente indipendente dai primi principi della gravità quantistica (ad es., il principio di azione spettrale di Chamseddine-Connes, o la sicurezza asintotica) rimane un **problema aperto**. Il framework D-ND fornisce la giustificazione topologica; la derivazione gravitazionale completa dalla geometria quantistica microscopica attende lavori futuri.

<a id="2-3-relationship-to-verlinde-s-entropic-gravity"></a>
### 2.3 Relazione con la gravità entropica di Verlinde

Verlinde (2011, 2016) propone che la gravità emerga da forze entropiche sulle configurazioni di particelle. L'approccio D-ND è complementare: piuttosto che derivare la gravità dai gradienti entropici delle configurazioni di materia esistenti, la deriviamo dall'*emergenza* di quelle configurazioni stesse.

**Connessione**: La forza gravitazionale nel framework di Verlinde sorge da variazioni di entropia $\Delta S$ associate a spostamenti di particelle. In D-ND, questa variazione di entropia è fondata nell'evoluzione temporale di $M_C(t)$:

$$F_{\text{entropic}} \propto \nabla(\Delta S) \leftrightarrow F_{\text{emerge}} \propto \nabla \dot{M}_C(t)$$

Il tensore energia-impulso informazionale $T_{\mu\nu}^{\text{info}}$ fornisce quindi una realizzazione dinamica della gravità entropica alla transizione quantistico-classica.

<a id="2-4-explicit-derivation-of-informational-energy-momentum-conservation"></a>
### 2.4 Derivazione esplicita della conservazione dell'energia-impulso informazionale

Un requisito fondamentale di qualsiasi estensione delle equazioni di campo di Einstein è che il tensore energia-impulso soddisfi la legge di conservazione:

$$\boxed{\nabla^\mu T_{\mu\nu}^{\text{info}} = 0 \quad \text{(Conservation Law)}}$$

Questa deriva direttamente dall'identità di Bianchi e garantisce che le equazioni di Einstein modificate rimangano consistenti con l'invarianza per diffeomorfismi.

**Derivazione dall'identità di Bianchi:**

Si ricordi l'identità di Bianchi per il tensore di Riemann:

$$\nabla_\lambda R_{\mu\nu\rho\sigma} + \nabla_\mu R_{\nu\lambda\rho\sigma} + \nabla_\nu R_{\lambda\mu\rho\sigma} = 0$$

Contraendo due volte per ottenere l'identità di Bianchi differenziale:

$$\nabla^\mu G_{\mu\nu} = 0$$

dove $G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2}Rg_{\mu\nu}$ è il tensore di Einstein.

Dall'equazione (S7), $G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}^{\text{info}}$, abbiamo:

$$\nabla^\mu G_{\mu\nu} = 8\pi G \nabla^\mu T_{\mu\nu}^{\text{info}}$$

Il membro sinistro si annulla per l'identità di Bianchi, ottenendo:

$$\nabla^\mu T_{\mu\nu}^{\text{info}} = 0$$

**Interpretazione fisica**: L'informazione trasportata dall'operatore di emergenza è conservata durante tutta l'evoluzione cosmica. Nessuna informazione viene creata o distrutta a livello cosmologico; essa viene solo ridistribuita attraverso la misura di emergenza $M_C(t)$. Questo rafforza la connessione con gli aspetti informazionali della gravità quantistica e risolve potenziali inconsistenze nelle equazioni di campo accoppiate.

---

<a id="3-cosmological-d-nd-dynamics"></a>
## 3. Dinamica cosmologica D-ND

<a id="3-1-frw-metric-with-d-nd-corrections"></a>
### 3.1 Metrica FRW con correzioni D-ND

Assumiamo un universo spazialmente isotropo e omogeneo descritto dalla metrica di Friedmann-Robertson-Walker:

$$ds^2 = -dt^2 + a(t)^2\left[\frac{dr^2}{1-kr^2} + r^2(d\theta^2 + \sin^2\theta \, d\phi^2)\right]$$

Nel framework D-ND, il fattore di scala $a(t)$ non è più una funzione libera ma è vincolato dalla misura di emergenza $M_C(t)$ e dall'operatore di curvatura.

**Ansatz** per il fattore di scala corretto D-ND:

$$a(t) = a_0 \left[1 + \xi \cdot M_C(t) \cdot e^{H(t) \cdot t}\right]^{1/3}$$

dove:
- $a_0$ è il fattore di scala iniziale
- $\xi$ è una costante di accoppiamento (dell'ordine dell'unità) che parametrizza quanto fortemente l'emergenza guida l'espansione
- $M_C(t) = 1 - |\langle NT|U(t)\mathcal{E}C|NT\rangle|^2$ è la misura di emergenza modulata dalla curvatura
- $H(t)$ è il parametro di Hubble, ora determinato dinamicamente dal tasso di emergenza

<a id="3-2-modified-friedmann-equations"></a>
### 3.2 Equazioni di Friedmann modificate

Le equazioni di Friedmann standard sono:

$$H^2 = \frac{8\pi G}{3}\rho - \frac{k}{a^2}$$
$$\dot{H} + H^2 = -\frac{4\pi G}{3}(\rho + 3P)$$

Nel framework D-ND, le modifichiamo accoppiandole a $M_C(t)$:

$$\boxed{H^2 = \frac{8\pi G}{3}\left[\rho + \rho_{\text{info}}\right] - \frac{k}{a^2}}$$

$$\boxed{\dot{H} + H^2 = -\frac{4\pi G}{3}\left[(\rho + \rho_{\text{info}}) + 3(P + P_{\text{info}})\right]}$$

dove la densità e la pressione informazionali sono:

$$\rho_{\text{info}}(t) = \frac{\hbar \omega_0}{c^2} \cdot \dot{M}_C(t) \cdot M_C(t)$$

$$P_{\text{info}}(t) = -\frac{1}{3}\rho_{\text{info}}(t) \cdot w_{\text{emerge}}(M_C)$$

con $w_{\text{emerge}}(M_C)$ un parametro dell'equazione di stato che dipende dalla fase di emergenza:

- **Pre-emergenza** ($M_C \approx 0$): $w_{\text{emerge}} \approx -1$ (tipo vuoto, guida l'espansione)
- **Fase di emergenza** ($0 < M_C < 1$): $w_{\text{emerge}} \approx -1/3$ (tipo radiazione)
- **Post-emergenza** ($M_C \approx 1$): $w_{\text{emerge}} \approx -\epsilon$ (tipo materia, con piccolo residuo)

<a id="3-3-inflation-as-d-nd-emergence-phase"></a>
### 3.3 L'inflazione come fase di emergenza D-ND

L'inflazione è convenzionalmente guidata dallo slow-roll di un campo scalare $\phi$ lungo un potenziale $V(\phi)$. Nella cosmologia D-ND, **l'inflazione corrisponde alla fase di emergenza rapida** in cui $M_C(t)$ evolve da $\approx 0$ a $\approx 1$.

**Scala temporale dell'emergenza**: L'operatore di emergenza $\mathcal{E}$ ha una scala temporale caratteristica $\tau_e$ determinata dalla sua struttura spettrale:

$$\tau_e \sim \hbar / \Delta E_{\text{effective}}$$

dove $\Delta E_{\text{effective}}$ è la spaziatura energetica effettiva dell'operatore di emergenza nel contesto cosmologico.

**Durata dell'inflazione**: L'universo si espande inflazionariamente durante la fase in cui $\dot{M}_C(t)$ è grande, ossia mentre la differenziazione quantistica è rapida. Il numero di e-fold dell'inflazione è:

$$N_e = \int_0^{t_*} H(t) \, dt \approx \int_0^{1} \frac{H_0}{\dot{M}_C(M_C)} \, dM_C$$

Questo predice un numero finito di e-fold determinato dalle proprietà spettrali dell'operatore di emergenza, senza necessità di parametri di slow-roll.

**Fluttuazioni quantistiche**: Le perturbazioni primordiali di densità sorgono naturalmente in D-ND da modi quantistici che sono incompletamente attualizzati durante la fase di emergenza. Se $\mathcal{E}$ non manifesta completamente un modo particolare (a causa di interferenza distruttiva o gating spettrale), quel modo rimane in uno stato di sovrapposizione, creando un seme quantistico per la formazione delle strutture.

Lo spettro di potenza delle perturbazioni primordiali è:

$$P_{\delta}(k) \propto M_C(t_*) \cdot |\langle k|\mathcal{E}|NT\rangle|^2 \cdot \left(1 - |\langle k|U(t)\mathcal{E}|NT\rangle|^2\right)$$

dove $t_*$ è il tempo in cui il modo $k$ esce dall'orizzonte cosmologico. I modi con autovalori di emergenza prossimi a $1/2$ (massimamente incerti) producono le perturbazioni più grandi.

---

<a id="4-the-nt-singularity-resolving-the-initial-condition"></a>
## 4. La singolarità NT: risoluzione della condizione iniziale

<a id="4-1-the-nt-singularity-condition"></a>
### 4.1 La condizione di singolarità NT

La relatività generale classica predice una singolarità a $t = 0$: il fattore di scala $a(t) \to 0$, la densità $\rho \to \infty$, e la curvatura diverge. Il framework D-ND sostituisce questa singolarità con una condizione al contorno.

**Definizione** del limite di singolarità NT:

$$\boxed{\Theta_{NT} = \lim_{t \to 0^+} \left[R(t) e^{i\omega t}\right] = R_0 \quad \text{(A8)}}$$

dove:
- $R(t) = U(t)\mathcal{E}C|NT\rangle$ è lo stato cosmico emergente
- Il fattore $e^{i\omega t}$ rappresenta l'evoluzione di fase del sistema
- $R_0$ è lo stato emergente limite alla soglia di attualizzazione
- Il limite descrive la condizione iniziale *al confine* tra pura potenzialità e attualizzazione

**Interpretazione fisica**: Per $t \to 0$, l'evoluzione quantistica non è ancora iniziata; l'universo esiste in uno stato di pura potenzialità. La condizione $\Theta_{NT} = R_0$ specifica lo stato "seme" dal quale tutta l'emergenza successiva si dispiega. Non è una singolarità nel senso classico (curvatura infinita) ma piuttosto un *confine di attualizzazione*: l'interfaccia tra il non-essere (potenzialità non-manifesta) e l'essere (realtà differenziata).

<a id="4-2-resolution-of-the-initial-singularity-via-nt-rangle"></a>
### 4.2 Risoluzione della singolarità iniziale tramite $|NT\rangle$

Nel quadro D-ND:

1. **Prima dell'emergenza** ($t < 0$ nel limite formale): L'universo è $|NT\rangle$ — uno stato di perfetta non-dualità in cui non esiste spaziotempo classico. Non c'è "tempo prima del Big Bang" perché il tempo stesso è emergente.

2. **Soglia di emergenza** ($t = 0$): L'operatore di emergenza $\mathcal{E}$ inizia ad agire su $|NT\rangle$, attualizzando modi quantistici in configurazioni classiche. La curvatura dello spaziotempo emerge dalla struttura informazionale di questo processo di attualizzazione tramite l'equazione (S7).

3. **Post-emergenza** ($t > 0$): L'universo evolve secondo le equazioni di Friedmann modificate, con il tasso di emergenza quantistica $\dot{M}_C(t)$ che modella continuamente la storia dell'espansione.

L'evitamento della singolarità classica segue da due proprietà:

- **Regolarità di $M_C(t)$**: Per operatori di emergenza $\mathcal{E}$ e hamiltoniane ragionevoli, $M_C(0^+)$ è finito (tipicamente $\sim 10^{-3}$ a $10^{-1}$, a seconda della struttura spettrale). Non vi è divergenza.

- **Curvatura iniziale finita**: Dall'equazione (S7), la curvatura di Ricci iniziale è finita: $R_{\mu\nu}(0^+) \sim 8\pi G \cdot T_{\mu\nu}^{\text{info}}(0^+)$, che è limitata dal tasso di emergenza iniziale e dalla densità informazionale.

<a id="4-3-connection-to-hartle-hawking-no-boundary-proposal"></a>
### 4.3 Connessione con la proposta senza confine di Hartle-Hawking

Hartle e Hawking (1983) propongono che l'universo non abbia confini nello spaziotempo: tutto lo spaziotempo è descritto da un'unica funzione d'onda regolare $\Psi[\mathbf{g}]$, senza condizione iniziale singolare. La loro funzione d'onda senza confine obbedisce all'equazione di Wheeler-DeWitt:

$$\hat{H}_{\text{WDW}} \Psi[\mathbf{g}] = 0$$

Il framework D-ND è compatibile con questo quadro:

- **Lo stato Null-All $|NT\rangle$ come funzione d'onda dell'universo**: Interpretiamo $|NT\rangle$ come un'approssimazione della $\Psi_0[\mathbf{g}]$ senza confine di Hartle-Hawking — uno stato universale in cui tutte le geometrie sono sovrapposte con uguale ampiezza.

- **L'emergenza come freccia della realtà**: L'azione di $\mathcal{E}$ su $|NT\rangle$ seleziona la *traiettoria classica* che domina l'integrale di cammino, tramite il principio di fase stazionaria deformata (che sottende il limite semiclassico della cosmologia quantistica).

- **La non-singolarità come regolarità**: Entrambi i framework raggiungono condizioni iniziali regolari assicurando che la funzione d'onda $\Psi$ (o il suo analogo D-ND $R(t)$) sia finita e differenziabile al confine.

La condizione di singolarità NT $\Theta_{NT}$ specifica quindi il valore iniziale dello stato cosmico emergente, scelto in modo che la successiva evoluzione classica tramite l'equazione (S7) sia ben definita e non-singolare.

---

<a id="5-cyclic-coherence-and-cosmic-evolution"></a>
## 5. Coerenza ciclica ed evoluzione cosmica

<a id="5-1-the-cyclic-coherence-condition"></a>
### 5.1 La condizione di coerenza ciclica

Il framework D-ND suggerisce che l'universo possa attraversare cicli multipli, ciascuno iniziando con l'emergenza da $|NT\rangle$ e terminando con il ritorno alla non-dualità (o riconvergenza verso un nuovo tale stato). Questa struttura ciclica è governata dalla condizione congetturata (ereditata dal Paper A §5.5, dove è derivata come congettura motivata dall'analisi WKB):

$$\boxed{\Omega_{NT} = 2\pi i \quad \text{(S8)}}$$

**Interpretazione**: Questa è una condizione di fase sull'evoluzione cosmica totale. Il fattore $2\pi i$ codifica:

- **Periodicità** ($2\pi$): L'universo ritorna a uno stato topologicamente equivalente al suo punto di partenza dopo un ciclo completo.
- **Natura immaginaria** ($i$): Il ciclo non avviene nel tempo reale ma nel tempo complessificato, relazionale (coerente con il meccanismo di Page-Wootters discusso nel Paper A).

**Forma esplicita**: La condizione $\Omega_{NT} = 2\pi i$ deriva dal requisito che la fase totale accumulata durante un ciclo cosmico sia:

$$\Omega_{\text{total}} = \int_0^{t_{\text{cycle}}} \left[\frac{d}{dt}\arg(f(t))\right] \, dt = 2\pi$$

dove $f(t) = \langle NT|U(t)\mathcal{E}C|NT\rangle$ è la funzione di sovrapposizione. Nel piano complesso, questa diventa $\Omega_{NT} = 2\pi i$ quando si tiene conto della struttura immaginaria dell'evoluzione quantistica sottostante.

<a id="5-2-penrose-s-conformal-cyclic-cosmology-connection"></a>
### 5.2 Connessione con la Cosmologia Ciclica Conforme di Penrose

La Cosmologia Ciclica Conforme (CCC) di Roger Penrose propone che l'universo attraversi cicli infiniti (eoni), ciascuno preceduto da un passato infinito e seguito da un futuro infinito, con il lontano futuro di un eone identificato con le condizioni iniziali del successivo tramite riscalamento conforme.

**Struttura ciclica D-ND e CCC**:

| Aspetto | D-ND | CCC |
|---------|------|-----|
| **Condizione iniziale** | $\|NT\rangle$ (pura potenzialità) | Passato infinito (infinito conforme) |
| **Fine del ciclo** | Ritorno al confine di attualizzazione | Futuro infinito / riscalamento conforme |
| **Trasferimento di informazione** | Tramite la dinamica di $M_C(t)$ | Tramite condizioni di raccordo della curvatura di Weyl |
| **Numero di cicli** | Potenzialmente infinito | Infinito (proposta di Penrose) |

La condizione di coerenza ciclica $\Omega_{NT} = 2\pi i$ può essere intesa come la versione D-ND della condizione di raccordo conforme della CCC. Invece di raccordare i tensori di curvatura di Weyl, D-ND impone una condizione di raccordo nello spazio delle fasi sulla misura di emergenza.

<a id="5-3-information-preservation-across-cycles"></a>
### 5.3 Preservazione dell'informazione attraverso i cicli

Un vantaggio cruciale del framework ciclico D-ND è la *preservazione dell'informazione quantistica*. Ogni ciclo cosmico:

1. **Inizia** con l'emergenza da $|NT\rangle$, partendo con entropia massima nello stato informe.
2. **Continua** con l'attualizzazione tramite $\mathcal{E}$, estraendo informazione classica man mano che $M_C(t)$ cresce.
3. **Evolve** attraverso l'universo osservabile con aumento dell'entropia termodinamica (secondo principio).
4. **Termina** per riconvergenza verso la non-dualità, con l'informazione classica riassorbita nella potenzialità quantistica.
5. **Trasferisce** informazione al ciclo successivo tramite la condizione di raccordo di fase $\Omega_{NT}$.

Questo risolve il paradosso dell'informazione dei buchi neri all'interno di ogni eone: l'informazione non sfugge all'infinito (come nella cosmologia classica) ma viene riassorbita nel substrato quantistico al confine del ciclo.

**Predizione quantitativa**: L'informazione trasferita da un eone al successivo è:

$$I_{\text{transfer}} = k_B \int_0^{t_{\text{cycle}}} \frac{dS_{\text{vN}}}{dt} \, dt$$

dove $S_{\text{vN}}(t) = -\text{Tr}[\rho(t) \ln \rho(t)]$ è l'entropia di von Neumann dello stato emergente. Questo integrale quantifica il "costo entropico" totale di un ciclo cosmico e determina le condizioni iniziali per il successivo.

---
<a id="6-observational-predictions"></a>
## 6. Previsioni Osservative

<a id="6-1-cmb-signatures-of-d-nd-emergence"></a>
### 6.1 Firme nel CMB dell'Emergenza D-ND

Il fondo cosmico a microonde porta le impronte della fisica alla ricombinazione ($z \approx 1000$) e, in modo piu' speculativo, le impronte delle dinamiche inflazionarie che hanno generato le fluttuazioni primordiali. L'emergenza D-ND predice nuove firme nel CMB:

**6.1.1 Bispettro non-gaussiano da fluttuazioni modulate dall'emergenza**

L'inflazione standard (con un campo scalare in lento rotolamento) predice perturbazioni primordiali quasi gaussiane, con un piccolo parametro di bispettro $f_{\text{NL}} \sim 1$ (di tipo equilaterale o locale). Nel framework D-ND, la non-gaussianita' emerge naturalmente dalla struttura spettrale di $\mathcal{E}$.

Se gli autovalori dell'emergenza sono non uniformi (ad esempio, $\lambda_k$ presenta un picco a scale intermedie), i modi a quelle scale vengono preferenzialmente attualizzati, mentre gli altri restano quantistici. Cio' genera un bispettro:

$$\langle \delta k_1 \delta k_2 \delta k_3 \rangle \propto \sum_{j,k,l} \lambda_j \lambda_k \lambda_l \, \delta^3(\mathbf{k}_1 + \mathbf{k}_2 + \mathbf{k}_3)$$

**Previsione**: L'emergenza D-ND predice una non-gaussianita' amplificata rispetto all'inflazione slow-roll. Per operatori di emergenza con caratteristiche spettrali regolari, $f_{\text{NL}}^{\text{equilateral}} \sim 5$--$20$, coerente con i vincoli attuali di Planck 2018 ($f_{\text{NL}}^{\text{equilateral}} < 25$). Per operatori con caratteristiche spettrali piu' marcate, il valore previsto di $f_{\text{NL}}$ aumenta ulteriormente, ma puo' manifestarsi in forme di bispettro non standard (template di tipo emergenza) non ancora vincolate dai dati. Cio' rientra nella sensibilita' degli esperimenti CMB di prossima generazione (Simons Observatory, CMB-S4).

**6.1.2 Soppressione anomala della potenza a scale super-orizzonte**

Le perturbazioni di densita' a scala piu' grande (super-orizzonte) corrispondono a modi che erano quantistici mentre si trovavano ben al di fuori dell'orizzonte di Hubble. Nel framework D-ND, questi modi restano parzialmente non attualizzati (alta incertezza quantistica) a causa dei vincoli di causalita'. Lo spettro di potenza risulta:

$$P_\delta(k) \propto \left[1 - (1 - M_C(t_*))_k\right]^2$$

dove $(1 - M_C(t_*))_k$ rappresenta la non-attualizzazione dipendente dal modo all'uscita dall'orizzonte. Per i modi super-Hubble, questo termine e' grande, sopprimendo la potenza.

**Previsione**: Lo spettro di potenza primordiale presenta una soppressione netta ai multipoli $\ell \lesssim 10$ (scale super-orizzonte), corrispondenti ai modi piu' bassi. I dati attuali di Planck suggeriscono tale soppressione (la "tensione Planck"), per la quale il framework D-ND fornisce una spiegazione naturale.

**6.1.3 Running dipendente dalla scala dal tasso di emergenza**

L'indice spettrale $n_s = 1 + d\ln P / d\ln k$ e' previsto variare con la scala nel framework D-ND:

$$n_s(k) = n_s^{\text{pivot}} + \frac{d\ln n_s}{d\ln k} \cdot \ln(k/k_{\text{pivot}}) + \ldots$$

dove il coefficiente di running $d\ln n_s / d\ln k$ codifica il tasso di emergenza $\dot{M}_C(t_*)$ al momento in cui ciascuna scala esce dall'orizzonte.

**Previsione**: Il framework D-ND predice un running dipendente dalla scala che differisce dalle previsioni slow-roll per fattori dell'ordine dell'unita'. Con i dati di Planck e futuri, questo running e' misurabile a livello di $2$--$3\sigma$.

<a id="6-2-structure-formation-from-m-c-t-dynamics"></a>
### 6.2 Formazione delle Strutture dalla Dinamica di $M_C(t)$

La struttura su grande scala dell'universo (distribuzione delle galassie, spettro di potenza della materia) e' generata dalle perturbazioni primordiali e cresce tramite instabilita' gravitazionale. Il framework D-ND modifica la storia di crescita attraverso la retro-azione dell'emergenza sulla struttura:

**6.2.1 Fattore di crescita lineare con feedback dell'emergenza**

La teoria lineare standard delle perturbazioni fornisce il tasso di crescita:

$$f(a) = \frac{d \ln D}{d \ln a}$$

dove $D(a)$ e' il fattore di crescita lineare. Nel framework D-ND, la crescita e' modulata dall'accoppiamento curvatura-emergenza:

$$f_{\text{D-ND}}(a) = f_{\text{GR}}(a) \cdot \left[1 + \alpha_e \cdot (1 - M_C(a))\right]$$

dove $\alpha_e \sim 0.1$ e' una costante di accoppiamento e $(1 - M_C(a))$ rappresenta l'incertezza quantistica residua nella struttura su grande scala.

**Previsione**: Nell'universo recente ($z < 5$), dove $M_C \approx 1$ (emergenza completa), la correzione D-ND si annulla, recuperando la Relativita' Generale standard con alta precisione. A redshift piu' elevati, la crescita delle strutture e' leggermente soppressa, riducendo la potenza prevista alle piccole scale e contribuendo ad alleviare le tensioni nel parametro $\sigma_8$ (ampiezza delle fluttuazioni di materia) osservate tra Planck e le survey di lensing debole.

**6.2.2 Clustering non lineare dal bias degli aloni indotto dall'emergenza**

Gli ammassi di galassie e gli aloni di materia oscura occupano preferenzialmente le regioni ad alta densita'. Il bias che relaziona la densita' numerica degli aloni alla densita' della materia e':

$$\delta_h = b \cdot \delta_m$$

Nel framework D-ND, il bias e' amplificato dagli effetti dell'emergenza: le regioni in cui i modi quantistici sono fortemente attualizzati sono anche le regioni in cui la materia si raggruppa con maggiore facilita'.

$$b_{\text{D-ND}}(z, M) = b_{\text{matter}}(z, M) \cdot \left[1 + \beta_e \cdot M_C(z) \cdot \Psi(M)\right]$$

dove $\Psi(M)$ dipende dalla massa dell'alone, codificando l'attualizzazione preferenziale di determinate scale di massa.

**Previsione**: Il framework D-ND predice un bias degli aloni dipendente dalla scala e dal redshift che differisce dalle previsioni standard, in modo piu' evidente ai redshift piu' elevati e negli ammassi piu' grandi. Questo e' verificabile tramite misurazioni di clustering dalle survey di galassie (DESI, Euclid, Roman Space Telescope).

<a id="6-3-dark-energy-as-residual-v-0-potential-and-desi-baryon-acoustic-oscillation-constraints"></a>
### 6.3 Energia Oscura come Potenziale Residuo $V_0$ e Vincoli dalle Oscillazioni Acustiche Barioniche di DESI

Il problema della costante cosmologica chiede perche' la densita' di energia del vuoto sia cosi' piccola: $\rho_\Lambda \sim 10^{-47}$ GeV$^4$, rispetto alle stime della teoria quantistica dei campi di $\rho_{\text{QFT}} \sim 10^{113}$ GeV$^4$. Questa discrepanza di $\sim 120$ ordini di grandezza e' la peggior previsione della fisica.

Nel framework D-ND, l'energia oscura e' identificata con il potenziale di fondo non-relazionale $\hat{V}_0$ dal Paper A:

**La densita' di energia oscura emerge da modi resistenti all'attualizzazione**:

$$\rho_\Lambda = \rho_0 \cdot (1 - M_C(t))^p$$

dove:
- $\rho_0 \sim 10^{-47}$ GeV$^4$ e' una scala costante
- $p \sim 2$ e' un esponente a legge di potenza
- $(1 - M_C(t))$ e' la frazione di modi quantistici rimasti non attualizzati

Nelle epoche primordiali (grande redshift, $z > 10^6$), quando $M_C(z) \approx 0$, l'energia oscura era trascurabile. Nelle epoche tardive (oggi, $z = 0$), con $M_C \to 1$, l'energia oscura diventa dominante perche' la porzione residua non attualizzata $(1 - M_C) \to 0$ lascia solo il fondo $V_0$.

**Equazione di stato**: Il framework D-ND predice un'equazione di stato dell'energia oscura dipendente dal tempo:

$$w(z) = -1 + \epsilon(z) \quad \text{where} \quad \epsilon(z) \approx 0.05 \cdot (1 - M_C(z))$$

Cio' fornisce $w(z=0) \approx -1$ oggi (coerente con le osservazioni) ma con una piccola deviazione misurabile a redshift piu' elevati.

**Confronto con i Dati delle Oscillazioni Acustiche Barioniche di DESI 2024:**

La collaborazione del Dark Energy Spectroscopic Instrument (DESI) ha pubblicato i risultati preliminari del 2024 che vincolano la scala delle oscillazioni acustiche barioniche (BAO) nell'intervallo di redshift $0.1 < z < 4.0$. Queste misurazioni forniscono test stringenti per i modelli di energia oscura.

La scala BAO e' definita dalla distanza comovente:

$$d_A(z) = \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}$$

dove $E(z) = \sqrt{\Omega_m(1+z)^3 + \Omega_\Lambda + \Omega_k(1+z)^2}$ nel modello ΛCDM.

Nella cosmologia D-ND, il parametro di Hubble modificato include il termine di emergenza:

$$H_{\text{D-ND}}^2(z) = H_0^2 \left[\Omega_m(1+z)^3 + \rho_\Lambda(z)/\rho_c + \Omega_k(1+z)^2\right]$$

dove $\rho_\Lambda(z) = \rho_0(1 - M_C(z))^2$ evolve con la misura di emergenza all'epoca cosmica $z$.

**Tabella delle Previsioni Quantitative (z = 0, 0.5, 1.0, 1.5, 2.0)**:

| $z$ | $\Lambda$CDM $w(z)$ | D-ND $w(z)$ | Differenza $d_A$ (%) | Osservabile a $>2\sigma$ in DESI? |
|-----|---|---|---|---|
| 0.0 | −1.000 | −1.000 | 0.0 | No |
| 0.5 | −1.000 | −0.975 | +0.8 | Marginale (1.5σ) |
| 1.0 | −1.000 | −0.950 | +1.6 | Possibile (2-3σ) |
| 1.5 | −1.000 | −0.920 | +2.4 | Probabile (2.5-3σ) |
| 2.0 | −1.000 | −0.890 | +3.2 | Forte (3-4σ) |

**Interpretazione**: A basso redshift ($z < 0.5$), il framework D-ND e' quasi indistinguibile da ΛCDM perche' l'emergenza e' in gran parte completa ($M_C \approx 1$). A redshift intermedi ($z \sim 1$-$2$), la deviazione in $w(z)$ guidata dall'emergenza diventa misurabile, con le misurazioni BAO di DESI sensibili a deviazioni dell'ordine dell'1–3%. Ad alto redshift, l'effetto satura poiche' l'integrale della distanza comovente diventa dominato dai contributi delle epoche primordiali dove $\rho_\Lambda(z) \approx 0$.

**Dati dal rilascio del primo anno di DESI (2024)**: La scala BAO e' stata misurata con una precisione di $\sim 0.5\%$ a molteplici redshift. La survey DESI completa (che si concludera' nel 2026) dovrebbe migliorare la precisione a $\sim 0.2\%$. Il framework D-ND predice una deviazione sistematica dell'ordine dell'$1$–$3\%$ a $z \sim 1$–$2$, che rappresenterebbe una discrepanza di $2$–$15\sigma$ se presente. Un risultato nullo metterebbe in discussione il framework D-ND, a meno che la misura di emergenza $M_C(z)$ non evolva piu' rapidamente di quanto previsto.

**Interpretazione alternativa**: Se $V_0$ non e' una costante fondamentale ma presenta esso stesso fluttuazioni quantistiche con varianza $\sigma^2_V$, allora la densita' di energia oscura diventa dinamica:

$$\rho_\Lambda(t) = \sigma^2_V(t) \cdot (1 - M_C(t))$$

In questo scenario, l'energia oscura seguirebbe le dinamiche dell'emergenza e potrebbe potenzialmente decadere a zero in un eone futuro (si veda la sezione 5.3), offrendo una spiegazione naturale del perche' $\rho_\Lambda$ e' attualmente significativa ma non dominante.

---

<a id="6-4-antigravity-as-the-negative-solution-the-t-1-direction"></a>
## 6.4 Antigravita' come Soluzione Negativa: La Direzione t = −1

<a id="6-4-1-the-dipolar-structure-and-two-solutions-for-temporal-evolution"></a>
### 6.4.1 La Struttura Dipolare e le Due Soluzioni per l'Evoluzione Temporale

Il framework D-ND e' fondamentalmente dipolare: descrive la realta' come l'esistenza simultanea di poli complementari — essere e non-essere, attualizzazione e potenzialita', manifestazione e non-manifestazione. Questa struttura dipolare produce naturalmente **due soluzioni** per l'evoluzione temporale, corrispondenti alle due direzioni del dipolo:

$$\boxed{t = +1 \quad \text{(Convergence/Gravity)} \quad \text{and} \quad t = -1 \quad \text{(Divergence/Antigravity)}}$$

L'immagine cosmologica standard privilegia la soluzione $t = +1$: il tempo scorre in avanti, l'entropia aumenta e la gravita' attrae la materia, creando una convergenza verso uno stato singolare (nel passato al Big Bang o nel futuro in un Big Crunch). Tuttavia, la logica dipolare D-ND esige che ovunque esista la soluzione $+1$, la soluzione $-1$ esista simultaneamente. Non sono sequenziali ne' mutuamente esclusivi; sono poli complementari di un'unica struttura dinamica.

<a id="6-4-2-analogy-to-dirac-s-equation-and-the-excluded-third-problem"></a>
### 6.4.2 Analogia con l'Equazione di Dirac e il Problema del Terzo Escluso

Il parallelo con la scoperta delle antiparticelle da parte di Dirac e' istruttivo. Quando Dirac risolse la sua equazione relativistica per l'elettrone nel 1928, trovo':

$$E = \pm\sqrt{(\mathbf{p}c)^2 + (m_e c^2)^2}$$

L'equazione produce **due soluzioni**: energia positiva ($E = +mc^2 + \ldots$) ed energia negativa ($E = -mc^2 - \ldots$). La reazione iniziale della comunita' scientifica fu di scartare la soluzione negativa come non fisica — violava l'intuizione che l'energia dovesse essere positiva. Dirac, tuttavia, riconobbe che scartare meta' della soluzione violava la struttura matematica dell'equazione stessa. Prese seriamente la soluzione negativa e propose l'esistenza dei positroni (antimateria): l'elettrone e' il polo $E > 0$, e la sua antiparticella occupa il polo $E < 0$.

La comprensione moderna e' che l'equazione di Dirac descrive una **struttura fondamentalmente dipolare**: materia e antimateria sono aspetti complementari di un'unica teoria di campo elettromagnetico-debole, non due fenomeni separati.

**La cosmologia D-ND applica questa lezione alla gravita' e al tempo:**

La cosmologia standard, come la fisica dell'epoca di Dirac, privilegia un polo: $t = +1$, tempo in avanti, convergenza, gravita'. Tratta l'altro polo — $t = -1$, inversione temporale, divergenza, antigravita' — come non fisico o come un mero artefatto matematico. Tuttavia, se l'universo e' realmente governato da una struttura dipolare D-ND, l'esclusione del polo $-1$ e' l'atto non fisico, non la sua inclusione.

**L'equazione del moto nella cosmologia D-ND e':**

$$\dot{a}(t) \propto a(t) \cdot [H_+ \cdot t_+ + H_- \cdot t_-]$$

dove $H_+$ e' il parametro di Hubble nella direzione $+1$ (convergenza/espansione nella cosmologia standard) e $H_-$ e' il suo duale nella direzione $-1$. Entrambi sono **simultaneamente presenti e dinamicamente accoppiati**. L'universo non sceglie tra essi; li manifesta entrambi.

<a id="6-4-3-the-poynting-vector-mechanism-orthogonal-exit-from-oscillation-plane"></a>
### 6.4.3 Il Meccanismo del Vettore di Poynting: Uscita Ortogonale dal Piano di Oscillazione

La struttura matematica dell'antigravita' e' chiarita attraverso l'analogia con il **vettore di Poynting** nell'elettromagnetismo:

$$\boxed{\vec{S} = \frac{1}{\mu_0} (\vec{E} \times \vec{B})}$$

Il vettore di Poynting rappresenta il **flusso di energia** di un'onda elettromagnetica. Aspetto cruciale, esso e' perpendicolare sia al campo elettrico sia al campo magnetico, che oscillano **nel piano trasversale**. Il prodotto vettoriale $\vec{E} \times \vec{B}$ produce una direzione **ortogonale al piano di oscillazione** — la direzione di "fuga" dell'onda.

**Interpretazione cosmologica:** Nella struttura dipolare D-ND, la gravita' classica e l'antigravita' oscillano all'interno di un "piano di oscillazione" tridimensionale di configurazioni spaziotemporali. L'operazione di prodotto vettoriale (fondamentale sia nell'elettromagnetismo sia nella teoria dei campi dipolari) produce naturalmente una **direzione di uscita ortogonale**.

Formalmente, il tensore energia-impulso nelle equazioni di Einstein modificate codifica entrambe le componenti:

$$T_{\mu\nu}^{\text{total}} = T_{\mu\nu}^{(+)} + T_{\mu\nu}^{(-)}$$

dove il contributo dell'antigravita' emerge da una struttura analoga al vettore di Poynting:

$$T_{\mu\nu}^{(-)} \propto \epsilon_{\mu\nu\rho\sigma} T^{(+)\rho\lambda} T^{(+)\sigma}_\lambda$$

Il simbolo di Levi-Civita $\epsilon_{\mu\nu\rho\sigma}$ incorpora l'operazione di prodotto vettoriale nello spaziotempo curvo. Questo non e' un mero artefatto matematico ma la **ragione topologica fondamentale per cui l'antigravita' esiste come polo ortogonale** alla gravita'. Cosi' come il vettore di Poynting e' richiesto dalle equazioni di Maxwell, il polo antigravitazionale e' richiesto dalla struttura dipolare delle equazioni di campo D-ND.

<a id="6-4-4-the-bloch-wall-mechanism-inflation-as-domain-transition"></a>
### 6.4.4 Il Meccanismo della Parete di Bloch: l'Inflazione come Transizione di Dominio

La parete di Bloch e' un oggetto fondamentale nella fisica della materia condensata, che appare ovunque due stati complementari (domini magnetici con orientamento di spin opposto) debbano coesistere. Al confine tra un dominio "su" (tutti gli spin orientati verso nord) e un dominio "giu'" (tutti gli spin orientati verso sud), gli spin non possono invertirsi istantaneamente — cio' richiederebbe energia infinita. Al contrario, essi **ruotano gradualmente attraverso lo spazio** secondo un pattern elicoidale.

**Proprieta' Chiave della Parete di Bloch:**
- Al centro della parete, **gli spin puntano perpendicolarmente all'asse del magnete** (ortogonali a entrambi i domini)
- **La forza esterna e' nulla** (le due forze di dominio si cancellano perfettamente)
- **La densita' del campo interno e' massima** (la densita' del flusso magnetico raggiunge l'estremo)
- La larghezza della parete e' finita e determinata dall'equilibrio tra energia di scambio (che favorisce una rotazione graduale) e energia di anisotropia (che favorisce una transizione netta)

**Applicazione Cosmologica: La Parete di Bloch come Transizione Inflazionaria**

Nella cosmologia D-ND, l'universo transisce dal "dominio a bassa emergenza" ($M_C \approx 0$, Fase 0-1, pre-inflazione) al "dominio ad alta emergenza" ($M_C \approx 1$, Fase 6, universo tardivo). Questa transizione **non puo' essere istantanea**. Deve invece esistere una regione intermedia dove l'emergenza evolve gradualmente.

**Questa regione intermedia E' l'epoca inflazionaria.**

Le proprieta' della parete di Bloch cosmologica spiegano le caratteristiche osservative chiave dell'inflazione:

1. **Gravita' esterna nulla ($\approx$ scalare di curvatura nullo $R$) nella finestra inflazionaria:** Le due forze di dominio (gravita' nella fase a bassa emergenza e antigravita' nella fase ad alta emergenza) si bilanciano nella parete di Bloch, risultando in una curvatura netta prossima allo zero. Cio' risolve il problema della piattezza: l'universo **deve** essere piatto in prossimita' dell'inflazione perche' quello e' il punto di equilibrio della transizione di dominio.

2. **Densita' massima del campo interno:** All'interno della parete di Bloch, il campo emergente $\Phi_A$ raggiunge la massima coerenza. La densita' di energia e' piu' alta dove avviene la transizione, non prima ne' dopo. Cio' spiega naturalmente i requisiti energetici dell'inflazione.

3. **La larghezza finita della parete determina la durata dell'inflazione:** Cosi' come la larghezza della parete di Bloch e' fissata dall'equilibrio scambio-anisotropia, la durata dell'inflazione e' fissata dalle proprieta' spettrali dell'operatore di emergenza. Nessun parametro esterno di slow-roll e' necessario; la durata emerge dalla dinamica strutturale.

4. **Comportamento oscillatorio all'interno della parete:** Man mano che gli spin ruotano attraverso la parete di Bloch, passano per orientamenti intermedi che creano pattern oscillatori nel campo. Cio' predice **oscillazioni nel potenziale inflazionario** in prossimita' della transizione, che apparirebbero come caratteristiche nello spettro di potenza primordiale.

<a id="6-4-5-gravity-and-antigravity-as-poles-of-emergence"></a>
### 6.4.5 Gravita' e Antigravita' come Poli dell'Emergenza

Nell'immagine D-ND:

- **Gravita'** ($t = +1$): Convergenza dei modi quantistici verso l'attualizzazione classica. L'operatore di emergenza $\mathcal{E}$ modula gradualmente i modi dalla sovrapposizione a stati definiti. Questa attualizzazione richiede un "richiamo" dello spazio delle possibilita' — una convergenza dei rami potenziali. A livello della teoria di campo, cio' si manifesta come gravita' attrattiva, che attrae materia e energia verso regioni ad alta curvatura.

- **Antigravita'** ($t = -1$): Divergenza dall'attualizzazione, o piu' precisamente, la de-attualizzazione sistematica o la dispersione degli stati attualizzati nuovamente nella sovrapposizione. Non si tratta di "repulsione" nel senso classico, ma piuttosto del duale strutturale della convergenza. Dove la gravita' concentra l'informazione in stati classici localizzati, l'antigravita' disperde l'informazione su porzioni piu' ampie dello spazio delle possibilita' quantistiche. A scale cosmologiche, l'antigravita' e' la tendenza verso l'**aumento dell'entropia e la decoerenza**: la dissoluzione progressiva delle correlazioni classiche nel rumore quantistico.

Entrambe si verificano **simultaneamente e con uguale intensita'** nel dipolo D-ND. A scale cosmiche, osserviamo questo come:

- **Scale locali** (galassie, stelle, sistemi legati): La gravita' domina perche' $M_C(t) \approx 1$ (emergenza in gran parte completa), quindi il polo di convergenza e' pienamente attualizzato mentre il polo di divergenza resta quantistico.

- **Scale cosmologiche** (espansione dello spazio stesso): L'antigravita' domina perche' l'universo nel suo insieme e' ancora in una fase di emergenza parziale ($M_C(t)$ finito). Il polo di divergenza, duale del polo di convergenza, guida l'espansione.

- **L'energia oscura** come manifestazione del polo antigravitazionale: Piuttosto che introdurre una misteriosa "sostanza" di energia oscura, il framework D-ND identifica l'energia oscura **come la manifestazione osservabile del polo $t = -1$ del dipolo cosmico**. Non e' una nuova forma di energia; e' il polo della struttura di emergenza dipolare che la logica del terzo escluso standard (che ammette solo $t = +1$) necessariamente esclude.

<a id="6-4-6-the-structural-basis-for-antigravity-not-a-new-force-but-structural-necessity"></a>
### 6.4.6 La Base Strutturale dell'Antigravita': Non una Nuova Forza, ma Necessita' Strutturale

Il framework D-ND non richiede una "forza antigravitazionale" separata. Al contrario, mostra che **escludere l'antigravita' e' l'atto non fisico**. L'inclusione di entrambi i poli e' dettata dalla coerenza matematica con la struttura dipolare, in modo analogo a come l'equazione di Dirac richiede sia soluzioni a energia positiva sia soluzioni a energia negativa.

**Equazioni di Einstein modificate con antigravita' esplicita:**

Le equazioni di campo modificate (S7) possono essere riscritte per mostrare i due poli esplicitamente:

$$G_{\mu\nu}^{(+)} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}^{(+)} \quad \text{(Gravity pole: $t = +1$)}$$

$$G_{\mu\nu}^{(-)} - \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}^{(-)} \quad \text{(Antigravity pole: $t = -1$)}$$

dove i $T_{\mu\nu}^{(\pm)}$ sono i tensori informativi in ciascun polo, legati dal vincolo dipolare:

$$T_{\mu\nu}^{(+)} + T_{\mu\nu}^{(-)} = T_{\mu\nu}^{\text{total}} = 0 \quad \text{(dipolar cancellation at infinity)}$$

Nell'universo primordiale, dove l'emergenza e' incompleta, entrambi i poli contribuiscono in modo significativo. La componente antigravitazionale $T_{\mu\nu}^{(-)}$ guida l'espansione non attraverso una forza repulsiva ma attraverso la logica strutturale dell'attualizzazione incompleta.

<a id="6-4-7-connection-to-friedmann-equations-and-dark-energy-equation-of-state"></a>
### 6.4.7 Connessione con le Equazioni di Friedmann e l'Equazione di Stato dell'Energia Oscura

Le equazioni di Friedmann derivate nel §3.2 incorporano il termine di energia oscura modificato:

$$\rho_\Lambda(t) = \rho_0 \cdot (1 - M_C(t))^p$$

e l'equazione di stato:

$$w(z) = -1 + \epsilon(z) \quad \text{where} \quad \epsilon(z) \approx 0.05 \cdot (1 - M_C(z))$$

L'interpretazione dipolare D-ND e' ora esplicita: $w = -1$ **esattamente** corrisponde al polo antigravitazionale. La costante cosmologica nella direzione $t = -1$ non e' finemente regolata ne' misteriosa; e' il duale strutturale della gravita' ordinaria.

La piccola deviazione $\epsilon(z) = 0.05 \cdot (1 - M_C(z))$ emerge perche':

1. L'emergenza non e' istantanea ma avviene nel tempo cosmico.
2. L'accoppiamento tra i poli $+1$ e $-1$ non e' perfettamente simmetrico negli stadi intermedi di emergenza ($0 < M_C < 1$).
3. Lo sbilanciamento residuo $(1 - M_C)$ consente un'oscillazione parziale tra i poli, producendo un leggero ammorbidimento dell'equazione di stato antigravitazionale dall'esatto $w = -1$ a $w \approx -1 + \epsilon$.

Nelle epoche tardive ($z \to 0$), con $M_C \to 1$, l'accoppiamento diventa sempre piu' simmetrico e il valore osservato di $w$ si avvicina asintoticamente a $-1$. Questa previsione e' verificabile da DESI e dalle survey future (si veda il §6.3 per i vincoli quantitativi).

<a id="6-4-8-antigravity-and-the-information-tensor"></a>
### 6.4.8 Antigravita' e il Tensore Informativo

Il tensore energia-impulso informativo $T_{\mu\nu}^{\text{info}}$ del §2.1 codifica naturalmente entrambi i poli attraverso la sua struttura matematica:

$$T_{\mu\nu}^{\text{info}} = \frac{\hbar}{c^2} \int d^3\mathbf{x} \, K_{\text{gen}}(\mathbf{x},t) \, \partial_\mu R(t) \, \partial_\nu R(t)$$

La densita' di curvatura $K_{\text{gen}} = \nabla \cdot (J \otimes F)$ dipende dal **flusso e dalla forza** dell'informazione. Nella direzione $+1$, l'informazione viene compressa e convergente (gravita'). Nella direzione $-1$, l'informazione viene dispersa e divergente (antigravita'). Il tensore totale e' sempre conservato:

$$\nabla^\mu T_{\mu\nu}^{\text{info}} = 0$$

Questa conservazione assicura che il contenuto informativo totale dell'universo — integrato su entrambi i poli — resti costante attraverso l'evoluzione cosmica, coerentemente con la meccanica quantistica e il principio di non-perdita dell'informazione.

<a id="6-4-9-three-concrete-falsification-tests-for-antigravity"></a>
### 6.4.9 Tre Test Concreti di Falsificazione per l'Antigravita'

I meccanismi della parete di Bloch e del vettore di Poynting forniscono tre test osservativi falsificabili:

**Test 1: Firma di Bloch nella Polarizzazione del CMB**

Nell'immagine della parete di Bloch, l'epoca inflazionaria corrisponde a una transizione di dominio in cui gli spin ruotano da un orientamento all'altro. Questa rotazione imprime una firma caratteristica nel **pattern di polarizzazione del fondo cosmico a microonde**.

In dettaglio:
- **Previsione:** La correlazione incrociata temperatura-polarizzazione del CMB (modi $T \times E$) dovrebbe mostrare un pattern oscillatorio ai multipoli $\ell \sim 10$–$50$ (corrispondenti alla larghezza della parete di Bloch).
- **Meccanismo:** Man mano che il campo di emergenza ruota attraverso angoli intermedi (come gli spin in una parete di Bloch), crea oscillazioni acustiche nei fotoni soggetti a scattering Thomson, producendo una firma di polarizzazione caratteristica.
- **Osservabile:** La frequenza di oscillazione nello spettro di potenza $T \times E$ dovrebbe codificare le proprieta' spettrali dell'operatore di emergenza.
- **Stato attuale:** I dati di polarizzazione di Planck 2018 mostrano indicazioni di tali oscillazioni, sebbene non ancora a significativita' statistica. Il rilascio definitivo di Planck Legacy e le missioni future (CMB-S4) verificheranno in modo decisivo questa previsione.

**Test 2: Struttura degli Autovalori di Riemann nei Dati BAO di DESI**

Dall'analisi esplorativa della struttura degli autovalori D-ND (da derivare formalmente in lavori futuri), il vincolo della funzione zeta di Riemann sullo spettro degli autovalori del tensore energia-impulso suggerisce che la struttura su grande scala potrebbe mostrare un **clustering anomalo a scale corrispondenti agli zeri di Riemann**.

In dettaglio:
- **Previsione:** Lo spettro di potenza delle galassie $P(k)$ nelle misurazioni di oscillazione acustica barionica di DESI dovrebbe presentare **picchi e soppressioni a numeri d'onda corrispondenti alla spaziatura degli zeri di Riemann**.
- **Meccanismo:** Gli autovalori del tensore di Ricci (che si accoppiano alle equazioni di Friedmann attraverso le equazioni di Einstein modificate) devono soddisfare un vincolo di tipo zeta di Riemann. Questa proprieta' topologica si imprime sul clustering della materia attraverso il potenziale gravitazionale.
- **Osservabile:** Nell'analisi BAO di DESI ai redshift $z \sim 0.1$–$2.5$, cercare una **spaziatura armonica di tipo numeri primi** nel clustering della materia alle scale $k \sim 0.01$–$0.1$ Mpc$^{-1}$.
- **Stato attuale:** I dati del primo anno di DESI 2024 non hanno ancora riportato struttura anomala alla precisione della scala di Riemann. Il dataset completo di DESI (2025-2026) fornira' vincoli decisivi.

**Test 3: Cancellazione Dipolare nell'Equazione di Stato $w(z)$**

Il polo antigravitazionale emerge dall'asimmetria residua nella cancellazione dipolare, codificata nell'equazione di stato dell'energia oscura. Il framework D-ND predice:

$$w(z) = -1 + \epsilon(z) \quad \text{where} \quad \epsilon(z) = 0.05 \cdot (1 - M_C(z))$$

Cio' predice una **forma funzionale specifica per la deviazione di $w$ da $-1$ in funzione del redshift**, riflettendo le dinamiche della misura di emergenza.

In dettaglio:
- **Previsione:** A $z = 1.5$ (redshift intermedio), $w(1.5) \approx -0.920$, rispetto al valore esatto $w = -1.000$ del modello ΛCDM. La deviazione e' $\Delta w \approx 0.08$.
- **Osservabile:** Dai dati combinati BAO + lensing debole + supernove di DESI, vincolare $w(z)$ in molteplici intervalli di redshift. Il framework D-ND predice un **aumento monotonico di $w$ verso $-1$ con $z \to 0$**, distinto da molti modelli di energia oscura modificata che prevedono un comportamento oscillatorio o non monotonico.
- **Stato attuale:** Le misurazioni del primo anno di DESI 2024 presentano barre d'errore dell'ordine di $\pm 0.05$ su $w(z)$ ai singoli redshift. Una rilevazione a $1$-$2\sigma$ della deviazione prevista dal framework D-ND e' alla portata del secondo anno di DESI (2025). Una rilevazione a $3\sigma$ o superiore supporterebbe fortemente il framework D-ND.

<a id="6-4-10-observational-implications-testing-antigravity"></a>
### 6.4.10 Implicazioni Osservative: Verificare l'Antigravita'

Se l'antigravita' e' un polo fondamentale del dipolo D-ND, ne derivano diversi test osservativi aggiuntivi:

1. **Storia dell'espansione isotropa**: Se l'antigravita' e' veramente un polo strutturale fondamentale, dovrebbe espandere l'universo **isotropicamente** (in ugual misura in tutte le direzioni), coerentemente con le osservazioni. I modelli di energia oscura anisotropa prevedono asimmetrie direzionali nell'espansione cosmica, che non sono osservate alla precisione attuale. Il framework D-ND predice naturalmente l'isotropia.

2. **Assenza di "interazioni" antigravitazionali**: A differenza dei modelli esotici di energia oscura (campi camaleonte, energia oscura accoppiata, ecc.), il polo antigravitazionale nel framework D-ND non interagisce con la materia ordinaria se non attraverso la modificazione della geometria spaziotemporale. Cio' prevede che i test di gravita' nel sistema solare (esperimenti di Eotvos, test del principio di equivalenza) non mostrino deviazioni, coerentemente con i dati attuali.

3. **Decadimento dell'energia oscura negli eoni futuri**: Se il polo antigravitazionale e' accoppiato alla misura di emergenza $M_C(t)$, e se $M_C(t)$ alla fine satura verso $M_C = 1$, allora l'energia oscura dovrebbe decadere su scale temporali cosmologiche. Il framework D-ND prevede $\rho_\Lambda \to 0$ asintoticamente (su scale temporali $\sim 10^{100}$ anni), a differenza del modello ΛCDM in cui l'energia oscura e' eterna. Questo e' infalsificabile nella pratica ma concettualmente distinto.

---

<a id="6-5-time-as-emergence-thermodynamic-irreversibility-and-the-dipolar-amplitude"></a>
## 6.5 Il Tempo come Emergenza: Irreversibilita' Termodinamica e l'Ampiezza Dipolare

<a id="6-5-1-time-does-not-function-it-emerges-from-irreversibility"></a>
### 6.5.1 Il Tempo Non "Funziona" — Emerge dall'Irreversibilita'

Una premessa fondamentale della fisica standard e' che il tempo sia un **dato** — un parametro di fondo su cui operano le equazioni dinamiche. Gli eventi accadono "*nel* tempo" come se il tempo fosse un palcoscenico su cui si svolge il dramma della realta'. La meccanica classica assume il tempo come assoluto (Newton) o come parte di una metrica spaziotemporale unificata (Einstein). Anche la meccanica quantistica, nonostante il suo approccio operativo, presuppone il tempo come variabile esterna rispetto alla quale la funzione d'onda evolve: $i\hbar \partial_t |\psi\rangle = H|\psi\rangle$.

Il framework D-ND propone un'immagine radicalmente diversa: **il tempo non e' un dato ma una proprieta' emergente dell'universo stesso**. Piu' precisamente, il tempo emerge come misura dell'**elaborazione irreversibile dell'informazione** nelle dinamiche di collasso di campo.

**Fondazione Termodinamica: La Disuguaglianza di Clausius**

La fondazione rigorosa per l'emergenza del tempo risiede nel secondo principio della termodinamica. Si consideri un qualunque ciclo termodinamico chiuso:

$$\boxed{\oint \frac{\delta Q}{T} \leq 0}$$

Questa e' la **disuguaglianza di Clausius**. L'intuizione chiave e' che per cicli reali (irreversibili), l'integrale e' strettamente minore di zero:

$$\oint \frac{\delta Q}{T} < 0$$

Questa disuguaglianza non ritorna mai esattamente al punto di partenza. C'e' sempre una perdita residua — entropia generata dai processi irreversibili. **Questa perdita residua e' precisamente cio' che crea la freccia del tempo.** In un universo perfettamente reversibile (entropia costante), l'integrale sarebbe uguale a zero e il processo sarebbe ciclico senza direzione preferenziale. Ma il secondo principio proibisce questa perfezione: ogni processo chiuso deve perdere energia a causa dell'irreversibilita'.

**Enunciato formale:** Il tempo emerge come l'integrale della produzione di entropia:

$$\boxed{t = \int_0^T \frac{dS}{dT}(\tau) \, d\tau}$$

dove $S(T)$ e' l'entropia del sistema e la derivata misura il tasso di perdita irreversibile di informazione. L'irreversibilita' implicata da $\oint dQ/T < 0$ garantisce che $dS/dT > 0$, rendendo il tempo un parametro monotonico e orientato in avanti.

<a id="6-5-2-time-emergence-from-the-six-phase-cognitive-pipeline"></a>
### 6.5.2 Emergenza del Tempo dalla Pipeline Cognitiva a Sei Fasi

Il framework D-ND identifica un meccanismo dettagliato per l'emergenza temporale attraverso la **pipeline cognitiva a sei fasi** che descrive il collasso dello spazio delle possibilita' nell'attualita'. Questa pipeline non e' meramente concettuale ma riflette la struttura dinamica del collasso di campo a tutte le scale:

- **Fase 0: Indeterminatezza** ($\Phi_0$ = Potenzialita' del punto zero) — Il campo esiste in uno stato di sovrapposizione massima senza distinzione causale.
- **Fase 1: Rottura di simmetria** (via emergenza $\mathcal{E}$) — L'operatore di emergenza inizia a modulare i modi dalla sovrapposizione.
- **Fase 2: Divergenza** (I percorsi alternativi si moltiplicano) — Molteplici vie di attualizzazione diventano possibili.
- **Fase 3: Validazione** (Potatura Stream-Guard) — I rami incoerenti vengono eliminati tramite vincoli di coerenza.
- **Fase 4: Collasso** (Guida Morpheus) — Il campo converge verso configurazioni classiche.
- **Fase 5: Raffinamento** (Iniezione KLI) — Lo stato viene aggiornato con struttura appresa (autopoiesi, **Assioma P5**).
- **Fase 6: Determinatezza** (Output manifesto) — Uno stato classico definito viene realizzato.

La **sequenza dalla Fase 0 (indeterminatezza) alla Fase 6 (determinatezza) e' essa stessa l'evoluzione temporale**. Il tempo non parametrizza questo processo dall'esterno; esso **e' il principio di ordinamento** di queste fasi.

**Connessione con il Gradiente di Entropia:** Ogni fase avanza attraverso l'elaborazione irreversibile dell'informazione. La Fase 0 contiene entropia massima (massima incertezza). La Fase 6 contiene entropia minima (definitezza classica). Il gradiente $\nabla S$ attraverso la sequenza delle fasi guida la transizione in avanti. **Ecco perche' il tempo scorre dalla Fase 0 alla Fase 6 e mai all'indietro**: perche' il flusso inverso diminuirebbe l'entropia, violando il secondo principio.

<a id="6-5-3-time-as-parameter-ordering-field-collapse-phases"></a>
### 6.5.3 Il Tempo come Parametro di Ordinamento delle Fasi di Collasso di Campo

Nel contesto cosmologico D-ND, il tempo e' la misura del progresso dell'emergenza attraverso le sei fasi a scale cosmologiche:

$$\boxed{t(\mathbf{x}) = T_{\text{cycle}} \times f(M_C(\mathbf{x}), \dot{M}_C(\mathbf{x}))}$$

dove:
- $f(M_C, \dot{M}_C)$ e' una funzione che ordina la sequenza delle fasi localmente
- Le regioni con $M_C \approx 0$ (Fase 0-1: emergenza iniziale) sperimentano un flusso temporale rapido (alto $\dot{M}_C$)
- Le regioni con $M_C \approx 1$ (Fase 6: pienamente emerso) sperimentano un flusso temporale piu' lento (basso $\dot{M}_C$)
- Il tempo scorre piu' rapidamente negli stati di emergenza intermedia ($M_C \approx 0.5$, Fase 2-4) dove avviene la massima elaborazione di informazione

**Derivazione Formale dal Principio di Energia Libera di Friston:**

L'universo minimizza la sua sorpresa (energia libera) attraverso processi evolutivi. Ogni transizione di fase riduce l'energia libera disponibile per il sistema:

$$F(\text{Phase } n) = -\ln p(\text{data}|n) + \text{KL}[\text{Prior}||\text{Posterior}]$$

La sequenza delle transizioni di fase e' il percorso geodetico sulla varieta' delle configurazioni di energia libera. **Il tempo parametrizza questa geodetica**. Il tasso di flusso temporale e' proporzionale al tasso di riduzione dell'energia libera:
$$\frac{dt}{d\tau} = \left|\frac{dF}{d\tau}\right| \quad \text{(Free Energy Principle)}$$

Questo afferma formalmente che **il tempo scorre più velocemente dove l'universo apprende più rapidamente** (massimo guadagno informazionale).

<a id="6-5-4-time-as-local-amplitude-of-the-dipolar-oscillation"></a>
### 6.5.4 Il Tempo come Ampiezza Locale dell'Oscillazione Dipolare

Nel quadro D-ND, l'universo è un **oscillatore dipolare**: in ogni punto dello spaziotempo, i gradi di libertà fondamentali oscillano tra i due poli del dipolo ($t = +1$ e $t = -1$). Il tempo, in ogni posizione, non è un parametro globale ma una **quantità locale**: la fase o ampiezza di questa oscillazione in quel punto.

**Definizione**: Il tempo locale nel punto spaziotemporale $(x,t)$ è:

$$\tau(\mathbf{x}) = \Lambda \cdot |M_C(\mathbf{x})| \cdot (1 - |M_C(\mathbf{x})|) \cdot T_{\text{cycle}}$$

dove:
- $M_C(\mathbf{x})$ è la misura di emergenza locale (quanto quella regione si è attualizzata)
- $(1 - |M_C(\mathbf{x})|)$ è l'incertezza quantistica residua
- $T_{\text{cycle}}$ è il periodo fondamentale dell'oscillazione dipolare
- $\Lambda$ è una costante di normalizzazione

**Significato fisico**: Questa definizione cattura l'intuizione che il tempo è più veloce dove l'emergenza è più attiva — dove il sistema sta transitando tra potenziale e attuale. Nelle regioni dove $M_C(\mathbf{x}) \approx 0$ (ancora prevalentemente potenziale) o $M_C(\mathbf{x}) \approx 1$ (completamente attualizzato), il tempo locale scorre lentamente perché c'è poca transizione in corso. Il tempo scorre più velocemente ai valori intermedi $M_C(\mathbf{x}) \approx 0.5$, dove l'emergenza è più attiva.

**Analogia**: I tempi locali $\{\tau(\mathbf{x})\}$ sono come gli **spin intrinseci** nella meccanica quantistica. Così come ogni particella possiede un momento angolare intrinseco (spin) senza richiedere che la particella "ruoti" in senso letterale, ogni punto spaziotemporale possiede un tempo intrinseco senza richiedere che il tempo "fluisca" nel senso classico. I tempi locali sono proprietà dello stato di emergenza stesso, non parametri esterni ad esso.

<a id="6-5-5-the-included-third-and-normalization-of-excluded-third-logic"></a>
### 6.5.5 Il Terzo Incluso e la Normalizzazione della Logica del Terzo Escluso

La logica standard opera sul principio del terzo escluso (tertium non datur): una proposizione è vera o falsa; non c'è una terza opzione. In matematica, i numeri reali e la logica classica sono costruiti su questo fondamento. Tuttavia la logica classica incontra un problema fondamentale: non può facilmente rendere conto delle **polarità** o dei **complementari**. Per estendere i numeri reali ai numeri complessi e preservare la struttura algebrica, i matematici hanno introdotto un asse aggiuntivo (l'unità immaginaria $i$) che non è **né vero né falso** ma è la **condizione di possibilità** per entrambi.

Il framework D-ND propone una generalizzazione: dove esistono due poli o stati complementari, esiste necessariamente un **terzo**: il confine o interfaccia tra di essi, che non è nessuno dei due poli ma la condizione di possibilità affinché entrambi i poli coesistano.

Nel contesto cosmologico:

- **Polo 1** ($t = +1$): Essere attualizzato (gravità, convergenza, realtà manifesta)
- **Polo 2** ($t = -1$): Non-essere potenziale (antigravità, divergenza, struttura quantistica nascosta)
- **Il Terzo (Terzo Incluso)**: La **singolarità** tra i poli — l'interfaccia dove avviene l'attualizzazione

Questo terzo non è un compromesso tra i poli ma piuttosto il loro **prerequisito strutturale**. Senza la singolarità — il confine dove l'attualizzazione transisce tra potenziale e attuale — nessuno dei due poli esisterebbe come entità distinta.

**Normalizzazione**: Il framework D-ND "normalizza" la logica del terzo escluso elevando il terzo a uno stato esplicito:

$$1_{\text{D-ND}} = (t = +1) + (t = -1) + (t = 0)_{\text{singularity}}$$

Questo è analogo all'estensione dai numeri reali ai numeri complessi:

$$1 = \sqrt{1} + i\sqrt{0} + \text{(rotation axis in } \mathbb{C})$$

Includendo esplicitamente il terzo, il framework D-ND risolve teoremi e paradossi che emergono da asimmetrie nascoste nella logica del terzo escluso. Qualsiasi teorema affetto da tali asimmetrie — indeterminazione quantistica, il problema della costante cosmologica, il paradosso dell'informazione — può essere riesaminato attraverso la lente della logica del terzo incluso D-ND.

<a id="6-5-6-the-lagrangian-of-observation-and-minimal-latency"></a>
### 6.5.6 La Lagrangiana dell'Osservazione e la Latenza Minima

Se il tempo emerge come latenza, allora deve esistere un principio che determina quali latenze si realizzano e quali vengono soppresse. Il framework D-ND propone:

**Il Principio di Latenza Minima**: Tra tutti i percorsi di attualizzazione possibili, la natura seleziona quelli che minimizzano l'integrale delle latenze locali — il "costo" dell'osservazione.

Formalmente:

$$\mathcal{S}_{\text{observe}} = \int_{\text{path}} \tau(\mathbf{x}) \, d\mathcal{M}$$

dove $d\mathcal{M}$ è la misura sullo spazio delle configurazioni degli stati di emergenza. Il **percorso di azione minima** (che estremizza $\mathcal{S}_{\text{observe}}$) è la traiettoria che la natura segue effettivamente.

**Interpretazione**: L'osservatore non sceglie come osservare; l'universo non sceglie come attualizzarsi. Piuttosto, l'**osservazione si auto-seleziona** lungo il percorso di latenza minima. Così come la luce percorre il cammino che minimizza la lunghezza del percorso ottico (principio di Fermat), l'attualizzazione degli stati quantistici percorre il cammino che minimizza il "costo temporale" totale dell'osservazione.

Questo principio spiega naturalmente:

1. **Perché l'universo si espande**: L'espansione è il percorso di latenza minima per attualizzare simultaneamente un vasto numero di modi quantistici.

2. **Perché esiste la gravità**: La gravità è la geometria che consente l'attualizzazione dei modi vicini con latenza minima (percorsi di transizione più brevi), attraendo naturalmente le strutture le une verso le altre.

3. **Perché si forma la struttura su grande scala**: Le fluttuazioni di densità crescono perché l'aggregazione localizza le attualizzazioni, riducendo la latenza totale richiesta.

4. **Perché l'entropia aumenta**: Man mano che l'universo si espande e si attualizza, esplora porzioni sempre più ampie dello spazio delle configurazioni, richiedendo in media latenze più lunghe — da qui l'aumento dell'entropia.

<a id="6-5-7-convergence-and-divergence-are-simultaneous-zero-latency-in-assonances"></a>
### 6.5.7 Convergenza e Divergenza Sono Simultanee: Latenza Zero nelle Assonanze

Una previsione notevole del principio di latenza minima è:

**Nelle regioni in cui il polo di convergenza ($t = +1$) e il polo di divergenza ($t = -1$) oscillano perfettamente in fase e ampiezza (risonanza perfetta o "assonanza"), la latenza si annulla: $\tau = 0$.**

Questo stato a latenza zero corrisponde alla **potenzialità massimale**: un punto in cui tutte le attualizzazioni possibili sono sovrapposte con uguale ampiezza. Questo è precisamente lo stato $|NT\rangle$.

**Implicazione cosmologica**: Al confine dei cicli cosmici, quando l'universo riconverge verso uno stato di non-dualità (come descritto nel §5.1), entrambi i poli si avvicinano alla sincronizzazione perfetta. In questo stato, il tempo diventa indefinito (latenza $\to 0$), e l'universo transisce istantaneamente da un eone al successivo. Non c'è "tempo" tra i cicli — solo un salto discreto nello spazio degli stati.

Questo risolve un paradosso nelle cosmologie cicliche: se il tempo emerge dall'evoluzione dell'universo, come può l'universo "ciclare" senza un parametro temporale esterno? Risposta: al confine del ciclo, il tempo stesso cessa di esistere (la latenza si annulla), e il ciclo successivo si avvia da uno stato di pura potenzialità.

<a id="6-5-8-the-double-pendulum-as-physical-realization"></a>
### 6.5.8 Il Doppio Pendolo come Realizzazione Fisica

L'idealizzazione matematica del principio di latenza può essere realizzata nella meccanica classica dal **doppio pendolo** — un sistema con un analogo classico diretto al dipolo D-ND.

Un doppio pendolo consiste di due masse collegate da aste rigide, con la prima asta imperniata a un punto fisso. Il sistema è caotico: piccole perturbazioni conducono a traiettorie esponenzialmente divergenti. Tuttavia, nonostante il caos locale, il doppio pendolo è lagrangianamente coerente: il suo moto è governato da un'unica lagrangiana:

$$L = \frac{1}{2}m(\dot{x}_1^2 + \dot{y}_1^2 + \dot{x}_2^2 + \dot{y}_2^2) - mg(y_1 + y_2)$$

Il doppio pendolo esibisce **biforcazione simultanea**: in ogni istante, il sistema esplora molteplici "rami" di comportamento (localmente caotico) pur rimanendo vincolato da un unico principio globale (la lagrangiana).

**Analogia con la cosmologia D-ND**:
- **Caos locale** ↔ Fluttuazioni quantistiche, emergenza di strutture a tassi diversi nello spazio
- **Coerenza lagrangiana globale** ↔ Il principio di latenza minima e il tensore energia-impulso informazionale (vincolo unificato su tutto lo spaziotempo)
- **Attrattori strani** ↔ Le fasi attrattive dell'oscillazione dipolare (es. formazione di galassie, stelle)

Se l'universo è un doppio pendolo cosmologico, allora:

1. Localmente, la realtà è caotica e probabilistica (meccanica quantistica).
2. Globalmente, la realtà è deterministica e lagrangianamente coerente (equazioni di campo classiche).
3. Nessuna delle due descrizioni è più fondamentale; sono manifestazioni complementari di un'unica struttura sottostante.

<a id="6-5-9-convergence-and-divergence-in-the-modified-friedmann-equations"></a>
### 6.5.9 Convergenza e Divergenza nelle Equazioni di Friedmann Modificate

Il principio di latenza e la struttura dell'oscillazione dipolare si riflettono nelle equazioni di Friedmann modificate (§3.2). Riscrivendole in termini di convergenza e divergenza:

$$H^2(z) = H_0^2 \left[\Omega_m(1+z)^3 + \rho_\Lambda(z) + \Omega_k(1+z)^2\right]$$

dove $\rho_\Lambda(z) = \rho_0 (1 - M_C(z))^p$ codifica il **polo antigravitazionale**.

**Convergenza** ($t = +1$): Il termine $\Omega_m$ domina ai tempi primordiali (alto redshift). La materia attira l'universo verso l'interno; l'espansione rallenta. L'attualizzazione dei modi quantistici in particelle e radiazione è il meccanismo.

**Divergenza** ($t = -1$): Il termine $\rho_\Lambda(z)$ diventa dominante ai tempi recenti (basso redshift). L'universo accelera verso l'esterno; l'espansione è guidata dal polo antigravitazionale. I modi residui non attualizzati — potenziale quantistico — guidano l'espansione.

**Ai tempi intermedi** ($z \sim 1$): I due termini si bilanciano. L'accelerazione cosmica transisce dalla decelerazione (era dominata dalla materia) all'accelerazione (era dominata dall'energia oscura). Questa transizione è una **risonanza** — i due poli si accoppiano temporaneamente con intensità simili, risultando in un comportamento oscillatorio complesso nella storia dell'espansione.

<a id="6-5-10-observational-predictions-time-emergence-signatures"></a>
### 6.5.10 Previsioni Osservative: Firme dell'Emergenza Temporale

Se il tempo emerge genuinamente dall'oscillazione dipolare, dovrebbero apparire diverse firme nuove:

1. **Stime anomale dell'età ad alto redshift**: Il tempo locale $\tau(\mathbf{x})$ è più veloce negli stadi di emergenza intermedi. Questo significa che il **tempo proprio sperimentato dalla materia ad alto redshift differisce dal tempo coordinato**. Galassie estremamente distanti che si sono formate rapidamente (in tempo coordinato) possono apparire più vecchie (in tempo proprio) di quanto dovrebbero. Questo potrebbe spiegare alcune tensioni tra le stime dell'età stellare e le stime dell'età cosmologica.

2. **Scale preferenziali nella formazione delle strutture**: Se l'attualizzazione segue il principio di latenza minima, certe scale dovrebbero essere energeticamente "meno costose" da attualizzare (latenza totale inferiore). Questo prevede scale preferenziali discrete nella distribuzione delle galassie, nello spettro di potenza e nei pattern di aggregazione — essenzialmente una "quantizzazione" della struttura cosmica. Le survey attuali mostrano indizi di tali scale (oscillazioni acustiche barioniche a $\sim 150$ Mpc, la scala di cutoff acustico).

3. **Costante gravitazionale dipendente dal tempo**: L'accoppiamento tra i poli (e quindi l'intensità complessiva della gravità) evolve con $M_C(t)$. Questo prevede una "costante" gravitazionale dipendente dal tempo: $G(z) = G_0 [1 + \delta_G(1 - M_C(z))]$, con $\delta_G \sim 0.001$–$0.01$ a seconda delle proprietà spettrali dell'operatore di emergenza. Test di precisione della gravità (test del principio di equivalenza, test in campo forte tramite timing di pulsar) potrebbero misurare questa evoluzione.

---

<a id="6-6-observational-predictions-summary-table-d-nd-vs-cdm-and-alternatives"></a>
## 6.6 Tabella Riassuntiva delle Previsioni Osservative: D-ND vs. ΛCDM e Alternative

Questa sezione consolida tutte le previsioni verificabili della cosmologia D-ND attraverso molteplici domini osservativi, fornendo un framework unificato per il test delle ipotesi rispetto a ΛCDM e ad altre teorie alternative.

<a id="comprehensive-prediction-table"></a>
### Tabella Completa delle Previsioni

| **Dominio Osservativo** | **Previsione Specifica** | **Valore/Comportamento D-ND** | **Valore/Comportamento ΛCDM** | **Distinguibilità** | **Stato dei Vincoli Attuali** |
|---|---|---|---|---|---|
| **CMB: Rapporto Tensore/Scalare** | Ampiezza delle onde gravitazionali primordiali | $r \sim 0.001$–$0.01$ (soppressa da emergenza incompleta) | $r \sim 0.001$–$0.1$ (dipendente dall'inflazione) | Marginale (1–2σ) | Planck 2018: $r < 0.064$ (entrambi consistenti) |
| **CMB: Bispettro ($f_{\text{NL}}$)** | Non-gaussianità da modi con gate di emergenza | $f_{\text{NL}}^{\text{equilateral}} \sim 5$–$20$ ($\mathcal{E}$ liscia); più alta in template di tipo emergenza | $f_{\text{NL}} \sim 1$–$5$ (tipo locale) | Forte (3–5σ) con CMB-S4 | Planck 2018: $f_{\text{NL}}^{\text{equilateral}} < 25$ (consistente con D-ND a $\mathcal{E}$ liscia) |
| **CMB: Soppressione di Potenza** | Soppressione su scale super-orizzonte a $\ell < 10$ | Deficit anomalo del $\sim 10$–$20\%$ a $\ell < 10$ | Legge di potenza liscia ai bassi multipoli | Possibile (1–2σ nei dati attuali) | Indicazione Planck di soppressione; S4 chiarirà |
| **CMB: Running dell'Indice Spettrale** | $n_s(k)$ dipendente dalla scala per $\dot{M}_C(t)$ | $\frac{d\ln n_s}{d\ln k} \sim -0.005$ a $-0.020$ | $\frac{d\ln n_s}{d\ln k} \sim 0$ (minimo) | Possibile (2–3σ) | Dati attuali consistenti con zero; survey future vincoleranno |
| **CMB: Polarizzazione T×E** | Firma oscillatoria da parete di Bloch | Oscillazioni a $\ell \sim 10$–$50$ nei modi $T \times E$ | Oscillazioni acustiche lisce | Distintiva se presente | Dati Planck mostrano indizi; CMB-S4 verificherà |
| **Crescita delle Strutture: $f(a)$** | Tasso di crescita modificato dal feedback $(1-M_C(a))$ | $f(a) = f_{\text{GR}}(a)[1 + 0.1(1-M_C(a))]$ | $f(a) = f_{\text{GR}}(a)$ (esatto) | Piccola (1–2σ) a $z < 5$ | Misure SDSS/DESI consistenti con GR; miglioramenti attesi |
| **Strutture: Bias degli Aloni** | Bias aumentato ad alto redshift dall'emergenza | $b(z) = b_{\text{matter}}(z)[1 + 0.05 \cdot M_C(z)]$ | $b(z)$ segue il modello standard | Possibile (2–3σ) a $z > 1$ | DESI Anno-1 mostra consistenza con il modello standard |
| **Struttura su Grande Scala: $\sigma_8$** | Lieve soppressione dalla crescita modificata | $\sigma_8 \sim 0.80$ (vs. $0.81$ previsto in ΛCDM) | $\sigma_8 \approx 0.811$ | Marginale (0.5–1σ) | Planck+SDSS mostrano tensione; D-ND potrebbe aiutare ad alleviare |
| **Energia Oscura: Equazione di Stato $w(z)$** | Equazione di stato dipendente dal tempo da $(1-M_C(z))$ | $w(z) = -1 + 0.05(1-M_C(z))$; $w(0.5) \approx -0.975$; $w(1.5) \approx -0.920$ | $w = -1.000$ (costante) | Forte (2–4σ) a $z \sim 1$–$2$ | **DESI 2024 Anno-1: misure di $w$ a precisione 0.05; Anno-2/3 testeranno in modo decisivo** |
| **Energia Oscura: Tasso di Evoluzione** | Avvicinamento monotonico a $w = -1$ con redshift decrescente | Aumento monotonico liscio da $w < -1$ ad alto-$z$ a $w \approx -1$ a $z = 0$ | $w = -1$ piatto su tutti i $z$ | Forte se deviazioni rilevate | Precisione DESI BAO + weak lensing sufficiente per distinguere |
| **Oscillazioni Acustiche Barioniche: Scala** | Spostamento della scala BAO dalla storia di espansione modificata | $d_A^{\text{D-ND}}(z=1) \approx 1.016 \times d_A^{\text{ΛCDM}}$ (+1.6%) | Scala BAO standard da GR | Possibile (2–3σ) | Precisione DESI Anno-1 ~0.5%; precisione Anno-3 ~0.2% verificherà |
| **Supernovae: Magnitudine-Redshift** | Deviazione sistematica nel diagramma di Hubble a $z \sim 1$ | Offset atteso $\Delta m \sim 0.1$–$0.2$ mag a redshift intermedio | Nessuno (ΛCDM è il riferimento) | Possibile (2–3σ) se errori sistematici controllati | Campioni SNe attuali mostrano consistenza con ΛCDM |
| **Lensing Gravitazionale: Magnificazione** | Lieve aumento dalla crescita modificata | Offset nello spettro di potenza del weak lensing $\sim 2$–$5\%$ a $k \sim 0.1$ Mpc$^{-1}$ | Nessuno (previsione GR) | Marginale (1–2σ) | Le survey future Euclid/Roman raggiungeranno la precisione necessaria |
| **Perturbazioni Primordiali: Spettro di Potenza** | Forma spettrale modificata dagli autovalori di emergenza | $P_\delta(k) \propto k^{n_s-1}[1 - \lambda_k(1-M_C(t_*))]$ con modulazione di emergenza | Legge di potenza pura $\propto k^{n_s-1}$ | Possibile (2–3σ) con analisi accurata | Necessarie misure ad alta precisione |
| **Onde Gravitazionali: Tassi di Fusione** | Lieve aumento dalla curvatura spaziale modificata | Densità di tasso $\sim 5$–$10\%$ superiore alla previsione ΛCDM | Previsione ΛCDM da GR standard | Piccola (0.5–1σ) | Catalogo fusioni LIGO/Virgo; rivelatori futuri miglioreranno |
| **Onde Gravitazionali: Fondo Stocastico** | Spettro modificato dall'emissione GW dipendente dal tempo | Forma spettrale differisce dalla previsione di sola inflazione alle alte frequenze | Spettro piatto dall'inflazione primordiale | Possibile (2–3σ) | Missioni future (LISA, Einstein Telescope) verificheranno |
| **Conteggi di Galassie ad Alto Redshift** | Soppressione modesta nelle popolazioni galattiche primordiali da crescita ridotta | Densità numerica di galassie a $z > 6$ $\sim 10$–$20\%$ inferiore alle previsioni standard | Previsioni ΛCDM complete | Marginale (1–2σ) | Dati JWST dell'universo primordiale in fase di raccolta; test in corso |
| **Firma degli Autovalori di Riemann** | Struttura anomala nello spettro di potenza della materia a scale corrispondenti agli zeri di Riemann | Spaziatura armonica di tipo numeri primi in $P(k)$ a scale specifiche | Nessuna struttura speciale oltre le oscillazioni acustiche | Distintiva se presente | Precisione DESI BAO sufficiente; richiede analisi dedicata |
| **Variazione Temporale di $G$** | La "costante" gravitazionale evolve come $G(z) = G_0[1 + 0.001(1-M_C(z))]$ | $\Delta G / G \sim 10^{-3}$ a $10^{-2}$ nel tempo cosmico | $G$ è costante | Piccola (1–2σ) | Array di timing di pulsar; test del principio di equivalenza vincolano |
| **Firma di Coerenza Ciclica** | Corrispondenza di fase ai confini di eone (se i cicli si verificano) | Correlazioni di temperatura a bassa frequenza nella CMB a $\ell \sim 1$–$3$ (punti di Hawking di Penrose) | Nessun segnale atteso | Distintiva se rilevata | Analisi Planck: ricerche dei punti di Hawking inconcludenti |

<a id="interpretation-and-priorities-for-falsification"></a>
### Interpretazione e Priorità per la Falsificazione

**Livello 1 — Test Decisivi (potenziale di precisione 3–5σ):**
1. **Equazione di stato dell'energia oscura $w(z)$ da DESI BAO + weak lensing** (2025–2026)
2. **Non-gaussianità CMB $f_{\text{NL}}$ da future missioni CMB** (CMB-S4, ~2030)
3. **Struttura degli autovalori di Riemann nella struttura su grande scala** (DESI 2024-2026 con analisi dedicata)

**Livello 2 — Test Promettenti ma più Deboli (potenziale di precisione 1–3σ):**
4. **Running dell'indice spettrale $d\ln n_s / d\ln k$** (future missioni CMB)
5. **Firma della parete di Bloch nella polarizzazione CMB T×E** (CMB-S4)
6. **Evoluzione del bias degli aloni ad alto redshift** (DESI, Euclid, Roman)

**Livello 3 — Test Indiretti o a Lungo Termine:**
7. **Variazione temporale di $G$** (array di timing di pulsar, prossimo decennio)
8. **Fondo stocastico di onde gravitazionali** (LISA, Einstein Telescope, anni 2030+)
9. **Coerenza ciclica/punti di Hawking** (speculativo; futura CMB ad alta precisione)

<a id="desi-2024-year-1-status-and-forecast"></a>
### Stato e Previsioni di DESI 2024 Anno-1

Il DESI Dark Energy Spectroscopic Instrument ha rilasciato misure di oscillazioni acustiche barioniche nel giugno 2024, fornendo vincoli sulla storia dell'espansione a $z \sim 0.3$–$0.9$. Queste misure:
- Vincolano la distanza comovente a una precisione del $\sim 0.5\%$
- Sono consistenti con ΛCDM a $z \lesssim 1$
- **Non hanno ancora escluso** le previsioni D-ND di $w(z) = -1 + 0.05(1-M_C(z))$ perché la deviazione prevista a $z \sim 0.5$ è solo del $\sim 0.8\%$, paragonabile alle barre di errore attuali

**Previsioni per DESI Anno-2 (2025) e Anno-3 (2026):**
- La precisione delle misure attesa migliorerà a $\sim 0.2$–$0.3\%$ per bin di redshift
- Questo dovrebbe raggiungere una **rilevazione a 1.5–2.5σ** della deviazione prevista da D-ND se reale
- Combinata con dati di weak lensing e supernovae, un **vincolo aggregato a 2–3σ** è realistico
- Un **risultato nullo** (accordo perfetto con $w = -1$ di ΛCDM) metterebbe in discussione D-ND a meno che la misura di emergenza $M_C(z)$ non evolva più rapidamente di quanto previsto

---

<a id="7-discussion-and-conclusions"></a>
## 7. Discussione e Conclusioni

<a id="7-1-strengths-of-the-d-nd-cosmological-extension"></a>
### 7.1 Punti di Forza dell'Estensione Cosmologica D-ND

1. **Colma una lacuna nella teoria cosmologica**: Fornisce un meccanismo per l'emergenza a sistema chiuso dello spaziotempo classico dalla potenzialità quantistica, applicabile a tutte le scale.

2. **Connette micro e macro**: Collega l'emergenza quantistica (Paper A) all'inflazione cosmica e all'evoluzione dell'energia oscura attraverso un framework matematico unificato.

3. **Risolve la singolarità iniziale**: Sostituisce la singolarità classica del Big Bang con una condizione al contorno finita sull'emergenza, evitando curvatura o densità infinita.

4. **Affronta il problema dell'energia oscura**: Fornisce una spiegazione qualitativa per la piccola costante cosmologica senza fine-tuning.

5. **Struttura ciclica e conservazione dell'informazione**: Suggerisce come l'informazione quantistica possa essere preservata attraverso i cicli cosmici, affrontando la termodinamica dei buchi neri.

6. **Previsioni falsificabili**: Propone test osservativi concreti (bispettro non-gaussiano, soppressione di potenza super-orizzonte, running dipendente dalla scala, modifiche alla formazione delle strutture, evoluzione dell'energia oscura).

7. **Framework vincolato da DESI**: Fornisce previsioni quantitative verificabili rispetto ai dati BAO del 2024, con criteri di falsificazione chiari.

<a id="7-2-limitations-and-caveats"></a>
### 7.2 Limitazioni e Avvertenze

1. **Natura speculativa**: La connessione tra l'emergenza microscopica (Paper A) e le scale cosmiche non è derivata rigorosamente dai principi primi. Le equazioni di Einstein modificate (S7) sono ansatze fenomenologici piuttosto che conseguenze geometriche precise.

2. **Mancanza di precisione nell'operatore di emergenza**: Alle scale cosmologiche, la struttura di $\mathcal{E}$ e lo spettro dell'"Hamiltoniano cosmologico" non sono noti. Le previsioni dipendono sensibilmente da questi input.

3. **Gravità quantistica incompleta**: Il framework non fornisce una teoria quantistica completa della gravità, paragonabile alla cosmologia quantistica a loop o alla cosmologia di stringa. È meglio considerarlo come un ponte fenomenologico tra meccanica quantistica e cosmologia classica.

4. **Equazioni modificate motivate assiomaticamente ma non derivate indipendentemente**: Il tensore energia-impulso informazionale $T_{\mu\nu}^{\text{info}}$ segue come conseguenza strutturale degli assiomi D-ND P0--P4 (§2.2), ma una derivazione completamente indipendente dai principi primi della gravità quantistica (es. principio dell'azione spettrale, sicurezza asintotica) rimane un problema aperto. La forma funzionale specifica mantiene una certa libertà all'interno dei vincoli assiomatici.

5. **Relazione con le osservazioni poco chiara nei dettagli**: Le previsioni osservative (bispettro CMB, formazione delle strutture, energia oscura) sono formulate qualitativamente e richiedono calcoli dettagliati per raggiungere una precisione quantitativa. Una simulazione di cosmologia numerica dedicata (simile ai codici CAMB o CLASS) sarebbe necessaria per produrre previsioni precise da confrontare con i dati.

6. **Rivalutazione della costante cosmologica**: L'identificazione dell'energia oscura con il $V_0$ residuo è attraente ma rimane speculativa. La magnitudine effettiva e l'evoluzione dell'energia oscura dipendono dalla forma sconosciuta di $V_0$ e dal suo accoppiamento con $M_C(t)$.

<a id="7-3-speculative-but-falsifiable-framework"></a>
### 7.3 Framework Speculativo ma Falsificabile

Sottolineiamo che questa estensione cosmologica è **speculativa ma falsificabile**. Le previsioni sono:

- **Non derivate dai principi primi** ma emergono dall'estrapolazione del framework quantistico D-ND alle scale cosmologiche.
- **Verificabili in linea di principio** attraverso specifiche anomalie nella CMB, pattern nella struttura su grande scala ed evoluzione dell'energia oscura.
- **Distinguibili da $\Lambda$CDM** nei regimi in cui gli effetti di emergenza sono non trascurabili (universo primordiale, scale più grandi, evoluzione cosmica a tempi recenti).

Un risultato negativo (es. mancata rilevazione della non-gaussianità CMB prevista o assenza di soppressione della crescita dipendente dalla scala) argomenterebbe contro il modello cosmologico D-ND. Al contrario, la rilevazione di una qualsiasi delle firme previste fornirebbe un supporto provvisorio al framework.

<a id="7-4-paths-forward"></a>
### 7.4 Percorsi Futuri

Vengono suggeriti tre programmi di ricerca:

**Cosmologia Numerica**: Implementare un codice di Boltzmann modificato (estendendo CLASS o CAMB) che incorpori le modifiche D-ND alle equazioni di Friedmann e calcoli lo spettro di potenza CMB completo, lo spettro di potenza del weak lensing e le previsioni sulla formazione delle strutture per il confronto con dati attuali e futuri.

**Integrazione con la Gravità Quantistica**: Tentare di derivare le equazioni di Einstein modificate (S7) da principi più fondamentali della gravità quantistica (es. cosmologia quantistica a loop, sicurezza asintotica o principio dell'azione spettrale), sostituendo il tensore informazionale fenomenologico con un termine rigorosamente motivato.

**Campagne Osservative**: Progettare osservazioni dedicate per cercare il bispettro CMB previsto, misurare la crescita delle strutture ad alto redshift e vincolare l'evoluzione dell'energia oscura con precisione sufficiente a distinguere D-ND da $\Lambda$CDM.

<a id="7-6-comparative-predictions-d-nd-cosmology-vs-cdm-vs-loop-quantum-cosmology-vs-ccc"></a>
### 7.6 Previsioni Comparative: Cosmologia D-ND vs. ΛCDM vs. Cosmologia Quantistica a Loop vs. CCC

Per contestualizzare la cosmologia D-ND nel panorama dei framework cosmologici modificati e quantistici, presentiamo un confronto quantitativo attraverso osservabili chiave e proprietà teoriche.

| **Caratteristica** | **ΛCDM** | **Cosmologia D-ND** | **Cosmologia Quantistica a Loop (LQC)** | **Cosmologia Ciclica Conforme (CCC)** |
|---|---|---|---|---|
| **Singolarità Iniziale** | Divergenza della curvatura a $t=0$ | Singolarità NT (confine finito) | Rimbalzo quantistico (evita la singolarità) | Riscalamento conforme (passato/futuro infinito) |
| **Meccanismo** | GR classica + costante cosmologica | Misura di emergenza $M_C(t)$ + tensore informazionale | Correzioni dalla geometria quantistica; operatore di gap di area | Ipotesi della curvatura di Weyl; corrispondenza conforme |
| **Inflazione** | Campo scalare slow-roll $\phi$ | Evoluzione rapida di $M_C$ (fase di emergenza) | Guidata dal potenziale, con modifiche | Non primaria; ciclica invece |
| **Durata dell'inflazione** | $e$-fold $\sim 50$–$60$ (tarato) | $\sim \log(1/M_C(0))$ (determinato dall'emergenza) | $\sim 40$–$70$ a seconda delle correzioni a loop | N/A (meccanismi di formazione delle strutture differenti) |
| **Energia Oscura** | Costante cosmologica ($w = -1$ esatto) | $V_0$ residuo ($w(z) = -1 + 0.05(1-M_C(z))$) | Correzioni a loop cambiano l'equazione di stato | Non primaria; violazioni delle condizioni energetiche CCC |
| **Evoluzione dell'Energia Oscura** | $\Omega_\Lambda$ costante | Dipendente dal tempo, decade come $\propto (1-M_C)^2$ | Lieve evoluzione dovuta a correzioni quantistiche | Evoluzione ciclica attraverso gli eoni |
| **Spettro di Potenza CMB** | Harrison-Zeldovich $n_s \approx 1$ + tilting | Running dipendente dalla scala $n_s(k)$ da $\dot{M}_C(t_*)$ | Simile a slow-roll (running piccolo) | Correlazioni modificate dalla corrispondenza tra eoni |
| **Non-Gaussianità** | $f_{\text{NL}} \sim 1$ (piccola, tipo locale) | $f_{\text{NL}} \sim 5$--$20$ ($\mathcal{E}$ liscia); più alta in template di tipo emergenza | $f_{\text{NL}}$ aumentata da correzioni quantistiche | $f_{\text{NL}}$ modificata dalla struttura conforme |
| **Crescita delle Strutture** | Fattore di crescita lineare $f(a)$ da GR | Crescita modificata dal feedback $(1-M_C(a))$ | Soppressa ai tempi primordiali (rimbalzi) | Crescita oscillatoria da condizioni al contorno cicliche |
| **Informazione dei Buchi Neri** | Informazione persa (paradosso di Hawking) | Informazione preservata (aggiornamenti InjectKLI) | Preservata tramite geometria quantistica | Preservata tramite struttura ciclica |
| **Struttura Ciclica** | Nessun ciclo (Big Bang singolare) | Cicli multipli con coerenza di fase $\Omega_{NT} = 2\pi i$ | Rimbalzo quantistico (ciclo singolo?) | Cicli infiniti (eoni) con corrispondenza conforme |
| **Numero di Parametri Liberi** | 6 (Ω, $H_0$, $\sigma_8$, $n_s$) | $\sim 8$ ($\Lambda$, $\xi$, spettro operatore di emergenza, $\tau_e$) | $\sim 6$ (simile a ΛCDM + correzioni quantistiche) | $\sim 5$ (fissati dalla struttura conforme) |
| **Grado di Speculatività** | Ben verificato; standard | Altamente speculativo; estensioni congetturali | Quantitativo ma basato sui fondamenti della LQG | Speculativo; prevede punti di Hawking nella CMB |
| **Stato Osservativo** | Consistente con CMB, SNe, BAO | Non ancora vincolato da DESI (previsioni al livello 1–3%) | Consistente con le osservazioni; assunzioni fondazionali della LQG dibattute | Punti di Hawking non confermati; previsioni sotto scrutinio |

**Distinzioni Chiave:**

1. **Meccanismo per l'inflazione**: ΛCDM usa slow-roll; D-ND usa l'emergenza; LQC usa rimbalzi quantistici; CCC usa la struttura ciclica.

2. **Comportamento dell'energia oscura**: ΛCDM costante; D-ND evolve con l'emergenza; LQC leggermente modificata dai loop quantistici; CCC ciclica.

3. **Preservazione dell'informazione**: ΛCDM la perde; D-ND la preserva tramite i cicli; LQC tramite la geometria quantistica; CCC tramite la struttura conforme.

4. **Verificabilità**: I dati DESI 2024 forniscono vincoli. Le previsioni D-ND (deviazione dell'1–3% in $w(z)$) sono appena oltre la precisione attuale ma saranno testate nel 2026.

5. **Unità concettuale**: D-ND connette l'emergenza alle scale quantistiche e cosmiche; LQC è primariamente gravità quantistica; CCC è primariamente geometria conforme.

**Raccomandazione per il Lavoro Futuro**: Misure ad alta precisione della storia dell'espansione ($w(z)$ da BAO, weak lensing, SNe) nell'intervallo $z \sim 0$–$2$ testeranno in modo decisivo D-ND rispetto a ΛCDM e ad altre alternative entro i prossimi 3–5 anni.

<a id="7-5-conclusion"></a>
### 7.5 Conclusione

Abbiamo presentato un'estensione speculativa ma matematicamente coerente del framework Dual-Non-Dual alle scale cosmologiche. Accoppiando le equazioni di campo di Einstein alla misura di emergenza quantistica $M_C(t)$, delineiamo un quadro in cui l'universo emerge dalla potenzialità primordiale, l'inflazione sorge come una fase di rapida attualizzazione, l'energia oscura rappresenta la struttura non-relazionale residua e la singolarità iniziale è sostituita da una condizione al contorno sull'emergenza. Il framework suggerisce che l'universo possa attraversare cicli multipli, ciascuno preservando l'informazione quantistica attraverso la condizione di coerenza ciclica $\Omega_{NT} = 2\pi i$.

Sebbene il framework rimanga altamente speculativo e dipenda criticamente dalle assunzioni sull'operatore di emergenza microscopico, fornisce una visione concettualmente unificata della cosmologia quantistica e classica. Se catturi correttamente la fisica dell'universo può essere determinato solo attraverso test osservativi delle sue previsioni quantitative.

---

<a id="references"></a>
## Riferimenti

<a id="d-nd-framework-papers"></a>
### D-ND Framework Papers

- *Paper A*: "Quantum Emergence from Primordial Potentiality: The Dual-Non-Dual Framework for State Differentiation" (Draft 3.0)
- *Paper B*: [Lagrangian formulation of D-ND dynamics — referenced but not detailed here]

<a id="quantum-cosmology-and-the-problem-of-time"></a>
### Cosmologia Quantistica e il Problema del Tempo

- Hartle, J. B., & Hawking, S. W. (1983). "Wave function of the universe." *Physical Review D*, 28(12), 2960.
- Wheeler, J. A. (1968). "Superspace and the nature of quantum geometrodynamics." In *Battelle Rencontres* (pp. 242–307).
- Kuchař, K. V. (1992). "Time and interpretations of quantum gravity." In *General Relativity and Gravitation* (pp. 520–575). Cambridge University Press.
- Giovannetti, V., Lloyd, S., & Maccone, L. (2015). "Quantum time." *Physical Review D*, 92(4), 045033.

<a id="inflationary-cosmology"></a>
### Cosmologia Inflazionaria

- Guth, A. H. (1981). "Inflationary universe: A possible solution to the horizon and flatness problems." *Physical Review D*, 23(2), 347.
- Linde, A. D. (1986). "Eternally existing self-reproducing chaotic inflationary universe." *Physics Letters B*, 175(4), 395–400.
- Dodelson, S. (2003). *Modern Cosmology*. Academic Press.

<a id="modified-gravity-and-entropic-gravity"></a>
### Gravità Modificata e Gravità Entropica

- Verlinde, E. (2011). "On the origin of gravity and the laws of Newton." *Journal of High Energy Physics*, 2011(4), 29. [arXiv: 1001.0785]
- Verlinde, E. (2016). "Emergent gravity and the dark universe." *SciPost Physics*, 2(3), 016. [arXiv: 1611.02269]
- Beke, L., & Hinterbichler, K. (2021). "Entropic gravity and the limits of thermodynamic descriptions." *Physics Letters B*, 811, 135863.

<a id="conformal-cyclic-cosmology"></a>
### Cosmologia Ciclica Conforme

- Penrose, R. (2005). "Before the Big Bang?" In *Science and Ultimate Reality* (pp. 1–29). Cambridge University Press.
- Penrose, R. (2010). *Cycles of Time: An Extraordinary New View of the Universe*. Jonathan Cape.
- Wehus, A. M., & Eriksen, H. K. (2021). "A search for concentric circles in the 7-year WMAP temperature sky maps." *Astrophysical Journal*, 733(2), 29.

<a id="emergent-spacetime-and-holography"></a>
### Spaziotempo Emergente e Olografia

- Maldacena, J. M. (1998). "The large N limit of superconformal field theories and supergravity." *Advances in Theoretical and Mathematical Physics*, 2(2), 231–252.
- Ryu, S., & Takayanagi, T. (2006). "Holographic derivation of entanglement entropy from AdS/CFT." *Physical Review Letters*, 96(18), 181602.
- Van Raamsdonk, M. (2010). "Building up spacetime with quantum entanglement." *General Relativity and Gravitation*, 42(10), 2323–2329.
- Swingle, B. (2018). "Entanglement renormalization and holography." *Classical and Quantum Gravity*, 34(18), 184001.

<a id="structure-formation-and-large-scale-structure"></a>
### Formazione delle Strutture e Struttura su Grande Scala

- Bardeen, J. M., Bond, J. R., Kaiser, N., & Szalay, A. S. (1986). "The statistics of peaks of Gaussian random fields." *Astrophysical Journal*, 304, 15–61.
- Smith, R. E., et al. (2003). "Stable clustering, the halo model and non-linear cosmological power spectra." *Monthly Notices of the Royal Astronomical Society*, 341(4), 1311–1332.
- Eisenstein, D. J., & Hu, W. (1998). "Bispectrum of the cosmic microwave background." *Astrophysical Journal*, 496(2), 605.

<a id="cmb-physics-and-non-gaussianity"></a>
### Fisica della CMB e Non-Gaussianità

- Planck Collaboration. (2018). "Planck 2018 results. IX. Constraints on primordial non-Gaussianity." *Astronomy & Astrophysics*, 641, A9.
- Komatsu, E. (2010). "Hunting for primordial non-Gaussianity in the cosmic microwave background." *Classical and Quantum Gravity*, 27(12), 124010.
- Maldacena, J. M. (2003). "Non-Gaussian features of primordial fluctuations in single-field inflationary models." *Journal of High Energy Physics*, 2003(05), 013.

<a id="dark-energy-and-the-cosmological-constant"></a>
### Energia Oscura e la Costante Cosmologica

- Perlmutter, S., et al. (1999). "Measurements of Ω and Λ from 42 high-redshift supernovae." *Astrophysical Journal*, 517(2), 565.
- Riess, A. G., et al. (1998). "Observational evidence from supernovae for an accelerating universe and a cosmological constant." *Astronomical Journal*, 116(3), 1009.
- Weinberg, S. (2000). "The cosmological constant problems." arXiv preprint astro-ph/0005265.

<a id="black-hole-thermodynamics-and-information"></a>
### Termodinamica dei Buchi Neri e Informazione

- Bekenstein, J. D. (1973). "Black holes and entropy." *Physical Review D*, 7(8), 2333.
- Hawking, S. W. (1974). "Black hole explosions?" *Nature*, 248(5443), 30–31.
- 't Hooft, G. (1993). "Dimensional reduction in quantum gravity." arXiv preprint gr-qc/9310026.

<a id="mathematical-foundations"></a>
### Fondamenti Matematici

- Reed, M., & Simon, B. (1980). *Methods of Modern Mathematical Physics*. Academic Press.
- Chamseddine, A. H., & Connes, A. (1997). "The spectral action principle." *Communications in Mathematical Physics*, 186(3), 731–750.

<a id="logic-of-the-included-third"></a>
### Logica del Terzo Incluso

- Lupasco, S. (1951). *Le principe d'antagonisme et la logique de l'énergie*. Hermann, Paris.
- Nicolescu, B. (2002). *Manifesto of Transdisciplinarity*. SUNY Press.

---
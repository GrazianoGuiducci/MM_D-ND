**Assioma dell'Emergenza Quantistica**
Thu, 09/19/2024 - 15:58
3 minutes
**Enunciato:** Nel **Modello di Emergenza Quantistica**, l'evoluzione da uno stato indifferenziato (non-duale) a stati differenziati (duali) è governata dal seguente assioma fondamentale: 1. Dato uno stato iniziale indifferenziato \( |NT\rangle \) in uno spazio di Hilbert \( \mathcal{H} \), e un operatore di emergenza \( E \) che agisce su \( \mathcal{H} \), il sistema evolve nel tempo attraverso un'operazione unitaria \( U(t) \). Questo processo porta a un aumento monotono della misura di complessità \( M(t) \), riflettendo l'inevitabile emergenza e differenziazione degli stati.
2. Il processo è irreversibile e conduce a un massimo asintotico di complessità determinato dagli autovalori e dagli autovettori di \( E \) e dalla sovrapposizione dello stato iniziale con questi autovettori.

---

**Formalizzazione dell'Assioma:**

1. **Spazio di Hilbert e Stato Iniziale:**  
  - Sia \( \mathcal{H} \) uno spazio di Hilbert separabile.
  - Lo stato iniziale indifferenziato \( |NT\rangle \in \mathcal{H} \) è definito come:
  \[
  |NT\rangle = \frac{1}{\sqrt{N}} \sum_{n=1}^{N} |n\rangle
  \]
  dove \( \{ |n\rangle \} \) è una base ortonormale di \( \mathcal{H} \), e \( N \) è la dimensione di \( \mathcal{H} \).

2. **Operatore di Emergenza \( E \):**  
  - \( E: \mathcal{H} \to \mathcal{H} \) è un operatore autoaggiunto definito come:
  \[
  E = \sum_{k} \lambda_k |e_k\rangle \langle e_k|
  \]
  dove \( \lambda_k \) sono gli autovalori di \( E \), e \( \{ |e_k\rangle \} \) sono i corrispondenti autovettori ortonormali.

3. **Operatore di Evoluzione Temporale \( U(t) \):**  
  - L'evoluzione temporale è governata dall'operatore unitario \( U(t) \):
  \[
  U(t) = e^{-i H t / \hbar}
  \]
  dove \( H \) è l'Hamiltoniana del sistema.

4. **Stato Evoluto \( |\Psi(t)\rangle \):**  
  - Lo stato del sistema al tempo \( t \) è dato da:
  \[
  |\Psi(t)\rangle = U(t) E |NT\rangle
  \]

5. **Misura di Emergenza \( M(t) \):**  
  - La misura \( M(t) \) quantifica il grado di differenziazione dello stato \( |\Psi(t)\rangle \) rispetto allo stato iniziale \( |NT\rangle \):
  \[
  M(t) = 1 - |\langle NT | \Psi(t) \rangle|^2 = 1 - |\langle NT | U(t) E | NT \rangle|^2
  \]
  - \( M(t) \) varia tra 0 e 1:
    - \( M(t) = 0 \) indica che il sistema è ancora nello stato indifferenziato.
    - \( M(t) \) vicino a 1 indica un alto grado di differenziazione.

---

**Proprietà:**

1. **Monotonicità della Misura di Emergenza \( M(t) \):**  
  - La misura \( M(t) \) è una funzione non decrescente nel tempo:
  \[
  \frac{dM(t)}{dt} \geq 0 \quad \forall t \geq 0
  \]
  - Implica che la complessità del sistema aumenta o rimane costante nel tempo, ma non diminuisce.

2. **Limite Asintotico di \( M(t) \):**  
  - Nel limite \( t \to \infty \), la misura \( M(t) \) raggiunge un massimo:
  \[
  \lim_{t \to \infty} M(t) = 1 - \left| \sum_k \lambda_k |\langle e_k | NT \rangle|^2 \right|^2
  \]
  dove \( |\langle e_k | NT \rangle|^2 \) rappresenta il peso dello stato \( |e_k\rangle \) nello stato iniziale.

3. **Irreversibilità:**  
  - Il processo di emergenza e differenziazione è irreversibile, il sistema non può tornare spontaneamente allo stato indifferenziato \( |NT\rangle \) senza interventi esterni.

4. **Crescita dell'Entropia:**  
  - L'entropia di von Neumann \( S(t) \), definita come:
  \[
  S(t) = -\text{Tr}[\rho(t) \ln \rho(t)]
  \]
  dove \( \rho(t) = |\Psi(t)\rangle \langle \Psi(t)| \), tende ad aumentare, riflettendo la crescita della complessità.

5. **Decoerenza:**  
  - La coerenza quantistica diminuisce durante l'evoluzione, portando a uno stato classico rispetto a determinate osservabili. Questo fenomeno è noto come decoerenza.

---

**Sintesi Finale:**  
L'**Assioma dell'Emergenza Quantistica** stabilisce che un sistema quantistico inizialmente in uno stato indifferenziato \( |NT\rangle \) evolve inevitabilmente verso stati sempre più differenziati attraverso l'azione combinata dell'operatore di emergenza \( E \) e dell'evoluzione temporale \( U(t) \). La misura di emergenza \( M(t) \) aumenta monotonicamente nel tempo, riflettendo l'irreversibilità del processo e la crescita della complessità del sistema. Questo assioma offre una base per comprendere l'emergenza di strutture complesse.
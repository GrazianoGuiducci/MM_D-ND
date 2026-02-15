================================================================================
D-ND COSMOLOGY FRAMEWORK - LATEX NOTATION QUICK REFERENCE CARD
================================================================================

All commands defined in: dnd_shared.sty
Usage in any paper: \usepackage{dnd_shared}

================================================================================
CORE D-ND PHYSICS
================================================================================

QUANTUM STATES & OPERATORS:
  \NT                       │NT⟩ Nested Topology quantum state
  \emerge                   ℰ Emergence operator
  \Hilbert                  ℋ Hilbert space (custom macro in template)
  \HilbertNT                ℋ_NT Hilbert space for NT (custom macro)
  \Obs                      𝒪 Observable operator (custom macro)

TIME-DEPENDENT QUANTITIES:
  \emeasure                 M(t) Emergence measurement function
  \orderparam               Z(t) Order parameter (ranges 0→1)
  \resultant                R(t) Resultant parameter
  \evolve{X}                X(t) Time evolution (custom macro)

================================================================================
COUPLING CONSTANTS & PARAMETERS
================================================================================

PRIMARY CONSTANTS:
  \Kgen                     K_gen Generator constant
  \chiDND                   χ_DND D-ND coupling constant
  \OmegaNT                  Ω_NT NT frequency
  \rhoDND                   ρ_DND D-ND density parameter
  \rhoLECO                  ρ_LECO Least Emerged Classical Oscillator density

LAMBDA VARIANTS (COUPLING STRENGTHS):
  \lambdak                  λ_k k-mode coupling
  \lambdaDND                λ_DND D-ND coupling strength
  \lambdaauto               λ_auto Autonomous coupling
  \lambdacosmo              λ_cosmo Cosmological coupling

VARIANCE & FLUCTUATIONS:
  \sigmaE                   σ²_ℰ Emergence variance
  \sigmaV                   σ²_V Velocity variance

TEMPORAL & COGNITIVE SCALES:
  \Tcog                     T_cog Cognitive timescale
  \GS                       G_S Singularity Constant

================================================================================
MATHEMATICAL OPERATORS (SUPPLEMENTARY MACROS)
================================================================================

QUANTUM MECHANICS:
  \expect{X}                ⟨X⟩ Expectation value
  \commutator{A}{B}         [A, B] Commutator bracket
  \anticommutator{A}{B}     {A, B} Anticommutator (anti-bracket)

CALCULUS:
  \pd{f}{x}                 ∂f/∂x Partial derivative
  \pdd{f}{x}{y}             ∂²f/∂x∂y Second mixed partial
  \frac{\partial}{\partial x}  Alternative: full notation

DERIVED OPERATORS:
  \emerge@squared           ℰ² Emergence operator squared
  \emerge@normalized        ℰ̃ Normalized emergence operator
  \emergence@threshold      ℰ_th Threshold emergence value
  \emergence@critical       ℰ* Critical emergence value
  \coupling@strength        α_couple Coupling strength parameter

================================================================================
QUANTUM MECHANICS NOTATION (via braket package)
================================================================================

Standard quantum notation (from \usepackage{braket}):
  \ket{\psi}                |ψ⟩ Ket vector
  \bra{\phi}                ⟨φ| Bra vector
  \braket{\phi}{\psi}       ⟨φ|ψ⟩ Inner product
  \braket{\phi|\hat{A}|\psi}  ⟨φ|Â|ψ⟩ Matrix element

With D-ND notation:
  \ket{\NT}                 |NT⟩ NT state (use with \NT macro)
  \braket{\text{classical}|\emerge|{\psi}}  Classical-emergence-quantum

================================================================================
THEOREM ENVIRONMENTS (all papers)
================================================================================

\begin{axiom}[Name]         Fundamental principles of D-ND
  content
\end{axiom}

\begin{theorem}[Name]       Major theoretical results
  content
\end{theorem}

\begin{proposition}[Name]   Intermediate results
  content
\end{proposition}

\begin{corollary}[Name]     Direct theorem consequences
  content
\end{corollary}

\begin{lemma}[Name]         Technical auxiliary results
  content
\end{lemma}

\begin{definition}[Name]    Key concept definitions
  content
\end{definition}

\begin{remark}              Commentary and notes
  content
\end{remark}

\begin{note}                Additional information
  content
\end{note}

WITH PROOFS:
\begin{theorem}[Name]
  Statement...
\end{theorem}
\begin{proof}
  Proof text...
\end{proof}

================================================================================
CROSS-REFERENCING (via cleveref package)
================================================================================

SMART REFERENCES (from cleveref):
  \cref{sec:axioms}         Renders as "Section 2" automatically
  \Cref{sec:axioms}         Renders as "Section 2" (capitalized)
  \cref{eq:order}           Renders as "Equation (3)" automatically
  \cref{fig:diagram}        Renders as "Figure 1" automatically
  \cref{tab:values}         Renders as "Table 2" automatically

CUSTOM HELPERS (in template):
  \secref{axioms}           Renders as "Section 2"
  \eqref@dnd{order}         Renders as "Eq. (3)"
  \figref{diagram}          Renders as "Fig. 1"
  \tabref{values}           Renders as "Table 2"

BASIC REFERENCES (alternative):
  \ref{label}               Just the number
  \label{sec:name}          Define label (put after section heading)
  \label{eq:name}           Define equation label (in equation)
  \label{fig:name}          Define figure label (with caption)

LABELING CONVENTION:
  \label{sec:section_name}        For sections
  \label{eq:equation_name}        For equations
  \label{fig:figure_name}         For figures
  \label{tab:table_name}          For tables
  \label{axiom:axiom_name}        For axioms
  \label{thm:theorem_name}        For theorems

================================================================================
MATHEMATICAL ENVIRONMENTS
================================================================================

BASIC EQUATIONS:
  \begin{equation}
    E = mc^2
    \label{eq:einstein}
  \end{equation}

EQUATION ARRAYS (aligned):
  \begin{align}
    y &= mx + b \\
    z &= x^2
    \label{eq:curves}
  \end{align}

UNNUMBERED EQUATIONS:
  \begin{equation*}
    a^2 + b^2 = c^2
  \end{equation*}

  \begin{align*}
    f(x) &= x^3 \\
    g(x) &= 2x
  \end{align*}

MULTIPLE COLUMNS (split within equation):
  \begin{equation}
    \begin{split}
      H &= \sum_i p_i^2/2m + V(q_i) \\
        &= T + V
    \end{split}
  \end{equation}

CASES:
  \[ f(x) = \begin{cases}
    x^2 & \text{if } x \geq 0 \\
    -x & \text{if } x < 0
  \end{cases} \]

================================================================================
MATH MODE SYMBOLS & FORMATTING
================================================================================

GREEK LETTERS (full list):
  \alpha \beta \gamma \delta \epsilon \zeta \eta \theta
  \iota \kappa \lambda \mu \nu \xi \omicron \pi
  \rho \sigma \tau \upsilon \phi \chi \psi \omega

CAPITALS:
  \Gamma \Delta \Theta \Lambda \Xi \Pi \Sigma \Upsilon \Phi \Psi \Omega

SCRIPT/CALLIGRAPHIC (via mathrsfs):
  \mathcal{A}  \mathcal{E}  \mathcal{H}  \mathcal{L}
  \mathscr{A}  \mathscr{E}  \mathscr{H}  (alternative fancy fonts)

BOLD & SPECIAL:
  \mathbf{v}     Bold vector
  \mathrm{d}x    Roman differential (not italic)
  \text{text}    Regular text in math mode

OPERATORS & RELATIONS:
  \times  ×  multiplication
  \cdot   ·  dot product / scalar multiplication
  \otimes ⊗  tensor product
  \oplus  ⊕  direct sum
  \equiv  ≡  equivalence
  \approx ≈  approximation
  \propto ∝  proportional to
  \sim    ~  similar
  \ll     ≪  much less than
  \gg     ≫  much greater than
  \pm     ±  plus or minus

SPACES IN MATH:
  \,      Thin space
  \:      Medium space
  \;      Thick space
  \quad   Large space (= font size)
  \qquad  Two large spaces

================================================================================
COMMON D-ND EQUATIONS (EXAMPLES)
================================================================================

EMERGENCE OPERATOR (from template):
  \begin{equation}
    \emerge \ket{\psi} = \lambda_{\emerge} \ket{\psi_{\text{classical}}}
  \end{equation}

ORDER PARAMETER EVOLUTION:
  \begin{equation}
    \frac{d}{dt}\orderparam = -\Gamma \left(\orderparam - \orderparam_{\text{eq}}\right)
  \end{equation}

EMERGENCE MEASURE:
  \begin{equation}
    \emeasure = \frac{1}{2}\left(1 + \tanh\left(\beta(\emerge - \emerge_{\text{th}})\right)\right)
  \end{equation}

CLASSICAL HAMILTONIAN LIMIT:
  \begin{equation}
    H_{\text{classical}} = \lim_{\orderparam \to 1} \emerge^{\dagger} H_{\text{quantum}} \emerge
  \end{equation}

RICCI CURVATURE:
  \begin{equation}
    R_{ij} = \frac{\partial^2 \ln g}{\partial \xi^i \partial \xi^j}
  \end{equation}

================================================================================
QUICK SYNTAX CHECKLISTS
================================================================================

BEFORE COMPILING:
  ☐ \usepackage{dnd_shared} is included
  ☐ All \NT, \emerge, etc. are used correctly
  ☐ All equations have \label{eq:...}
  ☐ All sections have \label{sec:...}
  ☐ All \cite{} keys exist in bibliography
  ☐ All \ref{} labels are defined
  ☐ Braces {} are balanced
  ☐ Dollar signs $ are paired

AFTER COMPILING (troubleshooting):
  ☐ Check for "undefined reference" warnings
  ☐ Check for "undefined control sequence" errors
  ☐ Check .log file for package warnings
  ☐ Verify PDF generation (compare modification times)
  ☐ Test all cross-references with \cref{}
  ☐ Verify bibliography appears in PDF
  ☐ Check that all figures are included

================================================================================
JOURNAL-SPECIFIC DOCUMENT CLASSES
================================================================================

Paper A - Physical Review A:
  \documentclass[aps,pra,11pt,notitlepage,nofootinbib]{revtex4-2}

Paper B - Physical Review E:
  \documentclass[aps,pre,11pt,notitlepage,nofootinbib]{revtex4-2}

Paper C - Journal of Mathematical Physics (AIP):
  \documentclass[11pt]{aip}

Paper D - Foundations of Physics (Springer):
  \documentclass{svjour3}

Paper E - Classical and Quantum Gravity (IOP):
  \documentclass{iopart}

Paper F - Quantum:
  \documentclass{quantumarticle}

Paper G - Cognitive Science (Springer):
  \documentclass{svjour3}

================================================================================
ESSENTIAL PACKAGES (loaded automatically by templates)
================================================================================

Math:        amsmath, amssymb, mathrsfs, braket, amsthm
References:  hyperref, cleveref
Bibliography: natbib
Formatting:  geometry, setspace, graphicx, float, inputenc, fontenc

All included in paper_A_template.tex (lines 17-44)

================================================================================
FILE LOCATIONS
================================================================================

Template directory:
  /sessions/pensive-sharp-curie/mnt/domain_D-ND_Cosmology/domain/AWARENESS/3_π_PRAGMATIC/latex/

Files:
  paper_A_template.tex           671 lines - Main template (reference)
  dnd_shared.sty                 107 lines - Notation package (all papers)
  README.md                      608 lines - Complete documentation
  JOURNAL_ADAPTATION_GUIDE.md    594 lines - Journal-specific instructions
  D-ND_NOTATION_CARD.txt         This file - Quick reference

Total: ~2000 lines of documentation + templates

================================================================================
COMPILATION COMMANDS
================================================================================

Full compilation cycle:
  pdflatex paper_A_template.tex
  bibtex paper_A_template
  pdflatex paper_A_template.tex
  pdflatex paper_A_template.tex

Verify PDF created:
  file paper_A_template.pdf
  du -h paper_A_template.pdf

View in PDF reader:
  xdg-open paper_A_template.pdf      (Linux)
  open paper_A_template.pdf          (macOS)

Clean auxiliary files:
  rm -f *.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk

================================================================================
TIPS & TRICKS
================================================================================

1. USE LABELS CONSISTENTLY:
   \section{...} \label{sec:name}
   \subsection{...} \label{subsec:name}
   \begin{equation} ... \label{eq:name} \end{equation}

2. PREFER \cref OVER \ref:
   ✓ \cref{eq:einstein} renders as "Equation (1)"
   ✗ Equation~\ref{eq:einstein} requires manual text

3. DEFINE CUSTOM NOTATION IN dnd_shared.sty:
   ✓ All papers use consistent notation
   ✗ Don't define notation in individual papers

4. USE DOUBLE BRACES FOR FRAGILE COMMANDS:
   ✓ \cite{{Smith, 2020}} for fragile contexts
   ✗ \cite{Smith, 2020} may fail in some contexts

5. ESCAPE SPECIAL CHARACTERS IN TEXT:
   ✓ \% \$ \{ \} \_ \& \#
   ✗ % $ { } _ & # (will cause errors)

6. USE \text{} FOR WORDS IN MATH MODE:
   ✓ $Z(t)_{\text{eq}}$ is the equilibrium value
   ✗ $Z(t)_eq$ renders incorrectly

7. ALIGN EQUATIONS CONSISTENTLY:
   ✓ Multiple equations: use \begin{align}
   ✗ Single equation: use \begin{equation}

8. TEST LOCAL COMPILATION BEFORE SUBMITTING:
   ✓ Compile successfully 3x
   ✓ Check .log file for warnings
   ✗ Submit without testing locally

================================================================================
TROUBLESHOOTING QUICK REFERENCE
================================================================================

PROBLEM: "Undefined control sequence \NT"
  CAUSE: \usepackage{dnd_shared} missing
  FIX: Add \usepackage{dnd_shared} after \usepackage{amsthm}

PROBLEM: "Unknown citation key"
  CAUSE: \cite{key} doesn't match bibliography entry
  FIX: Check spelling in .bib file (or \bibitem{key})

PROBLEM: "Reference undefined"
  CAUSE: \label{...} missing or \ref{...} doesn't match
  FIX: Add \label{eq:name} after equation, verify reference

PROBLEM: Document class not found
  CAUSE: TeX distribution incomplete
  FIX: Install texlive-full or specific journal package

PROBLEM: Theorem environments not numbered correctly
  CAUSE: \usepackage{amsthm} issue
  FIX: Verify amsthm comes after amsmath and amssymb

PROBLEM: PDF not generated
  CAUSE: Compilation errors in .log file
  FIX: Check .log file, fix LaTeX errors, recompile

================================================================================
REMEMBER
================================================================================

✓ Use dnd_shared.sty for all D-ND notation
✓ Keep document structure consistent across papers
✓ Use smart references with \cref{}
✓ Label everything with consistent prefixes
✓ Test compilation before submission
✓ Check .log file for warnings
✓ Version control your .tex files
✓ Comment your custom macros

The shared notation system ensures:
  • Consistent appearance across all 7 papers
  • Easy maintenance (edit dnd_shared.sty once)
  • Rapid adaptation to new journals
  • Professional presentation to reviewers

================================================================================
Last updated: 2025-02-14
For questions, see JOURNAL_ADAPTATION_GUIDE.md or README.md
================================================================================

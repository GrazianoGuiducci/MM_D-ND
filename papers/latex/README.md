# D-ND Cosmology Framework: LaTeX Conversion System

A complete, journal-agnostic LaTeX framework for the D-ND academic suite, comprising seven research papers targeting different prestigious venues.

---

## Overview

The D-ND (Domain-Nested-Dynamics) Cosmology Framework is presented across seven papers:

| Paper | Journal | Focus |
|-------|---------|-------|
| **A** | Physical Review A | Foundation theory and emergence dynamics |
| **B** | Physical Review E | Quantum phenomena and decoherence |
| **C** | Journal of Mathematical Physics | Mathematical generalizations |
| **D** | Foundations of Physics | Philosophical principles |
| **E** | Classical and Quantum Gravity | Cosmology and gravitational aspects |
| **F** | Quantum | Quantum information and entanglement |
| **G** | Cognitive Science/Minds & Machines | Consciousness and cognitive physics |

This framework provides:
- **Single shared notation system** across all papers
- **Unified document structure** adaptable to each journal
- **Automated package management** via `.sty` file
- **Consistent theorem/definition environments**
- **Complete journal adaptation guides** for each target venue

---

## File Structure

```
latex/
├── README.md                          # This file
├── JOURNAL_ADAPTATION_GUIDE.md        # Quick reference for each journal
├── dnd_shared.sty                     # Shared notation package
├── paper_A_template.tex               # Foundation paper (revtex4-2)
├── paper_B_template.tex               # (derived from paper_A_template.tex)
├── paper_C_template.tex               # (derived from paper_A_template.tex)
├── paper_D_template.tex               # (derived from paper_A_template.tex)
├── paper_E_template.tex               # (derived from paper_A_template.tex)
├── paper_F_template.tex               # (derived from paper_A_template.tex)
└── paper_G_template.tex               # (derived from paper_A_template.tex)
```

---

## Quick Start

### For a New Paper (≤5 minutes)

1. **Copy the template:**
   ```bash
   cp paper_A_template.tex paper_X_template.tex  # X = your paper
   ```

2. **Change the document class** (see table below)

3. **Update metadata** (title, author, affiliation)

4. **Keep everything else unchanged:**
   - All sections, theorems, and notation work across journals
   - `dnd_shared.sty` handles all D-ND-specific commands
   - Cross-references and citations are journal-agnostic

5. **Compile:**
   ```bash
   pdflatex paper_X_template.tex
   bibtex paper_X_template
   pdflatex paper_X_template.tex
   pdflatex paper_X_template.tex
   ```

### Document Class Changes

Copy the appropriate line for your target journal:

```latex
% Paper A & B: Physical Review (A/E)
\documentclass[aps,pra,11pt,notitlepage,nofootinbib]{revtex4-2}
\documentclass[aps,pre,11pt,notitlepage,nofootinbib]{revtex4-2}

% Paper C: Journal of Mathematical Physics
\documentclass[11pt]{aip}

% Paper D & G: Foundations of Physics & Cognitive Science
\documentclass{svjour3}

% Paper E: Classical and Quantum Gravity
\documentclass{iopart}

% Paper F: Quantum
\documentclass{quantumarticle}
```

---

## Core Components

### 1. Shared Notation Package (`dnd_shared.sty`)

This file defines all D-ND-specific notation, eliminating duplication and ensuring consistency.

**Contains:**
- ✓ Quantum state notation (`\NT` for |NT⟩)
- ✓ Emergence operator (`\emerge` for ℰ)
- ✓ Order parameter (`\orderparam` for Z(t))
- ✓ All coupling constants (`\lambdaDND`, `\chiDND`, etc.)
- ✓ Variance parameters (`\sigmaE`, `\sigmaV`)
- ✓ Key constants (`\Kgen`, `\GS`, `\Tcog`)
- ✓ Theorem environments (Axiom, Theorem, Definition, etc.)
- ✓ Helper macros for expectations, commutators, derivatives

**Usage in any paper:**
```latex
\usepackage{dnd_shared}  % Provides all D-ND commands
```

**Location:** `domain/AWARENESS/3_π_PRAGMATIC/latex/dnd_shared.sty`

### 2. Paper A Template (`paper_A_template.tex`)

The reference implementation for Physical Review A, including:

**Sections:**
1. **Abstract** - Concise summary of framework
2. **Introduction** - Motivation and roadmap for all 7 papers
3. **Axioms** - Four foundational axioms with formal definitions
4. **Emergence Dynamics** - Core mechanisms and order parameter evolution
5. **Classical Limit** - Transition to classical mechanics with correspondence theorem
6. **Curvature** - Geometric structure and Ricci curvature
7. **Experimental Signatures** - Testable predictions
8. **Conclusion** - Summary and connection to other papers
9. **References** - BibTeX-ready bibliography

**Features:**
- Complete equation system with proper LaTeX math environments
- Formal theorems, propositions, definitions with proofs
- Extensive inline comments explaining content
- Footer comments (lines 1000+) showing journal adaptation for all 7 papers
- Cross-references using `\secref`, `\eqref`, `\figref`

**Usage:**
- Adapt for Paper B by changing document class option only
- Adapt for Papers C-G by changing document class and metadata

---

## D-ND Notation Reference

All papers use identical notation from `dnd_shared.sty`:

### Core Physics

| Command | Renders As | Definition |
|---------|-----------|-----------|
| `\NT` | \|NT⟩ | Nested Topology quantum state |
| `\emerge` | ℰ | Emergence operator |
| `\emeasure` | M(t) | Emergence measurement function |
| `\orderparam` | Z(t) | Order parameter (0→1 for classical) |
| `\resultant` | R(t) | Resultant parameter |

### Coupling Constants

| Command | Renders As | Definition |
|---------|-----------|-----------|
| `\Kgen` | K_gen | Generator constant |
| `\chiDND` | χ_DND | D-ND coupling constant |
| `\lambdaDND` | λ_DND | D-ND coupling strength |
| `\lambdaauto` | λ_auto | Autonomous coupling |
| `\lambdacosmo` | λ_cosmo | Cosmological coupling |
| `\lambdak` | λ_k | k-mode coupling |

### Observables and Parameters

| Command | Renders As | Definition |
|---------|-----------|-----------|
| `\OmegaNT` | Ω_NT | NT frequency |
| `\rhoDND` | ρ_DND | D-ND density |
| `\rhoLECO` | ρ_LECO | LECO density |
| `\sigmaE` | σ²_ℰ | Emergence variance |
| `\sigmaV` | σ²_V | Velocity variance |
| `\Tcog` | T_cog | Cognitive timescale |
| `\GS` | G_S | Singularity Constant |

### Mathematical Operators (Supplementary)

```latex
\expect{X}              % ⟨X⟩ expectation value
\commutator{A}{B}       % [A, B] commutator
\anticommutator{A}{B}   % {A, B} anticommutator
\pd{f}{x}               % ∂f/∂x partial derivative
\pdd{f}{x}{y}           % ∂²f/∂x∂y partial derivative
\evolve{X}              % X(t) time evolution
```

---

## Theorem Environments

All papers use the same theorem environments (from `dnd_shared.sty`):

```latex
\begin{axiom}[name]
  Fundamental assumptions of D-ND framework
\end{axiom}

\begin{theorem}[name]
  Major theoretical results with implications
\end{theorem}

\begin{proposition}[name]
  Intermediate results leading to theorems
\end{proposition}

\begin{corollary}[name]
  Direct consequences of theorems
\end{corollary}

\begin{lemma}[name]
  Technical auxiliary results
\end{lemma}

\begin{definition}[name]
  Formal definitions of key concepts
\end{definition}

\begin{remark}
  Commentary on definitions or theorems
\end{remark}

\begin{note}
  Additional notes for readers
\end{note}
```

**Example:**
```latex
\begin{theorem}[Classical Correspondence]
In the limit where $\orderparam \to 1$, the D-ND equations reduce to
classical Hamiltonian mechanics.
\end{theorem}

\begin{proof}
... mathematical derivation ...
\end{proof}
```

---

## Section Structure (Inherited by All Papers)

Paper A (and recommended for all others):

```
1. Introduction
   1.1 Motivation and Context
   1.2 Paper Roadmap

2. Axioms of the D-ND Framework
   [4 formal axioms with definitions]

3. Emergence Dynamics
   3.1 The Emergence Operator
   3.2 Order Parameter Evolution
   3.3 Measurement and Emergence

4-5. Classical Limit and Effective Dynamics
   4.1 Transition to Classical Mechanics
   4.2 Correspondence with Classical Physics

6. Curvature and Differential Geometry
   6.1 Emergence Manifold
   6.2 Ricci Curvature

7. Experimental Signatures and Observations
   7.1 Signatures in Quantum Systems
   7.2 Cosmological Tests

8. Conclusion
```

---

## Journal-Specific Adaptation

Each target journal requires minimal changes. See **JOURNAL_ADAPTATION_GUIDE.md** for:

- ✓ Exact document class declarations
- ✓ Author/affiliation formatting
- ✓ Bibliography style recommendations
- ✓ Citation format specifications
- ✓ Detailed step-by-step adaptation instructions
- ✓ Estimated adaptation time for each journal
- ✓ Troubleshooting tips

**Quick reference:**

| Paper | Journal | Time | Effort | Class |
|-------|---------|------|--------|-------|
| A | Physical Review A | 0 min | None | revtex4-2 |
| B | Physical Review E | 5 min | Class change | revtex4-2 |
| C | Math Physics | 15 min | Class + author format | aip |
| D | Foundations | 20 min | Springer format | svjour3 |
| E | Quantum Gravity | 15 min | IOP format | iopart |
| F | Quantum | 5 min | Minimal | quantumarticle |
| G | Cognitive Science | 20 min | Springer format | svjour3 |

---

## Compilation Instructions

### Standard Workflow

```bash
# For any paper:
pdflatex paper_X_template.tex    # Generate .aux file
bibtex paper_X_template           # Process bibliography
pdflatex paper_X_template.tex    # Resolve references (1st pass)
pdflatex paper_X_template.tex    # Resolve references (2nd pass)
```

### With BibTeX File

If using a separate `references.bib` file:

```latex
% In document:
\bibliography{references}  % instead of \begin{thebibliography}
```

Then compile as above.

### Troubleshooting Compilation

**Error: Package not found**
```bash
# Ensure all required packages are installed:
# On Ubuntu/Debian:
sudo apt-get install texlive-full

# On macOS with MacTeX:
# Usually pre-installed; if not, use MacTeX installer
```

**Error: Document class not found**
```bash
# Install specific journal class files if needed:
# - revtex4-2: Usually in texlive-latex-extra
# - aip: May need manual installation
# - svjour3: Available from Springer
# - iopart: Available from IOP
# - quantumarticle: Available from Quantum journal
```

**Verify compilation:**
```bash
# Check if .pdf file is created:
ls -lh paper_A_template.pdf

# View file size (should be > 100KB):
file paper_A_template.pdf
```

---

## Key Features

### ✓ Journal Agnostic
- Single notation system across all seven papers
- Document class is the only element that changes
- All mathematical content is fully portable

### ✓ Consistent Cross-Referencing
- Uses `cleveref` for smart references: `\cref{sec:axioms}` → "Section 2"
- Hyperref integration for PDF navigation
- Consistent equation, figure, and table numbering

### ✓ Professional Formatting
- Appropriate theorems with automatic numbering
- Proper equation alignment and numbering
- Bibliography with citation support

### ✓ Notation Consistency
- All D-ND commands defined in one place (`dnd_shared.sty`)
- Changing notation requires editing only one file
- Automatic consistency across all seven papers

### ✓ Easy Maintenance
- Comments throughout explain structure
- Journal-specific notes in template footer (lines 900+)
- Clear separation of content and formatting

---

## Extended Use Cases

### Adding a New Journal

To add a new target journal:

1. **Create new template:**
   ```bash
   cp paper_A_template.tex paper_NEW_template.tex
   ```

2. **Update document class:**
   ```latex
   \documentclass[options]{newjournal}
   ```

3. **Adjust metadata:**
   ```latex
   \author{...}
   \affiliation{...}
   ```

4. **Test compilation** and update `JOURNAL_ADAPTATION_GUIDE.md`

### Modifying D-ND Notation

To add or change notation:

1. **Edit `dnd_shared.sty`:**
   ```latex
   \newcommand{\mynewcommand}{\text{definition}}
   ```

2. **Use in any paper:**
   ```latex
   This appears as $\mynewcommand$ in all papers
   ```

3. **No need to update individual papers** - they all inherit from the .sty file

### Creating Derived Works

To create a variation on a paper (e.g., extended version, condensed version):

1. Copy the base template
2. Keep all notation and theorem environments unchanged
3. Modify only the content and sections as needed
4. Compile using the same journal's document class

---

## Dependencies and Requirements

### LaTeX Packages Used

**Core Mathematics:**
- `amsmath` - Mathematical environments
- `amssymb` - Extended math symbols
- `mathrsfs` - Script fonts (ℰ, ℱ, etc.)
- `braket` - Quantum notation (⟨|⟩)
- `amsthm` - Theorem environments

**Document Control:**
- `hyperref` - PDF links and navigation
- `cleveref` - Smart cross-references
- `natbib` - Bibliography management

**Formatting:**
- `geometry` - Page layout
- `setspace` - Line spacing
- `graphicx` - Graphics inclusion
- `float` - Figure positioning

**Journal Classes:**
- `revtex4-2` (Physical Review journals)
- `aip` (AIP journals)
- `svjour3` (Springer journals)
- `iopart` (IOP journals)
- `quantumarticle` (Quantum journal)

### System Requirements

- **TeX Distribution:** TeXLive 2020+ or MacTeX 2020+
- **Disk Space:** ~100 MB for full TeXLive
- **Memory:** 512 MB RAM minimum
- **Compilation Time:** 30-60 seconds per paper

### Recommended Setup

```bash
# Ubuntu/Debian
sudo apt-get install texlive-full texlive-bibtex-extra

# macOS
brew install basictex  # or install MacTeX directly

# After installation, update packages:
tlmgr update --all
```

---

## Workflow Example: Adapting Paper A to Paper C

### Step 1: Copy Template
```bash
cp paper_A_template.tex paper_C_template.tex
```

### Step 2: Change Document Class

**Before:**
```latex
\documentclass[aps,pra,11pt,notitlepage,nofootinbib]{revtex4-2}
```

**After:**
```latex
\documentclass[11pt]{aip}
```

### Step 3: Update Author Format

**Before:**
```latex
\author{D-ND Research Collective}
\affiliation{Department of Theoretical Physics, Foundation Research Institute}
```

**After:**
```latex
\author{D-ND Research Collective}
\address{Department of Theoretical Physics, Foundation Research Institute}
```

### Step 4: Keep Everything Else

- All sections unchanged
- All `\NT`, `\emerge`, `\chiDND` commands still work
- All theorems and definitions format automatically
- All equations and cross-references compile unchanged

### Step 5: Compile

```bash
pdflatex paper_C_template.tex
bibtex paper_C_template
pdflatex paper_C_template.tex
pdflatex paper_C_template.tex
```

**That's it!** The paper is now formatted for Journal of Mathematical Physics.

---

## Best Practices

1. **Always keep `dnd_shared.sty` updated** - It's the source of truth for notation
2. **Use `\cref{}` instead of `\ref{}`** - Automatic text (Section, Equation, etc.)
3. **Label everything:** `\label{sec:...}`, `\label{eq:...}`, `\label{fig:...}`
4. **Use consistent label prefixes** - Makes cross-referencing easier
5. **Test locally before submission** - Catch formatting issues early
6. **Keep bibliography in BibTeX format** - Easier to maintain and switch styles
7. **Document custom commands** - Add comments for non-standard notation
8. **Version control your .tex files** - Track changes to content, not formatting

---

## Support

For issues with:

| Topic | See |
|-------|-----|
| D-ND notation | `dnd_shared.sty` (lines 1-150) |
| Theorem formatting | `paper_A_template.tex` (lines 250-450) |
| Journal adaptation | `JOURNAL_ADAPTATION_GUIDE.md` (this directory) |
| LaTeX documentation | Package documentation (e.g., `texdoc amsmath`) |
| Specific journal requirements | Journal submission guidelines |

---

## Summary

This LaTeX framework enables:

- **Single source** of notation and structure
- **Rapid adaptation** to new journals (≤5-20 minutes)
- **Consistent formatting** across all seven papers
- **Professional appearance** for high-impact journals
- **Easy maintenance** through shared `.sty` file

By using `dnd_shared.sty` and maintaining consistent document structure, all seven D-ND papers can be created, edited, and maintained with minimal duplicated effort while ensuring complete notational consistency.

---

**Files in this directory:**

```
paper_A_template.tex               27 KB    Main template (revtex4-2)
dnd_shared.sty                     4.6 KB   Shared notation package
JOURNAL_ADAPTATION_GUIDE.md        17 KB    Quick reference for all journals
README.md                          12 KB    This file
```

**Total package size:** ~61 KB (easily version controlled)

**Generated PDF per paper:** ~300-500 KB (varies by content)

---

*Last updated: 2025-02-14*
*D-ND Research Collective*

# D-ND Academic Suite: Journal Adaptation Guide

## Overview

This guide provides quick-reference instructions for adapting the Paper A template to each of the seven target journals in the D-ND academic suite.

## Master Template File

**Base template:** `paper_A_template.tex`
**Shared package:** `dnd_shared.sty` (included in all papers)

All papers use the same notation, theorem environments, and structure. Only the document class and journal-specific metadata change.

---

## Paper A: Physical Review A (revtex4-2)

**Status:** Reference implementation (no changes needed)

### Document Class
```latex
\documentclass[aps,pra,11pt,notitlepage,nofootinbib]{revtex4-2}
```

### Author/Affiliation
```latex
\author{D-ND Research Collective}
\affiliation{Department of Theoretical Physics, Foundation Research Institute}
```

### Key Features
- Automatic two-column formatting for journals
- Native support for `\cite{}` and `\ref{}`
- `thebibliography` environment for references
- Automatically handles header/footer for Physical Review A

### No Additional Changes Required
Use `paper_A_template.tex` as-is.

---

## Paper B: Physical Review E (revtex4-2)

**Status:** Minimal adaptation (sister journal to A)

### Document Class Change Only
```latex
% Change FROM:
\documentclass[aps,pra,11pt,notitlepage,nofootinbib]{revtex4-2}

% Change TO:
\documentclass[aps,pre,11pt,notitlepage,nofootinbib]{revtex4-2}
```

### Everything Else Remains Identical
- Author/affiliation format: same
- Sections and theorems: same
- Bibliography format: same
- Packages and notation: same

### Adaptation Time: 5 minutes

**Process:**
1. Copy `paper_A_template.tex` to `paper_B_template.tex`
2. Change `[aps,pra,` to `[aps,pre,` in documentclass
3. Adjust title and abstract for Paper B content
4. Keep all sections, commands, and notation unchanged

---

## Paper C: Journal of Mathematical Physics (aip)

**Status:** Moderate adaptation required

### Document Class
```latex
% Change FROM:
\documentclass[aps,pra,11pt,notitlepage,nofootinbib]{revtex4-2}

% Change TO:
\documentclass[11pt]{aip}
```

### Author/Affiliation Format
```latex
% Change FROM:
\author{D-ND Research Collective}
\affiliation{Department of Theoretical Physics, Foundation Research Institute}

% Change TO:
\author{D-ND Research Collective}
\address{Department of Theoretical Physics, Foundation Research Institute}
```

### Package Adjustments
- Most packages in template are compatible
- May remove: `nofootinbib` option (revtex4-2 specific)
- Keep all math and theorem packages unchanged

### Bibliography Format
AIP typically uses `\bibliography{}` command:
```latex
% Replace:
\begin{thebibliography}{99}
...\bibitem...
\end{thebibliography}

% With:
\bibliography{references}  % and use BibTeX file
```

### Theorem Styling
- AIP has specific theorem styling
- `amsthm` package automatically handles it
- Theorem environment names (Axiom, Theorem, etc.) remain unchanged

### Key Differences
- Single-column format (typically)
- Higher mathematical rigor emphasis
- Stricter equation numbering standards

### Adaptation Time: 15 minutes

**Process:**
1. Copy `paper_A_template.tex` to `paper_C_template.tex`
2. Change documentclass to `{11pt}{aip}`
3. Change `\affiliation{}` to `\address{}`
4. Consider using BibTeX for bibliography
5. Verify theorem formatting displays correctly

---

## Paper D: Foundations of Physics (springer / svjour3)

**Status:** Moderate adaptation required

### Document Class
```latex
% Change FROM:
\documentclass[aps,pra,11pt,notitlepage,nofootinbib]{revtex4-2}

% Change TO:
\documentclass{svjour3}
```

### Author/Affiliation Format
```latex
% Change FROM:
\author{D-ND Research Collective}
\affiliation{Department of Theoretical Physics, Foundation Research Institute}

% Change TO:
\author{D-ND Research Collective \and Co-Author Name}
\address{Department of Theoretical Physics, Foundation Research Institute}
\email{contact@institute.org}
```

### Title and Metadata
```latex
\title{Paper Title}
\subtitle{Optional subtitle}  % Can add if needed
\titlerunning{Short title for headers}  % Optional
\authorrunning{Author names for headers}  % Optional
```

### Package Adjustments
- Remove: `[aps,pra,11pt,notitlepage,nofootinbib]` options
- Keep all math packages unchanged
- Add if needed: `\usepackage{spbasic}` for Springer-specific formatting

### Theorem Styling
- Springer has specific theorem formatting
- `amsthm` package works, but Springer's `spthm` package offers alternatives
- Keep current theorem definitions (they'll adapt automatically)

### Bibliography Format
```latex
% Springer recommends:
\bibliography{references}  % Use BibTeX with plainnat style
```

### Citation Style
Springer typically uses numeric citations. Ensure citations match journal style.

### Key Differences
- Two-column format standard
- Springer-specific metadata in header/footer
- Different header/footer management
- Citation numbering may differ

### Adaptation Time: 20 minutes

**Process:**
1. Copy `paper_A_template.tex` to `paper_D_template.tex`
2. Change documentclass to `{svjour3}`
3. Update author/affiliation to Springer format
4. Add `\titlerunning` and `\authorrunning` if desired
5. Switch to BibTeX bibliography with `plainnat` style
6. Test theorem formatting

---

## Paper E: Classical and Quantum Gravity (iopart)

**Status:** Moderate adaptation required

### Document Class
```latex
% Change FROM:
\documentclass[aps,pra,11pt,notitlepage,nofootinbib]{revtex4-2}

% Change TO:
\documentclass{iopart}
```

### Author/Affiliation/Email Format
```latex
% Change FROM:
\author{D-ND Research Collective}
\affiliation{Department of Theoretical Physics, Foundation Research Institute}

% Change TO:
\author{D-ND Research Collective$^1$}
\address{$^1$Department of Theoretical Physics, Foundation Research Institute}
\ead{contact@institute.org}
```

### Abstract Format
```latex
% Keep same:
\begin{abstract}
...content...
\end{abstract}

% Can add keywords:
\begin{keywords}
Quantum gravity, D-ND framework, emergence
\end{keywords}
```

### Package Adjustments
- Remove revtex4-2 specific options
- Keep all math and theorem packages
- iopart provides its own formatting for references

### Bibliography Format
IOP typically accepts:
```latex
\begin{thebibliography}{99}
% or use \bibliography{references}
```

### Theorem Styling
- `amsthm` package works well with iopart
- Theorem numbering will follow iopart standards automatically

### Key Differences
- Popular for relativity, quantum gravity, mathematical physics
- Two-column format standard
- Specific formatting for equations and references
- Special support for quantum gravity notation

### Adaptation Time: 15 minutes

**Process:**
1. Copy `paper_A_template.tex` to `paper_E_template.tex`
2. Change documentclass to `{iopart}`
3. Update author/affiliation to iopart format with superscript addresses
4. Add `\ead{}` for email
5. Keep `\begin{thebibliography}` or switch to BibTeX
6. Optionally add keywords section

---

## Paper F: Quantum (quantumarticle)

**Status:** Minimal adaptation (highly compatible)

### Document Class
```latex
% Change FROM:
\documentclass[aps,pra,11pt,notitlepage,nofootinbib]{revtex4-2}

% Change TO:
\documentclass{quantumarticle}
```

### Author/Affiliation Format
```latex
% Change FROM:
\author{D-ND Research Collective}
\affiliation{Department of Theoretical Physics, Foundation Research Institute}

% Change TO:
\author{D-ND Research Collective}
\affiliation{Department of Theoretical Physics, Foundation Research Institute}
```
(Same format - very compatible!)

### Abstract and Keywords
```latex
\begin{abstract}
...content...
\end{abstract}

\begin{keywords}
quantum information, emergence, D-ND framework
\end{keywords}
```

### Package Adjustments
- `quantumarticle` is specifically designed for modern quantum research
- All packages in template are compatible
- Hyperref is already configured well

### Bibliography Format
Quantumarticle supports both BibTeX and `thebibliography`:
```latex
% Option 1: BibTeX
\bibliography{references}

% Option 2: Manual (thebibliography) - already in template
\begin{thebibliography}{99}
...\bibitem...
\end{thebibliography}
```

### Special Features
- Excellent PDF rendering
- Cross-referencing with `cleveref` very well supported
- Open-access publication model
- Modern, clean typesetting

### Key Differences
- Quantum journal (modern, open-access)
- Single-column format standard
- Exceptional cross-reference support
- Very compatible with hyperref

### Adaptation Time: 5 minutes

**Process:**
1. Copy `paper_A_template.tex` to `paper_F_template.tex`
2. Change documentclass to `{quantumarticle}`
3. Keep author/affiliation format (already compatible)
4. Add keywords section
5. Keep bibliography as-is
6. All other sections unchanged

---

## Paper G: Cognitive Science / Minds and Machines (springer / svjour3)

**Status:** Moderate adaptation (same as Paper D, different content)

### Document Class
```latex
% Change FROM:
\documentclass[aps,pra,11pt,notitlepage,nofootinbib]{revtex4-2}

% Change TO:
\documentclass{svjour3}
```

### Author/Affiliation Format
```latex
% Same as Paper D:
\author{D-ND Research Collective \and Co-Author Name}
\address{Department of Theoretical Physics, Foundation Research Institute}
\email{contact@institute.org}
```

### Title and Metadata
```latex
\title{D-ND Framework and Cognitive Science}
\subtitle{Bridging Physics and Consciousness Studies}
\titlerunning{D-ND and Cognitive Science}
\authorrunning{DND Research Collective}
```

### Package Adjustments
- Same as Paper D
- All math and theorem packages fully compatible

### Specific Content Adjustments
For cognitive science emphasis:
```latex
% Enhance philosophical/cognitive sections
% Add discussion of:
% - Conscious emergence (Section 3)
% - Cognitive timescale implications (Paper G specific)
% - Neural correlates of emergence
% - Philosophy of mind connections
```

### Bibliography Format
Same as Paper D (Springer standard).

### Key Differences
- Interdisciplinary journal (physics + cognitive science)
- Emphasis on consciousness and emergence
- More philosophical discussion expected
- Bridge between physics and mind

### Adaptation Time: 20 minutes

**Process:**
1. Copy `paper_A_template.tex` to `paper_G_template.tex`
2. Change documentclass to `{svjour3}` (like Paper D)
3. Update author/affiliation to Springer format
4. Enhance title to emphasize cognitive aspects
5. Expand Section 3 (Emergence) with cognitive implications
6. Add subsections on consciousness and neural emergence
7. Switch to BibTeX with `plainnat` style
8. Keep all mathematical sections (they apply to cognitive physics too)

---

## Quick Adaptation Checklist

### For All Papers (≤5 minutes):
- [ ] Change `\documentclass` line
- [ ] Update `\author`, `\title`, `\affiliation` (or equivalent)
- [ ] Keep `\usepackage{dnd_shared}` unchanged
- [ ] Keep all sections and theorems unchanged
- [ ] Keep all D-ND notation from `dnd_shared.sty`

### For AIP, IOP, Springer (≤15 minutes):
- [ ] Complete "All Papers" checklist above
- [ ] Adjust address/affiliation format
- [ ] Review theorem formatting
- [ ] Consider BibTeX bibliography setup
- [ ] Add journal-specific metadata if needed

### For Content Specialization (≤30 minutes):
- [ ] Complete checklist items above
- [ ] Reorder/add sections for journal emphasis
- [ ] Adjust abstract for journal focus
- [ ] Customize keywords for journal scope
- [ ] Verify citation style matches journal standards

---

## File Organization

```
domain/AWARENESS/3_π_PRAGMATIC/latex/
├── dnd_shared.sty                      # Shared package (used by all)
├── paper_A_template.tex                # Physical Review A (reference)
├── paper_B_template.tex                # Physical Review E (adapt from A)
├── paper_C_template.tex                # Journal of Mathematical Physics (aip)
├── paper_D_template.tex                # Foundations of Physics (springer)
├── paper_E_template.tex                # Classical and Quantum Gravity (iopart)
├── paper_F_template.tex                # Quantum (quantumarticle)
├── paper_G_template.tex                # Cognitive Science (springer)
└── JOURNAL_ADAPTATION_GUIDE.md         # This file
```

---

## Document Class Reference

| Paper | Journal | Class | Options |
|-------|---------|-------|---------|
| A | Physical Review A | revtex4-2 | `[aps,pra,11pt,notitlepage]` |
| B | Physical Review E | revtex4-2 | `[aps,pre,11pt,notitlepage]` |
| C | J. Mathematical Physics | aip | `[11pt]` |
| D | Foundations of Physics | svjour3 | (none) |
| E | Classical & Quantum Gravity | iopart | (none) |
| F | Quantum | quantumarticle | (none) |
| G | Cognitive Science | svjour3 | (none) |

---

## Commands and Notation

All seven papers use the same D-ND notation defined in `dnd_shared.sty`:

### Core D-ND Commands
```latex
\NT              % |NT⟩ quantum state
\emerge          % ℰ emergence operator
\emeasure        % M(t) emergence measurement
\orderparam      % Z(t) order parameter
\resultant       % R(t) resultant parameter
\Kgen            % K_gen generator constant
\chiDND          % χ_DND DND coupling
\OmegaNT         % Ω_NT NT frequency
\rhoDND          % ρ_DND DND density
\rhoLECO         % ρ_LECO LECO density
\sigmaE          % σ²_ℰ emergence variance
\sigmaV          % σ²_V velocity variance
\lambdak         % λ_k k-mode coupling
\lambdaDND       % λ_DND DND coupling
\lambdaauto      % λ_auto autonomous coupling
\lambdacosmo     % λ_cosmo cosmological coupling
\Tcog            % T_cog cognitive timescale
\GS              % G_S singularity constant
```

### Theorem Environments (all papers)
```latex
\begin{axiom}
\begin{theorem}
\begin{proposition}
\begin{corollary}
\begin{lemma}
\begin{definition}
\begin{remark}
\begin{note}
```

---

## Troubleshooting

### Issue: Document class not found
**Solution:** Ensure the LaTeX distribution includes the class file. For Springer, install `svjour3.cls`. For iopart, install `iopart.cls`.

### Issue: Theorem environments not displaying correctly
**Solution:** Verify `\usepackage{amsthm}` is included and not conflicting with journal-specific theorem packages.

### Issue: D-ND commands undefined
**Solution:** Ensure `dnd_shared.sty` is in the same directory as the .tex file, and `\usepackage{dnd_shared}` is called before theorem definitions.

### Issue: Bibliography formatting differs from journal requirements
**Solution:** Use BibTeX with the appropriate style file:
- Physical Review: `revtex4-2` provides automatic styling
- AIP: Use `aip.bst` or `plainnat`
- Springer: Use `plainnat` or `spmpsci`
- IOP: Use `iopart.bst`
- Quantum: Use `plainnat` or `plain`

### Issue: PDF compilation fails
**Solution:** Check for:
1. Missing packages (install with `\usepackage{}`)
2. Incompatible package combinations (check documentation)
3. Class file path issues (ensure classes are in TeX path)
4. Syntax errors in custom commands (check `dnd_shared.sty`)

---

## Compilation Examples

### Compile Paper A (revtex4-2)
```bash
pdflatex paper_A_template.tex
bibtex paper_A_template
pdflatex paper_A_template.tex
pdflatex paper_A_template.tex
```

### Compile Paper C (aip)
```bash
pdflatex paper_C_template.tex
bibtex paper_C_template
pdflatex paper_C_template.tex
pdflatex paper_C_template.tex
```

### Compile Paper D (springer)
```bash
pdflatex paper_D_template.tex
bibtex paper_D_template
pdflatex paper_D_template.tex
pdflatex paper_D_template.tex
```

---

## Best Practices

1. **Keep `dnd_shared.sty` unchanged** - Use for all papers
2. **Test each journal's class** - Verify document compiles before content creation
3. **Use `\cref{}` and `\Cref{}`** - From `cleveref` for smart cross-references
4. **Maintain notation consistency** - All D-ND commands from shared package
5. **Document journal-specific changes** - Comment out changes for easy switching
6. **Use BibTeX** - Better bibliography management across journal formats
7. **Generate PDFs early** - Catch formatting issues before final submission

---

## Support and Maintenance

For questions about:
- **D-ND notation:** See `dnd_shared.sty`
- **Theorem formatting:** Check `paper_A_template.tex` Sections 2-3
- **Journal requirements:** Consult journal submission guidelines
- **LaTeX syntax:** Refer to package documentation (amsmath, amsthm, etc.)

---

*Last updated: 2025-02-14*
*For the D-ND Research Collective*

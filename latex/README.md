# How to Edit & Compile This Paper

## 1 — Install LaTeX (one-time)

Open **Terminal** and run:

```bash
brew install --cask basictex
```

> If you don't have Homebrew yet, install it first:
> ```bash
> /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
> ```

After BasicTeX finishes installing, **restart Terminal**, then install the packages this paper needs:

```bash
eval "$(/usr/libexec/path_helper)"

tlmgr --usermode install \
  booktabs float microtype biblatex biblatex-ieee \
  fira stix2-type1 stix2-otf \
  cleveref caption colortbl enumitem fancyhdr \
  hyperref xurl bookmark titlesec titling \
  mdframed zref needspace xifthen ifmtarg etoolbox \
  fontaxes adjustbox collectbox environ trimspaces \
  csquotes logreq listings listingsutf8 \
  lastpage lettrine setspace sidecap supertabular \
  pbalance sttools tcolorbox tikzfill \
  footmisc ragged2e multirow here
```

This takes about 2–3 minutes.

---

## 2 — Install the VS Code Extension

1. Open VS Code
2. Go to **Extensions** (⇧⌘X)
3. Search for **LaTeX Workshop** (by James Yu) and install it

---

## 3 — Open and Compile

1. Open the `latex/` folder in VS Code (**File → Open Folder**)
2. Open `main.tex`
3. Press **⌘⇧P** → type **LaTeX Workshop: Build LaTeX project** → Enter

The PDF will open automatically in a side panel. Any time you save `main.tex` (⌘S), it recompiles automatically.

---

## 4 — Manual compile (Terminal)

If you prefer the terminal, run this from inside the `latex/` folder:

```bash
eval "$(/usr/libexec/path_helper)"
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Then open `main.pdf` with any PDF viewer.

> **Quick rebuild** (text edits only, no new citations):
> ```bash
> pdflatex main.tex
> ```

---

## File Structure

```
latex/
├── main.tex          ← the paper (edit this)
├── rho.bib           ← bibliography entries
├── rho-class/        ← journal template (don't edit)
│   ├── rho.cls
│   ├── rhobabel.sty
│   └── rhoenvs.sty
└── figures/          ← all images used in the paper
```

---

## Overleaf (no install needed)

If you'd rather skip the local setup, zip the entire `latex/` folder and upload it to [overleaf.com](https://www.overleaf.com) via **New Project → Upload Project**. It compiles in the browser with no installation required.

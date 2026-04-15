# LUAD Transcriptomic Biomarker Discovery

This project is a publication-style cancer transcriptomics analysis focused on **lung adenocarcinoma (LUAD)**. It uses public TCGA/UCSC Xena RNA-seq resources to build a reproducible workflow for:

- tumor-versus-normal differential-expression screening,
- pathway enrichment,
- sparse classification,
- and survival-oriented biomarker prioritization.

## Portfolio value

This project is designed to read well for:

- biotech and bioinformatics internships,
- undergraduate research applications,
- computational biology portfolio reviews,
- and early translational-data-science roles.

It stands out because it shows more than plotting ability. The workflow goes from public data acquisition to statistical testing, machine-learning evaluation, survival analysis, and publishable reporting in a single reproducible project.

## Why this project matters

For biotech, molecular biology, translational research, and bioinformatics internships, this project is a much stronger signal than a generic classroom analysis because it demonstrates:

- public cancer-omics data acquisition,
- reproducible computational pipeline development,
- quantitative analysis of RNA-seq data,
- pathway-level biological interpretation,
- and publication-style scientific communication.

## Project structure

- `index.qmd` — main paper
- `appendix.qmd` — reproducibility notes and extensions
- `src/analysis.py` — complete Python pipeline
- `references.bib` — bibliography for the paper
- `styles.css` — light publication styling
- `requirements.txt` — Python dependencies

## Setup

Create and activate a Python environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Install Quarto if you do not already have it:

- https://quarto.org/docs/download/

Confirm Quarto is available:

```bash
quarto check
```

## Run the project

From the project directory:

```bash
quarto render
```

The render process will:

1. download the LUAD expression, clinical, and survival files if they are not already cached;
2. run the transcriptomic analysis pipeline;
3. save all figures and tables into `results/`;
4. build the publication into `docs/`.

Open the site locally with:

```bash
open docs/index.html
```

## Publish

### Recommended: GitHub Pages

This project is already configured with `output-dir: docs`, which makes GitHub Pages the easiest publishing path.

```bash
git init
git add .
git commit -m "Initial LUAD biomarker project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
git push -u origin main
```

Then in GitHub:

1. open your repository settings;
2. go to **Pages**;
3. set the source to **Deploy from a branch**;
4. choose the `main` branch and the `/docs` folder.

Your project site will then publish from the rendered HTML in `docs/`.

### Alternative: Quarto Pub

If you want the fastest no-GitHub option:

```bash
quarto publish quarto-pub
```

This is simple for sharing a portfolio link, but GitHub Pages is usually better if you also want the repository itself to be visible to recruiters.

## Notes

- The LUAD `HiSeqV2` matrix is distributed as log2-transformed normalized expression, so this project uses transparent screening-style differential analysis rather than a raw-count negative-binomial model.
- Enrichment analysis is coded to run automatically when internet access is available.
- If one of the UCSC Xena endpoints changes, update the URL lists near the top of `src/analysis.py`.

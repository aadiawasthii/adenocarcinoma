# LUAD Transcriptomic Biomarker Discovery

This project is a publication-style cancer transcriptomics analysis focused on **lung adenocarcinoma (LUAD)**. It uses public TCGA/UCSC Xena RNA-seq resources to build a reproducible workflow for:

- tumor-versus-normal differential-expression screening,
- pathway enrichment,
- sparse classification,
- and survival-oriented biomarker prioritization.




## Notes

- The LUAD `HiSeqV2` matrix is distributed as log2-transformed normalized expression, so this project uses transparent screening-style differential analysis rather than a raw-count negative-binomial model.
- Enrichment analysis is coded to run automatically when internet access is available.
- If one of the UCSC Xena endpoints changes, update the URL lists near the top of `src/analysis.py`.

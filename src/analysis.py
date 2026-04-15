from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns

try:
    import gseapy as gp
except Exception:
    gp = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
ENRICH_DIR = RESULTS_DIR / "enrichment"

EXPRESSION_URLS = [
    "https://tcga.xenahubs.net/download/TCGA.LUAD.sampleMap/HiSeqV2.gz",
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LUAD.sampleMap%2FHiSeqV2.gz",
]
CLINICAL_URLS = [
    "https://tcga.xenahubs.net/download/TCGA.LUAD.sampleMap/LUAD_clinicalMatrix.gz",
    "https://tcga.xenahubs.net/download/TCGA.LUAD.sampleMap/LUAD_clinicalMatrix",
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LUAD.sampleMap%2FLUAD_clinicalMatrix.gz",
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LUAD.sampleMap%2FLUAD_clinicalMatrix",
]
SURVIVAL_URLS = [
    "https://toil.xenahubs.net/download/TCGA_survival_data",
    "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA_survival_data",
]

SAMPLE_TYPE_MAP = {
    "01": "Primary tumor",
    "11": "Solid tissue normal",
}

PRIMARY_COHORT = "LUAD"
PRIMARY_DISEASE = "lung adenocarcinoma"


@dataclass
class PipelineOutputs:
    sample_summary: pd.DataFrame
    differential_expression: pd.DataFrame
    classifier_metrics: dict
    top_biomarker: dict
    stage_available: bool
    enrichment_available: bool


def ensure_directories() -> None:
    for path in [RAW_DIR, PROC_DIR, FIG_DIR, TABLE_DIR, ENRICH_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def download_first_available(urls: Iterable[str], destination: Path, timeout: int = 120) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    last_error = None
    for url in urls:
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with destination.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if destination.exists() and destination.stat().st_size > 0:
                return destination
        except Exception as exc:  # pragma: no cover - network-dependent
            last_error = exc
            if destination.exists():
                destination.unlink(missing_ok=True)
            continue

    raise RuntimeError(f"Unable to download {destination.name}. Last error: {last_error}")


def ensure_input_data(force_download: bool = False) -> dict[str, Path]:
    ensure_directories()
    expr_path = RAW_DIR / "TCGA.LUAD.sampleMap_HiSeqV2.gz"
    clin_path = RAW_DIR / "TCGA.LUAD.sampleMap_LUAD_clinicalMatrix"
    surv_path = RAW_DIR / "TCGA_survival_data"

    if force_download:
        for path in [expr_path, clin_path, surv_path]:
            path.unlink(missing_ok=True)

    download_first_available(EXPRESSION_URLS, expr_path)
    download_first_available(CLINICAL_URLS, clin_path)
    download_first_available(SURVIVAL_URLS, surv_path)
    return {"expression": expr_path, "clinical": clin_path, "survival": surv_path}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
    return df


def parse_tcga_sample_type(sample_id: str) -> str | None:
    sample_id = str(sample_id)
    match = re.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-(\d{2})", sample_id)
    if not match:
        return None
    return SAMPLE_TYPE_MAP.get(match.group(1))


def load_expression_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, sep="\t", compression="infer")
    gene_col = raw.columns[0]
    expr = raw.rename(columns={gene_col: "gene_id"}).set_index("gene_id")
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.loc[expr.notna().any(axis=1)]

    gene_map = expr.index.to_series().str.split("|", n=1, expand=True)
    feature_meta = pd.DataFrame({"gene_id": expr.index, "gene_symbol": gene_map[0].values})
    feature_meta["gene_symbol"] = feature_meta["gene_symbol"].replace({"?": np.nan})
    feature_meta = feature_meta.dropna(subset=["gene_symbol"])

    expr = expr.loc[feature_meta["gene_id"]]
    expr.index = feature_meta["gene_symbol"].values
    expr = expr.groupby(expr.index).median()

    sample_types = pd.Series({col: parse_tcga_sample_type(col) for col in expr.columns}, name="sample_type")
    keep_samples = sample_types[sample_types.isin(SAMPLE_TYPE_MAP.values())].index.tolist()
    expr = expr[keep_samples].transpose()
    expr.index.name = "sample"

    sample_meta = pd.DataFrame({
        "sample": expr.index,
        "sample_type": [sample_types[s] for s in expr.index],
        "patient_id": [str(s)[:12] for s in expr.index],
    })
    sample_meta["is_tumor"] = (sample_meta["sample_type"] == "Primary tumor").astype(int)
    return expr, sample_meta


def load_clinical_matrix(path: Path) -> pd.DataFrame:
    clinical = pd.read_csv(path, sep="\t", compression="infer")
    clinical = _standardize_columns(clinical)
    sample_col = clinical.columns[0]
    clinical = clinical.rename(columns={sample_col: "sample"})
    clinical["sample"] = clinical["sample"].astype(str)
    clinical["patient_id"] = clinical["sample"].str[:12]
    return clinical


def load_survival_data(path: Path) -> pd.DataFrame:
    survival = pd.read_csv(path, sep="\t")
    survival = _standardize_columns(survival)

    if "sample" not in survival.columns:
        sample_like = [c for c in survival.columns if c in {"sampleid", "sample_id", "barcode", "submitter_id"}]
        if sample_like:
            survival = survival.rename(columns={sample_like[0]: "sample"})
        else:
            survival = survival.rename(columns={survival.columns[0]: "sample"})

    survival["sample"] = survival["sample"].astype(str)
    survival["patient_id"] = survival["sample"].str[:12]
    return survival


def merge_metadata(sample_meta: pd.DataFrame, clinical: pd.DataFrame, survival: pd.DataFrame) -> pd.DataFrame:
    merged = sample_meta.merge(clinical, on=["sample", "patient_id"], how="left", suffixes=("", "_clinical"))
    merged = merged.merge(survival, on=["patient_id"], how="left", suffixes=("", "_survival"))
    return merged


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(np.nan_to_num(pvals, nan=1.0))
    ranked = pvals[order]
    adjusted = np.empty(n, dtype=float)
    cumulative = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        cumulative = min(cumulative, val)
        adjusted[i] = cumulative
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def compute_differential_expression(expr: pd.DataFrame, sample_meta: pd.DataFrame) -> pd.DataFrame:
    y = sample_meta.set_index("sample").loc[expr.index, "is_tumor"].values.astype(bool)
    tumor = expr.loc[y, :]
    normal = expr.loc[~y, :]

    mean_tumor = tumor.mean(axis=0)
    mean_normal = normal.mean(axis=0)
    std_tumor = tumor.std(axis=0, ddof=1)
    std_normal = normal.std(axis=0, ddof=1)
    n_tumor = tumor.shape[0]
    n_normal = normal.shape[0]
    pooled_sd = np.sqrt(((n_tumor - 1) * (std_tumor ** 2) + (n_normal - 1) * (std_normal ** 2)) / max(n_tumor + n_normal - 2, 1))
    pooled_sd = pooled_sd.replace(0, np.nan)

    t_stat, pvals = stats.ttest_ind(tumor.values, normal.values, axis=0, equal_var=False, nan_policy="omit")
    fdr = bh_fdr(np.asarray(pvals, dtype=float))

    de = pd.DataFrame({
        "gene_symbol": expr.columns,
        "mean_tumor": mean_tumor.values,
        "mean_normal": mean_normal.values,
        "log2_fold_change": (mean_tumor - mean_normal).values,
        "t_statistic": t_stat,
        "p_value": pvals,
        "fdr": fdr,
        "cohens_d": ((mean_tumor - mean_normal) / pooled_sd).values,
        "abs_log2_fc": np.abs((mean_tumor - mean_normal).values),
    })
    de = de.replace([np.inf, -np.inf], np.nan)
    de = de.sort_values(["fdr", "abs_log2_fc"], ascending=[True, False]).reset_index(drop=True)
    return de


def summarize_samples(sample_meta: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    summary = sample_meta.groupby("sample_type").agg(samples=("sample", "count"), patients=("patient_id", "nunique"))
    stage_col = find_stage_column(metadata)
    if stage_col:
        summary.loc[:, "non_missing_stage_annotations"] = metadata.groupby("sample_type")[stage_col].apply(lambda s: s.notna().sum())
    return summary.reset_index()


def find_stage_column(metadata: pd.DataFrame) -> str | None:
    candidates = [
        "pathologic_stage",
        "ajcc_pathologic_tumor_stage",
        "tumor_stage",
        "pathologic_t_stage",
        "clinical_stage",
    ]
    for col in candidates:
        if col in metadata.columns:
            return col
    for col in metadata.columns:
        if "stage" in col and metadata[col].notna().sum() > 20:
            return col
    return None


def select_top_heatmap_genes(de: pd.DataFrame, up_n: int = 15, down_n: int = 15) -> list[str]:
    up = de[(de["fdr"] < 0.01) & (de["log2_fold_change"] > 1)].head(up_n)["gene_symbol"].tolist()
    down = de[(de["fdr"] < 0.01) & (de["log2_fold_change"] < -1)].head(down_n)["gene_symbol"].tolist()
    genes = [g for g in up + down if isinstance(g, str)]
    return genes


def cross_validated_classifier(expr: pd.DataFrame, sample_meta: pd.DataFrame, de: pd.DataFrame, top_n: int = 50) -> dict:
    feature_genes = de.loc[de["fdr"] < 0.01, "gene_symbol"].head(top_n).tolist()
    if len(feature_genes) < 5:
        feature_genes = de["gene_symbol"].head(top_n).tolist()

    X = expr[feature_genes].values
    y = sample_meta.set_index("sample").loc[expr.index, "is_tumor"].values

    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, penalty="l1", solver="liblinear", random_state=42)),
    ])

    aucs, aps = [], []
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []

    for train_idx, test_idx in splitter.split(X, y):
        pipeline.fit(X[train_idx], y[train_idx])
        probs = pipeline.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], probs))
        aps.append(average_precision_score(y[test_idx], probs))
        fpr, tpr, _ = roc_curve(y[test_idx], probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    metrics = {
        "n_features": len(feature_genes),
        "features": feature_genes,
        "mean_roc_auc": float(np.mean(aucs)),
        "sd_roc_auc": float(np.std(aucs, ddof=1)),
        "mean_average_precision": float(np.mean(aps)),
        "sd_average_precision": float(np.std(aps, ddof=1)),
        "mean_fpr": mean_fpr.tolist(),
        "mean_tpr": mean_tpr.tolist(),
    }
    return metrics


def compute_gene_auc(expr: pd.DataFrame, sample_meta: pd.DataFrame, genes: list[str]) -> pd.Series:
    y = sample_meta.set_index("sample").loc[expr.index, "is_tumor"].values
    aucs = {}
    for gene in genes:
        try:
            aucs[gene] = roc_auc_score(y, expr[gene].values)
        except Exception:
            continue
    return pd.Series(aucs)


def find_survival_columns(survival_df: pd.DataFrame) -> tuple[str | None, str | None]:
    time_candidates = ["os_time", "os_time_days", "overall_survival_time", "time", "time_to_event", "days_to_death", "days_to_last_followup"]
    status_candidates = ["os", "vital_status", "status", "event", "os_event", "overall_survival"]

    time_col = next((c for c in time_candidates if c in survival_df.columns), None)
    status_col = next((c for c in status_candidates if c in survival_df.columns), None)

    return time_col, status_col


def normalize_status(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = set(series.dropna().astype(float).unique().tolist())
        if unique_vals.issubset({0.0, 1.0}):
            return series.astype(float)
        if unique_vals.issubset({1.0, 2.0}):
            return series.astype(float).map({1.0: 0.0, 2.0: 1.0})
    mapping = {
        "alive": 0.0,
        "deceased": 1.0,
        "dead": 1.0,
        "living": 0.0,
        "yes": 1.0,
        "no": 0.0,
    }
    return series.astype(str).str.lower().map(mapping)


def is_interpretable_gene_symbol(gene_symbol: str) -> bool:
    if not isinstance(gene_symbol, str) or not gene_symbol:
        return False
    upper = gene_symbol.upper()
    if upper.startswith("LOC"):
        return False
    if re.search(r"\d+ORF\d+", upper):
        return False
    return True


def evaluate_prognostic_candidates(expr: pd.DataFrame, sample_meta: pd.DataFrame, de: pd.DataFrame, survival: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    tumor_samples = sample_meta.loc[sample_meta["sample_type"] == "Primary tumor", ["sample", "patient_id"]].drop_duplicates()
    tumor_expr = expr.loc[tumor_samples["sample"]].copy()
    tumor_expr["patient_id"] = tumor_samples.set_index("sample").loc[tumor_expr.index, "patient_id"].values
    tumor_expr = tumor_expr.groupby("patient_id").mean(numeric_only=True)

    survival = survival.copy()
    time_col, status_col = find_survival_columns(survival)
    if not time_col or not status_col:
        return {"available": False, "reason": "Could not identify survival columns."}, pd.DataFrame()

    survival = survival[["patient_id", time_col, status_col]].drop_duplicates(subset=["patient_id"])
    survival[time_col] = pd.to_numeric(survival[time_col], errors="coerce")
    survival[status_col] = normalize_status(survival[status_col])
    survival = survival.dropna(subset=[time_col, status_col])
    survival = survival[survival[time_col] > 0]

    ranked_upregulated = de[(de["fdr"] < 0.01) & (de["log2_fold_change"] > 1)]["gene_symbol"].tolist()
    if not ranked_upregulated:
        ranked_upregulated = de[de["log2_fold_change"] > 0]["gene_symbol"].tolist()

    preferred_genes = [gene for gene in ranked_upregulated if is_interpretable_gene_symbol(gene)]
    candidate_genes = preferred_genes[:25] or ranked_upregulated[:25]
    selection_strategy = (
        "interpretable_named_genes"
        if preferred_genes
        else "all_significant_upregulated_genes"
    )

    rows = []
    for gene in candidate_genes:
        if gene not in tumor_expr.columns:
            continue
        joined = survival.merge(tumor_expr[[gene]], left_on="patient_id", right_index=True, how="inner")
        if joined.shape[0] < 40:
            continue
        median_val = joined[gene].median()
        joined["group"] = np.where(joined[gene] >= median_val, "High", "Low")
        if joined["group"].nunique() < 2:
            continue

        high = joined[joined["group"] == "High"]
        low = joined[joined["group"] == "Low"]
        logrank = logrank_test(high[time_col], low[time_col], high[status_col], low[status_col])

        cph_df = joined[[time_col, status_col, gene]].rename(columns={time_col: "time", status_col: "status", gene: "expression"})
        try:
            cph = CoxPHFitter()
            cph.fit(cph_df, duration_col="time", event_col="status")
            hr = float(math.exp(cph.params_["expression"]))
            cox_p = float(cph.summary.loc["expression", "p"])
        except Exception:
            hr = np.nan
            cox_p = np.nan

        rows.append({
            "gene_symbol": gene,
            "n_patients": int(joined.shape[0]),
            "logrank_p": float(logrank.p_value),
            "hazard_ratio": hr,
            "cox_p": cox_p,
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return {"available": False, "reason": "No candidate genes had sufficient survival information."}, pd.DataFrame()

    summary = summary.sort_values(["logrank_p", "cox_p"], ascending=[True, True]).reset_index(drop=True)
    top_gene = summary.iloc[0]["gene_symbol"]
    top_info = {
        "available": True,
        **summary.iloc[0].to_dict(),
        "time_col": time_col,
        "status_col": status_col,
        "selection_strategy": selection_strategy,
    }
    return top_info, summary


def save_summary_json(sample_summary: pd.DataFrame, de: pd.DataFrame, classifier_metrics: dict, biomarker: dict) -> None:
    up = int(((de["fdr"] < 0.05) & (de["log2_fold_change"] > 1)).sum())
    down = int(((de["fdr"] < 0.05) & (de["log2_fold_change"] < -1)).sum())
    summary = {
        "sample_counts": sample_summary.to_dict(orient="records"),
        "n_significant_upregulated": up,
        "n_significant_downregulated": down,
        "classifier": classifier_metrics,
        "top_biomarker": biomarker,
    }
    with (TABLE_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def make_sample_distribution_plot(sample_summary: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4.5))
    ax = sns.barplot(data=sample_summary, x="sample_type", y="samples")
    ax.set_xlabel("")
    ax.set_ylabel("Samples")
    ax.set_title("LUAD cohort composition")
    ax.bar_label(ax.containers[0], padding=3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sample_counts.png", dpi=300)
    plt.close()


def make_pca_plot(expr: pd.DataFrame, sample_meta: pd.DataFrame) -> None:
    variances = expr.var(axis=0).sort_values(ascending=False)
    top_genes = variances.head(min(1000, len(variances))).index.tolist()
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(expr[top_genes].values)

    plot_df = sample_meta.set_index("sample").loc[expr.index].copy()
    plot_df["PC1"] = pcs[:, 0]
    plot_df["PC2"] = pcs[:, 1]

    plt.figure(figsize=(7, 5.5))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="sample_type", s=55, alpha=0.85)
    plt.title("PCA of LUAD transcriptomes")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.legend(title="Group")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pca.png", dpi=300)
    plt.close()


def make_volcano_plot(de: pd.DataFrame) -> None:
    plot_df = de.copy()
    plot_df["neg_log10_fdr"] = -np.log10(plot_df["fdr"].clip(lower=1e-300))
    plot_df["class"] = "Not significant"
    plot_df.loc[(plot_df["fdr"] < 0.05) & (plot_df["log2_fold_change"] > 1), "class"] = "Upregulated"
    plot_df.loc[(plot_df["fdr"] < 0.05) & (plot_df["log2_fold_change"] < -1), "class"] = "Downregulated"

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="log2_fold_change", y="neg_log10_fdr", hue="class", s=18, alpha=0.8, palette={
        "Upregulated": "#B22222",
        "Downregulated": "#1f77b4",
        "Not significant": "#BEBEBE",
    })
    plt.axvline(1, color="black", linestyle="--", linewidth=1)
    plt.axvline(-1, color="black", linestyle="--", linewidth=1)
    plt.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=1)
    plt.xlabel("Tumor - normal log2 expression difference")
    plt.ylabel("-log10(FDR)")
    plt.title("Differential-expression landscape in LUAD")

    for _, row in pd.concat([
        plot_df[plot_df["class"] == "Upregulated"].nlargest(6, "neg_log10_fdr"),
        plot_df[plot_df["class"] == "Downregulated"].nlargest(6, "neg_log10_fdr"),
    ]).iterrows():
        plt.text(row["log2_fold_change"], row["neg_log10_fdr"], row["gene_symbol"], fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "volcano.png", dpi=300)
    plt.close()


def make_heatmap(expr: pd.DataFrame, sample_meta: pd.DataFrame, de: pd.DataFrame) -> None:
    genes = select_top_heatmap_genes(de)
    if len(genes) < 6:
        return
    plot_mat = expr[genes].copy()
    plot_mat = (plot_mat - plot_mat.mean()) / plot_mat.std(ddof=0)
    plot_mat = plot_mat.clip(-3, 3)
    plot_mat = plot_mat.T
    col_colors = sample_meta.set_index("sample").loc[expr.index, "sample_type"].map({
        "Primary tumor": "#B22222",
        "Solid tissue normal": "#1f77b4",
    })

    g = sns.clustermap(
        plot_mat,
        col_cluster=False,
        row_cluster=True,
        cmap="vlag",
        figsize=(10, 8),
        xticklabels=False,
        yticklabels=True,
        col_colors=col_colors,
    )
    legend_handles = [
        Line2D([0], [0], marker="s", color="w", label="Primary tumor", markerfacecolor="#B22222", markersize=10),
        Line2D([0], [0], marker="s", color="w", label="Solid tissue normal", markerfacecolor="#1f77b4", markersize=10),
    ]
    g.ax_heatmap.legend(handles=legend_handles, title="Sample type", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    g.fig.suptitle("Top dysregulated genes in LUAD", y=1.02)
    g.savefig(FIG_DIR / "heatmap_top_genes.png", dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def make_classifier_roc_plot(metrics: dict) -> None:
    fpr = np.array(metrics["mean_fpr"])
    tpr = np.array(metrics["mean_tpr"])
    mean_auc = metrics["mean_roc_auc"]
    sd_auc = metrics["sd_roc_auc"]

    plt.figure(figsize=(6.5, 5.5))
    plt.plot(fpr, tpr, linewidth=2.5, label=f"Mean ROC (AUC = {mean_auc:.3f} ± {sd_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("Cross-validated tumor-vs-normal classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "classifier_roc.png", dpi=300)
    plt.close()


def make_km_plot(expr: pd.DataFrame, sample_meta: pd.DataFrame, survival: pd.DataFrame, biomarker: dict) -> None:
    if not biomarker.get("available"):
        return
    gene = biomarker["gene_symbol"]
    time_col = biomarker["time_col"]
    status_col = biomarker["status_col"]

    tumor_samples = sample_meta.loc[sample_meta["sample_type"] == "Primary tumor", ["sample", "patient_id"]].drop_duplicates()
    tumor_expr = expr.loc[tumor_samples["sample"]].copy()
    tumor_expr["patient_id"] = tumor_samples.set_index("sample").loc[tumor_expr.index, "patient_id"].values
    tumor_expr = tumor_expr.groupby("patient_id").mean(numeric_only=True)

    survival = survival.copy()
    survival[time_col] = pd.to_numeric(survival[time_col], errors="coerce")
    survival[status_col] = normalize_status(survival[status_col])
    joined = survival.merge(tumor_expr[[gene]], left_on="patient_id", right_index=True, how="inner")
    joined = joined.dropna(subset=[time_col, status_col, gene])
    joined = joined[joined[time_col] > 0]
    cutoff = joined[gene].median()
    joined["group"] = np.where(joined[gene] >= cutoff, "High expression", "Low expression")

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(7, 5.5))
    for group, color in [("High expression", "#B22222"), ("Low expression", "#1f77b4")]:
        subset = joined[joined["group"] == group]
        kmf.fit(subset[time_col], subset[status_col], label=group)
        kmf.plot(ci_show=False, color=color, linewidth=2.2)

    plt.title(f"Overall survival by {gene} expression in LUAD")
    plt.xlabel("Days")
    plt.ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kaplan_meier_top_biomarker.png", dpi=300)
    plt.close()


def make_stage_plot(expr: pd.DataFrame, metadata: pd.DataFrame, biomarker: dict) -> bool:
    if not biomarker.get("available"):
        return False
    stage_col = find_stage_column(metadata)
    if not stage_col or biomarker["gene_symbol"] not in expr.columns:
        return False

    tumor_meta = metadata[metadata["sample_type"] == "Primary tumor"].copy()
    tumor_meta = tumor_meta[["sample", stage_col]].dropna()
    tumor_meta = tumor_meta[tumor_meta[stage_col].astype(str).str.len() > 0]

    plot_df = tumor_meta.copy()
    plot_df[biomarker["gene_symbol"]] = expr.loc[plot_df["sample"], biomarker["gene_symbol"]].values
    stage_order = sorted(plot_df[stage_col].astype(str).unique().tolist())

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=plot_df, x=stage_col, y=biomarker["gene_symbol"], order=stage_order)
    sns.stripplot(data=plot_df, x=stage_col, y=biomarker["gene_symbol"], order=stage_order, color="black", alpha=0.35, size=2.5)
    plt.xticks(rotation=25, ha="right")
    plt.title(f"{biomarker['gene_symbol']} expression across LUAD stages")
    plt.xlabel("Pathologic stage")
    plt.ylabel("Expression (log2 norm count + 1)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "stage_boxplot_top_biomarker.png", dpi=300)
    plt.close()
    return True


def run_enrichment(de: pd.DataFrame) -> bool:
    if gp is None:
        return False

    up_genes = de[(de["fdr"] < 0.05) & (de["log2_fold_change"] > 1)]["gene_symbol"].head(250).tolist()
    down_genes = de[(de["fdr"] < 0.05) & (de["log2_fold_change"] < -1)]["gene_symbol"].head(250).tolist()
    if len(up_genes) < 10 and len(down_genes) < 10:
        return False

    libraries = ["MSigDB_Hallmark_2020", "KEGG_2021_Human", "GO_Biological_Process_2023"]
    success = False

    for direction, genes in [("up", up_genes), ("down", down_genes)]:
        if len(genes) < 10:
            continue
        outdir = ENRICH_DIR / direction
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            enr = gp.enrichr(gene_list=genes, gene_sets=libraries, organism="Human", outdir=str(outdir), cutoff=0.5)
            res = enr.results.copy()
            res.to_csv(outdir / f"{direction}_enrichment.csv", index=False)
            top = res.sort_values("Adjusted P-value").head(10).copy()
            top = top.iloc[::-1]

            plt.figure(figsize=(9, 5.5))
            plt.barh(top["Term"], -np.log10(top["Adjusted P-value"].clip(lower=1e-300)))
            plt.xlabel("-log10(adjusted p)")
            plt.title(f"Top enriched pathways: {direction}regulated genes")
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"enrichment_{direction}.png", dpi=300)
            plt.close()
            success = True
        except Exception:
            continue

    return success


def write_tables(sample_summary: pd.DataFrame, de: pd.DataFrame, classifier_metrics: dict, biomarker: dict, survival_table: pd.DataFrame) -> None:
    sample_summary.to_csv(TABLE_DIR / "sample_summary.csv", index=False)
    de.head(100).to_csv(TABLE_DIR / "top_differential_expression.csv", index=False)
    de.to_csv(PROC_DIR / "differential_expression_full.csv", index=False)
    pd.DataFrame([classifier_metrics]).drop(columns=["mean_fpr", "mean_tpr", "features"], errors="ignore").to_csv(TABLE_DIR / "classifier_summary.csv", index=False)
    pd.DataFrame([biomarker]).to_csv(TABLE_DIR / "top_biomarker.csv", index=False)
    if not survival_table.empty:
        survival_table.to_csv(TABLE_DIR / "survival_candidates.csv", index=False)


def run_pipeline(force_download: bool = False) -> PipelineOutputs:
    ensure_directories()
    data_paths = ensure_input_data(force_download=force_download)

    expr, sample_meta = load_expression_matrix(data_paths["expression"])
    clinical = load_clinical_matrix(data_paths["clinical"])
    survival = load_survival_data(data_paths["survival"])
    metadata = merge_metadata(sample_meta, clinical, survival)

    sample_summary = summarize_samples(sample_meta, metadata)
    de = compute_differential_expression(expr, sample_meta)
    classifier_metrics = cross_validated_classifier(expr, sample_meta, de, top_n=50)
    top_biomarker, survival_table = evaluate_prognostic_candidates(expr, sample_meta, de, survival)

    auc_series = compute_gene_auc(expr, sample_meta, de.head(200)["gene_symbol"].tolist())
    if top_biomarker.get("available"):
        top_gene = top_biomarker["gene_symbol"]
        if top_gene in auc_series.index:
            top_biomarker["diagnostic_auc"] = float(auc_series[top_gene])
        row = de.loc[de["gene_symbol"] == top_gene].head(1)
        if not row.empty:
            top_biomarker["log2_fold_change"] = float(row.iloc[0]["log2_fold_change"])
            top_biomarker["fdr"] = float(row.iloc[0]["fdr"])
            top_biomarker["cohens_d"] = float(row.iloc[0]["cohens_d"])

    write_tables(sample_summary, de, classifier_metrics, top_biomarker, survival_table)
    save_summary_json(sample_summary, de, classifier_metrics, top_biomarker)

    make_sample_distribution_plot(sample_summary)
    make_pca_plot(expr, sample_meta)
    make_volcano_plot(de)
    make_heatmap(expr, sample_meta, de)
    make_classifier_roc_plot(classifier_metrics)
    make_km_plot(expr, sample_meta, survival, top_biomarker)
    stage_available = make_stage_plot(expr, metadata, top_biomarker)
    enrichment_available = run_enrichment(de)

    return PipelineOutputs(
        sample_summary=sample_summary,
        differential_expression=de,
        classifier_metrics=classifier_metrics,
        top_biomarker=top_biomarker,
        stage_available=stage_available,
        enrichment_available=enrichment_available,
    )


def cli() -> None:
    parser = argparse.ArgumentParser(description="LUAD transcriptomic biomarker discovery pipeline")
    parser.add_argument("command", nargs="?", default="run", choices=["run"], help="Pipeline action")
    parser.add_argument("--force-download", action="store_true", help="Re-download input files even if cached locally")
    args = parser.parse_args()
    if args.command == "run":
        outputs = run_pipeline(force_download=args.force_download)
        print("Pipeline complete.")
        print(outputs.sample_summary)
        if outputs.top_biomarker.get("available"):
            print(f"Top candidate biomarker: {outputs.top_biomarker['gene_symbol']}")
        else:
            print("No survival-supported biomarker could be selected.")


if __name__ == "__main__":
    cli()

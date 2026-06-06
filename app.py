
import os
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Gaming Study Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .small-note {font-size: 0.92rem; color: #5b6472;}
    .section-title {
        font-size: 1.15rem; font-weight: 700; margin-top: 1rem;
        margin-bottom: 0.35rem; color: #111827;
    }
    .subtle-box {
        background: #f8fafc; border: 1px solid #e5e7eb;
        border-radius: 14px; padding: 0.85rem 1rem;
    }
    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e5e7eb; border-radius: 16px;
        padding: 0.9rem 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e5e7eb; border-radius: 16px;
        padding: 0.7rem 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding-top: 8px;
        padding-bottom: 8px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CONSTANTS
# =========================================================
RAW_DEFAULT_CANDIDATES = [
    "GamingStudy_data.csv",
    "GamingStudy_data(6).csv",
    "GamingStudy_data(5).csv",
    "GamingStudy_data(4).csv",
    "GamingStudy_data(3).csv",
    "GamingStudy_data(2).csv",
    "GamingStudy_data(1).csv",
]
CLEAN_DEFAULT_CANDIDATES = [
    "GamingStudy_LatentProfileAnalysis_Clean.csv",
    "GamingStudy_LatentProfileAnalysis_Clean(4).csv",
    "GamingStudy_LatentProfileAnalysis_Clean(3).csv",
    "GamingStudy_LatentProfileAnalysis_Clean(2).csv",
]
LPA_FILES = {
    "model_fit": ["02_Model_Fit_Indices.csv"],
    "profile_prevalence": ["03_Profile_Prevalence.csv"],
    "profile_means": ["04_Profile_Means.csv"],
    "chi_square": ["05_ChiSquare_Results.csv"],
    "anova": ["06_ANOVA_Results.csv"],
    "assignments": ["07_Profile_Assignments.csv"],
    "labels": ["08_Profile_Labels.csv"],
    "sample_characteristics": ["01_Sample_Characteristics.csv"],
    "report": ["LPA_Report.txt"],
}
SPIN_ITEMS = [f"SPIN{i}" for i in range(1, 18)]


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def find_first_existing(patterns: List[str], search_dirs: Optional[List[Path]] = None) -> Optional[Path]:
    if search_dirs is None:
        search_dirs = [Path.cwd(), Path("/mnt/data")]
    for directory in search_dirs:
        if not directory.exists():
            continue
        for pat in patterns:
            matches = list(directory.glob(pat))
            if matches:
                return matches[0]
    return None


def read_text_file(path_or_buffer) -> str:
    if path_or_buffer is None:
        return ""
    if hasattr(path_or_buffer, "getvalue"):
        raw = path_or_buffer.getvalue()
        for enc in ("utf-8", "latin1", "cp1252"):
            try:
                return raw.decode(enc, errors="replace")
            except Exception:
                continue
        return raw.decode("utf-8", errors="replace")
    path = Path(path_or_buffer)
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            return path.read_text(encoding=enc, errors="replace")
        except Exception:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def read_csv_safely(path_or_buffer) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    if hasattr(path_or_buffer, "getvalue"):
        # UploadedFile in Streamlit
        for enc in encodings:
            try:
                path_or_buffer.seek(0)
                return pd.read_csv(path_or_buffer, low_memory=False, encoding=enc)
            except Exception:
                continue
        path_or_buffer.seek(0)
        return pd.read_csv(path_or_buffer, low_memory=False, encoding="latin1")
    else:
        for enc in encodings:
            try:
                return pd.read_csv(path_or_buffer, low_memory=False, encoding=enc)
            except Exception:
                continue
        return pd.read_csv(path_or_buffer, low_memory=False, encoding="latin1")


def infer_file_role(filename: str) -> str:
    name = filename.lower()
    if "latentprofileanalysis_clean" in name or "lpa_clean" in name:
        return "clean"
    if "gamingstudy_data" in name and "clean" not in name:
        return "raw"
    if "02_model_fit_indices" in name:
        return "model_fit"
    if "03_profile_prevalence" in name:
        return "profile_prevalence"
    if "04_profile_means" in name:
        return "profile_means"
    if "05_chisquare_results" in name:
        return "chi_square"
    if "06_anova_results" in name:
        return "anova"
    if "07_profile_assignments" in name:
        return "assignments"
    if "08_profile_labels" in name:
        return "labels"
    if "01_sample_characteristics" in name:
        return "sample_characteristics"
    if "lpa_report" in name:
        return "report"
    if name.endswith(".r"):
        return "r_script"
    if name.endswith(".txt"):
        return "text"
    return "other"


def top_categories(df: pd.DataFrame, col: str, top_n: int = 15) -> pd.DataFrame:
    s = df[col].astype("string").fillna("Missing").str.strip()
    counts = s.value_counts(dropna=False)
    if len(counts) <= top_n:
        out = counts.reset_index()
        out.columns = [col, "Count"]
        return out
    top = counts.head(top_n)
    rest = counts.iloc[top_n:].sum()
    out = top.reset_index()
    out.columns = [col, "Count"]
    out = pd.concat([out, pd.DataFrame({col: [f"Other ({len(counts) - top_n} categories)"], "Count": [rest]})], ignore_index=True)
    return out


def numeric_summary(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "Variable": col,
            "N": int(x.notna().sum()),
            "Mean": float(x.mean()) if x.notna().any() else np.nan,
            "SD": float(x.std()) if x.notna().any() else np.nan,
            "Min": float(x.min()) if x.notna().any() else np.nan,
            "Median": float(x.median()) if x.notna().any() else np.nan,
            "Max": float(x.max()) if x.notna().any() else np.nan,
            "Missing": int(x.isna().sum()),
        })
    return pd.DataFrame(rows)


def compute_missingness(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum().sort_values(ascending=False)
    out = miss.reset_index()
    out.columns = ["Variable", "Missing_Count"]
    out["Missing_Percent"] = (out["Missing_Count"] / len(df) * 100).round(2)
    return out


def plot_missingness(df: pd.DataFrame, title: str, top_n: int = 15):
    miss = compute_missingness(df)
    miss = miss[miss["Missing_Count"] > 0].head(top_n)
    if miss.empty:
        st.info("No missing values detected in the selected dataset.")
        return None
    fig = px.bar(
        miss.sort_values("Missing_Count"),
        x="Missing_Count",
        y="Variable",
        orientation="h",
        text="Missing_Percent",
        title=title,
        labels={"Missing_Count": "Missing values", "Variable": "Variable", "Missing_Percent": "% missing"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_categorical_distribution(df: pd.DataFrame, col: str, title: str, top_n: int = 12):
    if col not in df.columns:
        st.warning(f"Column '{col}' not found.")
        return None
    freq = top_categories(df, col, top_n=top_n)
    freq = freq.sort_values("Count", ascending=False)
    fig = px.bar(
        freq,
        x="Count",
        y=col,
        orientation="h",
        text="Count",
        title=title,
        labels={col: col, "Count": "Count"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_numeric_histogram(df: pd.DataFrame, col: str, title: str, nbins: int = 40):
    if col not in df.columns:
        st.warning(f"Column '{col}' not found.")
        return None
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    if x.empty:
        st.info(f"No numeric data available for {col}.")
        return None
    fig = px.histogram(
        x=x,
        nbins=nbins,
        title=title,
        labels={"x": col, "count": "Count"},
    )
    fig.add_vline(x=float(x.mean()), line_dash="dash", line_color="#2563eb")
    fig.add_vline(x=float(x.median()), line_dash="dot", line_color="#10b981")
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_item_means(df: pd.DataFrame, title: str = "Mean SPIN Item Scores"):
    items = [c for c in SPIN_ITEMS if c in df.columns]
    if not items:
        return None
    means = pd.DataFrame({
        "Item": items,
        "Mean": [pd.to_numeric(df[c], errors="coerce").mean() for c in items]
    })
    fig = px.bar(
        means,
        x="Item",
        y="Mean",
        text="Mean",
        title=title,
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_heatmap_profile_means(profile_means: pd.DataFrame, title: str = "Profile Heatmap of SPIN Item Means"):
    if profile_means is None or profile_means.empty:
        return None
    df = profile_means.copy()
    item_col = "SPIN_Item" if "SPIN_Item" in df.columns else "Item"
    val_col = "Mean" if "Mean" in df.columns else "Value"
    df[item_col] = df[item_col].astype(str)
    profile_col = "Profile"
    if "SPIN_Item_Number" in df.columns:
        df = df.sort_values(["Profile", "SPIN_Item_Number"])
    pivot = df.pivot(index=profile_col, columns=item_col, values=val_col)
    pivot = pivot.reindex(columns=SPIN_ITEMS, fill_value=np.nan)
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        title=title,
        labels=dict(x="SPIN Items", y="Profile", color="Mean"),
    )
    fig.update_layout(height=550, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_profile_lines(profile_means: pd.DataFrame, title: str = "Profile Line Plot of SPIN Item Means"):
    if profile_means is None or profile_means.empty:
        return None
    df = profile_means.copy()
    item_col = "SPIN_Item" if "SPIN_Item" in df.columns else "Item"
    val_col = "Mean" if "Mean" in df.columns else "Value"
    if "SPIN_Item_Number" in df.columns:
        df = df.sort_values(["Profile", "SPIN_Item_Number"])
    else:
        df[item_col] = pd.Categorical(df[item_col], categories=SPIN_ITEMS, ordered=True)
    fig = px.line(
        df,
        x=item_col,
        y=val_col,
        color="Profile",
        markers=True,
        title=title,
        labels={item_col: "SPIN Items", val_col: "Mean"},
    )
    fig.update_layout(height=540, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="Profile")
    return fig


def plot_model_fit(model_fit: pd.DataFrame):
    if model_fit is None or model_fit.empty:
        return None
    df = model_fit.copy()
    if "Converged" in df.columns:
        df = df[df["Converged"] == True]
    if df.empty:
        return None
    melted = df[["Profiles", "AIC", "BIC"]].melt(id_vars="Profiles", var_name="Criterion", value_name="Value")
    fig = px.line(
        melted,
        x="Profiles",
        y="Value",
        color="Criterion",
        markers=True,
        title="Model Selection Plot (AIC & BIC)",
        labels={"Profiles": "Number of Profiles", "Value": "Information Criterion"},
    )
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_entropy(model_fit: pd.DataFrame):
    if model_fit is None or model_fit.empty or "Entropy" not in model_fit.columns:
        return None
    df = model_fit.copy()
    if "Converged" in df.columns:
        df = df[df["Converged"] == True]
    if df.empty:
        return None
    fig = px.bar(
        df,
        x="Profiles",
        y="Entropy",
        title="Entropy by Profile Solution",
        text="Entropy",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(yaxis=dict(range=[0, 1]), height=420, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_prevalence(profile_prevalence: pd.DataFrame, labels: Optional[pd.DataFrame] = None):
    if profile_prevalence is None or profile_prevalence.empty:
        return None
    df = profile_prevalence.copy()
    if labels is not None and not labels.empty and "Profile" in labels.columns:
        label_map = labels.set_index("Profile").get("Suggested_Label", pd.Series(dtype=str))
        df["Label"] = df["Profile"].map(label_map).fillna(df["Profile"].astype(str))
        df["Display"] = "Profile " + df["Profile"].astype(str) + ": " + df["Label"].astype(str)
    else:
        df["Display"] = "Profile " + df["Profile"].astype(str)
    df = df.sort_values("Percent", ascending=True)
    fig = px.bar(
        df,
        x="Percent",
        y="Display",
        orientation="h",
        text="Percent",
        title="Profile Prevalence",
        labels={"Percent": "% of sample", "Display": "Profile"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def profile_narrative(labels_df: pd.DataFrame, prevalence_df: pd.DataFrame, max_items: int = 3) -> pd.DataFrame:
    if labels_df is None or labels_df.empty or prevalence_df is None or prevalence_df.empty:
        return pd.DataFrame()
    df = labels_df.copy()
    if "Profile" not in df.columns:
        return pd.DataFrame()
    df = df.merge(prevalence_df[["Profile", "n", "Percent"]], on="Profile", how="left")
    narratives = []
    for _, row in df.iterrows():
        label = row.get("Suggested_Label", f"Profile {row['Profile']}")
        high_cols = [c for c in ["Mean_Performance", "Mean_Interaction", "Mean_Physiological", "Overall_SPIN_Mean", "Mean_All_Items"] if c in df.columns]
        # Simple readable summary from available values
        values = {
            "Performance": row.get("Mean_Performance", np.nan),
            "Interaction": row.get("Mean_Interaction", np.nan),
            "Physiological": row.get("Mean_Physiological", np.nan),
        }
        top_dims = sorted([(k, v) for k, v in values.items() if pd.notna(v)], key=lambda x: x[1], reverse=True)
        top_dim = top_dims[0][0] if top_dims else "General"
        narratives.append({
            "Profile": int(row["Profile"]),
            "Suggested_Label": label,
            "Top_Dimension": top_dim,
            "Mean_All_Items": row.get("Mean_All_Items", np.nan),
            "Prevalence_%": row.get("Percent", np.nan),
        })
    return pd.DataFrame(narratives)


def short_plain_language_fit_interpretation(model_fit: pd.DataFrame) -> str:
    if model_fit is None or model_fit.empty:
        return "No model-fit file was loaded, so the app cannot comment on AIC, BIC, or entropy yet."
    df = model_fit.copy()
    if "Converged" in df.columns:
        df = df[df["Converged"] == True]
    if df.empty:
        return "No converged model was found."
    best = df.sort_values(["BIC", "AIC"]).iloc[0]
    entropy = best.get("Entropy", np.nan)
    min_size = best.get("Minimum_Profile_Percent", np.nan)
    parts = [
        f"The app highlights the {int(best['Profiles'])}-profile solution as the best-fitting option by BIC.",
        f"Entropy = {entropy:.3f}" if pd.notna(entropy) else "Entropy is unavailable",
    ]
    if pd.notna(min_size):
        parts.append(f"smallest profile = {min_size:.3f}%")
    if pd.notna(entropy) and entropy >= 0.70 and pd.notna(min_size) and min_size >= 5:
        parts.append("This means profile separation is acceptable and no class is too tiny to interpret safely.")
    elif pd.notna(entropy) and entropy < 0.70:
        parts.append("Classification is acceptable but should be interpreted more cautiously because entropy is below .70.")
    elif pd.notna(min_size) and min_size < 5:
        parts.append("One profile is very small, so that class may be less stable.")
    return " ".join(parts)


def top_spin_items_for_profile(profile_means: pd.DataFrame, profile_id: int, n_items: int = 3) -> str:
    if profile_means is None or profile_means.empty:
        return ""
    df = profile_means.copy()
    item_col = "SPIN_Item" if "SPIN_Item" in df.columns else "Item"
    val_col = "Mean" if "Mean" in df.columns else "Value"
    df = df[df["Profile"] == profile_id].sort_values(val_col, ascending=False)
    items = df[item_col].astype(str).head(n_items).tolist()
    return ", ".join(items)


def plain_profile_interpretation(labels_df: pd.DataFrame, prevalence_df: pd.DataFrame, profile_means: pd.DataFrame) -> pd.DataFrame:
    if labels_df is None or labels_df.empty:
        return pd.DataFrame()
    out = []
    for _, row in labels_df.sort_values("Profile").iterrows():
        pid = int(row["Profile"])
        label = row.get("Suggested_Label", f"Profile {pid}")
        top_items = top_spin_items_for_profile(profile_means, pid, 3)
        prevalence_row = prevalence_df[prevalence_df["Profile"] == pid]
        prev = float(prevalence_row["Percent"].iloc[0]) if not prevalence_row.empty else np.nan
        out.append({
            "Profile": pid,
            "Label": label,
            "Top_SPIN_Items": top_items,
            "Prevalence_%": prev,
        })
    return pd.DataFrame(out)


def dataframe_download(df: pd.DataFrame, name: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {name} as CSV",
        data=csv,
        file_name=f"{name}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================================================
# FILE DISCOVERY / UPLOADER
# =========================================================
st.sidebar.title("🎮 Gaming Study Dashboard")
st.sidebar.caption("Upload files or let the app auto-detect files in the current folder.")

uploaded_files = st.sidebar.file_uploader(
    "Upload CSV / TXT / R files",
    type=["csv", "txt", "r"],
    accept_multiple_files=True,
)

use_local_files = st.sidebar.checkbox(
    "Auto-load files already in the folder",
    value=True,
)

local_found: Dict[str, Path] = {}
if use_local_files:
    for key, patterns in RAW_DEFAULT_CANDIDATES and [("raw", RAW_DEFAULT_CANDIDATES)] or []:
        pass

    found_raw = find_first_existing([f"*{name}*" for name in RAW_DEFAULT_CANDIDATES])
    found_clean = find_first_existing([f"*{name}*" for name in CLEAN_DEFAULT_CANDIDATES])
    if found_raw:
        local_found["raw"] = found_raw
    if found_clean:
        local_found["clean"] = found_clean
    for key, patterns in LPA_FILES.items():
        found = find_first_existing([f"*{name}*" for name in patterns])
        if found:
            local_found[key] = found

# Build a lookup from uploaded files
uploaded_lookup: Dict[str, object] = {}
if uploaded_files:
    for uf in uploaded_files:
        uploaded_lookup[uf.name.lower()] = uf

# Resolve final file objects (uploaded > local)
resolved_files: Dict[str, object] = {}

def resolve_file(role: str, patterns: List[str]):
    # uploaded
    for uploaded_name, uf in uploaded_lookup.items():
        if any(p.lower() in uploaded_name for p in patterns):
            return uf
    # local
    if role in local_found:
        return local_found[role]
    return None


raw_file = resolve_file("raw", [x.lower().replace(".csv", "") for x in RAW_DEFAULT_CANDIDATES])
clean_file = resolve_file("clean", [x.lower().replace(".csv", "") for x in CLEAN_DEFAULT_CANDIDATES])
model_fit_file = resolve_file("model_fit", ["02_model_fit_indices"])
profile_prev_file = resolve_file("profile_prevalence", ["03_profile_prevalence"])
profile_means_file = resolve_file("profile_means", ["04_profile_means"])
chi_sq_file = resolve_file("chi_square", ["05_chisquare_results"])
anova_file = resolve_file("anova", ["06_anova_results"])
assignments_file = resolve_file("assignments", ["07_profile_assignments"])
labels_file = resolve_file("labels", ["08_profile_labels"])
sample_char_file = resolve_file("sample_characteristics", ["01_sample_characteristics"])
report_file = resolve_file("report", ["lpa_report"])

text_files = []
for uf in uploaded_files or []:
    if infer_file_role(uf.name) in {"text", "r_script"}:
        text_files.append(uf)
# also local .r/.txt files
if use_local_files:
    for p in Path.cwd().glob("*.r"):
        text_files.append(p)
    for p in Path.cwd().glob("*.txt"):
        text_files.append(p)
    for p in Path("/mnt/data").glob("*.r"):
        text_files.append(p)
    for p in Path("/mnt/data").glob("*.txt"):
        text_files.append(p)

# Dedupe preserving order
seen = set()
dedup_text_files = []
for f in text_files:
    key = getattr(f, "name", str(f))
    if key not in seen:
        dedup_text_files.append(f)
        seen.add(key)
text_files = dedup_text_files


# =========================================================
# LOAD DATA FRAMES
# =========================================================
def load_any_dataframe(obj):
    if obj is None:
        return None
    try:
        return read_csv_safely(obj)
    except Exception:
        return None


raw_df = load_any_dataframe(raw_file)
clean_df = load_any_dataframe(clean_file)
model_fit_df = load_any_dataframe(model_fit_file)
profile_prev_df = load_any_dataframe(profile_prev_file)
profile_means_df = load_any_dataframe(profile_means_file)
chi_sq_df = load_any_dataframe(chi_sq_file)
anova_df = load_any_dataframe(anova_file)
assignments_df = load_any_dataframe(assignments_file)
labels_df = load_any_dataframe(labels_file)
sample_char_df = load_any_dataframe(sample_char_file)
report_text = read_text_file(report_file) if report_file is not None else ""


# =========================================================
# HEADER
# =========================================================
st.title("🎮 Gaming Study Visual Dashboard")
st.caption(
    "A lay-friendly dashboard for raw data, cleaned data, and Latent Profile Analysis (LPA) outputs."
)

if raw_df is not None or clean_df is not None:
    st.info(
        "This app works best when you upload both the raw dataset and the cleaned dataset. "
        "If LPA output files are also uploaded, the dashboard will display model fit, profile prevalence, profile means, and demographic comparisons."
    )

# =========================================================
# QUICK METRICS
# =========================================================
m1, m2, m3, m4, m5 = st.columns(5)

raw_n = len(raw_df) if raw_df is not None else np.nan
clean_n = len(clean_df) if clean_df is not None else np.nan
retain_pct = (clean_n / raw_n * 100) if pd.notna(raw_n) and pd.notna(clean_n) and raw_n else np.nan
spin_count = len([c for c in SPIN_ITEMS if (clean_df is not None and c in clean_df.columns) or (raw_df is not None and c in raw_df.columns)])
profile_count = int(profile_prev_df["Profile"].nunique()) if profile_prev_df is not None and "Profile" in profile_prev_df.columns else (
    int(model_fit_df["Profiles"].max()) if model_fit_df is not None and "Profiles" in model_fit_df.columns else np.nan
)

with m1:
    st.metric("Raw N", f"{int(raw_n):,}" if pd.notna(raw_n) else "—")
with m2:
    st.metric("Clean N", f"{int(clean_n):,}" if pd.notna(clean_n) else "—")
with m3:
    st.metric("Retention", f"{retain_pct:.1f}%" if pd.notna(retain_pct) else "—")
with m4:
    st.metric("SPIN items", f"{spin_count}/17")
with m5:
    st.metric("Profiles", f"{profile_count}" if pd.notna(profile_count) else "—")

# =========================================================
# MAIN TABS
# =========================================================
tab_overview, tab_rawclean, tab_spin, tab_lpa, tab_docs = st.tabs(
    ["Overview", "Raw vs Clean", "SPIN & Symptoms", "LPA Results", "Documents / Scripts"]
)

# =========================================================
# TAB 1: OVERVIEW
# =========================================================
with tab_overview:
    st.subheader("Overview")
    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown(
            """
            <div class="subtle-box">
            <b>How to read this dashboard</b><br>
            • <b>Raw data</b> shows what the dataset looked like before cleaning.<br>
            • <b>Clean data</b> shows the analysis-ready sample used for the final model.<br>
            • <b>LPA results</b> show the hidden profiles of social anxiety found among gamers.<br>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if raw_df is not None and clean_df is not None:
            comp = pd.DataFrame(
                {
                    "Dataset": ["Raw", "Clean"],
                    "N": [len(raw_df), len(clean_df)],
                    "Columns": [raw_df.shape[1], clean_df.shape[1]],
                    "Missing cells": [int(raw_df.isna().sum().sum()), int(clean_df.isna().sum().sum())],
                }
            )
            st.dataframe(comp, use_container_width=True, hide_index=True)
            dataframe_download(comp, "dataset_comparison")
        elif clean_df is not None:
            st.success("Clean dataset loaded. Upload the raw dataset too for a before/after comparison.")
        elif raw_df is not None:
            st.success("Raw dataset loaded. Upload the cleaned dataset too for a before/after comparison.")
        else:
            st.warning("No dataset found yet. Upload the CSV files or place them in the same folder as this app.")

    with right:
        if raw_df is not None and clean_df is not None:
            fig = px.bar(
                pd.DataFrame({
                    "Stage": ["Raw", "Clean"],
                    "N": [len(raw_df), len(clean_df)]
                }),
                x="Stage",
                y="N",
                text="N",
                title="Sample Size Before vs After Cleaning",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "This chart shows how many participants remained after cleaning. A smaller but cleaner sample is often better for analysis than a larger noisy one."
            )

    st.markdown("### What files are currently available?")
    file_status = pd.DataFrame(
        [
            ["Raw CSV", "Loaded" if raw_df is not None else "Not loaded"],
            ["Clean CSV", "Loaded" if clean_df is not None else "Not loaded"],
            ["Model fit", "Loaded" if model_fit_df is not None else "Not loaded"],
            ["Profile prevalence", "Loaded" if profile_prev_df is not None else "Not loaded"],
            ["Profile means", "Loaded" if profile_means_df is not None else "Not loaded"],
            ["Chi-square", "Loaded" if chi_sq_df is not None else "Not loaded"],
            ["ANOVA", "Loaded" if anova_df is not None else "Not loaded"],
            ["Profile labels", "Loaded" if labels_df is not None else "Not loaded"],
            ["Assignments", "Loaded" if assignments_df is not None else "Not loaded"],
            ["Report text", "Loaded" if report_text else "Not loaded"],
        ],
        columns=["File", "Status"],
    )
    st.dataframe(file_status, use_container_width=True, hide_index=True)

# =========================================================
# TAB 2: RAW VS CLEAN
# =========================================================
with tab_rawclean:
    st.subheader("Raw vs Clean")
    if raw_df is None or clean_df is None:
        st.warning("Upload both raw and clean CSV files to unlock the full before/after view.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Raw data")
            miss_fig = plot_missingness(raw_df, "Top Missing Values in Raw Dataset", top_n=15)
            if miss_fig is not None:
                st.plotly_chart(miss_fig, use_container_width=True)
                st.caption(
                    "This chart shows where data problems are concentrated. Large missingness here usually means a variable is harder to use for analysis."
                )

        with col2:
            st.markdown("#### Clean data")
            miss_fig2 = plot_missingness(clean_df, "Top Missing Values in Clean Dataset", top_n=15)
            if miss_fig2 is not None:
                st.plotly_chart(miss_fig2, use_container_width=True)
                st.caption(
                    "After cleaning, the main analysis variables should be complete. Some non-core variables may still contain missing values."
                )
            else:
                st.success("No missing values shown in the clean dataset for the top variables displayed.")

        st.markdown("#### Categorical variables in the clean dataset")
        cat_options = [c for c in ["Gender", "Age_Category", "Hours_Category", "Playstyle", "Work", "Degree", "SPIN_Severity"] if c in clean_df.columns]
        if cat_options:
            selected_cat = st.selectbox("Choose a variable", cat_options, index=0)
            top_n = st.slider("Top categories to display", 5, 25, 12, 1)
            fig_cat = plot_categorical_distribution(clean_df, selected_cat, f"Distribution of {selected_cat}", top_n=top_n)
            if fig_cat is not None:
                st.plotly_chart(fig_cat, use_container_width=True)
                st.caption(
                    "This chart tells you which categories are most common in the cleaned sample. Bars that are very small usually represent rare groups."
                )
        else:
            st.info("No categorical variables were found in the clean dataset.")

        st.markdown("#### Raw playstyle responses")
        if "Playstyle" in raw_df.columns:
            raw_playstyle = top_categories(raw_df, "Playstyle", top_n=18)
            fig_play = px.bar(
                raw_playstyle.sort_values("Count", ascending=False),
                x="Count",
                y="Playstyle",
                orientation="h",
                text="Count",
                title="Raw Playstyle Responses (Top Categories)",
            )
            fig_play.update_traces(textposition="outside")
            fig_play.update_layout(height=600, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_play, use_container_width=True)
            st.caption(
                "The raw playstyle field contains many free-text answers. The app groups rare responses into 'Other' so the main pattern stays readable."
            )
            st.dataframe(raw_playstyle, use_container_width=True, hide_index=True)
            dataframe_download(raw_playstyle, "raw_playstyle_top_categories")

        st.markdown("#### Cleaned sample details")
        if clean_df is not None:
            cols_to_show = [c for c in ["Gender", "Age", "Age_Category", "Hours", "Hours_Category", "Playstyle", "Work", "Degree", "SPIN_Total", "SPIN_Severity"] if c in clean_df.columns]
            st.dataframe(clean_df[cols_to_show].head(25), use_container_width=True)
            st.caption("This preview shows what the final analysis-ready dataset looks like.")

# =========================================================
# TAB 3: SPIN & SYMPTOMS
# =========================================================
with tab_spin:
    st.subheader("SPIN & Symptoms")
    if clean_df is None and raw_df is None:
        st.warning("Load a dataset first.")
    else:
        source_df = clean_df if clean_df is not None else raw_df
        st.markdown(
            """
            <div class="subtle-box">
            <b>Plain-language guide:</b> SPIN is the symptom scale used in the study. A higher total score means more social anxiety symptoms.
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)

        with c1:
            if source_df is not None and "SPIN_Total" in source_df.columns:
                hist_fig = plot_numeric_histogram(source_df, "SPIN_Total", "Distribution of SPIN Total Score")
                if hist_fig is not None:
                    st.plotly_chart(hist_fig, use_container_width=True)
                    st.caption(
                        "Most readers should look for where the curve is concentrated. If the distribution leans to the right, more severe symptoms are more common."
                    )

        with c2:
            if source_df is not None and "SPIN_Severity" in source_df.columns:
                sev_fig = plot_categorical_distribution(source_df, "SPIN_Severity", "SPIN Severity Categories", top_n=10)
                if sev_fig is not None:
                    st.plotly_chart(sev_fig, use_container_width=True)
                    st.caption(
                        "This chart shows how many participants fall into each severity group, from minimal to very severe."
                    )

        st.markdown("#### Mean score for each SPIN item")
        item_fig = plot_item_means(source_df, "Average Score per SPIN Item")
        if item_fig is not None:
            st.plotly_chart(item_fig, use_container_width=True)
            st.caption(
                "Items with taller bars are the ones endorsed more strongly on average. These items often help explain which symptoms are most common."
            )

        st.markdown("#### Clean-data summary for SPIN-related variables")
        summary_cols = [c for c in ["SPIN_Total"] + [c for c in SPIN_ITEMS if c in source_df.columns] if c in source_df.columns]
        summary = numeric_summary(source_df, summary_cols)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        dataframe_download(summary, "spin_numeric_summary")

# =========================================================
# TAB 4: LPA RESULTS
# =========================================================
with tab_lpa:
    st.subheader("LPA Results")
    if model_fit_df is None and profile_prev_df is None and profile_means_df is None:
        st.warning(
            "Upload the LPA output CSV files (model fit, prevalence, profile means, labels, chi-square, ANOVA) to unlock this tab."
        )
    else:
        st.markdown(
            """
            <div class="subtle-box">
            <b>How to read the LPA section:</b><br>
            • <b>AIC/BIC</b> = lower is better.<br>
            • <b>Entropy</b> = closer to 1 means profiles are cleaner and easier to separate.<br>
            • <b>Prevalence</b> = how common each profile is in the sample.<br>
            • <b>Heatmap / line plot</b> = darker or higher lines mean more symptoms on that SPIN item.<br>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Model fit
        if model_fit_df is not None and not model_fit_df.empty:
            st.markdown("### Model fit")
            fit_fig = plot_model_fit(model_fit_df)
            ent_fig = plot_entropy(model_fit_df)
            cc1, cc2 = st.columns(2)
            with cc1:
                if fit_fig is not None:
                    st.plotly_chart(fit_fig, use_container_width=True)
            with cc2:
                if ent_fig is not None:
                    st.plotly_chart(ent_fig, use_container_width=True)

            st.dataframe(model_fit_df, use_container_width=True)
            dataframe_download(model_fit_df, "model_fit_indices")
            st.success(short_plain_language_fit_interpretation(model_fit_df))

            if "Reviewer_Eligibility" in model_fit_df.columns:
                st.caption(
                    "The table already includes reviewer-style notes about whether each profile solution is good, acceptable, or too small to trust easily."
                )

        # Prevalence
        if profile_prev_df is not None and not profile_prev_df.empty:
            st.markdown("### Profile prevalence")
            prev_fig = plot_prevalence(profile_prev_df, labels_df)
            if prev_fig is not None:
                st.plotly_chart(prev_fig, use_container_width=True)
            st.dataframe(profile_prev_df, use_container_width=True, hide_index=True)
            dataframe_download(profile_prev_df, "profile_prevalence")
            st.caption(
                "The biggest bar shows the most common profile. If one bar is much larger than the others, most gamers fall into that symptom pattern."
            )

        # Labels and profile narratives
        if labels_df is not None and not labels_df.empty:
            st.markdown("### Profile labels and quick interpretation")
            narrative_df = plain_profile_interpretation(labels_df, profile_prev_df, profile_means_df)
            if not narrative_df.empty:
                st.dataframe(narrative_df, use_container_width=True, hide_index=True)
                dataframe_download(narrative_df, "profile_plain_language_summary")
                st.caption(
                    "These labels are simple descriptions. They do not diagnose anyone; they just summarize the dominant symptom pattern."
                )

        # Profile means
        if profile_means_df is not None and not profile_means_df.empty:
            st.markdown("### Profile patterns across SPIN items")
            c1, c2 = st.columns(2)
            with c1:
                heat_fig = plot_heatmap_profile_means(profile_means_df)
                if heat_fig is not None:
                    st.plotly_chart(heat_fig, use_container_width=True)
                    st.caption(
                        "Darker cells usually mean that profile has higher symptom scores on that SPIN item."
                    )
            with c2:
                line_fig = plot_profile_lines(profile_means_df)
                if line_fig is not None:
                    st.plotly_chart(line_fig, use_container_width=True)
                    st.caption(
                        "This line plot makes it easy to compare profiles item-by-item. A profile that sits higher overall usually reflects more severe social anxiety."
                    )
            st.dataframe(profile_means_df, use_container_width=True, hide_index=True)
            dataframe_download(profile_means_df, "profile_means")

            # Plain-language top items by profile
            if labels_df is not None and "Profile" in labels_df.columns:
                st.markdown("### Quick profile notes for non-technical readers")
                notes = []
                for _, row in labels_df.sort_values("Profile").iterrows():
                    pid = int(row["Profile"])
                    label = row.get("Suggested_Label", f"Profile {pid}")
                    top_items = top_spin_items_for_profile(profile_means_df, pid, 3)
                    notes.append({
                        "Profile": pid,
                        "Label": label,
                        "Most elevated items": top_items if top_items else "Not available"
                    })
                notes_df = pd.DataFrame(notes)
                st.dataframe(notes_df, use_container_width=True, hide_index=True)

        # Statistical comparisons
        st.markdown("### Demographic differences across profiles")
        cc1, cc2 = st.columns(2)
        with cc1:
            if chi_sq_df is not None and not chi_sq_df.empty:
                st.markdown("#### Chi-square results")
                st.dataframe(chi_sq_df, use_container_width=True, hide_index=True)
                dataframe_download(chi_sq_df, "chi_square_results")
                sig = chi_sq_df[chi_sq_df["Significant"] == True] if "Significant" in chi_sq_df.columns else chi_sq_df
                if not sig.empty:
                    fig = px.bar(
                        chi_sq_df,
                        x="Variable",
                        y="Cramers_V" if "Cramers_V" in chi_sq_df.columns else "Chi_Square",
                        color="Significant" if "Significant" in chi_sq_df.columns else None,
                        title="Effect Size by Categorical Variable (Cramer's V)",
                    )
                    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "Cramer's V shows how strong the differences are. A small p-value tells you the groups differ; Cramer's V tells you how large the difference is."
                    )

            if anova_df is not None and not anova_df.empty:
                st.markdown("#### ANOVA results")
                st.dataframe(anova_df, use_container_width=True, hide_index=True)
                dataframe_download(anova_df, "anova_results")
                fig2 = px.bar(
                    anova_df,
                    x="Variable",
                    y="Eta_Squared" if "Eta_Squared" in anova_df.columns else "F",
                    title="Effect Size by Continuous Variable",
                )
                fig2.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig2, use_container_width=True)
                st.caption(
                    "Eta-squared shows how much of the variation is explained by profile membership. Small values mean the difference is real but not very large."
                )

        with cc2:
            if assignments_df is not None and not assignments_df.empty:
                st.markdown("#### Profile assignments preview")
                cols_preview = [c for c in ["Profile", "Age_Category", "Hours_Category", "Gender", "Work", "Degree", "SPIN_Total", "SPIN_Severity"] if c in assignments_df.columns]
                st.dataframe(assignments_df[cols_preview].head(25), use_container_width=True)
                st.caption(
                    "This preview shows which participants were assigned to which profile. It is useful if you want to compare profile membership with other variables."
                )

        # Summary interpretation box
        st.markdown("### Plain-language summary")
        if model_fit_df is not None and profile_prev_df is not None and labels_df is not None:
            summary_lines = []
            if "Profiles" in model_fit_df.columns and "BIC" in model_fit_df.columns:
                best = model_fit_df.sort_values(["BIC", "AIC"]).iloc[0]
                summary_lines.append(
                    f"The dashboard points to a {int(best['Profiles'])}-profile solution as the most attractive model by BIC."
                )
            if not profile_prev_df.empty:
                largest = profile_prev_df.sort_values("Percent", ascending=False).iloc[0]
                largest_label = None
                if labels_df is not None and "Profile" in labels_df.columns and "Suggested_Label" in labels_df.columns:
                    matched = labels_df.loc[labels_df["Profile"] == largest["Profile"], "Suggested_Label"]
                    largest_label = matched.iloc[0] if not matched.empty else None
                if largest_label:
                    summary_lines.append(
                        f"The largest group is Profile {int(largest['Profile'])} ({largest_label}), making up about {largest['Percent']:.1f}% of the sample."
                    )
                else:
                    summary_lines.append(
                        f"The largest group is Profile {int(largest['Profile'])}, making up about {largest['Percent']:.1f}% of the sample."
                    )
            st.success(" ".join(summary_lines))

# =========================================================
# TAB 5: DOCUMENTS / SCRIPTS
# =========================================================
with tab_docs:
    st.subheader("Documents / Scripts")
    st.markdown(
        """
        <div class="subtle-box">
        This tab is for previewing supporting files such as the R scripts and written notes.
        It helps non-coders see what the analysis workflow looked like without opening R.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if text_files:
        chosen = st.selectbox(
            "Choose a text / script file to preview",
            [getattr(f, "name", str(f)) for f in text_files],
        )
        selected_obj = None
        for f in text_files:
            if getattr(f, "name", str(f)) == chosen:
                selected_obj = f
                break

        text = read_text_file(selected_obj)
        st.code(text[:25000], language="r" if str(chosen).lower().endswith(".r") else "text")
        if len(text) > 25000:
            st.caption("Preview truncated for readability. Download the original file if you need the full text.")
    else:
        st.info("No text or R script files were found. Upload them if you want previews here.")

    st.markdown("### File map")
    file_map = pd.DataFrame(
        [
            ["Raw dataset", getattr(raw_file, "name", str(raw_file)) if raw_file is not None else "—"],
            ["Clean dataset", getattr(clean_file, "name", str(clean_file)) if clean_file is not None else "—"],
            ["Model fit", getattr(model_fit_file, "name", str(model_fit_file)) if model_fit_file is not None else "—"],
            ["Profile prevalence", getattr(profile_prev_file, "name", str(profile_prev_file)) if profile_prev_file is not None else "—"],
            ["Profile means", getattr(profile_means_file, "name", str(profile_means_file)) if profile_means_file is not None else "—"],
            ["Chi-square", getattr(chi_sq_file, "name", str(chi_sq_file)) if chi_sq_file is not None else "—"],
            ["ANOVA", getattr(anova_file, "name", str(anova_file)) if anova_file is not None else "—"],
            ["Labels", getattr(labels_file, "name", str(labels_file)) if labels_file is not None else "—"],
            ["Assignments", getattr(assignments_file, "name", str(assignments_file)) if assignments_file is not None else "—"],
        ],
        columns=["File type", "Loaded file"],
    )
    st.dataframe(file_map, use_container_width=True, hide_index=True)


# =========================================================
# SIDEBAR FOOTER
# =========================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### What the visuals mean")
st.sidebar.markdown(
    """
    - **Bigger bars** = more people in that category  
    - **Higher AIC/BIC lines** = worse fit  
    - **Lower AIC/BIC lines** = better fit  
    - **Higher heatmap values** = more symptoms on that item  
    - **Small p-values** = groups differ, but look at effect sizes too  
    """
)
st.sidebar.caption("Tip: upload the CSV files from your LPA output folder to unlock the full dashboard.")


# End of app

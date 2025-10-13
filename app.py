# ========== UNIFIED APP.PY (STOCSY + DAFDISCOVERY + STREAMLIT) ==========

# === Original STOCSY(mode).py functions ===

import os
import mpld3
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import stats
from scipy.optimize import curve_fit
from PIL import Image

def STOCSY(target, X, rt_values, mode="linear"):
    """
    Structured STOCSY: Compute correlation and covariance between a target signal and a matrix of signals.
    """

    import os
    import mpld3
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from scipy import stats
    from scipy.optimize import curve_fit
    import pandas as pd

    def exp_model(x, a, b, c):
        return a * np.exp(-b * x) + c

    def sin_model(x, a, b, c, d):
        return a * np.sin(b * x + c) + d

    def sigmoid_model(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def gauss_model(x, a, mu, sigma, c):
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

    if isinstance(target, float):
        idx = np.abs(rt_values - target).idxmin()
        target_vect = X.iloc[idx]
    else:
        target_vect = target

    corr = []
    for i in range(X.shape[0]):
        x = target_vect.values
        y = X.iloc[i].values
        try:
            if mode == "linear":
                r = np.corrcoef(x, y)[0, 1]
            elif mode == "exponential":
                popt, _ = curve_fit(exp_model, x, y, maxfev=10000)
                r = np.corrcoef(y, exp_model(x, *popt))[0, 1]
            elif mode == "sinusoidal":
                guess_freq = 1 / (2 * np.pi)
                popt, _ = curve_fit(sin_model, x, y, p0=[1, guess_freq, 0, 0], maxfev=10000)
                r = np.corrcoef(y, sin_model(x, *popt))[0, 1]
            elif mode == "sigmoid":
                x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
                y_scaled = (y - np.min(y)) / (np.max(y) - np.min(y))
                popt, _ = curve_fit(sigmoid_model, x_scaled, y_scaled, p0=[1, 1, 0.5], maxfev=10000)
                r = np.corrcoef(y_scaled, sigmoid_model(x_scaled, *popt))[0, 1]
            elif mode == "gaussian":
                mu_init = x[np.argmax(y)]
                sigma_init = np.std(x)
                popt, _ = curve_fit(gauss_model, x, y, p0=[1, mu_init, sigma_init, 0], maxfev=10000)
                r = np.corrcoef(y, gauss_model(x, *popt))[0, 1]
            else:
                raise ValueError("Invalid mode")
        except Exception:
            r = 0
        corr.append(r)

    corr = np.array(corr)
    covar = (target_vect - target_vect.mean()) @ (X.T - np.tile(X.T.mean(), (X.T.shape[0], 1))) / (X.T.shape[0] - 1)

    x = np.linspace(0, len(covar), len(covar))
    y = covar
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, axs = plt.subplots(1, 1, figsize=(16, 4), sharex=True, sharey=True)
    norm = plt.Normalize(corr.min(), corr.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(corr)
    lc.set_linewidth(2)
    axs.add_collection(lc)
    fig.colorbar(lc, ax=axs)

    # === ZOOM X e Y com base na região NMR ===
    if isinstance(rt_values, pd.Series):
        nmr_len = int(len(rt_values) / X.shape[1]) if X.shape[1] > 1 else len(rt_values)
        nmr_start = rt_values.iloc[0]
        nmr_end = rt_values.iloc[nmr_len - 1]
        x_start = int((nmr_start - rt_values.min()) / (rt_values.max() - rt_values.min()) * len(covar))
        x_end = int((nmr_end - rt_values.min()) / (rt_values.max() - rt_values.min()) * len(covar))
        if x_start > x_end:
            x_start, x_end = x_end, x_start
        y_nmr_region = y[x_start:x_end]
        y_min = np.min(y_nmr_region)
        y_max = np.max(y_nmr_region)
        axs.set_xlim(x_end, x_start)  # eixo ppm decrescente
        axs.set_ylim(y_min, y_max)
    else:
        axs.set_xlim(x.min(), x.max())
        axs.set_ylim(y.min(), y.max())

    # Ticks do eixo X (ppm)
    min_rt = rt_values.min()
    max_rt = rt_values.max()
    ticksx = []
    tickslabels = []
    if max_rt < 30:
        ticks = np.linspace(math.ceil(min_rt), int(max_rt), int(max_rt) - math.ceil(min_rt) + 1)
    else:
        ticks = np.linspace(math.ceil(min_rt / 10.0) * 10,
                             math.ceil(max_rt / 10.0) * 10 - 10,
                             math.ceil(max_rt / 10.0) - math.ceil(min_rt / 10.0))
    currenttick = 0
    for rt_val in rt_values:
        if currenttick < len(ticks) and rt_val > ticks[currenttick]:
            position = int((rt_val - min_rt) / (max_rt - min_rt) * x.max())
            if position < len(x):
                ticksx.append(x[position])
                tickslabels.append(ticks[currenttick])
            currenttick += 1
    plt.xticks(ticksx, tickslabels, fontsize=12)

    axs.set_xlabel('ppm', fontsize=14)
    axs.set_ylabel(f"Covariance with \n signal at {target:.2f} ppm", fontsize=14)
    axs.set_title(f'STOCSY from signal at {target:.2f} ppm ({mode} model)', fontsize=16)

    text = axs.text(1, 1, '')
    lnx = plt.plot([60, 60], [0, 1.5], color='black', linewidth=0.3)
    lny = plt.plot([0, 100], [1.5, 1.5], color='black', linewidth=0.3)
    lnx[0].set_linestyle('None')
    lny[0].set_linestyle('None')

    def hover(event):
        if event.inaxes == axs:
            inv = axs.transData.inverted()
            maxcoord = axs.transData.transform((x[0], 0))[0]
            mincoord = axs.transData.transform((x[-1], 0))[0]
            rt_val = ((maxcoord - mincoord) - (event.x - mincoord)) / (maxcoord - mincoord) * (max_rt - min_rt) + min_rt
            i = int(((maxcoord - mincoord) - (event.x - mincoord)) / (maxcoord - mincoord) * len(covar))
            if 0 <= i < len(covar):
                cov_val = covar[i]
                cor_val = corr[i]
                text.set_visible(True)
                text.set_position((event.xdata, event.ydata))
                text.set_text(f'{rt_val:.2f} min, covariance: {cov_val:.6f}, correlation: {cor_val:.2f}')
                lnx[0].set_data([event.xdata, event.xdata], [y_min, y_max])
                lnx[0].set_linestyle('--')
                lny[0].set_data([x[0], x[-1]], [cov_val, cov_val])
                lny[0].set_linestyle('--')
        else:
            text.set_visible(False)
            lnx[0].set_linestyle('None')
            lny[0].set_linestyle('None')
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    if not os.path.exists('images'):
        os.mkdir('images')
    plt.savefig(f"images/stocsy_from_{target}_{mode}.pdf", transparent=True, dpi=300)
    html_str = mpld3.fig_to_html(fig)
    with open(f"images/stocsy_interactive_{target}min_{mode}.html", "w") as f:
        f.write(html_str)

    plt.show()
    return corr, covar, fig

# Função para exibir o scatter plot de MS STOCSY
import plotly.express as px

def show_stocsy_ms_correlation_plot(msinfo_corr, label=None):
    """
    Robust MS STOCSY scatter plot (RT × m/z, colored by correlation).
    - Tolerates different casings and separators in column names.
    - Prefers 'corr_<label>' if label is provided; otherwise first 'corr_' column.
    """
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import streamlit as st
    import re

    st.subheader("💥 Correlation Plot of MS Features")

    if not isinstance(msinfo_corr, pd.DataFrame) or msinfo_corr.empty:
        st.warning("No MSinfo/correlation data to plot.")
        return

    df = msinfo_corr.copy()

    # -------- helpers to normalize names & find columns ----------------------
    def norm(s: str) -> str:
        s = str(s)
        s = s.lower()
        s = re.sub(r"[^\w]+", " ", s)   # replace non-alnum with spaces
        s = re.sub(r"\s+", " ", s).strip()
        return s

    name_map = {c: norm(c) for c in df.columns}

    def pick_col(predicate, blacklist=None):
        for orig, nm in name_map.items():
            if blacklist and any(b in nm for b in blacklist):
                continue
            if predicate(nm):
                return orig
        return None

    # ---- find ID / m/z / RT columns (metadata) ------------------------------
    id_col = pick_col(lambda nm: ("row" in nm and "id" in nm) or nm == "id")
    mz_col = pick_col(lambda nm: ("mz" in nm or "m z" in nm), blacklist=["corr", "covar"])
    rt_col = pick_col(lambda nm:
                      ("retention" in nm and "time" in nm) or
                      nm == "rt" or
                      nm.startswith("rt "),
                      blacklist=["corr", "covar"])

    # ---- find correlation column --------------------------------------------
    corr_col = None
    corr_candidates = [c for c in df.columns if name_map[c].startswith("corr ")]
    if label:
        wanted = f"corr {norm(label)}"
        for c in corr_candidates:
            if name_map[c] == wanted:
                corr_col = c
                break
    if corr_col is None and corr_candidates:
        corr_col = corr_candidates[0]

    # ---- sanity checks -------------------------------------------------------
    missing = []
    for nm, col in (("row ID", id_col), ("m/z", mz_col), ("retention time", rt_col), ("correlation", corr_col)):
        if col is None:
            missing.append(nm)
    if missing:
        st.warning(f"⚠️ Could not find required column(s): {', '.join(missing)}.\n"
                   f"Available columns: {list(df.columns)}")
        return

    # ---- ensure numeric ------------------------------------------------------
    for c in (mz_col, rt_col, corr_col):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[mz_col, rt_col, corr_col]).copy()
    if df.empty:
        st.warning("⚠️ No rows left after cleaning NaNs in m/z, RT, or correlation.")
        return

    # ---- build scatter -------------------------------------------------------
    try:
        fig = px.scatter(
            df,
            x=rt_col,
            y=mz_col,
            color=corr_col,
            size=(df[corr_col].abs() ** 2),
            opacity=0.75,
            hover_data=[id_col, rt_col, mz_col, corr_col],
            color_continuous_scale=px.colors.sequential.Jet,
            title=f"Correlation Plot of MS Features (STOCSY{f' – {label}' if label else ''})"
        )
        fig.update_layout(font=dict(size=16), height=520)
        st.plotly_chart(fig, use_container_width=True)

        html_plot = fig.to_html(full_html=False)
        st.download_button(
            f"⬇️ Download MS Correlation Plot (HTML){f' — {label}' if label else ''}",
            data=html_plot,
            file_name=f"stocsy_correlation_MS{f'_{label}' if label else ''}.html",
            mime="text/html"
        )
    except Exception as e:
        st.warning("⚠️ Could not display MS correlation plot.")
        st.exception(e)


# === Original dafdiscovery_process.py functions (cleaned) ===

"""
Author: Ricardo M. Borges (and collaborators)
License: MIT
"""

# === Dependencies ===
import numpy as np
import pandas as pd
import warnings
import matplotlib as plt
import matplotlib.pyplot as pltpy
import plotly.graph_objects as go
plt.rcParams['pdf.fonttype'] = 42 #set up for Acrobat Illustrator manipulation
plt.rcParams['ps.fonttype'] = 42  #set up for Acrobat Illustrator manipulation
import os
import os.path

# =========================
# 🔍 1. METADATA HANDLING
# =========================
def analyze_metadata(metadata_df):
    """
    Analyze the Metadata DataFrame and determine:
    - ordered_samples: list of sample IDs
    - ordered_nmr_files: list of NMR filenames (if present)
    - ordered_ms_files: list of MS filenames (if present)
    - ordered_bio_files: list of BioActivity filenames (if present)
    - option: selected Option number (1 to 5)
    - data_in_use: list of data types present (e.g., ['NMR', 'MS'])
    """

    if 'Samples' not in metadata_df.columns:
        print("❌ Error: 'Samples' column is required in the Metadata file.")
        return None, None, None, None, None, None

    ordered_samples = metadata_df['Samples'].tolist()

    has_nmr = 'NMR_filename' in metadata_df.columns
    has_ms = 'MS_filename' in metadata_df.columns
    has_bio = 'BioAct_filename' in metadata_df.columns

    ordered_nmr_files = metadata_df['NMR_filename'].tolist() if has_nmr else None
    ordered_ms_files = metadata_df['MS_filename'].tolist() if has_ms else None
    ordered_bio_files = metadata_df['BioAct_filename'].tolist() if has_bio else None

    if has_nmr and has_ms and has_bio:
        option = 1
        data_in_use = ['NMR', 'MS', 'BioAct']
        print("🧬 Merging data from: NMR + MS + BioActivity → Option 1")

    elif has_nmr and has_ms:
        option = 2
        data_in_use = ['NMR', 'MS']
        print("🧪 Merging data from: NMR + MS → Option 2")

    elif has_nmr and has_bio:
        option = 3
        data_in_use = ['NMR', 'BioAct']
        print("🔬 Merging data from: NMR + BioActivity → Option 3")

    elif has_ms and has_bio:
        option = 4
        data_in_use = ['MS', 'BioAct']
        print("⚗️ Merging data from: MS + BioActivity → Option 4")

    elif has_nmr:
        option = 5
        data_in_use = ['NMR']
        print("📈 Working with NMR data only → Option 5")

    else:
        print("❌ Error: Metadata must contain at least one of the following columns: 'NMR_filename', 'MS_filename', or 'BioAct_filename'")
        return None, None, None, None, None, None

    print(f"✅ Data types detected: {data_in_use}")
    return ordered_samples, ordered_nmr_files, ordered_ms_files, ordered_bio_files, option, data_in_use



# ===============================
# ⚙️ 2. DATA MERGING BY OPTION
# ===============================
import os
import numpy as np
import pandas as pd

def _merge_with_msinfo(MergeDF_numeric: pd.DataFrame,
                       n_nmr_rows: int,
                       n_ms_rows: int,
                       n_bio_rows: int,
                       MSinfo: pd.DataFrame | None) -> pd.DataFrame:
    """
    Return a copy of MergeDF with the 3 MS info columns inserted on the MS rows.
    Non-MS rows get NaN in those 3 columns.
    """
    if MSinfo is None or not isinstance(MSinfo, pd.DataFrame) or MSinfo.empty:
        # nothing to add
        return MergeDF_numeric.copy()

    total_rows = n_nmr_rows + n_ms_rows + n_bio_rows
    block = pd.DataFrame({
        "row ID": [np.nan]*total_rows,
        "row m/z": [np.nan]*total_rows,
        "row retention time": [np.nan]*total_rows,
    })

    ms_start = n_nmr_rows
    ms_end   = n_nmr_rows + n_ms_rows
    block.iloc[ms_start:ms_end, :] = MSinfo.reset_index(drop=True).values

    # prepend the info columns
    return pd.concat([block, MergeDF_numeric.reset_index(drop=True)], axis=1)


def prepare_data_by_option(option, Ordered_Samples,
                           NMR=None, Ordered_NMR_filename=None,
                           MS=None, Ordered_MS_filename=None,
                           BioAct=None, Ordered_BioAct_filename=None):

    import os
    import numpy as np
    import pandas as pd

    # --- helpers --------------------------------------------------------------
    def _col_lookup(df, patterns, must_exist=True):
        """
        Find a single column in df whose normalized name matches any of 'patterns'.
        Normalization: lower, strip, collapse spaces, replace '-' with ' '.
        """
        def norm(s):
            return str(s).lower().strip().replace("-", " ").replace("_", " ")
        cols_norm = {norm(c): c for c in df.columns}
        for p in patterns:
            p_norm = norm(p)
            # exact
            if p_norm in cols_norm:
                return cols_norm[p_norm]
            # contains
            for k, orig in cols_norm.items():
                if p_norm in k:
                    return orig
        if must_exist:
            raise KeyError(f"Could not find any of columns matching {patterns} in: {list(df.columns)}")
        return None

    def _extract_ms_info(ms_df):
        """
        Robustly extract MS metadata columns and standardize names:
        ['row ID','row m/z','row retention time'].
        Fallback: take the first three columns if named variants are not found.
        """
        if ms_df is None or ms_df.empty:
            return None, None, None

        # Try robust matching
        try:
            col_id = _col_lookup(ms_df, ["row id", "feature id", "id"])
            col_mz = _col_lookup(ms_df, ["row m/z", "mz", "m z"])
            col_rt = _col_lookup(ms_df, ["row retention time", "retention time", "rt"])
            msinfo = ms_df[[col_id, col_mz, col_rt]].copy()
        except Exception:
            # Fallback: first 3 columns as given
            msinfo = ms_df.iloc[:, :3].copy()

        # Standardize column names
        msinfo.columns = ["row ID", "row m/z", "row retention time"]
        return msinfo, "row ID", "row m/z"

    # -------------------------------------------------------------------------

    ppm = None
    new_axis = None
    MSinfo = None

    # ===== Option 1: NMR + MS + BioAct ======================================
    if option == 1:
        # NMR
        ppm = NMR["Unnamed: 0"]
        NMR = NMR[Ordered_NMR_filename]
        NMR.rename(columns=dict(zip(Ordered_NMR_filename, Ordered_Samples)), inplace=True)

        # MS (keep metadata!)
        MSinfo, _id_col, _mz_col = _extract_ms_info(MS)
        MSdata = MS.drop(MS.columns[:3], axis=1)  # drop the first 3 (metadata) columns
        MSdata = MSdata[Ordered_MS_filename]
        MSdata.rename(columns=dict(zip(Ordered_MS_filename, Ordered_Samples)), inplace=True)

        # BioAct
        BioActdata = BioAct.iloc[:, 1:]
        BioActdata = BioActdata[Ordered_BioAct_filename]
        BioActdata.rename(columns=dict(zip(Ordered_BioAct_filename, Ordered_Samples)), inplace=True)

        # Merge (BioAct last)
        MergeDF = pd.concat([NMR, MSdata / 1e8, BioActdata], ignore_index=True)

        gap = ppm.values[-1] - ppm.values[-2]
        start = ppm.values[-1] + gap
        end = start + ((len(MSdata) + len(BioActdata)) * gap)
        axis2 = np.arange(start, end, gap)
        new_axis = pd.concat([ppm, pd.Series(axis2)], ignore_index=True)

        filename = "MergeDF_NMR_MS_BioAct.csv"

    # ===== Option 2: NMR + MS ===============================================
    elif option == 2:
        # NMR
        ppm = NMR["Unnamed: 0"]
        NMR = NMR[Ordered_NMR_filename]
        NMR.rename(columns=dict(zip(Ordered_NMR_filename, Ordered_Samples)), inplace=True)

        # MS (keep metadata!)
        MSinfo, _id_col, _mz_col = _extract_ms_info(MS)
        MSdata = MS.drop(MS.columns[:3], axis=1)
        MSdata = MSdata[Ordered_MS_filename]
        MSdata.rename(columns=dict(zip(Ordered_MS_filename, Ordered_Samples)), inplace=True)

        # Merge
        MergeDF = pd.concat([NMR, MSdata / 1e8], ignore_index=True)

        gap = ppm.values[-1] - ppm.values[-2]
        start = ppm.values[-1] + gap
        end = start + len(MSdata) * gap
        axis2 = np.arange(start, end, gap)
        new_axis = pd.concat([ppm, pd.Series(axis2)], ignore_index=True)

        filename = "MergeDF_NMR_MS.csv"

    # ===== Option 3: NMR + BioAct ===========================================
    elif option == 3:
        ppm = NMR["Unnamed: 0"]
        NMR = NMR[Ordered_NMR_filename]
        NMR.rename(columns=dict(zip(Ordered_NMR_filename, Ordered_Samples)), inplace=True)

        BioActdata = BioAct.iloc[:, 1:]
        BioActdata = BioActdata[Ordered_BioAct_filename]
        BioActdata.rename(columns=dict(zip(Ordered_BioAct_filename, Ordered_Samples)), inplace=True)

        MergeDF = pd.concat([NMR, BioActdata / 20000], ignore_index=True)

        gap = ppm.values[-1] - ppm.values[-2]
        start = ppm.values[-1] + gap
        end = start + len(BioActdata) * gap
        axis2 = np.arange(start, end, gap)
        new_axis = pd.concat([ppm, pd.Series(axis2)], ignore_index=True)

        filename = "MergeDF_NMR_BioAct.csv"

    # ===== Option 4: MS + BioAct ============================================
    elif option == 4:
        # MS (keep metadata!)
        MSinfo, _id_col, _mz_col = _extract_ms_info(MS)
        MSdata = MS.drop(MS.columns[:3], axis=1)
        MSdata = MSdata[Ordered_MS_filename]
        MSdata.rename(columns=dict(zip(Ordered_MS_filename, Ordered_Samples)), inplace=True)

        # BioAct
        BioActdata = BioAct.iloc[:, 1:]
        BioActdata = BioActdata[Ordered_BioAct_filename]
        BioActdata.rename(columns=dict(zip(Ordered_BioAct_filename, Ordered_Samples)), inplace=True)

        MergeDF = pd.concat([MSdata, BioActdata], ignore_index=True)

        # Construct a simple synthetic axis (index-like) for MS+BioAct
        new_axis = pd.Series(np.arange(0, len(MSdata) + len(BioActdata)))
        filename = "MergeDF_MS_BioAct.csv"

    # ===== Option 5: NMR only ===============================================
    elif option == 5:
        ppm = NMR["Unnamed: 0"]
        NMR = NMR[Ordered_NMR_filename]
        NMR.rename(columns=dict(zip(Ordered_NMR_filename, Ordered_Samples)), inplace=True)

        MergeDF = NMR
        new_axis = ppm
        filename = "MergeDF_NMR.csv"

    else:
        raise ValueError("❌ Invalid option. Option must be between 1 and 5.")

    # ===== Tail (common epilogue) ============================================
    if not os.path.exists('data'):
        os.makedirs('data')

    # Always persist MSinfo if present
    if MSinfo is not None and isinstance(MSinfo, pd.DataFrame) and not MSinfo.empty:
        MSinfo.to_csv("data/MSinfo_extracted.csv", index=False)

    MergeDF.to_csv(f"data/{filename}", sep=",", index=False)
    print(f"✅ Data merged and saved to 'data/{filename}'")
    if MSinfo is not None:
        print("✅ MSinfo (first 3 MS columns) saved to 'data/MSinfo_extracted.csv'")

    return MergeDF, new_axis, MSinfo



# =======================================
# 📈 3. STOCSY ANALYSIS AND CORRELATIONS
# =======================================
import os
import pandas as pd

def run_stocsy_and_export(driver, MergeDF, axis, MSinfo=None, mode="linear", output_prefix="default"):
    """
    Runs STOCSY using a selected driver, calculates correlation and covariance,
    maps results to MSinfo (if available), and exports them to a CSV file.

    Parameters:
    -----------
    driver : float
        The driver value (can be a ppm or synthetic value from new_axis).
    MergeDF : pd.DataFrame
        The merged data matrix with NMR, MS, and/or BioAct data.
    axis : pd.Series
        The full axis of variables (including ppm, MS and/or BioAct).
    MSinfo : pd.DataFrame or None
        The MSinfo dataframe containing "row ID", "row m/z", "row retention time". Can be None.
    output_prefix : str
        Prefix to use in naming output files (e.g., "fromBioAct", or a ppm value).
    
    Returns:
    --------
    corr : np.ndarray
    covar : np.ndarray
    MSinfo_corr : pd.DataFrame or None
    fig : matplotlib.figure.Figure
    """

    # Run STOCSY
    corr, covar, fig = STOCSY(float(driver), MergeDF, axis, mode=mode)

    # Create DataFrames
    corrDF = pd.DataFrame(corr, columns=[f'corr_{output_prefix}'])
    covarDF = pd.DataFrame(covar, columns=[f'covar_{output_prefix}'])

    # Handle case where MSinfo is available
    MSinfo_corr = None
    if MSinfo is not None and isinstance(MSinfo, pd.DataFrame):
        try:
            corrDF_MS = corrDF.iloc[-len(MSinfo):].reset_index(drop=True)
            covarDF_MS = covarDF.iloc[-len(MSinfo):].reset_index(drop=True)

            MSinfo_corr = pd.concat([MSinfo.reset_index(drop=True), corrDF_MS], axis=1)
            MSinfo_corr_covar = pd.concat([MSinfo_corr, covarDF_MS], axis=1)

            # Export to CSV
            if not os.path.exists("data"):
                os.makedirs("data")

            corr_filename = f"data/MSinfo_corr_{output_prefix}.csv"
            covar_filename = f"data/MSinfo_corr_covar_{output_prefix}.csv"

            MSinfo_corr.to_csv(corr_filename, sep=",", index=False)
            MSinfo_corr_covar.to_csv(covar_filename, sep=",", index=False)

            print(f"✅ Correlation file saved to {corr_filename}")
            print(f"✅ Correlation + Covariance file saved to {covar_filename}")
        except Exception as e:
            print(f"⚠️ Could not export MSinfo correlation files: {e}")
            MSinfo_corr = None

    return corr, covar, MSinfo_corr, fig




def auto_stocsy_driver_run(MergeDF, new_axis,  MSinfo, data_in_use, mode="linear", driver_value=None):
    """
    Run STOCSY depending on data availability.
    
    If BioAct is present, runs with the last axis value as driver.
    Otherwise, requires a user-supplied driver_value (ppm or MS feature).

    Parameters:
    -----------
    MergeDF : pd.DataFrame
        Merged dataset containing NMR, MS, BioAct.
    new_axis : pd.Series
        Full axis used for STOCSY (ppm + MS + BioAct).
    MSinfo : pd.DataFrame
        MS info for annotation and export.
    data_in_use : list
        Data sources included (e.g., ['NMR', 'MS'] or ['MS', 'BioAct']).
    driver_value : float or None
        Value to use as driver (optional if BioAct is present).

    Returns:
    --------
    corr, covar, MSinfo_corr, fig
    """

    # Case 1: BioAct present → use last axis value
    if "BioAct" in data_in_use:
        driver = float(new_axis.values[-1])
        prefix = "fromBioAct"
        print(f"🧬 Using BioActivity as driver (value = {driver})")

    # Case 2: No BioAct → user must supply driver
    else:
        if driver_value is None:
            raise ValueError("❌ No BioAct available. Please provide a driver_value (ppm or MS index).")
        driver = float(driver_value)
        prefix = f"{driver_value}ppm" if "NMR" in data_in_use else f"MS_{driver_value}"
        print(f"🔍 Using manual driver = {driver}")

    # Run STOCSY logic
    corr, covar, MSinfo_corr, fig = run_stocsy_and_export(
        driver=driver,
        MergeDF=MergeDF,
        axis=new_axis,
	mode="linear",
        MSinfo=MSinfo,
        output_prefix=prefix
    )

    return corr, covar, MSinfo_corr, fig




# =======================================
# 📈 3. NMR DATA PROCESSING
# =======================================

# ... to be added...


# === Original Streamlit app.py ===

# ===== Auto-installer for most packages =====
import subprocess
import sys
import io

required_packages = [
    "streamlit", "pandas", "numpy", "matplotlib",
    "scipy", "plotly", "mpld3"
]

def install_missing_packages(packages):
    for package in packages:
        try:
            __import__(package.split("==")[0])
        except ImportError:
            print(f"📦 Installing: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_missing_packages(required_packages)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from matplotlib.collections import LineCollection
from scipy import stats

# ====== Init session_state ======
for key in ["merged_df", "axis", "msinfo", "corr", "covar", "msinfo_corr"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.set_page_config(layout="wide")
st.title("DAFdiscovery: NMR–MS–BioActivity Integration")

# Load the logo
logo = Image.open("static/dafDISCOVERY_icon.png")
# Display the logo in the sidebar or header
st.image(logo, width=250)

# PayPal donate button
st.markdown("""
<hr>
<center>
<p>To support the app development:</p>
<a href="https://www.paypal.com/donate/?business=2FYTFNDV4F2D4&no_recurring=0&item_name=Support+with+%245+→+Send+receipt+to+tlc2chrom.app@gmail.com+with+your+login+email+→+Access+within+24h!&currency_code=USD" target="_blank">
    <img src="https://www.paypalobjects.com/en_US/i/btn/btn_donate_SM.gif" alt="Donate with PayPal button" border="0">
</a>
</center>
""", unsafe_allow_html=True)

st.markdown("""
  
""")
#st.stop()

# Citation
st.markdown("""
📖 **Please Cite:**  
Borges RM, das Neves Costa F, Chagas FO, *et al.*  
**Data Fusion-based Discovery (DAFdiscovery) pipeline to aid compound annotation and bioactive compound discovery across diverse spectral data.**  
*Phytochemical Analysis.* 2023; 34(1): 48–55.  
[https://doi.org/10.1002/pca.3178](https://doi.org/10.1002/pca.3178)
""")

# Tutorial
st.markdown("""
<hr>
<center>
<p><strong>Need help?</strong> Read the tutorial:</p>
<a href="https://github.com/RicardoMBorges/DAFdiscovery_st/blob/main/tutorial.md" target="_blank">
    <img src="https://img.shields.io/badge/%20Open%20Tutorial-blue?style=for-the-badge&logo=readthedocs" alt="Open Tutorial">
</a>
</center>
""", unsafe_allow_html=True)

st.markdown("""
  
""")


# --- Upload Metadata ---
st.header("📁 Step 1: Upload Metadata")
metadata_file = st.file_uploader("Upload your Metadata CSV file:", type="csv")

if metadata_file:
    metadata_df = pd.read_csv(metadata_file)
    st.success("✅ Metadata loaded.")
    st.markdown("📋 Preview do Metadata CSV:")
    st.markdown(metadata_df.head().to_html(index=False), unsafe_allow_html=True)


    (
        ordered_samples,
        ordered_nmr_files,
        ordered_ms_files,
        ordered_bio_files,
        option,
        data_in_use
    ) = analyze_metadata(metadata_df)

    st.markdown(f"**Detected Option**: {option} — Using: {', '.join(data_in_use)}")

    # --- Upload data files based on metadata ---
    st.header("📂 Step 2: Upload Required Data Files")

    nmr_data = ms_data = bio_data = None

    if "NMR" in data_in_use:
        nmr_data_file = st.file_uploader("Upload NMR data CSV", type="csv")
        if nmr_data_file:
            nmr_data = pd.read_csv(nmr_data_file)
            st.subheader("🧪 NMR Preview")

            ppm = nmr_data[nmr_data.columns[0]]
            st.markdown("**📁 NMR column headers:** " + ", ".join(nmr_data.columns))
            fig = go.Figure()
            for col in nmr_data.columns[1:]:
                fig.add_trace(go.Scatter(x=ppm, y=nmr_data[col], mode='lines', name=col, opacity=0.5))
            fig.update_layout(title="NMR Raw Spectra", xaxis_title="ppm", yaxis_title="Intensity",
                              xaxis=dict(autorange='reversed'), height=400)
            st.plotly_chart(fig, use_container_width=True)

            html_nmr = pio.to_html(fig, full_html=False)
            st.download_button("⬇️ Download NMR Plot (HTML)", data=html_nmr,
                               file_name="nmr_plot.html", mime="text/html")

    if "MS" in data_in_use:
        ms_data_file = st.file_uploader("Upload MS data CSV", type="csv")
        if ms_data_file:
            ms_data = pd.read_csv(ms_data_file)
            st.markdown("**📁 MS column headers:** " + ", ".join(ms_data.columns))

    if "BioAct" in data_in_use:
        bio_data_file = st.file_uploader("Upload BioActivity data CSV", type="csv")
        if bio_data_file:
            bio_data = pd.read_csv(bio_data_file)
            st.markdown("**📁 BioActivity column headers:** " + ", ".join(bio_data.columns))

    # --- Merge and STOCSY ---
    if st.button("▶️ Run Merge and STOCSY"):
        if ("NMR" in data_in_use and nmr_data is None) or ("MS" in data_in_use and ms_data is None) or ("BioAct" in data_in_use and bio_data is None):
            st.warning("⚠️ Please upload all required data files before running.")
        else:
            st.session_state.merged_df, st.session_state.axis, st.session_state.msinfo = prepare_data_by_option(
                option,
                Ordered_Samples=ordered_samples,
                NMR=nmr_data, Ordered_NMR_filename=ordered_nmr_files,
                MS=ms_data, Ordered_MS_filename=ordered_ms_files,
                BioAct=bio_data, Ordered_BioAct_filename=ordered_bio_files
            )

            st.success("✅ Data merged successfully.")
            st.markdown("📋 Preview of merged dataset:")#.head().to_html(index=False), unsafe_allow_html=True)
            st.markdown(st.session_state.merged_df.head().to_html(index=False), unsafe_allow_html=True)

            st.header("🧪 Step 3: Run STOCSY from BioAct")
            import traceback
            try:
                st.session_state.corr, st.session_state.covar, st.session_state.msinfo_corr, fig = auto_stocsy_driver_run(
                    MergeDF=st.session_state.merged_df,
                    new_axis=st.session_state.axis,
                    MSinfo=st.session_state.msinfo,
                    data_in_use=data_in_use,
                    mode="linear",
                    driver_value=None
                )
                st.session_state.fig = fig  # <-- Salva a figura no estado
                st.success("✅ STOCSY (BioActivity as driver) complete.")
                if "MS" in data_in_use and st.session_state.msinfo_corr is not None:
                    show_stocsy_ms_correlation_plot(st.session_state.msinfo_corr, label="fromBioAct")

                # Download do CSV com resultados
                st.download_button("⬇️ Download Correlation Results (BioAct)",
                                   data=st.session_state.msinfo_corr.to_csv(index=False),
                                   file_name="STOCSY_results_bioact.csv",
                                   mime="text/csv")

                # Exibição da figura do STOCSY e botão de download
                if fig:
                    st.pyplot(fig)
                    html_plot = mpld3.fig_to_html(fig)
                    st.download_button("⬇️ Download STOCSY Plot (HTML)",
                                       data=html_plot,
                                       file_name="stocsy_plot_bioact.html",
                                       mime="text/html")
                    # Caminho para o PDF salvo automaticamente
                    pdf_path = f"images/stocsy_from_BioAct_linear.pdf"
                    if os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as f:
                            st.download_button("⬇️ Download STOCSY Plot (PDF)",
                                               data=f,
                                               file_name="stocsy_plot_bioact.pdf",
                                               mime="application/pdf")

            except Exception as e:
                st.error(f"❌ STOCSY with BioAct failed:\n\n{traceback.format_exc()}")
                

    if st.session_state.corr is not None and "NMR" in data_in_use and isinstance(st.session_state.axis, pd.Series):
        st.subheader("🧲 STOCSY Projection (NMR) from BioActivity")
        try:
            corr_array = np.asarray(st.session_state.corr)
            if corr_array.ndim == 0 or corr_array.size == 0:
                st.warning("⚠️ STOCSY correlation array is empty or invalid.")
            else:
                default_idx = int(np.argmax(corr_array))
                if default_idx < len(st.session_state.axis) and len(st.session_state.merged_df.columns) > 1:
                    default_ppm = st.session_state.axis.iloc[default_idx]

                    st.markdown(f"🔍 Shape do merged_df: `{st.session_state.merged_df.shape}`")
                    #st.markdown("🔍 Primeiros valores do eixo (axis):")
                    #st.markdown("```text\n" + st.session_state.axis.head().to_string() + "\n```")

                    try:
                        corr_, covar_, fig = STOCSY(default_ppm, st.session_state.merged_df, st.session_state.axis)

                        if fig:
                            plt.close(fig) 
                            st.pyplot(fig)
                            html_plot = mpld3.fig_to_html(fig)
                            st.download_button("⬇️ Download STOCSY NMR Plot (HTML)",
                                            data=html_plot,
                                            file_name="stocsy_nmr_plot_bioact.html",
                                            mime="text/html",
                                            key="download_html_nmr")
                        else:
                            st.warning("⚠️ STOCSY returned no valid figure to plot.")

                    except Exception as stocsy_error:
                        st.error("❌ Erro na função STOCSY.")
                        st.exception(stocsy_error)

        except Exception as e:
            st.warning("⚠️ Não foi possível plotar a projeção STOCSY do NMR.")
            st.exception(e)


# ===============================
# 📌 Run STOCSY with Manual Driver
# ===============================
if st.session_state.merged_df is not None:
    st.header("📌 Run STOCSY with Manual Driver")
    with st.expander("🛠️ Custom STOCSY: Select a Manual Driver", expanded=False):
        st.markdown("""
        Choose a ppm value (for NMR) or an index from the MS region on the combined data axis.
Use this feature when you want to force the correlation analysis of a specific signal.
        """)

        # Tipo de driver
        driver_type = st.radio("Select the driver type:", ["NMR (ppm)", "MS (index)"], horizontal=True)

        driver_input = st.text_input("Enter the driver value (e.g., 2.54 for ppm, or 102 for MS index):", "")

        run_manual = st.button("▶️ Run STOCSY with Manual Driver")

        if run_manual:
            if driver_input.strip() == "":
                st.warning("⚠️ Enter a valid value for the driver.")
            else:
                try:
                    driver_value = float(driver_input)
                    prefix = f"{driver_value}ppm" if driver_type == "NMR (ppm)" else f"MS_{driver_value}"

                    st.info(f"🔍 Running STOCSY with driver = `{driver_value}` ({driver_type})")

                    corr_, covar_, msinfo_corr_, fig_manual = run_stocsy_and_export(
                        driver=driver_value,
                        MergeDF=st.session_state.merged_df,
                        axis=st.session_state.axis,
                        MSinfo=st.session_state.msinfo,
                        mode="linear",
                        output_prefix=prefix
                    )

                    if fig_manual:
                        st.pyplot(fig_manual)
                        html_manual = mpld3.fig_to_html(fig_manual)
                        st.download_button("⬇️ Download STOCSY Manual Plot (HTML)",
                                        data=html_manual,
                                        file_name=f"stocsy_manual_{prefix}.html",
                                        mime="text/html",
                                        key=f"download_html_manual_{prefix}")

                        # PDF path
                        pdf_manual_path = f"images/stocsy_from_{driver_value}_{'linear'}.pdf"
                        if os.path.exists(pdf_manual_path):
                            with open(pdf_manual_path, "rb") as f:
                                st.download_button("⬇️ Download STOCSY Manual Plot (PDF)",
                                                data=f,
                                                file_name=f"stocsy_manual_{prefix}.pdf",
                                                mime="application/pdf",
                                                key=f"download_pdf_manual_{prefix}")

                        # CSV results
                        st.download_button("⬇️ Download STOCSY Manual CSV Results",
                                        data=msinfo_corr_.to_csv(index=False),
                                        file_name=f"stocsy_results_{prefix}.csv",
                                        mime="text/csv",
                                        key=f"download_csv_manual_{prefix}")

                        # ✅ ADDED: Scatter Plot of MS Features
                        if "MS" in data_in_use and msinfo_corr_ is not None:
                            show_stocsy_ms_correlation_plot(msinfo_corr_, label=prefix)

                        st.success("✅ STOCSY with manual driver completed.")

                except Exception as e:
                    st.error("❌ Error running STOCSY with manual driver.")
                    st.exception(e)





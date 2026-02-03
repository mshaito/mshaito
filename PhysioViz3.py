# ---------------------------------------------------------------
# 🧭 PhysioViz — Multimodal Physiological Signal Dashboard (vNext)
# Author: Mohammad Shaito
#
# WHAT THIS APP DOES :
# - You can either GENERATE synthetic (fake-but-realistic) physiological signals
#   or UPLOAD your own CSV file with the required columns.
# - The app shows interactive charts for EEG, ECG, and EDA.
# - You can toggle each signal ON/OFF and smooth the curves (in seconds).
# - You can download any chart as a PNG via the Plotly camera icon.
#
# HOW TO USE (quick steps):
# 1) LEFT SIDEBAR → pick "Generate Synthetic Data" or "Upload Existing Data".
# 2) For Generate: adjust sliders and click "🚀 Generate Synthetic Data"
#    (or turn ON "⚡ Auto-generate on change" to refresh automatically).
# 3) The charts appear in the main area (4 viewing modes via tabs).
# 4) UNDER the charts you'll find 🛠️ Display Options to hide/show signals
#    and set the smoothing window. Click "Apply display options" to update.
# Patch highlights in this build:
# ✅ Theme switch resets CSS before applying (prevents Light/Dark “sticking”)
# ✅ Auto-generate OFF no longer clears charts immediately after toggling OFF
# ✅ Per-group downsampling for fairer plots on large selections
# ✅ Mode banners/hints are force-cleared after data load/generation
# ✅ Deprecated `st.experimental_rerun()` → modern `st.rerun()`
# ✅ CSV validator + error download + 500k preview cap (from prior build)
# ---------------------------------------------------------------

# ============= 1) IMPORTS =============
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
import time
import io
from plotly import io as pio
# ============= 2) CONSTANTS & PAGE SETUP =============
APP_NAME   = "PhysioViz"
PAGE_TITLE = f"{APP_NAME} — Multimodal Signal Dashboard"

PREVIEW_ROWS = 20
MAX_PREVIEW_ROWS = 500_000        # hard cap: skip preview when exceeded
PLOT_DOWNSAMPLE_LIMIT = 200_000   # guard: thin very large plotted selections
ALLOWED_TASKS = {"Rest", "N-Back", "VR Task"}

st.set_page_config(page_title=PAGE_TITLE, layout="wide")



# --- THEME PICKER (sidebar) ---
def _collapse_about():
    # bump counter so About expander label changes invisibly → forces re-collapsed
    st.session_state["about_version"] = st.session_state.get("about_version", 0) + 1

if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "Auto"

theme_choice = st.sidebar.selectbox("Theme", ["Auto", "Light", "Dark"],   # Would render a theme dropdown
                                    key="theme_mode",on_change=_collapse_about,   
                                    help="Switch between light/dark appearance. Auto follows Streamlit default."
)
st.session_state.theme_choice = theme_choice



# --- APPLY THEME (CSS + Plotly) ---
def apply_theme(theme: str):
    if theme.lower() == "dark":
        pio.templates.default = "plotly_dark"
        st.markdown("""
        <style>
        .stApp { background-color:#0e1117; color:#e1e1e6; }
        [data-testid="stSidebar"] { background-color:#0e1117 !important; border-right:1px solid #1f2430; }
        [data-testid="stSidebar"] * { color:#e1e1e6 !important; }
        </style>
        """, unsafe_allow_html=True)

    elif theme.lower() == "light":
        pio.templates.default = "plotly"
        st.markdown("""
        <style>
        .stApp { background:#ffffff; color:#1f2328; }
        [data-testid="stSidebar"] { background:#f7f7f9 !important; border-right:1px solid #e6e8eb; }
        [data-testid="stSidebar"] * { color:#1f2328 !important; }
        </style>
        """, unsafe_allow_html=True)

    elif theme.lower() == "auto":
        # Use system preference via CSS media query
        pio.templates.default = "plotly"  # default → updated dynamically by CSS
        st.markdown("""
        <style>
        /* Dark mode */
        @media (prefers-color-scheme: dark) {
            .stApp { background-color:#0e1117; color:#e1e1e6; }
            [data-testid="stSidebar"] { background-color:#0e1117 !important; border-right:1px solid #1f2430; }
            [data-testid="stSidebar"] * { color:#e1e1e6 !important; }
        }
        /* Light mode */
        @media (prefers-color-scheme: light) {
            .stApp { background:#ffffff; color:#1f2328; }
            [data-testid="stSidebar"] { background:#f7f7f9 !important; border-right:1px solid #e6e8eb; }
            [data-testid="stSidebar"] * { color:#1f2328 !important; }
        }
        </style>
        """, unsafe_allow_html=True)

apply_theme(theme_choice)




# --- GLOBAL CSS (with tighter header gutters and smaller title margin) ---
st.markdown(
    """
<style>
div.stButton > button, div.stDownloadButton > button { width: 100% !important; }
@media (max-width: 767px){ .block-container { padding-top: 1.6rem !important; } }
@media (min-width: 768px){ .block-container { padding-top: 1.2rem !important; } }

/* Title styling */
.pd-title{
  font-weight:800; letter-spacing:.2px; line-height:1.15;
  font-size:clamp(28px,5.2vw,48px);
  margin-bottom:.35rem;   /* tighter so About sits closer */
}
@media (max-width: 767px){
  .pd-title{ font-size:clamp(24px,7.2vw,34px); }
}

/* Info banners */
.mode-banner{
  background:#eaf4ff; border:1px solid #cde3ff;
  padding:14px 16px; border-radius:12px; margin:10px 0 8px 0; font-size:1.05rem;
}
.mobile-tip{
  background:#eaf4ff; border:1px solid #cde3ff; padding:12px 14px;
  border-radius:8px; margin:8px 0 6px 0; font-size:0.95rem;
}
@media (min-width:768px){ .mobile-tip{ display:none; } }

/* Tighter gutters so title and About are snug */
div[data-testid="column"]{ padding-left:8px; padding-right:8px; }
</style>
""",
    unsafe_allow_html=True,
)
st.markdown("""
<style>
/* tighter header spacing + align expander with title baseline */
.pd-header details { margin-top: .15rem; }
</style>
""", unsafe_allow_html=True)
# ============= 3) SESSION STATE =============
st.session_state.setdefault("data_df", None)
st.session_state.setdefault("source", None)
st.session_state.setdefault("dataset_params", None)
st.session_state.setdefault("synthetic_params_last_gen", None)
st.session_state.setdefault("empty_hint_shown", False)

st.session_state.setdefault("show_eeg", True)
st.session_state.setdefault("show_ecg", True)
st.session_state.setdefault("show_eda", True)
st.session_state.setdefault("smooth_seconds", 0.0)

st.session_state.setdefault("auto_generate", False)
st.session_state.setdefault("suppress_initial_autogen", True)

st.session_state.setdefault("about_version", 0)     # for auto-collapse of About
st.session_state.setdefault("prev_mode", None)
st.session_state.setdefault("theme_mode", "Auto")   # Auto / Light / Dark        # (theme mode key)
st.session_state.setdefault("plotly_template", "plotly")  # plotly / plotly_dark / plotly_white

def _user_interacted():
    st.session_state["suppress_initial_autogen"] = False


# ============= 4) THEME HANDLING (optional) =============
def _css_reset():
    st.markdown(
        """
        <style>
          :root { color-scheme: normal; }
          body, .block-container { background-color: initial !important; color: inherit !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def apply_theme(mode: str):
    _css_reset()
    if mode == "Auto":
        st.session_state["plotly_template"] = "plotly"
        return
    if mode == "Dark":
        st.session_state["plotly_template"] = "plotly_dark"
        st.markdown(
            """
            <style>
            :root { color-scheme: dark; }
            body, .block-container { background-color:#0e1117 !important; color:#e0e0e0 !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.session_state["plotly_template"] = "plotly_white"
        st.markdown(
            """
            <style>
            :root { color-scheme: light; }
            body, .block-container { background-color:#ffffff !important; color:#222 !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

# ============= 5) HEADER ROW (title + wide About, same row) =============
def _render_header_row():
    # invisible suffix to force a "new" expander when about_version changes
    _about_suffix = "\u2063" * int(st.session_state.get("about_version", 0))  # zero-width joiner

    # Center the header content and keep title+about snug:
    # [pad | title | about | pad]
    try:
        padL, col_title, col_about, padR = st.columns([1, 6, 12, 1], gap="small")
    except TypeError:
        padL, col_title, col_about, padR = st.columns([1, 6, 12, 1])

    with col_title:
        st.markdown('<h1 class="pd-title"> 🧭 PhysioViz </h1>', unsafe_allow_html=True)

    with col_about:
        # Wrap in a class so we can tweak spacing if needed
        with st.container():
            st.markdown('<div class="pd-header">', unsafe_allow_html=True)
            with st.expander(f"📘 About PhysioViz{_about_suffix}", expanded=False):
                st.markdown(
                    """
**PhysioViz helps you visualize three physiological signals:**

- 🧪 **Synthetic data generation** (simulate signals; no hardware needed)  
- 📤 **Upload your own CSV file** with the required format  

---

### 🔹 What are the tasks?
These are experimental “conditions” that affect the signals:

- **Rest** — baseline, idle condition  
- **N-Back** — a working-memory task (respond when current matches 2 steps back)  
- **VR Task** — a rhythm/action VR game, used here to simulate higher engagement  

---

### 🔹 What are the signals?
- **EEG (µV)** — very small brain electrical activity  
  *(one synthetic channel per row; includes random noise and slow drift)*  
- **ECG (bpm)** — heart rate derived from cardiac activity  
- **EDA (µS)** — skin conductance linked to sympathetic arousal  

---

### 🔹 View Modes (4 ways to explore)
1. 🎯 **Single selection** — one participant & one task  
2. 👥 **All participants** — everyone for one chosen task  
3. 🔄 **Compare tasks** — compare different tasks for one participant  
4. 🧑‍🤝‍🧑 **Compare participants** — choose a subset of participants for one task  

---

### 🔹 Synthetic generation (how it behaves)
The simulator creates time-stamped signals in realistic ranges:

- **EEG:** ~8–10 µV ± noise & task offsets  
- **ECG:** ~70–80 bpm ± noise & task offsets  
- **EDA:** ~0.4–0.7 µS ± noise & task offsets  

⚠️ **Important:** After changing sliders (participants, duration, sampling rate, tasks),  
click **“🚀 Generate Synthetic Data”** to refresh — unless you enable  
**⚡ Auto-generate on change**, which updates automatically.

**💡 Reproducibility tip — “Use fixed random seed”:**  
Turn this ON to make synthetic data **repeatable**. With the same settings and seed, you’ll get the **exact same dataset** (and it loads faster due to caching). Turn it OFF to get **new variations** each time.

---

### 🔹 CSV requirements
If you upload your own data, the file must have these exact columns:

- **Timestamp** → ISO format (`2025-08-20T14:05:30Z`)  
- **Participant** → e.g., `P01`, `P02`  
- **Task** → e.g., `Rest`, `N-Back`, `VR Task`  
- **EEG_Signal (µV)** • **ECG_Signal (bpm)** • **EDA_Signal (µS)**  

---

### 👤 Developed by
Mohammad A. Shaito (📧 mshaito78@gmail.com). ©2025  

🙏 **Acknowledgment**  
Thanks to discussions on multimodal datasets at the Heracleia Human-Centered Computing Lab (CSE@UTA).  
PhysioViz is independently developed and uses only synthetic data for demonstration.  

ℹ️ **Note:** PhysioViz was formerly called *“biosignaldashboard.”*
"""
                )
            st.markdown('</div>', unsafe_allow_html=True)

# render header row immediately
_render_header_row()
st.markdown(
    '<div class="mobile-tip">📱 <b>:</b> Tap the <b>&raquo;</b> icon (top-left) to open the sidebar.</div>',
    unsafe_allow_html=True,
)


st.markdown('</div>', unsafe_allow_html=True)



# ============= 6) DATA GENERATION HELPERS =============
def _make_rng(seed):
    return np.random.default_rng(seed)

def _synth_once(
    num_participants: int,
    duration_min: int,
    sampling_rate: int,
    tasks: list,
    seed=None,
) -> pd.DataFrame:
    total_samples = duration_min * 60 * sampling_rate
    start_utc = datetime.now(timezone.utc)
    timestamps = [start_utc + timedelta(seconds=i / sampling_rate) for i in range(total_samples)]
    rng = _make_rng(seed)
    task_offsets = {
        "Rest": {"EEG": 0.0, "ECG": -2.0, "EDA": -0.05},
        "N-Back": {"EEG": 0.3, "ECG": 1.0, "EDA": 0.05},
        "VR Task": {"EEG": 0.5, "ECG": 2.0, "EDA": 0.10},
    }
    rows = []
    for p in range(num_participants):
        pid = f"P{p+1:02d}"
        for task in tasks:
            o = task_offsets.get(task, {"EEG": 0.0, "ECG": 0.0, "EDA": 0.0})
            eeg = rng.normal(9.0 + o["EEG"] - 0.15 * p, 1.0, total_samples)
            ecg = rng.normal(74.0 + o["ECG"] + 0.75 * p, 2.0, total_samples)
            eda = rng.normal(0.55 + o["EDA"] + 0.02 * p, 0.06, total_samples)
            # add a gentle low-frequency drift to EEG (~1 cycle/min)
            drift = 0.15 * np.sin(2 * np.pi * (np.arange(total_samples) / (sampling_rate * 60.0)))
            eeg = eeg + drift
            for i in range(total_samples):
                rows.append(
                    {
                        "Timestamp": timestamps[i],
                        "Participant": pid,
                        "Task": task,
                        "EEG_Signal": float(eeg[i]),
                        "ECG_Signal": float(ecg[i]),
                        "EDA_Signal": float(eda[i]),
                    }
                )
    df = pd.DataFrame(rows)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    return df

@st.cache_data(show_spinner=False)
def _synth_cached(
    num_participants: int,
    duration_min: int,
    sampling_rate: int,
    tasks_tuple: tuple,
    seed: int,
) -> pd.DataFrame:
    return _synth_once(num_participants, duration_min, sampling_rate, list(tasks_tuple), seed=seed)

def estimate_dt_seconds(df_all: pd.DataFrame) -> float:
    try:
        dts = df_all.sort_values("Timestamp")["Timestamp"].diff().dropna().dt.total_seconds()
        med = float(dts.median())
        return med if np.isfinite(med) and med > 0 else 1.0
    except Exception:
        return 1.0

def apply_smoothing_per_group(dfg: pd.DataFrame, window_points: int) -> pd.DataFrame:
    if window_points <= 1:
        return dfg
    dfg = dfg.sort_values("Timestamp").copy()
    if st.session_state.get("show_eeg", True):
        dfg["EEG_Signal"] = dfg["EEG_Signal"].rolling(window_points, center=True, min_periods=1).mean()
    if st.session_state.get("show_ecg", True):
        dfg["ECG_Signal"] = dfg["ECG_Signal"].rolling(window_points, center=True, min_periods=1).mean()
    if st.session_state.get("show_eda", True):
        dfg["EDA_Signal"] = dfg["EDA_Signal"].rolling(window_points, center=True, min_periods=1).mean()
    return dfg

def _maybe_downsample_group(dfg: pd.DataFrame) -> pd.DataFrame:
    n = len(dfg)
    if n <= PLOT_DOWNSAMPLE_LIMIT:
        return dfg
    step = max(1, n // PLOT_DOWNSAMPLE_LIMIT)
    return dfg.iloc[::step, :]

# ============= 7) SIDEBAR =============
st.sidebar.header("Step1- Choose Data Input Mode")

mode_options = ["🔬 Generate Synthetic Data", "📤 Upload Existing Data"]
input_mode = st.sidebar.radio(
    "Select how you want to load data:",
    mode_options,
    key="mode_radio",
    help="Pick 'Generate' or 'Upload'.",
    on_change=_collapse_about,  # keeps About collapsed on any change
)

# Helper flags for mode handling
is_generate = input_mode.startswith("🔬")
has_data = st.session_state.get("data_df") is not None

# --- Mode-Specific Instructions (now that input_mode exists) ---
st.markdown("### 👋 Welcome to PhysioViz")
if is_generate:
    st.info(
        "Use the **left sidebar** then:\n\n"
        "**Step 1:** Choose Data input --  *Generate Synthetic Data* is selected by default .  \n"
        "**Step 2:** Adjust the sliders and task options.  \n"
        "**Step 3:** Click 🚀 *Generate Synthetic Data* in the sidebar."
    )
else:
    st.info(
        "📤 You’ve selected **Upload Existing Data**.  \n\n"
        "Please **upload your CSV file using the left sidebar** — PhysioViz will automatically validate and load your data once uploaded."
    )


# Reset on mode change
if st.session_state["prev_mode"] is not None and st.session_state["prev_mode"] != input_mode:
    st.session_state["data_df"] = None
    st.session_state["source"] = None
    st.session_state["dataset_params"] = None
    st.session_state["synthetic_params_last_gen"] = None
    st.session_state["empty_hint_shown"] = False
    st.session_state["suppress_initial_autogen"] = True
st.session_state["prev_mode"] = input_mode

mode_banner = st.empty()
main_hint = st.empty()
data_info = st.container()

# ---------- Generate Mode ----------
if input_mode.startswith("🔬"):
    #st.sidebar.subheader("🧪 Synthetic Data Settings")
    st.sidebar.subheader("Step 2 — Synthetic Data Settings")
    num_participants = st.sidebar.slider(
        "Number of Participants", 1, 12, 3, key="num_participants", on_change=_collapse_about
    )
    duration_min = st.sidebar.slider(
        "Duration (minutes)", 1, 30, 5, key="duration_minutes", on_change=_collapse_about
    )
    sampling_rate = st.sidebar.slider(
        "Sampling Rate (Hz)", 1, 50, 10, key="sampling_rate", on_change=_collapse_about
    )
    selected_tasks = st.sidebar.multiselect(
        "Select Tasks",
        ["Rest", "N-Back", "VR Task"],
        default=["Rest", "N-Back"],
        key="selected_tasks",
        on_change=_collapse_about,
    )

    use_fixed_seed = st.sidebar.toggle(
        "Use fixed random seed",
        value=False,
        key="use_fixed_seed",
        help="Turn ON for reproducible synthetic data.",
        on_change=_collapse_about,
    )
    seed_value = 42
    if use_fixed_seed:
        seed_value = int(
            st.sidebar.number_input(
                "Seed value", min_value=0, value=42, step=1, key="seed_value", on_change=_collapse_about
            )
        )

    st.sidebar.toggle(
        "⚡ Auto-generate on change",
        key="auto_generate",
        help="When ON, new data is generated whenever you change settings.",
        on_change=_collapse_about,
    )
    autogen = bool(st.session_state["auto_generate"])

    ui_params = (
        int(num_participants),
        int(duration_min),
        int(sampling_rate),
        tuple(sorted(selected_tasks)),
    )
    last_params = st.session_state.get("synthetic_params_last_gen")

    # If user just turned Auto-generate from ON→OFF, pin current UI so display doesn't clear on next tweak
    if st.session_state.get("_prev_autogen") is None:
        st.session_state["_prev_autogen"] = autogen
    if st.session_state["_prev_autogen"] and (not autogen):
        st.session_state["dataset_params"] = ui_params
    st.session_state["_prev_autogen"] = autogen

    def _generate_and_store():
        if not selected_tasks:
            data_info.error("❌ Please select at least one task before generating.")
            return
        _collapse_about()
        with st.spinner("Generating synthetic signals…"):
            if use_fixed_seed:
                df_new = _synth_cached(
                    num_participants,
                    duration_min,
                    sampling_rate,
                    tuple(sorted(selected_tasks)),
                    seed_value,
                )
            else:
                df_new = _synth_once(
                    num_participants,
                    duration_min,
                    sampling_rate,
                    sorted(selected_tasks),
                    seed=None,
                )
        st.session_state["data_df"] = df_new
        st.session_state["source"] = "synthetic"
        st.session_state["synthetic_params_last_gen"] = ui_params
        st.session_state["dataset_params"] = ui_params
        _user_interacted()
        mode_banner.empty(); main_hint.empty()
        data_info.success("✅ Synthetic data generated successfully! Scroll down to explore.")
    

    # Sidebar “refresh” button (always visible in Generate mode)
    if is_generate:
        st.sidebar.subheader("Step 3 — Click below Generate your dataset.")
        if st.sidebar.button(
            "🚀 Generate Synthetic Data",
            key="sidebar_generate",
            use_container_width=True,
        ):
            _generate_and_store()

    
    if autogen and (st.session_state["suppress_initial_autogen"] is False) and (last_params != ui_params):
        _generate_and_store()
    else:
        if (not autogen) and (st.session_state.get("data_df") is not None) and (last_params != ui_params):
            with main_hint.container():
                st.info("ℹ️ **Auto-generate is OFF.** Click **🚀 Generate Synthetic Data** to refresh. or ( **Turn ⚡ Auto-generate ON**).")
        #elif (not autogen) and (st.session_state.get("data_df") is None):
            #with main_hint.container():
               # st.info("ℹ️ Auto-generate is OFF. Adjust settings, then click **🚀 Generate Synthetic Data**.")


# ---------- Upload Mode (with schema validation) ----------
else:
    uploaded = st.sidebar.file_uploader(
        "Upload CSV File",
        type=["csv"],
        key="uploader",
        on_change=_collapse_about,
        help="CSV must include: Timestamp, Participant, Task, EEG_Signal, ECG_Signal, EDA_Signal.",
    )
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            required = [
                "Timestamp", "Participant", "Task",
                "EEG_Signal", "ECG_Signal", "EDA_Signal",
            ]
            missing = [c for c in required if c not in df_up.columns]
            if missing:
                data_info.error(f"❌ Missing required columns: {missing}")
            elif df_up.empty:
                data_info.error("❌ The uploaded CSV has no rows.")
            else:
                # Vectorized schema checks
                ts_parsed = pd.to_datetime(df_up["Timestamp"], errors="coerce", utc=False)
                part_ok = df_up["Participant"].astype(str).str.len() > 0
                task_ok = df_up["Task"].isin(ALLOWED_TASKS)
                eeg_num = pd.to_numeric(df_up["EEG_Signal"], errors="coerce")
                ecg_num = pd.to_numeric(df_up["ECG_Signal"], errors="coerce")
                eda_num = pd.to_numeric(df_up["EDA_Signal"], errors="coerce")

                ok_mask = (
                    ts_parsed.notna() & part_ok.fillna(False) & task_ok.fillna(False)
                    & eeg_num.notna() & ecg_num.notna() & eda_num.notna()
                )

                # Build row-level error report (first 1000 rows to keep UI snappy)
                bad_idx = df_up.index[~ok_mask]
                errors_list = []
                for i in bad_idx[:1000]:
                    issues = []
                    if not ts_parsed.notna().iloc[i]: issues.append("Invalid Timestamp")
                    if not part_ok.fillna(False).iloc[i]: issues.append("Missing/empty Participant")
                    if not task_ok.fillna(False).iloc[i]: issues.append("Invalid Task")
                    if not eeg_num.notna().iloc[i]: issues.append("EEG_Signal not numeric")
                    if not ecg_num.notna().iloc[i]: issues.append("ECG_Signal not numeric")
                    if not eda_num.notna().iloc[i]: issues.append("EDA_Signal not numeric")
                    errors_list.append({"row": int(i), "issues": "; ".join(issues)})
                errors_df = pd.DataFrame(errors_list)

                # Attach parsed/typed columns
                df_up["Timestamp"] = ts_parsed
                df_up["EEG_Signal"] = eeg_num
                df_up["ECG_Signal"] = ecg_num
                df_up["EDA_Signal"] = eda_num

                # Finalize valid set
                valid_df = df_up.loc[ok_mask].copy()

                # Timestamp → UTC-aware
                if getattr(valid_df["Timestamp"].dt, "tz", None) is None:
                    valid_df["Timestamp"] = valid_df["Timestamp"].dt.tz_localize("UTC")
                else:
                    valid_df["Timestamp"] = valid_df["Timestamp"].dt.tz_convert("UTC")

                n_bad = len(df_up) - len(valid_df)
                if n_bad > 0:
                    data_info.warning(f"⚠️ Found {n_bad:,} invalid row(s). They will be skipped.")
                    with st.expander("See row-level schema errors"):
                        st.dataframe(errors_df, use_container_width=True, height=260)
                        if not errors_df.empty:
                            buf = io.StringIO(); errors_df.to_csv(buf, index=False)
                            st.download_button(
                                "⬇️ Download full error report (CSV)",
                                data=buf.getvalue().encode("utf-8"),
                                file_name="schema_errors.csv",
                                mime="text/csv",
                            )

                if len(valid_df) == 0:
                    data_info.error("❌ All rows were invalid after validation.")
                else:
                    st.session_state["data_df"] = valid_df
                    st.session_state["source"] = "upload"
                    st.session_state["dataset_params"] = ("upload",)
                    st.session_state["synthetic_params_last_gen"] = None
                    _user_interacted()
                    mode_banner.empty(); main_hint.empty()
                    data_info.success(
                        f"✅ CSV uploaded and validated! Keeping {len(valid_df):,} row(s). Scroll down to explore."
                    )
        except Exception as e:
            data_info.error(f"❌ Failed to read CSV: {e}")

# Advanced tools
with st.sidebar.expander("Advanced"):
    st.caption("Reset clears the active dataset from memory.")
    if st.button("♻️ Reset Dataset (clear memory)", use_container_width=True):
        _collapse_about()
        st.session_state["data_df"] = None
        st.session_state["source"] = None
        st.session_state["dataset_params"] = None
        st.session_state["synthetic_params_last_gen"] = None
        st.session_state["empty_hint_shown"] = False
        st.session_state["suppress_initial_autogen"] = True
        st.rerun()

if st.session_state.get("data_df") is not None:
    _csv = st.session_state["data_df"].to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "⬇️ Download Current Dataset (CSV)",
        _csv,
        "PhysioViz_signals.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============= 8) MODE BANNERS =============
df_check = st.session_state.get("data_df")
mode_banner.empty()
if df_check is None or (isinstance(df_check, pd.DataFrame) and df_check.empty):
    if input_mode.startswith("🔬"):
        mode_banner.markdown(
            '<div class="mode-banner">🧪 You are in <b>Generate Mode </b> — tweak controls to auto-generate (when ON), or click  <b>🚀 Generate Synthetic Data.</b></div>',
            unsafe_allow_html=True,
        )
    else:
        mode_banner.markdown(
            '<div class="mode-banner">📤 You are in <b>Upload Mode</b> — please upload a CSV file to continue.</div>',
            unsafe_allow_html=True,
        )

# ============= 9) IF NO DATA → NUDGE & STOP =============
df = st.session_state.get("data_df", None)
if df is None or (isinstance(df, pd.DataFrame) and df.empty):
    if not st.session_state["empty_hint_shown"]:
        st.markdown(
            """
        <style>
          .pulse-red { background:#fff1f1; border:1px solid #ffc7c7; color:#7a1f1f;
            padding:14px 16px; border-radius:10px; margin:10px 0 8px 0;
            font-size:1.05rem; font-weight:600; animation:pulseGlow 1.1s ease-in-out 0s 6; }
          @keyframes pulseGlow {
            0%{box-shadow:0 0 0 0 rgba(255,0,0,.45);}
            50%{box-shadow:0 0 0 12px rgba(255,0,0,0);}
            100%{box-shadow:0 0 0 0 rgba(255,0,0,0);}
          }
        </style>""",
            unsafe_allow_html=True,
        )
        alert_box = st.empty()
        with alert_box.container():
            st.markdown(
                '<div class="pulse-red">👋 <b>Load or generate data using the sidebar to begin</b></div>',
                unsafe_allow_html=True,
            )
        time.sleep(2)
        alert_box.empty()
        st.session_state["empty_hint_shown"] = True
    st.stop()

# Ensure UTC timestamps
try:
    if df["Timestamp"].dt.tz is None:
        df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC")
    else:
        df["Timestamp"] = df["Timestamp"].dt.tz_convert("UTC")
except Exception:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)

# ============= 10) CLEARING GUARD (Auto-gen OFF) =============
if input_mode.startswith("🔬"):
    current_ui_params = (
        int(st.session_state["num_participants"]),
        int(st.session_state["duration_minutes"]),
        int(st.session_state["sampling_rate"]),
        tuple(sorted(st.session_state["selected_tasks"])),
    )
    if (
        st.session_state.get("source") == "synthetic"
        and st.session_state.get("dataset_params") is not None
        and st.session_state.get("data_df") is not None
        and current_ui_params != st.session_state.get("dataset_params")
        and not st.session_state.get("auto_generate", False)
    ):
        st.info(
            " Synthetic Settings Changed."
        )
        st.stop()

# ============= 11) MAIN VIEW =============
src = st.session_state.get("source", "?")
n_rows = len(df)
n_parts = df["Participant"].nunique()
n_tasks = df["Task"].nunique()
st.markdown(
    f"**Data Source:** `{src}` • **Rows:** `{n_rows:,}` • **Participants:** `{n_parts}` • **Tasks:** `{n_tasks}`"
)

# Preview (hard cap)
st.markdown("### 👀 Data Preview")
if n_rows > MAX_PREVIEW_ROWS:
    st.info(
        f"Dataset is large ({n_rows:,} rows). Preview table is skipped (>{MAX_PREVIEW_ROWS:,}). "
        "Use filters/tabs below to focus smaller slices."
    )
else:
    st.dataframe(df.head(PREVIEW_ROWS), use_container_width=True, height=280)

participants = sorted(df["Participant"].dropna().unique().tolist())
tasks = sorted(df["Task"].dropna().unique().tolist())

# --- Plot helper with per-group downsampling ---
def plot_lines(title: str, df_plot: pd.DataFrame, color_by: str = "Participant", legend_side: str = "right"):
    if df_plot.empty:
        st.warning("Nothing to plot for this selection.")
        return

    df_plot = df_plot.sort_values("Timestamp")
    dt_seconds = estimate_dt_seconds(df_plot)
    window_points = max(1, int(round(st.session_state.get("smooth_seconds", 0.0) / dt_seconds)))
    if window_points % 2 == 0:
        window_points += 1

    total_n = len(df_plot)
    if total_n > PLOT_DOWNSAMPLE_LIMIT:
        st.info(f"Large selection ({total_n:,} rows) — downsampled per group for plotting.")

    fig = go.Figure()
    for label, dfg in df_plot.groupby(color_by):
        dfg = apply_smoothing_per_group(_maybe_downsample_group(dfg), window_points)
        if st.session_state.get("show_eeg", True):
            fig.add_trace(go.Scatter(x=dfg["Timestamp"], y=dfg["EEG_Signal"], mode="lines",
                                     name=f"EEG | {label}", line=dict(width=1.2)))
        if st.session_state.get("show_ecg", True):
            fig.add_trace(go.Scatter(x=dfg["Timestamp"], y=dfg["ECG_Signal"], mode="lines",
                                     name=f"ECG | {label}", line=dict(width=1.2)))
        if st.session_state.get("show_eda", True):
            fig.add_trace(go.Scatter(x=dfg["Timestamp"], y=dfg["EDA_Signal"], mode="lines",
                                     name=f"EDA | {label}", line=dict(width=1.2)))

    if legend_side == "right":
        legend_cfg = dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(size=11))
        margin_cfg = dict(l=16, r=140, t=40, b=28)
    else:
        legend_cfg = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=11))
        margin_cfg = dict(l=16, r=16, t=56, b=28)

    y_bits = []
    if st.session_state.get("show_eeg", True): y_bits.append("EEG (µV)")
    if st.session_state.get("show_ecg", True): y_bits.append("ECG (bpm)")
    if st.session_state.get("show_eda", True): y_bits.append("EDA (µS)")
    y_axis_label = " • ".join(y_bits) if y_bits else "Signal Value"

    fig.update_layout(
        title=title, xaxis_title="Time", yaxis_title=y_axis_label, height=480,
        margin=margin_cfg, legend=legend_cfg, hovermode="x unified",
        template=st.session_state.get("plotly_template", "plotly"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(
        fig, use_container_width=True,
        config={"displayModeBar": True, "responsive": True,
                "toImageButtonOptions": {"format":"png","filename":title.replace(" ","_"),
                                         "height":600,"width":1000,"scale":2}}
    )

# ============= 12) TABS =============
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Single selection", "👥 All participants (for a task)",
     "🔄 Compare tasks (for a participant)", "🧑‍🤝‍🧑 Compare participants (for a task)"]
)

with tab1:
    st.subheader("🎯 Single selection — one participant & one task")
    pick_p = st.selectbox("Select Participant", participants, index=0, key="single_pick_p")
    pick_t = st.selectbox("Select Task", tasks, index=0, key="single_pick_t")
    df_sel = df[(df["Participant"] == pick_p) & (df["Task"] == pick_t)]
    st.markdown("### &nbsp;")
    plot_lines(f"Signals | {pick_t} — {pick_p}", df_sel, color_by="Task", legend_side="top") if not df_sel.empty else st.warning("No matching rows.")

with tab2:
    st.subheader("👥 Show all participants for a selected task")
    pick_task = st.selectbox("Select Task", tasks, index=0, key="allP_task")
    df_task = df[df["Task"] == pick_task]
    st.markdown("### &nbsp;")
    plot_lines(f"Signals | {pick_task} — All Participants", df_task, color_by="Participant") if not df_task.empty else st.warning("No rows for that task.")

with tab3:
    st.subheader("🔄 Compare tasks for a selected participant")
    pick_participant = st.selectbox("Select Participant", participants, index=0, key="taskCmp_part")
    df_part = df[df["Participant"] == pick_participant]
    st.markdown("### &nbsp;")
    plot_lines(f"Signals | {pick_participant} — Compare Tasks", df_part, color_by="Task") if not df_part.empty else st.warning("No rows for that participant.")

with tab4:
    st.subheader("🧑‍🤝‍🧑 Compare participants for a task")
    pick_task_cmp = st.selectbox("Select Task", tasks, index=0, key="cmp_task")
    part_for_task = sorted(df.loc[df["Task"] == pick_task_cmp, "Participant"].unique().tolist())
    if not part_for_task:
        st.info("No participants available for the selected task.")
    else:
        pick_participants = st.multiselect(
            "Select Participants to include",
            options=part_for_task,
            default=part_for_task,
            key="cmp_participants",
        )
        if not pick_participants:
            st.warning("Please select at least one participant to display.")
        else:
            df_cmp = df[(df["Task"] == pick_task_cmp) & (df["Participant"].isin(pick_participants))]
            st.markdown("### &nbsp;")
            plot_lines(f"Signals | {pick_task_cmp} — Compare Participants", df_cmp, color_by="Participant") if not df_cmp.empty else st.warning("No rows match that combination.")

# ============= 13) DISPLAY OPTIONS =============
st.markdown("### 🛠️ Display Options")
with st.form("display_options_form", clear_on_submit=False):
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        eeg_val = st.checkbox("EEG", value=st.session_state.get("show_eeg", True))
    with c2:
        ecg_val = st.checkbox("ECG", value=st.session_state.get("show_ecg", True))
    with c3:
        eda_val = st.checkbox("EDA", value=st.session_state.get("show_eda", True))
    with c4:
        smooth_val = st.slider(
            "Smoothing window (seconds)", 0.0, 5.0,
            value=float(st.session_state.get("smooth_seconds", 0.0)), step=0.1
        )
    if not (eeg_val or ecg_val or eda_val):
        st.info("At least one signal must be visible — re-enabling **EEG**.")
        eeg_val = True
    submitted = st.form_submit_button("Apply display options", use_container_width=True)

if submitted:
    _collapse_about()
    st.session_state["show_eeg"] = bool(eeg_val)
    st.session_state["show_ecg"] = bool(ecg_val)
    st.session_state["show_eda"] = bool(eda_val)
    st.session_state["smooth_seconds"] = float(smooth_val)
    st.rerun()

# ============= 14) FOOTER =============
st.markdown(
    "<div style='text-align:center; opacity:.65; font-size:.9rem; margin-top:1rem;'>PhysioViz — Multimodal Signal Dashboard</div>",
    unsafe_allow_html=True,
)

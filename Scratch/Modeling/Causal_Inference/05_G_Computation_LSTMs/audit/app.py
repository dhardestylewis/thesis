"""
app.py  –  PropTech Causal Trajectory Dashboard
================================================
Pure SQLite reader. No GPU. No PyTorch.
Sliders filter case_meta → SQL WHERE → aggregate case_preds → render surface.
Any slider combination is instant — computed from already-stored per-case data.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sqlite3, os, time

st.set_page_config(layout="wide", page_title="PropTech Causal Surfaces")

DB_PATH = "dashboard_cache/surfaces.db"
ALL_DOSES = np.linspace(0.0, 1.0, 11).tolist()
TIMES_ALL = np.arange(55)


def make_surface(grid, z_title, title, colorscale):
    valid = ~np.all(np.isnan(grid), axis=1)
    g = grid[valid, :]
    d_vals = [ALL_DOSES[i] for i, v in enumerate(valid) if v]
    if len(d_vals) < 2:
        return None
    fig = go.Figure(data=[go.Surface(
        z=g, x=TIMES_ALL, y=d_vals,
        colorscale=colorscale, connectgaps=True
    )])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="Time (Periods)",
                   yaxis_title="Petition Dose",
                   zaxis_title=z_title),
        template="plotly_dark", height=560,
        margin=dict(l=0, r=0, t=45, b=0)
    )
    return fig


# ─── DB ──────────────────────────────────────────────────────────────────────
if not os.path.exists(DB_PATH):
    st.title("PropTech Causal Trajectory Simulator")
    st.warning("Start `precompute_dashboard.py` to begin GPU precomputation.")
    st.stop()

con = sqlite3.connect(DB_PATH, check_same_thread=False)

# ─── Check overall progress ───────────────────────────────────────────────────
total_rows = con.execute("SELECT COUNT(*) FROM case_preds").fetchone()[0]
status_row = con.execute("SELECT value FROM progress WHERE key='status'").fetchone()
gpu_done   = status_row and status_row[0] == "done"
total_cases_stored = con.execute("SELECT COUNT(DISTINCT case_number) FROM case_preds").fetchone()[0]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("📐 Cohort Filters")
if not gpu_done:
    st.sidebar.info(f"⚙️ GPU computing: {total_cases_stored} cases done so far. Filters work on available data!")

# Pull ranges from what's actually stored
ht_range = con.execute("SELECT MIN(requested_ht), MAX(requested_ht) FROM case_meta").fetchone()
import math
lat_range = con.execute("SELECT MIN(latitude), MAX(latitude) FROM case_meta WHERE latitude IS NOT NULL").fetchone()
lon_range = con.execute("SELECT MIN(longitude), MAX(longitude) FROM case_meta WHERE longitude IS NOT NULL").fetchone()
yr_range  = con.execute("SELECT MIN(filing_year), MAX(filing_year) FROM case_meta").fetchone()

if ht_range[0] is None:
    st.warning("GPU is still initializing case metadata. Check back shortly.")
    time.sleep(4)
    st.rerun()

# Height filter — only applies to cases that explicitly requested a height
height_upzone_only = st.sidebar.checkbox(
    "Height Upzone Applications Only",
    value=False,
    help="Excludes the ~63% of cases that did not specify a height in their application (use-only rezonings)."
)
min_ht, max_ht = st.sidebar.slider(
    "Requested Height (ft)  [applies only if height requested]",
    float(0), float(800),
    (float(0), float(800)), step=5.0
)
min_year, max_year = st.sidebar.slider(
    "Filing Year",
    int(yr_range[0]), int(yr_range[1]),
    (int(yr_range[0]), int(yr_range[1]))
)
lat_min_f = math.floor(lat_range[0] * 100) / 100
lat_max_f = math.ceil(lat_range[1] * 100) / 100
lon_min_f = math.floor(lon_range[0] * 100) / 100
lon_max_f = math.ceil(lon_range[1] * 100) / 100

min_lat, max_lat = st.sidebar.slider(
    "Latitude", lat_min_f, lat_max_f, (lat_min_f, lat_max_f), step=0.01, format="%.2f"
)
min_lon, max_lon = st.sidebar.slider(
    "Longitude", lon_min_f, lon_max_f, (lon_min_f, lon_max_f), step=0.01, format="%.2f"
)
include_no_location = st.sidebar.checkbox("Include cases without GPS coordinates", value=True)

district_opts = [r[0] for r in con.execute(
    "SELECT DISTINCT council_dist FROM case_meta WHERE council_dist IS NOT NULL ORDER BY council_dist"
).fetchall()]
sel_districts = st.sidebar.multiselect("Council District(s)", district_opts,
                                        default=[], placeholder="All districts")

auto_refresh = st.sidebar.checkbox("Auto-refresh while GPU computing", value=not gpu_done)

# ─── Filter cases via SQL ─────────────────────────────────────────────────────
district_clause = ""
district_params = []
if sel_districts:
    placeholders = ",".join("?" * len(sel_districts))
    district_clause = f"AND council_dist IN ({placeholders})"
    district_params = sel_districts

# Check if height_requested column exists (may have been added post-creation)
meta_cols = [r[1] for r in con.execute("PRAGMA table_info(case_meta)").fetchall()]
has_ht_requested = "height_requested" in meta_cols

height_clause = ""
if height_upzone_only and has_ht_requested:
    height_clause = f"AND height_requested = 1 AND requested_ht BETWEEN {min_ht} AND {max_ht}"
else:
    height_clause = f"AND (requested_ht IS NULL OR requested_ht BETWEEN {min_ht} AND {max_ht})"
if include_no_location:
    loc_clause = "(latitude IS NULL OR (latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?))"
else:
    loc_clause = "(latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?)"
loc_params = [min_lat, max_lat, min_lon, max_lon]

query = f"""
    SELECT case_number FROM case_meta
    WHERE filing_year BETWEEN ? AND ?
      AND {loc_clause}
      {height_clause}
      {district_clause}
"""
params = [min_year, max_year] + loc_params + district_params
cohort_cases = [r[0] for r in con.execute(query, params).fetchall()]

cohort_n = len(cohort_cases)
st.sidebar.metric("Matching Cases", cohort_n)

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("PropTech Causal Trajectory Simulator")
pct_gpu = total_rows / max(total_cases_stored * 11 * 51, 1) if total_cases_stored else 0
st.markdown(
    f"**Cohort:** `{cohort_n}` cases  |  "
    f"**GPU Store:** `{total_cases_stored}` cases × 11 doses × 51 steps "
    f"({'✅ Complete' if gpu_done else f'⚙️ {total_rows:,} rows written, still running...'})"
)

if cohort_n == 0:
    st.warning("No cases match these filters. Widen the sliders.")
    if auto_refresh and not gpu_done:
        time.sleep(4)
        st.rerun()
    st.stop()

# ─── Aggregate case_preds for the cohort ─────────────────────────────────────
if cohort_n == 0:
    st.stop()

# Pull only rows for cases in this cohort that are already computed
placeholders = ",".join("?" * min(cohort_n, 999))  # SQLite limit
# For large cohorts use a temp table approach
con.execute("DROP TABLE IF EXISTS _cohort_filter")
con.execute("CREATE TEMP TABLE _cohort_filter (case_number TEXT PRIMARY KEY)")
con.executemany("INSERT OR IGNORE INTO _cohort_filter VALUES (?)", [(c,) for c in cohort_cases])
con.commit()

rows = con.execute("""
    SELECT dose, t,
           AVG(surv)          as surv,
           AVG(ht_delta)      as ht,
           AVG(cum_tok)       as tok,
           AVG(cum_comm)      as comm,
           AVG(cum_coun)      as coun
    FROM case_preds
    WHERE case_number IN (SELECT case_number FROM _cohort_filter)
    GROUP BY dose, t
    ORDER BY dose, t
""").fetchall()

if not rows:
    st.info("The GPU hasn't computed these cases yet. Check back in a moment.")
    if auto_refresh and not gpu_done:
        time.sleep(4)
        st.rerun()
    st.stop()

df = pd.DataFrame(rows, columns=["dose","t","surv","ht","tok","comm","coun"])
cells_available = len(rows)
total_cells = 11 * 51

st.sidebar.progress(
    min(cells_available / total_cells, 1.0),
    text=f"Surface coverage: {cells_available}/{total_cells} cells"
)

def build_grid(col):
    pivot = df.pivot(index="dose", columns="t", values=col)
    return pivot.reindex(index=ALL_DOSES).values

grid_surv = build_grid("surv")
grid_ht   = build_grid("ht")
grid_tok  = build_grid("tok")
grid_comm = build_grid("comm")
grid_coun = build_grid("coun")

# ─── Render 5 surfaces in 2-column grid ──────────────────────────────────────
col1, col2 = st.columns(2)

SURFACES = [
    (grid_surv, "Approval Probability [0-1]",       "① Survival: Approval Probability",           "RdBu"),
    (grid_ht,   "Height Concession (%)",             "② Height: Concession from Requested (%)",          "Magma"),
    (grid_tok,  "Total Paperwork (tokens)",          "③ Bureaucratic Friction: Total Pipeline Paperwork", "Inferno"),
    (grid_comm, "Commission Hearings (count)",        "④ Commission Hearing Rate",                   "Viridis"),
    (grid_coun, "Council Hearings (count)",           "⑤ Council Hearing Rate",                      "Cividis"),
]

for i, (grid, z_title, title, cscale) in enumerate(SURFACES):
    fig = make_surface(grid, z_title, title, cscale)
    if fig:
        if i % 2 == 0:
            col1.plotly_chart(fig, use_container_width=True)
        else:
            col2.plotly_chart(fig, use_container_width=True)

if cells_available < total_cells:
    st.info(f"Surface {cells_available/total_cells*100:.0f}% complete. Updating live as GPU finishes.")

if auto_refresh and not gpu_done:
    time.sleep(4)
    st.rerun()

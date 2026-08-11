"""
live_friction_api.py  (production-ready)

Environment variables:
  PORT              default 8001
  API_ROOT          path to thesis root (where Data/ lives)
  CORS_ORIGINS      comma-sep list of allowed origins, default *
  DEFAULT_MODE      "cached" (default) or "live" — which source the map opens with
  SKIP_LIVE         "1" to skip loading the causal forests (cached mode only)

Two self-consistent artifact sets are served side by side so they can be compared
in the browser. /geometries always returns the file that matches the parcel order
of /predict for the same mode, so parcel index i means the same parcel in both:

  mode=cached   inference_cache.npy      + austin_base_geometries_cached.fgb  (271,567)
                Precomputed 2026-05-11; the surface the site shipped with.
  mode=live     causal_models.pkl/X_base + austin_base_geometries.fgb         (284,958)
                Current forests, evaluated per request (~13 s per new height,
                instant for dose-only changes).
"""

import http.server
import socket
import urllib.parse
import numpy as np
import joblib
import json
import os
import threading
import time
from pathlib import Path

ROOT = Path(os.environ.get("API_ROOT", r"c:\Users\dhl\data\Thesis\thesis"))
PORT = int(os.environ.get("PORT", 8001))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
DEFAULT_MODE = os.environ.get("DEFAULT_MODE", "cached")
SKIP_LIVE = os.environ.get("SKIP_LIVE", "0") == "1"

DATA = ROOT / "Data/Zoning_Cases"
CACHE_PATH = DATA / "inference_cache.npy"
MODELS_PATH = DATA / "causal_models.pkl"
X_BASE_PATH = DATA / "X_base.npy"
GEOM_LIVE = DATA / "austin_base_geometries.fgb"           # pairs with X_base.npy
GEOM_CACHED = DATA / "austin_base_geometries_cached.fgb"   # pairs with inference_cache.npy

# Column order written by pipelines/08j_generate_modern_LUI_baseline.py
X_BASE_COLS = [
    'Delta_Requested_Height', 'latitude', 'longitude',
    'median_household_income', 'race_white', 'race_black', 'race_hispanic',
    'renter_share', 'rent_burden', 'total_population', 'median_age',
    'appraised_value', 'building_age',
    'mortgage_rate_30yr', 'fed_funds_rate', 'local_unemployment_rate',
    'knn_petition_rate_1km', 'dist_petition_rate_lag1',
    'cumulative_min_signer_dist', 'cumulative_signers_outside_200ft',
    'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2',
    'cumulative_petition_attempted', 'cumulative_mobilization_failure',
    'fire_hazard_severity', 'slope_degree', 'is_imagine_corridor',
    'P_withdraw',
]

# Feature order the forests were fitted on (pipelines/08i_train_and_save_model.py).
# The mediator columns present in X_base are deliberately excluded from X.
EX_ANTE_COLS = [
    'Delta_Requested_Height', 'latitude', 'longitude',
    'median_household_income', 'race_white', 'race_black', 'race_hispanic',
    'renter_share', 'rent_burden', 'total_population', 'median_age',
    'appraised_value', 'building_age',
    'mortgage_rate_30yr', 'fed_funds_rate', 'local_unemployment_rate',
    'fire_hazard_severity', 'slope_degree', 'is_imagine_corridor',
    'knn_petition_rate_1km', 'dist_petition_rate_lag1',
]

# ── Load the cached hypergrid ────────────────────────────────────────────────
hypergrid = None
if CACHE_PATH.exists() and GEOM_CACHED.exists():
    print(f"Loading hypergrid cache from {CACHE_PATH}...", flush=True)
    hypergrid = np.load(CACHE_PATH, mmap_mode='r')
    print(f"  cached mode ready (grid {hypergrid.shape})", flush=True)
elif CACHE_PATH.exists():
    # Without the geometry that matches the cache, parcel indices in /predict
    # would not line up with /geometries, silently mislabelling the whole map.
    print(f"cached mode unavailable: {GEOM_CACHED.name} missing", flush=True)
else:
    print(f"cached mode unavailable: {CACHE_PATH.name} missing", flush=True)

# ── Load the current causal forests ──────────────────────────────────────────
cf_joint = cf_withd = survival_clf = None
X_ex_ante = None
if SKIP_LIVE:
    print("SKIP_LIVE=1 — not loading causal forests.", flush=True)
elif MODELS_PATH.exists() and X_BASE_PATH.exists() and GEOM_LIVE.exists():
    print(f"Loading causal forests from {MODELS_PATH}...", flush=True)
    models = joblib.load(MODELS_PATH)
    cf_joint = models['cf_joint']
    cf_withd = models['cf_withd']
    survival_clf = models['survival_clf']

    # X_base carries the full LUI confounder set; reorder to the fitted subset.
    X_base = np.load(X_BASE_PATH)
    ex_idx = [X_BASE_COLS.index(c) for c in EX_ANTE_COLS]
    X_ex_ante = np.ascontiguousarray(X_base[:, ex_idx], dtype=np.float64)
    del X_base
    print(f"  live mode ready (X {X_ex_ante.shape}, joint adds P_withdraw)", flush=True)
else:
    print("live mode unavailable: models, X_base or geometry missing", flush=True)

MODES = {}
if hypergrid is not None:
    MODES['cached'] = {
        'label': 'Cached surface (2026-05-11)',
        'note': 'Precomputed hypergrid; the surface the site shipped with.',
        'parcels': int(hypergrid.shape[1]),
        'geometry': str(GEOM_CACHED),
        'latency': 'instant',
    }
if X_ex_ante is not None:
    MODES['live'] = {
        'label': 'Current forests (live)',
        'note': 'causal_models.pkl evaluated per request; mediators excluded from X.',
        'parcels': int(len(X_ex_ante)),
        'geometry': str(GEOM_LIVE),
        'latency': '~13 s per new height, instant per dose',
    }

if not MODES:
    raise SystemExit("ABORT: neither cached nor live mode could be loaded.")
if DEFAULT_MODE not in MODES:
    DEFAULT_MODE = next(iter(MODES))

# Cheap alignment check — a mismatch here would mislabel every parcel on the map.
try:
    import pyogrio
    for mode, cfg in MODES.items():
        n_geom = pyogrio.read_info(cfg['geometry'])['features']
        if n_geom != cfg['parcels']:
            raise SystemExit(f"ABORT: mode '{mode}' geometry has {n_geom:,} features "
                             f"but /predict returns {cfg['parcels']:,} parcels.")
        print(f"  {mode}: {n_geom:,} parcels, geometry alignment verified", flush=True)
except ImportError:
    print("  pyogrio not installed — skipping geometry alignment check", flush=True)

# Per-height memo for live mode (dose scaling is O(1) on top of it)
live_lock = threading.Lock()
live_height = None
live_marginal_joint = None
live_marginal_withd = None


def marginals(mode, height):
    """Return (delay, attrition, withdrawal) marginal effects for T0=0 -> T1=1."""
    global live_height, live_marginal_joint, live_marginal_withd

    if mode == 'cached':
        # Nearest cached height tier (grid covers 5-120 ft in 1 ft steps)
        h_idx = int(np.clip(round(height) - 5, 0, hypergrid.shape[0] - 1))
        return hypergrid[h_idx, :, 0], hypergrid[h_idx, :, 1], hypergrid[h_idx, :, 2]

    # Serialised: the live path mutates the shared X matrix in place, so two
    # concurrent requests at different heights would corrupt each other.
    with live_lock:
        if live_height != height:
            print(f"live: height={height} -> re-evaluating causal forests...", flush=True)
            X_ex_ante[:, 0] = height  # Delta_Requested_Height
            # Withdrawal hurdle feeds the joint model as its last feature.
            p_withd = survival_clf.predict_proba(X_ex_ante)[:, 1]
            X_joint = np.column_stack([X_ex_ante, p_withd])
            live_marginal_joint = cf_joint.effect(X_joint, T0=0.0, T1=1.0)
            live_marginal_withd = cf_withd.effect(X_ex_ante, T0=0.0, T1=1.0)
            live_height = height
        return live_marginal_joint[:, 1], live_marginal_joint[:, 0], live_marginal_withd


class FrictionAPIHandler(http.server.BaseHTTPRequestHandler):

    # Browsers open speculative sockets and leave them idle. Without a timeout a
    # single one of those wedges the server; with threads, it only costs a thread.
    timeout = 30

    def log_message(self, format, *args):
        pass

    def handle_one_request(self):
        # An idle preconnect or a tab closed mid-download is normal traffic here,
        # not an error worth a traceback.
        try:
            super().handle_one_request()
        except (TimeoutError, ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            self.close_connection = True

    def _cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', CORS_ORIGINS)
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _send_bytes(self, payload, content_type):
        self.send_response(200)
        self._cors_headers()
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, code, message=''):
        self.send_response(code)
        self._cors_headers()
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        if message:
            self.wfile.write(message.encode())

    def _mode(self, query):
        mode = query.get('mode', [DEFAULT_MODE])[0]
        return mode if mode in MODES else None

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        if parsed.path == '/health':
            self._send_bytes(b'ok', 'text/plain')
            return

        if parsed.path == '/modes':
            body = json.dumps({
                'default': DEFAULT_MODE,
                'modes': {m: {k: v for k, v in cfg.items() if k != 'geometry'}
                          for m, cfg in MODES.items()},
            }).encode()
            self._send_bytes(body, 'application/json')
            return

        if parsed.path == '/geometries':
            mode = self._mode(query)
            if mode is None:
                self._error(400, 'unknown mode')
                return
            try:
                with open(MODES[mode]['geometry'], 'rb') as f:
                    data = f.read()
            except FileNotFoundError:
                self._error(404, 'geometry missing')
                return
            self._send_bytes(data, 'application/octet-stream')
            return

        if parsed.path == '/predict':
            mode = self._mode(query)
            if mode is None:
                self._error(400, 'unknown mode')
                return
            try:
                dose = float(query.get('dose', [0.20])[0])
                height = float(query.get('height', [29.0])[0])
            except ValueError:
                self._error(400, 'dose and height must be numbers')
                return

            t0 = time.time()
            m_delay, m_attrition, m_withd = marginals(mode, height)

            # O(1): scale marginals by dose
            cate_delay = m_delay * dose
            cate_attr = m_attrition * dose
            cate_risk = m_withd * dose

            # Pack: [delay0, height0, withd0, delay1, ...]
            out = np.empty((MODES[mode]['parcels'] * 3,), dtype=np.float32)
            out[0::3] = np.clip(cate_delay, -365, 3650)  # delay
            out[1::3] = np.clip(cate_attr, -500, 1500)   # height attrition
            out[2::3] = np.clip(cate_risk, -1.0, 1.0)    # withdrawal risk

            self._send_bytes(out.tobytes(), 'application/octet-stream')
            print(f"Served /predict (mode={mode}, h={height}, dose={dose:.2f}) "
                  f"in {time.time() - t0:.3f}s", flush=True)
            return

        self._error(404)


if __name__ == '__main__':
    class Server(http.server.ThreadingHTTPServer):
        daemon_threads = True
        # Not reusing the address: on Windows SO_REUSEADDR lets a second instance
        # bind the same port and silently steal requests. Fail loudly instead.
        allow_reuse_address = False
        # Dual-stack. Browsers resolve "localhost" to ::1 first, so an IPv4-only
        # bind makes fetch() fail while curl on 127.0.0.1 still works.
        address_family = socket.AF_INET6

        def server_bind(self):
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            super().server_bind()

    try:
        server = Server(("::", PORT), FrictionAPIHandler)
    except OSError as exc:
        print(f"IPv6 bind failed ({exc}); falling back to IPv4 only.", flush=True)
        Server.address_family = socket.AF_INET
        Server.server_bind = http.server.ThreadingHTTPServer.server_bind
        server = Server(("0.0.0.0", PORT), FrictionAPIHandler)

    with server as httpd:
        print(f"Live Friction API serving on port {PORT} "
              f"(modes: {', '.join(MODES)}; default {DEFAULT_MODE})", flush=True)
        httpd.serve_forever()

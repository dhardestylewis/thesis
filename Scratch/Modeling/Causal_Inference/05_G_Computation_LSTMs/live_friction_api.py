"""
live_friction_api.py  (production-ready)

Environment variables:
  PORT              default 8001
  API_ROOT          path to thesis root (where Data/ lives)
  CORS_ORIGINS      comma-sep list of allowed origins, default *
"""

import http.server
import socketserver
import urllib.parse
import numpy as np
import joblib
import os
import time
from pathlib import Path

ROOT = Path(os.environ.get("API_ROOT", r"c:\Users\dhl\data\Thesis\thesis"))
PORT = int(os.environ.get("PORT", 8001))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")

print(f"Loading Causal Models from {ROOT}...", flush=True)
models = joblib.load(ROOT / "Data/Zoning_Cases/causal_models.pkl")
cf_joint = models['cf_joint']
cf_withd = models['cf_withd']
X_base = np.load(ROOT / "Data/Zoning_Cases/X_base.npy")

print(f"Loaded X_base shape: {X_base.shape}", flush=True)

# State cache for O(1) dose updates
cached_height = None
cached_marginal_joint = None
cached_marginal_withd = None


class FrictionAPIHandler(http.server.SimpleHTTPRequestHandler):

    def log_message(self, format, *args):
        # Suppress per-request noise; only keep important prints
        pass

    def _cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', CORS_ORIGINS)
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        global cached_height, cached_marginal_joint, cached_marginal_withd

        parsed = urllib.parse.urlparse(self.path)

        # ── Health check ──────────────────────────────────────────────────
        if parsed.path == '/health':
            self.send_response(200)
            self._cors_headers()
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'ok')
            return

        # ── Serve FlatGeobuf geometry ─────────────────────────────────────
        if parsed.path == '/geometries':
            fgb_path = ROOT / "Data/Zoning_Cases/austin_base_geometries.fgb"
            try:
                with open(fgb_path, 'rb') as f:
                    data = f.read()
                self.send_response(200)
                self._cors_headers()
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
            return

        # ── Serve live CATE predictions ───────────────────────────────────
        if parsed.path == '/predict':
            query = urllib.parse.parse_qs(parsed.query)
            try:
                dose   = float(query.get('dose',   [0.20])[0])
                height = float(query.get('height', [29.0])[0])
            except ValueError:
                self.send_response(400)
                self.end_headers()
                return

            t0 = time.time()

            if cached_height != height:
                print(f"Height={height} → re-evaluating Causal Forest...", flush=True)
                X_base[:, 0] = height
                cached_marginal_joint = cf_joint.effect(X_base, T0=0.0, T1=1.0)
                cached_marginal_withd = cf_withd.effect(X_base, T0=0.0, T1=1.0)
                cached_height = height
                print(f"  Done in {time.time()-t0:.2f}s", flush=True)

            # O(1): scale cached marginal by dose
            cate_multi = cached_marginal_joint * dose
            cate_w     = cached_marginal_withd  * dose

            # Pack: [delay0, height0, withd0, delay1, ...]
            out = np.empty((len(X_base) * 3,), dtype=np.float32)
            out[0::3] = np.clip(cate_multi[:, 1], -365,  3650)  # delay
            out[1::3] = np.clip(cate_multi[:, 0], -500,  1500)  # height attrition
            out[2::3] = np.clip(cate_w,            -1.0,  1.0)  # withdrawal

            buf = out.tobytes()
            print(f"Served /predict (dose={dose:.2f}, h={height}) → {len(buf)/1e6:.2f} MB in {time.time()-t0:.3f}s", flush=True)

            self.send_response(200)
            self._cors_headers()
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(buf)))
            self.end_headers()
            self.wfile.write(buf)
            return

        self.send_response(404)
        self.end_headers()


if __name__ == '__main__':
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), FrictionAPIHandler) as httpd:
        print(f"Live Friction API serving on port {PORT}", flush=True)
        httpd.serve_forever()

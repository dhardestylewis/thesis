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

# Load baseline data
X_base = np.load(ROOT / "Data/Zoning_Cases/X_base.npy")
print(f"Loaded X_base shape: {X_base.shape}", flush=True)

# Load Hypergrid Cache if available (Instant API mode)
CACHE_PATH = ROOT / "Data/Zoning_Cases/inference_cache.npy"
hypergrid = None
if CACHE_PATH.exists():
    print(f"Loading Hypergrid Cache from {CACHE_PATH}...", flush=True)
    hypergrid = np.load(CACHE_PATH)
    print(f"  Instant API Mode enabled (Grid: {hypergrid.shape})", flush=True)
else:
    print(f"Cache missing. Loading Causal Models from {ROOT} (Slow Mode)...", flush=True)
    models = joblib.load(ROOT / "Data/Zoning_Cases/causal_models.pkl")
    cf_joint = models['cf_joint']
    cf_withd = models['cf_withd']

# State cache for Slow Mode
cached_height = None
cached_marginal_joint = None
cached_marginal_withd = None


class FrictionAPIHandler(http.server.SimpleHTTPRequestHandler):

    def log_message(self, format, *args):
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

        if parsed.path == '/health':
            self.send_response(200)
            self._cors_headers()
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'ok')
            return

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

            if hypergrid is not None:
                # INSTANT MODE: Find nearest cached height tier (5-120 range)
                h_idx = int(np.clip(round(height) - 5, 0, hypergrid.shape[0] - 1))
                m_delay     = hypergrid[h_idx, :, 0]
                m_attrition = hypergrid[h_idx, :, 1]
                m_withd     = hypergrid[h_idx, :, 2]
            else:
                # SLOW MODE: Evaluate model if height changed
                if cached_height != height:
                    print(f"Height={height} → re-evaluating Causal Forest...", flush=True)
                    X_base[:, 0] = height
                    cached_marginal_joint = cf_joint.effect(X_base, T0=0.0, T1=1.0)
                    cached_marginal_withd = cf_withd.effect(X_base, T0=0.0, T1=1.0)
                    cached_height = height
                m_delay     = cached_marginal_joint[:, 1]
                m_attrition = cached_marginal_joint[:, 0]
                m_withd     = cached_marginal_withd

            # O(1): Scale marginals by dose
            cate_delay = m_delay * dose
            cate_attr  = m_attrition * dose
            cate_risk  = m_withd * dose

            # Pack: [delay0, height0, withd0, delay1, ...]
            out = np.empty((len(X_base) * 3,), dtype=np.float32)
            out[0::3] = np.clip(cate_delay, -365,  3650)  # delay
            out[1::3] = np.clip(cate_attr,  -500,  1500)  # height attrition
            out[2::3] = np.clip(cate_risk,  -1.0,  1.0)   # withdrawal risk

            buf = out.tobytes()
            self.send_response(200)
            self._cors_headers()
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(buf)))
            self.end_headers()
            self.wfile.write(buf)
            
            print(f"Served /predict (h={height}, dose={dose:.2f}) in {time.time()-t0:.3f}s", flush=True)
            return

        self.send_response(404)
        self.end_headers()

        self.send_response(404)
        self.end_headers()


if __name__ == '__main__':
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), FrictionAPIHandler) as httpd:
        print(f"Live Friction API serving on port {PORT}", flush=True)
        httpd.serve_forever()

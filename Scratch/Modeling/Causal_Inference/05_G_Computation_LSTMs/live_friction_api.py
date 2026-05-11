"""
live_friction_api.py

A native Python HTTP API that dynamically serves the Predictive Friction Surface.
Uses model caching: re-evaluates the Random Forest when `height` changes (~3s),
but caches the marginal effect so `dose` changes are INSTANTANEOUS (0.01s).
"""

import http.server
import socketserver
import urllib.parse
import numpy as np
import joblib
from pathlib import Path
import json
import time

ROOT = Path(r"c:\Users\dhl\data\Thesis\thesis")

print("Loading Causal Models and Base Data into RAM...", flush=True)
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
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        global cached_height, cached_marginal_joint, cached_marginal_withd
        
        parsed_url = urllib.parse.urlparse(self.path)
        
        # Serve the geometries
        if parsed_url.path == '/geometries':
            fgb_path = ROOT / "Data/Zoning_Cases/austin_base_geometries.fgb"
            try:
                with open(fgb_path, 'rb') as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_response(404)
                self.end_headers()
            return
            
        # Serve the CATE predictions
        if parsed_url.path == '/predict':
            query = urllib.parse.parse_qs(parsed_url.query)
            try:
                dose = float(query.get('dose', [0.20])[0])
                height = float(query.get('height', [29.0])[0])
            except ValueError:
                self.send_response(400)
                self.end_headers()
                return

            t0 = time.time()
            
            # If height changed, we MUST re-evaluate the random forest trees
            if cached_height != height:
                print(f"Height changed to {height}. Re-evaluating Causal Forest (expect 2-4s)...", flush=True)
                X_base[:, 0] = height # Update Delta_Requested_Height
                
                # Compute marginal effect (T1=1.0)
                # This is the heavy computation
                cached_marginal_joint = cf_joint.effect(X_base, T0=0.0, T1=1.0)
                cached_marginal_withd = cf_withd.effect(X_base, T0=0.0, T1=1.0)
                cached_height = height
                
            # If only dose changed, inference is O(1) math multiplication!
            cate_multi = cached_marginal_joint * dose
            cate_w = cached_marginal_withd * dose
            
            # Extract delay, height attrition, withdrawal
            height_preds = cate_multi[:, 0]
            delay_preds = cate_multi[:, 1]
            withd_preds = cate_w
            
            # Pack into a single 1D Float32Array: [delay0, height0, withd0, delay1, height1, withd1, ...]
            # We clip the values to sensible bounds
            out_array = np.empty((len(X_base) * 3,), dtype=np.float32)
            out_array[0::3] = np.clip(delay_preds, -365, 3650)
            out_array[1::3] = np.clip(height_preds, -500, 1500)
            out_array[2::3] = np.clip(withd_preds, -1.0, 1.0)
            
            buf = out_array.tobytes()
            
            print(f"Served /predict (dose={dose}, height={height}) in {time.time() - t0:.3f}s. Buffer: {len(buf)/1024/1024:.2f}MB")
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(buf)))
            self.end_headers()
            self.wfile.write(buf)
            return

        self.send_response(404)
        self.end_headers()

if __name__ == '__main__':
    PORT = 8001
    with socketserver.TCPServer(("", PORT), FrictionAPIHandler) as httpd:
        print(f"Live Friction API serving on port {PORT}", flush=True)
        httpd.serve_forever()

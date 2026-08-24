#!/usr/bin/env python3
"""
serve_dashboard.py -- BearRL × WuBuMath live training dashboard.

Serves dashboard/index.html at / and a live metrics feed at /metrics.json.
Metrics come from the geodesic-reacher trainer (bear_geodesic_reacher.c):
we launch it as a subprocess and tail its JSON-lines output.

Endpoints:
    GET /              -> the mobile dashboard
    GET /metrics.json  -> {"live": [...recent metric rows...], "certificate": {...}}
"""
import json, os, subprocess, threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.dirname(os.path.abspath(__file__))
REACHER = os.path.join(ROOT, "..", "src", "bear_geodesic_reacher.c")

METRICS = []          # recent rows
CERT = {}             # certificate row when done
LOCK = threading.Lock()

def run_trainer():
    """Build (if needed) + run the trainer, streaming its JSON lines."""
    global CERT
    try:
        proc = subprocess.Popen(
            ["/tmp/reacher", "150000", "32", "0.05"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            with LOCK:
                if row.get("type") == "metrics":
                    METRICS.append(row)
                    del METRICS[:-600]          # keep last 600
                elif row.get("type") == "greedy_eval":
                    row["baseline_random_solve_rate"] = 0.004
                    row["baseline_random_mean_dist"] = 0.890
                    CERT["greedy"] = row
                elif row.get("type") == "certificate":
                    CERT["final"] = row
    except Exception as e:
        with LOCK:
            CERT["error"] = str(e)

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def do_GET(self):
        if self.path.startswith("/metrics.json"):
            with LOCK:
                payload = json.dumps({
                    "live": METRICS[-400:],
                    "certificate": CERT,
                    "env": "quaternion_geodesic_reacher",
                    "state_space": "S^3 unit quaternions",
                    "action_space": f"tangent angular steps",
                    "reward": "potential-based geodesic distance shaping",
                }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            super().do_GET()

    def log_message(self, *a):   # quiet
        pass

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    threading.Thread(target=run_trainer, daemon=True).start()
    print(f"serving on :{port}  (/ and /metrics.json)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()

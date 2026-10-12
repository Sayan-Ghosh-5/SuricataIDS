"""
decision_engine.py
Turns a false-positive probability into one of three actions.
Alerts are NEVER deleted -- only re-prioritised (slide 13 design principle).

   p_fp <= T_LOW            -> LIKELY GENUINE      -> ESCALATE
   T_LOW < p_fp < T_HIGH    -> UNCERTAIN           -> HUMAN REVIEW
   p_fp >= T_HIGH           -> LIKELY FALSE POS.   -> DEPRIORITIZE
"""
import json, time, sys, itertools
from pathlib import Path
import joblib
import pandas as pd

from eve_parser import FeatureExtractor, NUMERIC, CATEGORICAL

T_LOW, T_HIGH = 0.25, 0.75
ESCALATE, REVIEW, DEPRIORITIZE = "ESCALATE", "HUMAN_REVIEW", "DEPRIORITIZE"


def decide(p_fp: float, t_low: float = T_LOW, t_high: float = T_HIGH) -> str:
    if p_fp >= t_high:
        return DEPRIORITIZE
    if p_fp <= t_low:
        return ESCALATE
    return REVIEW


class AlertTriage:
    def __init__(self, model_path, t_low=T_LOW, t_high=T_HIGH):
        self.model = joblib.load(model_path)
        self.model.named_steps["clf"].n_jobs = 1      # single-row scoring is faster without worker overhead
        self.fx = FeatureExtractor()
        self.t_low, self.t_high = t_low, t_high

    def process(self, ev: dict):
        row = self.fx.extract(ev)
        if row is None:
            return None
        p = float(self.model.predict_proba(pd.DataFrame([row])[NUMERIC + CATEGORICAL])[0, 1])
        return {
            "timestamp": ev["timestamp"], "src_ip": ev["src_ip"], "dest_ip": ev["dest_ip"],
            "dest_port": ev["dest_port"], "signature": ev["alert"]["signature"],
            "severity": ev["alert"]["severity"],
            "fp_probability": round(p, 3), "action": decide(p, self.t_low, self.t_high),
        }


def follow(path, from_start=False):
    """tail -F style generator so the engine can run live next to Suricata."""
    with open(path) as f:
        if not from_start:
            f.seek(0, 2)
        while True:
            line = f.readline()
            if line:
                yield json.loads(line)
            else:
                time.sleep(0.5)


if __name__ == "__main__":
    import argparse
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Triage Suricata EVE alerts with the ML false-positive model")
    ap.add_argument("eve", nargs="?", default=str(root / "data" / "eve.json"), help="path to eve.json")
    ap.add_argument("--live", action="store_true", help="follow the file like tail -F (run beside Suricata)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N events (quick demo)")
    ap.add_argument("--t-low", type=float, default=T_LOW)
    ap.add_argument("--t-high", type=float, default=T_HIGH)
    args = ap.parse_args()

    triage = AlertTriage(root / "models" / "rf_fp_model.joblib", args.t_low, args.t_high)
    stream = follow(args.eve, from_start=True) if args.live else (json.loads(l) for l in open(args.eve))
    if args.limit:
        stream = itertools.islice(stream, args.limit)
    out = root / "data" / "triaged_alerts.jsonl"
    n = 0
    with open(out, "w") as fo:
        for ev in stream:
            r = triage.process(ev)
            if r:
                fo.write(json.dumps(r) + "\n"); fo.flush(); n += 1
                if n <= 5:
                    print(r)
    print(f"Triaged {n} alerts -> {out}")

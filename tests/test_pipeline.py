"""Run:  python tests/test_pipeline.py     (plain asserts, no pytest needed)"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eve_parser import FeatureExtractor, load_eve, NUMERIC, CATEGORICAL
from decision_engine import decide, AlertTriage, ESCALATE, REVIEW, DEPRIORITIZE


def sample(ts="2026-09-01T09:00:10.000+0000", src="203.0.113.9", dport=22):
    return {"timestamp": ts, "flow_id": 1, "event_type": "alert", "src_ip": src, "src_port": 4444,
            "dest_ip": "192.168.10.5", "dest_port": dport, "proto": "TCP",
            "alert": {"signature_id": 2009582, "signature": "ET SCAN NMAP", "category": "Attempted Information Leak", "severity": 2},
            "flow": {"pkts_toserver": 4, "pkts_toclient": 2, "bytes_toserver": 400, "bytes_toclient": 120,
                     "start": "2026-09-01T09:00:07.000+0000"}}


def t01_non_alert_ignored():
    assert FeatureExtractor().extract({"event_type": "flow"}) is None

def t02_feature_values():
    r = FeatureExtractor().extract(sample())
    assert r["total_pkts"] == 6 and r["total_bytes"] == 520 and abs(r["flow_duration"] - 3.0) < 1e-6
    assert r["src_internal"] == 0 and r["dst_internal"] == 1 and r["dst_well_known"] == 1

def t03_alert_frequency_window():
    fx = FeatureExtractor()
    fx.extract(sample("2026-09-01T09:00:00.000+0000", dport=22))
    fx.extract(sample("2026-09-01T09:00:05.000+0000", dport=80))
    r = fx.extract(sample("2026-09-01T09:00:10.000+0000", dport=443))
    assert r["alerts_src_60s"] == 3 and r["distinct_ports_src_60s"] == 3
    r = fx.extract(sample("2026-09-01T09:05:00.000+0000"))           # 5 min later -> window expired
    assert r["alerts_src_60s"] == 1

def t04_decision_examples_from_deck():
    assert decide(0.91) == DEPRIORITIZE and decide(0.12) == ESCALATE and decide(0.52) == REVIEW

def t05_threshold_boundaries():
    assert decide(0.25) == ESCALATE and decide(0.75) == DEPRIORITIZE and decide(0.2501) == REVIEW

def t06_end_to_end_alert_is_never_dropped():
    triage = AlertTriage(ROOT / "models" / "rf_fp_model.joblib")
    out = triage.process(sample())
    assert out is not None and out["action"] in (ESCALATE, REVIEW, DEPRIORITIZE) and 0 <= out["fp_probability"] <= 1

def t07_full_file_row_count():
    n_lines = sum(1 for _ in open(ROOT / "data" / "eve.json"))
    assert len(load_eve(ROOT / "data" / "eve.json")) == n_lines == 20000

def t08_dashboard_routes():
    from dashboard import app
    c = app.test_client()
    assert c.get("/").status_code == 200 and c.get("/?action=ESCALATE").status_code == 200
    assert isinstance(c.get("/api/alerts").get_json(), list)

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("t0")]
    failed = 0
    for n, f in tests:
        try:
            f(); print(f"PASS  {n}")
        except Exception as e:
            failed += 1; print(f"FAIL  {n}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)

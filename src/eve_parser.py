"""
eve_parser.py
Reads Suricata EVE JSON alert events and converts each one into a feature row.

Features (matches slide 12 of the Phase 1 deck):
  protocol, src/dst ports, signature id, severity, category, packet count,
  byte count, flow duration, connection frequency, alert frequency, network context.
"""
import json
from collections import defaultdict, deque
from datetime import datetime
import pandas as pd

WINDOW = 60.0   # seconds used for the frequency features


def _ts(s: str) -> float:
    return datetime.strptime(s[:23], "%Y-%m-%dT%H:%M:%S.%f").timestamp()


def _is_internal(ip: str) -> int:
    return int(ip.startswith(("10.", "192.168.", "172.16.")))


class FeatureExtractor:
    """Stateful: keeps a sliding 60 s window per source IP for the frequency features."""

    def __init__(self, window: float = WINDOW):
        self.window = window
        self.alerts_by_src = defaultdict(deque)     # src_ip -> deque[(ts, dest_port, sid)]

    def extract(self, ev: dict):
        if ev.get("event_type") != "alert":
            return None
        a, fl = ev["alert"], ev.get("flow", {})
        ts = _ts(ev["timestamp"])
        q = self.alerts_by_src[ev["src_ip"]]
        while q and ts - q[0][0] > self.window:
            q.popleft()
        q.append((ts, ev["dest_port"], a["signature_id"]))

        p_ts, p_tc = fl.get("pkts_toserver", 0), fl.get("pkts_toclient", 0)
        b_ts, b_tc = fl.get("bytes_toserver", 0), fl.get("bytes_toclient", 0)
        dur = max(0.0, ts - _ts(fl["start"])) if "start" in fl else 0.0
        return {
            "flow_id": ev.get("flow_id"),
            "proto": ev.get("proto", "NA"),
            "app_proto": ev.get("app_proto", "none"),
            "category": a.get("category", "unknown"),
            "signature_id": a["signature_id"],
            "severity": a.get("severity", 3),
            "src_port": ev.get("src_port", 0),
            "dest_port": ev["dest_port"],
            "pkts_toserver": p_ts, "pkts_toclient": p_tc,
            "bytes_toserver": b_ts, "bytes_toclient": b_tc,
            "total_pkts": p_ts + p_tc,
            "total_bytes": b_ts + b_tc,
            "bytes_per_pkt": (b_ts + b_tc) / max(1, p_ts + p_tc),
            "flow_duration": dur,
            "alerts_src_60s": len(q),                                   # alert frequency
            "distinct_ports_src_60s": len({x[1] for x in q}),           # connection frequency / scan-likeness
            "distinct_sigs_src_60s": len({x[2] for x in q}),
            "src_internal": _is_internal(ev["src_ip"]),                 # network context
            "dst_internal": _is_internal(ev["dest_ip"]),
            "dst_well_known": int(ev["dest_port"] < 1024),
        }


def load_eve(path: str) -> pd.DataFrame:
    fx, rows = FeatureExtractor(), []
    with open(path) as f:
        for line in f:
            row = fx.extract(json.loads(line))
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


NUMERIC = ["signature_id", "severity", "src_port", "dest_port", "pkts_toserver", "pkts_toclient",
           "bytes_toserver", "bytes_toclient", "total_pkts", "total_bytes", "bytes_per_pkt",
           "flow_duration", "alerts_src_60s", "distinct_ports_src_60s", "distinct_sigs_src_60s",
           "src_internal", "dst_internal", "dst_well_known"]
CATEGORICAL = ["proto", "app_proto", "category"]

"""
generate_dataset.py
Creates a SIMULATED Suricata EVE-JSON alert stream plus ground-truth labels.

Why simulated?  Phase 1 has no VM capture yet.  This script lets the whole
pipeline (parser -> model -> decision engine -> evaluation) be built and
tested end-to-end.  It is replaced by real eve.json + labels in Phase 2.

Output : data/eve.json    (one JSON event per line, same shape as Suricata EVE)
         data/labels.csv  (flow_id, label)   label: 1 = genuine attack, 0 = false positive
"""
import json, random, csv, datetime as dt
from pathlib import Path

random.seed(42)
OUT = Path(__file__).resolve().parent.parent / "data"

# (sid, signature, category, severity, app_proto, P(genuine | this signature))
SIGNATURES = [
    (2010937, "ET SCAN Suspicious inbound to MySQL port 3306", "Potentially Bad Traffic", 2, None, 0.80),
    (2002910, "ET SCAN Potential VNC Scan 5900-5920",          "Attempted Information Leak", 2, None, 0.85),
    (2100498, "GPL ATTACK_RESPONSE id check returned root",    "Potentially Bad Traffic", 2, "http", 0.75),
    (2014726, "ET WEB_SERVER SQL Injection Attempt",           "Web Application Attack", 1, "http", 0.70),
    (2013504, "ET MALWARE Possible CnC Beacon",                "A Network Trojan was detected", 1, "tls", 0.80),
    (2012648, "ET POLICY Dropbox Client Broadcasting",         "Not Suspicious Traffic", 3, None, 0.04),
    (2013028, "ET POLICY curl User-Agent Outbound",            "Attempted Information Leak", 3, "http", 0.10),
    (2210044, "SURICATA STREAM Packet with broken ack",        "Generic Protocol Command Decode", 3, None, 0.06),
    (2200074, "SURICATA IPv4 truncated packet",                "Generic Protocol Command Decode", 3, None, 0.05),
    (2027865, "ET INFO Observed DNS Query to .xyz TLD",        "Potentially Bad Traffic", 2, "dns", 0.22),
    (2025275, "ET INFO Windows Update / Cloud Sync Check",     "Misc activity", 3, "http", 0.03),
    (2009582, "ET SCAN NMAP -sS window 1024",                  "Attempted Information Leak", 2, None, 0.90),
]
WEIGHTS = [3, 3, 2, 4, 3, 20, 18, 20, 14, 10, 10, 3]       # how often each fires (benign noise dominates)
P_FP_EPISODE = 0.012                                       # chance of a benign burst (chatty/misconfigured host)
LABEL_NOISE = 0.02                                         # fraction of ground-truth labels flipped (annotation error)
P_ATTACK_EPISODE = 0.02                                    # chance that an iteration is a multi-alert attack burst

INTERNAL = [f"192.168.10.{i}" for i in range(2, 60)]
EXTERNAL = [f"203.0.113.{i}" for i in range(2, 250)] + [f"198.51.100.{i}" for i in range(2, 250)]
SERVERS  = ["192.168.10.5", "192.168.10.6", "192.168.10.20"]


def make_event(ts, flow_id, sig, genuine, attacker):
    """Genuine and false-positive alerts deliberately OVERLAP: insider attacks, stealthy
    low-and-slow attacks, noisy benign external traffic -- so no single feature gives the label away."""
    sid, name, cat, sev, app, _ = sig
    stealthy = genuine and random.random() < 0.15          # attack that looks benign
    noisy = (not genuine) and random.random() < 0.12       # benign traffic that looks hostile
    attack_like = (genuine and not stealthy) or noisy
    if attack_like:
        src = attacker or random.choice(EXTERNAL)
        if genuine and random.random() < 0.20:             # lateral movement / insider
            src = random.choice(INTERNAL)
        dst = random.choice(SERVERS) if random.random() < 0.85 else random.choice(INTERNAL)
        proto = random.choices(["TCP", "UDP"], [0.88, 0.12])[0]
        dport = random.choice([22, 80, 443, 3306, 445, 3389, 5900, 8080, 53])
        pkts_ts = max(1, int(random.lognormvariate(1.8, 1.0)))
        pkts_tc = max(0, int(random.lognormvariate(1.1, 1.0)))
        bpp = random.gauss(150, 80)
        dur = random.expovariate(1 / 5.0)
    else:
        src = random.choice(INTERNAL) if random.random() < 0.55 else random.choice(EXTERNAL)
        r = random.random()
        dst = random.choice(SERVERS) if r < 0.40 else (random.choice(INTERNAL) if r < 0.60 else random.choice(EXTERNAL[:30]))
        proto = random.choices(["TCP", "UDP"], [0.68, 0.32])[0]
        dport = random.choice([53, 80, 123, 443, 443, 8080, 5353, 1900, 3306, 22, 445])
        pkts_ts = max(1, int(random.lognormvariate(1.9, 1.1)))
        pkts_tc = max(0, int(random.lognormvariate(1.7, 1.0)))
        bpp = random.gauss(190, 100)
        dur = random.expovariate(1 / 9.0)
    bpp = max(40, bpp)
    start = ts - dt.timedelta(seconds=dur)
    return {
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000",
        "flow_id": flow_id,
        "event_type": "alert",
        "src_ip": src, "src_port": random.randint(1025, 65000),
        "dest_ip": dst, "dest_port": dport,
        "proto": proto,
        **({"app_proto": app} if app else {}),
        "alert": {"action": "allowed", "gid": 1, "signature_id": sid, "rev": 1,
                  "signature": name, "category": cat, "severity": sev},
        "flow": {"pkts_toserver": pkts_ts, "pkts_toclient": pkts_tc,
                 "bytes_toserver": int(pkts_ts * bpp), "bytes_toclient": int(pkts_tc * bpp * 1.4),
                 "start": start.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"},
    }


def main(n_alerts=20000):
    OUT.mkdir(exist_ok=True)
    t = dt.datetime(2026, 9, 1, 9, 0, 0)
    attackers = random.sample(EXTERNAL, 25)
    events, labels = [], []
    flow_id = 1_000_000
    while len(events) < n_alerts:
        t += dt.timedelta(seconds=random.expovariate(1 / 2.0))
        if random.random() < P_ATTACK_EPISODE:
            # attack burst: one attacker fires several alerts in a short window
            attacker = random.choice(attackers)
            for _ in range(random.randint(3, 12)):
                t += dt.timedelta(seconds=random.uniform(0.05, 1.5))
                flow_id += 1
                s = random.choices(SIGNATURES, weights=[x[5] for x in SIGNATURES])[0]
                events.append(make_event(t, flow_id, s, True, attacker)); labels.append((flow_id, 1))
        elif random.random() < P_FP_EPISODE / (1 - P_ATTACK_EPISODE):
            host, sig = random.choice(INTERNAL), random.choices(SIGNATURES[5:10], WEIGHTS[5:10])[0]
            for _ in range(random.randint(3, 15)):
                t += dt.timedelta(seconds=random.uniform(0.05, 2.0))
                flow_id += 1
                events.append(make_event(t, flow_id, sig, False, host)); labels.append((flow_id, 0))
        else:
            sig = random.choices(SIGNATURES, WEIGHTS)[0]
            genuine = random.random() < sig[5]
            flow_id += 1
            attacker = random.choice(attackers) if genuine else None
            events.append(make_event(t, flow_id, sig, genuine, attacker)); labels.append((flow_id, int(genuine)))
    events, labels = events[:n_alerts], labels[:n_alerts]
    labels = [(f, 1 - l if random.random() < LABEL_NOISE else l) for f, l in labels]
    with open(OUT / "eve.json", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    with open(OUT / "labels.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["flow_id", "label"]); w.writerows(labels)
    pos = sum(l for _, l in labels)
    print(f"Wrote {len(events)} alerts  |  genuine={pos}  false_positive={len(labels)-pos}")


if __name__ == "__main__":
    main()

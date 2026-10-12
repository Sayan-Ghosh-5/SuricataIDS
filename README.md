# AI-Assisted Suricata IDS - ML false-positive reduction (prototype)

Suricata stays the primary IDS. This ML layer reads Suricata's EVE JSON alerts and labels each one
ESCALATE / HUMAN_REVIEW / DEPRIORITIZE. Alerts are never deleted.

    pip install -r requirements.txt
    python src/generate_dataset.py      # SIMULATED eve.json + labels.csv (replace with real VM capture later)
    python src/train_model.py           # trains Random Forest -> models/rf_fp_model.joblib
    python src/evaluate.py              # baseline vs proposed, metrics, figures -> figures/, data/results.json
    python src/decision_engine.py --limit 2000 --t-high 0.915   # triage -> data/triaged_alerts.jsonl
    python src/dashboard.py             # http://127.0.0.1:5000
    python tests/test_pipeline.py       # 8 checks

Live mode next to Suricata:
    suricata -c /etc/suricata/suricata.yaml -i eth0
    python src/decision_engine.py /var/log/suricata/eve.json --live

Real data: put real Suricata output in data/eve.json and ground truth in data/labels.csv (flow_id,label; 1 = genuine, 0 = false positive).

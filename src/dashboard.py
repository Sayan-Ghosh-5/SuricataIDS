"""
dashboard.py  --  minimal SOC dashboard (Flask).
Shows the triaged alert queue: ESCALATE first, then HUMAN_REVIEW, then DEPRIORITIZE.
Nothing is hidden or deleted; deprioritized alerts stay visible through the filter.

Run:  python src/dashboard.py        then open http://127.0.0.1:5000
"""
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify

ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__, template_folder=str(ROOT / "templates"))
ORDER = {"ESCALATE": 0, "HUMAN_REVIEW": 1, "DEPRIORITIZE": 2}


def load_alerts():
    path = ROOT / "data" / "triaged_alerts.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f]


@app.route("/")
def index():
    alerts = load_alerts()
    flt = request.args.get("action", "ALL")
    counts = {k: sum(a["action"] == k for a in alerts) for k in ORDER}
    total = len(alerts)
    shown = [a for a in alerts if flt == "ALL" or a["action"] == flt]
    shown.sort(key=lambda a: (ORDER[a["action"]], a["fp_probability"]))
    reduction = round(100 * counts["DEPRIORITIZE"] / total, 1) if total else 0
    return render_template("dashboard.html", alerts=shown[:25], counts=counts, total=total,
                           reduction=reduction, flt=flt)


@app.route("/api/alerts")
def api_alerts():
    return jsonify(load_alerts())


if __name__ == "__main__":
    app.run(debug=True)

"""Run a reproducible experiment and export independently scored results."""
import argparse
import csv
import json
from pathlib import Path

from testbed.generator import generate
from testbed.deinterleavers import FrequencyGateDeinterleaver
from testbed.evaluation import evaluate

def save_csv(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="scenarios/baseline.json")
    ap.add_argument("--output", default="results/baseline")
    ap.add_argument("--gate-hz", type=float, default=500_000)
    args = ap.parse_args()
    scenario = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    observations, truth = generate(scenario)
    # The deinterleaver sees only observations, never the truth dictionary.
    assignments = FrequencyGateDeinterleaver(args.gate_hz).process(observations)
    metrics = evaluate(truth, assignments)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out/"observations.csv", ["pdw_id","toa_s","frequency_hz","pulse_width_s","aoa_deg","amplitude_dbfs"],
             [[p.pdw_id,p.toa_s,p.frequency_hz,p.pulse_width_s,p.aoa_deg,p.amplitude_dbfs] for p in observations])
    save_csv(out/"truth.csv", ["pdw_id","true_emitter"], sorted(truth.items()))
    save_csv(out/"assignments.csv", ["pdw_id","estimated_track"], sorted(assignments.items()))
    (out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
    for ax, labels, title in [(axes[0], truth, "Ground truth"), (axes[1], assignments, "Estimated association")]:
        for label in sorted(set(labels.values())):
            selected = [p for p in observations if labels[p.pdw_id] == label]
            ax.scatter([p.toa_s for p in selected], [p.frequency_hz/1e9 for p in selected], s=3, label=label)
        ax.set_ylabel("RF (GHz)")
        ax.set_title(title)
        ax.legend(markerscale=3, loc="best")
    axes[-1].set_xlabel("TOA (s)")
    fig.tight_layout()
    fig.savefig(out/"assignments.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

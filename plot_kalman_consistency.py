"""Plot branch-labelled Kalman diagnostics; never pool alternative beam states.

The plotted sample is prior rank zero. With --run-result, retain primary-track
candidates from that track's source family, including gate rejections. These
remain selected, harmonic-fitted observations, not an IID chi-square experiment.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2


def summarize(rows):
    output = {"outcomes": dict(Counter(r["outcome"] for r in rows)), "by_dof": {}}
    for dof in sorted({r["dof"] for r in rows if r["nis"] is not None}):
        group = [r for r in rows if r["dof"] == dof]
        output["by_dof"][str(dof)] = {}
        for label, subset in (("all_candidates", group),
                              ("accepted_branches", [r for r in group if r["outcome"] == "accepted"])):
            values = np.array([r["nis"] for r in subset])
            output["by_dof"][str(dof)][label] = {
                "count": len(values), "mean_nis": float(values.mean()) if len(values) else None,
                "reference_mean": dof,
                "inside_95_percent": float(100 * np.mean(values <= chi2.ppf(.95, dof))) if len(values) else None,
                "inside_99_73_percent": float(100 * np.mean(values <= chi2.ppf(.9973, dof))) if len(values) else None}
    return output


def acf(values, max_lag=10):
    values = np.asarray(values, float)
    centered = values - values.mean()
    denominator = centered @ centered
    if len(values) < 3 or denominator == 0:
        return [], []
    lags = list(range(1, min(max_lag + 1, len(values))))
    return lags, [float(centered[:-lag] @ centered[lag:] / denominator) for lag in lags]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--run-result")
    parser.add_argument("--output", default="results/kalman_consistency.png")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.log).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["prior_hypothesis_rank"] == 0]
    names = {}
    if args.run_result:
        result = json.loads(Path(args.run_result).read_text())
        names = {track_id: family for family, track_id in result["track_ids"].items()}
        sources = {e["observation_id"]: e["source_family"] for e in result["events"]}
        rows = [r for r in rows if r["track_id"] in names and
                (r["outcome"] == "missed" or sources.get(r["observation_id"]) == names[r["track_id"]])]
    if not rows:
        parser.error("No rank-zero candidate records match the requested tracks")
    report = {"sample": "Prior rank zero; same source-family primary candidates when run result supplied",
              "limitations": "Upstream selection, harmonic minimization, overlapping summaries and beam selection violate IID chi-square assumptions. Accepted-only NIS is gate-truncated. ACF uses event lag, not uniform time. No automatic Q/R tuning.",
              "tracks": {}}
    fig, ax = plt.subplots(4, 3, figsize=(17, 16), constrained_layout=True)
    cov_fig, cov_ax = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    origin = min(r["time_s"] for r in rows if r["nis"] is not None)
    for track_id in sorted({r["track_id"] for r in rows}):
        track_rows = [r for r in rows if r["track_id"] == track_id]
        label = names.get(track_id, track_id)
        stats = summarize(track_rows)
        report["tracks"][label] = stats
        candidates = [r for r in track_rows if r["nis"] is not None]
        if not candidates:
            continue
        for dof in sorted({r["dof"] for r in candidates}):
            vals = [r["nis"] for r in candidates if r["dof"] == dof]
            limit = chi2.ppf(.999, dof)
            counts, edges = np.histogram(vals, bins=30, range=(0, limit))
            ax[0, 0].stairs(counts / (len(vals) * np.diff(edges)), edges,
                            label=f"{label}, df={dof}, n={len(vals)}, tail={sum(v > limit for v in vals)}")
            grid = np.linspace(0, limit, 300)
            ax[0, 0].plot(grid, chi2.pdf(grid, dof), '--', label=f"χ²({dof}) reference")
        for outcome, marker in (("accepted", '.'), ("rejected", 'x'), ("not_selected", '+')):
            subset = [r for r in candidates if r["outcome"] == outcome]
            ax[0, 1].scatter([r["time_s"] - origin for r in subset],
                            [r["nis"] / r["dof"] for r in subset], marker=marker, s=14, label=f"{label} {outcome}")
        for dof, st in stats["by_dof"].items():
            val = st["all_candidates"]
            ax[0, 2].scatter([val["mean_nis"] / int(dof)], [val["inside_95_percent"]], label=f"{label}, df={dof}")
        stats["pre_gate_components"] = {}
        for axis in ("aoa_deg", "pw_s", "pri_base_s"):
            components = [r["axes"][axis] for r in candidates if axis in r["axes"]]
            if components:
                stats["pre_gate_components"][axis] = {
                    "count": len(components),
                    "mean_nis_component": float(np.mean([c["innovation"] ** 2 / c["S"] for c in components])),
                    "mean_gain": float(np.mean([c["K"] for c in components]))}
        stats["innovations"] = {}
        for col, axis in enumerate(("aoa_deg", "pw_s", "pri_base_s")):
            accepted = [r for r in candidates if r["outcome"] == "accepted" and axis in r["axes"]]
            if not accepted:
                continue
            times = [r["time_s"] - origin for r in accepted]
            components = [r["axes"][axis] for r in accepted]
            white = [c["innovation"] / np.sqrt(c["S"]) for c in components]
            lags, correlation = acf(white)
            stats["innovations"][axis] = {"count": len(white),
                "raw_mean": float(np.mean([c["innovation"] for c in components])),
                "standardized_mean": float(np.mean(white)), "acf_lags": lags, "acf": correlation}
            ax[1, col].plot(times, white, '.-', ms=2, lw=.6, label=label)
            ax[1, col].axhline(0, color='gray', lw=.5)
            ax[1, col].set_title(f"{axis}: ν / √S (accepted)")
            ax[2, col].plot(lags, correlation, '.-', label=label)
            ax[2, col].set_title(f"{axis}: innovation ACF, event lag")
            ax[3, col].plot(times, [c["K"] for c in components], label=f"{label} K")
            ax[3, col].set_title(f"{axis}: K and covariance fractions")
            for key, style in (("P_predicted", '--'), ("R", ':'), ("S", '-.')):
                cov_ax[col].plot(times, [c[key] for c in components], style, lw=.9, label=f"{label} {key}")
                cov_ax[col].set_title(f"{axis}: P⁻, R, S (native units squared)")
                cov_ax[col].set_xlabel("Seconds from first monitored candidate")
                cov_ax[col].set_yscale("log")
                ax[3, col].plot(times, [c[key] / c["S"] for c in components], style, lw=.7, label=f"{label} {key}/S")
    ax[0, 0].set_title("Pre-gate NIS distribution by df")
    ax[0, 1].set_title("Candidate NIS / df; rejections retained")
    ax[0, 1].axhline(1, color='gray', ls='--')
    ax[0, 1].set_yscale('symlog', linthresh=1)
    ax[0, 1].set_ylim(bottom=0)
    ax[0, 1].set_xlabel("Seconds from first monitored candidate")
    ax[0, 2].set_title("Mean NIS / df vs % inside 95% gate")
    ax[0, 2].axvline(1, color='gray', ls='--')
    ax[0, 2].axhline(95, color='gray', ls='--')
    for panel in ax.flat:
        panel.grid(alpha=.2)
        handles, labels = panel.get_legend_handles_labels()
        if handles:
            panel.legend(fontsize=6)
    fig.suptitle("Kalman consistency — selected observations; conditional χ² comparison", fontsize=14)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    for panel in cov_ax:
        panel.grid(alpha=.2)
        if panel.get_legend_handles_labels()[0]:
            panel.legend(fontsize=8)
    cov_fig.savefig(output.with_name(output.stem + "_covariance.png"), dpi=150)
    plt.close(cov_fig)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
    print(output)


if __name__ == '__main__':
    main()

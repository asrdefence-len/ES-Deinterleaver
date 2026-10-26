"""Inspect candidate PRI peaks without using simulator truth for discovery."""
import argparse
import csv
import json
from pathlib import Path
from testbed.generator import generate
from testbed.pri_discovery import difference_histogram, candidate_peaks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="scenarios/baseline.json")
    parser.add_argument("--output", default="results/pri_diagnostic")
    parser.add_argument("--max-pdws", type=int, default=2500,
                        help="Limit diagnostic pairwise computation")
    args = parser.parse_args()
    if args.max_pdws < 2:
        parser.error("--max-pdws must be at least 2")
    scenario = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    observations, _truth = generate(scenario)
    observations = observations[:args.max_pdws]
    histogram = difference_histogram(observations)
    peaks = candidate_peaks(histogram)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "histogram.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["interval_us", "pair_count"])
        writer.writerows([(x * 1e6, count) for x, count in histogram])
    with (output / "candidate_pris.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_interval_us", "pair_count"])
        writer.writerows([(p.interval_s * 1e6, p.count) for p in peaks])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot([x * 1e6 for x, _ in histogram], [n for _, n in histogram])
    for peak in peaks[:6]:
        ax.axvline(peak.interval_s * 1e6, linestyle="--", alpha=0.5)
    ax.set(xlabel="TOA difference (microseconds)", ylabel="Number of PDW pairs",
           title="TOA difference histogram (candidate PRI peaks, not identified emitters)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "pri_histogram.png", dpi=160)
    plt.close(fig)
    print(f"Analysed {len(observations)} PDWs; strongest candidate intervals (microseconds):")
    for p in peaks:
        print(f"  {p.interval_s * 1e6:8.2f} : {p.count} pairs")
    print(f"Saved plot and CSVs to {output}")


if __name__ == "__main__":
    main()

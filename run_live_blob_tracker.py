"""Run the whole-blob tracker with a live AOA/time truth overlay.

The display is an operational waterfall: AOA is horizontal, simulated time
is vertical, and the newest observations appear at the top.  Estimated track
histories are solid; majority truth-emitter histories are dashed.  Truth is
used only for display and evaluation after each association decision.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import time

import h5py
import numpy as np

from compare_aoa_batch import batch_aoa_populations, rf_pw_groups
from compare_multiscale_rf import group_window
from extend_config_pdws import feature_names
from testbed.blob_observation import summarize_blob_pri
from testbed.pipeline_provenance import run_manifest
from testbed.blob_tracker import BlobEmitterTracker


def _track_number(track_id: str) -> int:
    try:
        return int(track_id.removeprefix("E"))
    except ValueError:
        return 0


class Waterfall:
    def __init__(self, trail_s: float, dropouts: list[dict]):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        self.plt = plt
        self.trail_s = trail_s
        self.dropouts = dropouts
        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        self.legend_handles = [
            Line2D([0], [0], color="tab:blue", lw=1.5, label="Estimated track"),
            Line2D([0], [0], color="black", lw=1., ls="--", alpha=.55,
                   label="Majority truth emitter"),
            Line2D([0], [0], color="tab:red", lw=7, alpha=.16,
                   label="Injected truth dropout"),
        ]
        plt.ion()
        plt.show(block=False)

    def draw(self, rows: list[dict], now_s: float, active_tracks: int,
             pdws_processed: int) -> None:
        ax = self.ax
        ax.clear()
        oldest = max(0., now_s - self.trail_s)
        visible = [row for row in rows if row["time_s"] >= oldest]
        tracks, truths = defaultdict(list), defaultdict(list)
        for row in visible:
            if row["track_id"]:
                tracks[row["track_id"]].append((row["time_s"], row["aoa_deg"]))
            if row["truth_emitter"] is not None:
                truths[row["truth_emitter"]].append((row["time_s"], row["aoa_deg"]))

        palette = self.plt.get_cmap("tab20")
        for track_id, points in tracks.items():
            points.sort()
            colour = palette(_track_number(track_id) % 20)
            ax.plot([p[1] for p in points], [p[0] for p in points], "o-",
                    color=colour, ms=2.5, lw=1., alpha=.82)
            if points and points[-1][0] >= now_s - min(.5, self.trail_s / 8):
                ax.annotate(track_id, (points[-1][1], points[-1][0]),
                            xytext=(3, 0), textcoords="offset points", fontsize=7,
                            color=colour, clip_on=True)
        for points in truths.values():
            points.sort()
            ax.plot([p[1] for p in points], [p[0] for p in points], "--",
                    color="black", lw=.65, alpha=.30)
        for dropout in self.dropouts:
            start, end = dropout["start_s"], dropout["end_s"]
            if end >= oldest and start <= now_s:
                ax.axhspan(max(oldest, start), min(now_s, end), color="tab:red", alpha=.07)

        unique_tracks = len({row["track_id"] for row in rows if row["track_id"]})
        ax.set(xlim=(-180, 180), ylim=(oldest, max(now_s, oldest + 1e-6)),
               xlabel="AOA (degrees)", ylabel="Simulated time (s) — newest at top")
        ax.set_xticks(np.arange(-180, 181, 45))
        ax.grid(True, alpha=.18)
        ax.set_title(f"S2B live emitter tracks  |  t={now_s:5.2f}s  |  "
                     f"active={active_tracks}  |  IDs created={unique_tracks}  |  "
                     f"PDWs={pdws_processed:,}")
        ax.legend(handles=self.legend_handles, loc="lower left", fontsize=8)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def pause(self, seconds: float) -> None:
        self.plt.pause(max(.001, seconds))

    def save(self, path: Path) -> None:
        self.fig.savefig(path, dpi=160, bbox_inches="tight")


def load_dropouts(handle: h5py.File) -> list[dict]:
    raw = handle.attrs.get("s2b_extension_manifest_json")
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode()
    try:
        return list(json.loads(str(raw)).get("dropouts", []))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = ["window", "observation_id", "time_s", "aoa_deg", "pulse_count",
              "track_id", "association_status", "truth_emitter", "truth_purity_pct"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--window-size", type=int, default=10000)
    parser.add_argument("--grouping", choices=("multiscale", "hdbscan"), default="hdbscan")
    parser.add_argument("--toa-scale", type=float, default=1e-6)
    parser.add_argument("--frequency-scale", type=float, default=1e6)
    parser.add_argument("--pw-scale", type=float, default=1e-6)
    parser.add_argument("--max-tracks", type=int, default=256)
    parser.add_argument("--max-gap", type=float, default=.3)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--trail", type=float, default=10., help="visible history in simulated seconds")
    parser.add_argument("--playback-rate", type=float, default=10.,
                        help="simulated seconds per wall-clock second; use 1 for real time, 0 for fastest")
    parser.add_argument("--no-live", action="store_true", help="process without opening the live plot")
    parser.add_argument("--output", default="results/config101_60s_live")
    args = parser.parse_args()
    if (args.window_size < 8 or args.max_tracks < 1 or args.beam_width < 1
            or args.max_gap <= 0 or args.trail <= 0 or args.playback_rate < 0):
        parser.error("Invalid positive processing/display setting")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tracker = BlobEmitterTracker(beam_width=args.beam_width,
        max_tracks=args.max_tracks, max_gap_s=args.max_gap,
        history_limit=max(256, args.max_tracks * 2))
    manifest = run_manifest(args.file, grouping=args.grouping,
        measurement_mode="histogram", settings=vars(args), tracker=tracker)
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows: list[dict] = []
    started = time.monotonic()
    first_time = None

    with h5py.File(args.file, "r") as handle:
        names = feature_names(handle)
        required = {"ToA", "Frequency", "PulseWidth", "AoA"}
        if not required.issubset(names):
            parser.error(f"Missing required features: {sorted(required - set(names))}")
        labels_available = "labels" in handle
        dropouts = load_dropouts(handle)
        display = None if args.no_live else Waterfall(args.trail, dropouts)
        total_windows = (len(handle["data"]) + args.window_size - 1) // args.window_size

        for window in range(total_windows):
            start = window * args.window_size
            stop = min(start + args.window_size, len(handle["data"]))
            if stop - start < 8:
                continue
            data = np.asarray(handle["data"][start:stop], float)
            populations, population_count = batch_aoa_populations(data[:, names.index("AoA")])
            blob_ids = (group_window(data, names, populations, population_count, 8, .20, 6., 8)
                        if args.grouping == "multiscale"
                        else rf_pw_groups(data, names, populations, population_count))
            selected = blob_ids >= 0
            observations = (summarize_blob_pri(data[selected], names, blob_ids[selected],
                f"W{window}", toa_scale=args.toa_scale,
                frequency_scale=args.frequency_scale, pw_scale=args.pw_scale)
                if np.any(selected) else ())
            truth = (np.asarray(handle["labels"][start:stop]).reshape(-1)
                     if labels_available else None)

            for observation in observations:
                options = tracker.process(observation)
                best = dict(tracker.hypotheses[0].recent_assignments).get(observation.observation_id)
                truth_emitter, truth_purity = None, None
                if truth is not None:
                    counts = Counter(truth[blob_ids == observation.blob_id].tolist())
                    if counts:
                        truth_emitter, majority = counts.most_common(1)[0]
                        truth_emitter = int(truth_emitter)
                        truth_purity = 100. * majority / observation.pulse_count
                rows.append({
                    "window": window,
                    "observation_id": observation.observation_id,
                    "time_s": .5 * (observation.start_s + observation.end_s),
                    "aoa_deg": observation.aoa_deg,
                    "pulse_count": observation.pulse_count,
                    "track_id": best or "",
                    "association_status": options["status"],
                    "truth_emitter": truth_emitter,
                    "truth_purity_pct": truth_purity,
                })

            now_s = float(np.max(data[:, names.index("ToA")]) * args.toa_scale)
            tracker.advance_to(now_s)
            tracker.complete_window(f"W{window}", now_s)
            if first_time is None:
                first_time = now_s
            active = len(tracker.hypotheses[0].tracks)
            print(f"window {window + 1}/{total_windows}: t={now_s:.3f}s, "
                  f"active={active}, observations={len(observations)}, PDWs={stop:,}", flush=True)
            if display is not None:
                display.draw(rows, now_s, active, stop)
            if args.playback_rate > 0:
                target_elapsed = (now_s - first_time) / args.playback_rate
                wait = max(0., target_elapsed - (time.monotonic() - started))
                if display is not None:
                    display.pause(wait)
                elif wait:
                    time.sleep(wait)

        if not rows:
            parser.error("No grouped observations were produced")
        write_rows(output / "live_track_history.csv", rows)
        if display is not None:
            last_time = max(row["time_s"] for row in rows)
            display.draw(rows, last_time, len(tracker.hypotheses[0].tracks), len(handle["data"]))
            display.save(output / "final_waterfall.png")
            display.pause(.25)

    summary = {
        "file": args.file,
        "pdws": int(stop),
        "windows": total_windows,
        "observations": len(rows),
        "unique_track_ids": len({row["track_id"] for row in rows if row["track_id"]}),
        "final_active_tracks": len(tracker.hypotheses[0].tracks),
        "dropouts": dropouts,
        "playback_rate": args.playback_rate,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "tracker_snapshot.json").write_text(json.dumps(tracker.snapshot(), indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

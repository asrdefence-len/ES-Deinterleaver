"""Run controlled observation-level tracker experiments (not raw-PDW trials).

Run: python compare_imm_emitter_tracker.py
"""
from testbed.imm_emitter_tracker import IMMEmitterTracker
from testbed.pulse_train_observation import PulseTrainObservation


def obs(name, start, pri_us, rf_ghz, aoa=45.0, stagger=None):
    intervals = stagger if stagger is not None else [pri_us] * 20
    toas = [start]
    for interval in intervals:
        toas.append(toas[-1] + interval * 1e-6)
    frequencies = tuple([rf_ghz * 1e9] * len(toas))
    return PulseTrainObservation(
        schema_version=1, train_id=name, start_toa_s=toas[0],
        end_toa_s=toas[-1], pri_s=sum(intervals) / len(intervals) * 1e-6,
        pri_sigma_s=2e-6, frequency_hz=rf_ghz * 1e9,
        frequency_sigma_hz=100_000, aoa_deg=aoa, aoa_sigma_deg=0.3,
        pulse_width_s=10e-6, pulse_width_sigma_s=0.1e-6,
        pulse_count=len(toas), pdw_ids=tuple(range(len(toas))),
        toa_sequence_s=tuple(toas), frequency_sequence_hz=frequencies,
        pri_intervals_s=tuple(v * 1e-6 for v in intervals),
    )


def main():
    cases = {
        "stable": [obs("a", 0, 1000, 9.4), obs("b", 0.1, 1000, 9.4)],
        "PRI change and return": [obs("a", 0, 1000, 9.4),
                                  obs("b", 0.1, 750, 9.4),
                                  obs("c", 0.2, 1000, 9.4)],
        "frequency switch": [obs("a", 0, 1000, 9.4),
                             obs("b", 0.1, 1000, 9.45)],
        "joint change": [obs("a", 0, 1000, 9.4),
                         obs("b", 0.1, 750, 9.45)],
        "stagger": [obs("a", 0, 1000, 9.4),
                    obs("b", 0.1, 1000, 9.4, stagger=[900, 1000, 1100]*8)],
        "overlapping new train": [obs("a", 0, 1000, 9.4),
                                  obs("b", 0.005, 750, 9.4)],
    }
    for label, observations in cases.items():
        tracker = IMMEmitterTracker()
        ids = [tracker.process(o) for o in observations]
        print(f"\\n{label}: assignments={ids}")
        for track in tracker.snapshot()["tracks"]:
            print(f"  {track['emitter_id']}: PRI={track['pri_models']}, "
                  f"RF={track['frequency_models']}")
    print("\\nCaution: probabilities are heuristic and not calibrated; "
          "these tests do not validate raw-PDW deinterleaving.")


if __name__ == "__main__":
    main()

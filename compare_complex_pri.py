"""Stress-test current PRI scoring on complex patterns and missed pulses.

This is diagnostic, not a calibrated classification benchmark. No ground-truth
parameters are passed to the scoring function.
Run: python compare_complex_pri.py
"""
from collections import Counter
from testbed.complex_pri_scenarios import KINDS, generate_observation
from testbed.signal_models import PRI_MODELS, pri_evidence


def main():
    trials = 100
    print("Complex PRI model stress test (100 trials per condition)")
    print("NOTE: current models are heuristic; 'change' and 'stagger' do not")
    print("yet explicitly marginalise over missed-pulse counts or stagger phase.")
    for detection in (1.0, 0.95, 0.70):
        print(f"\nDetection probability: {detection:.0%}")
        print(f"{'Pattern':<14} {'constant':>9} {'change':>9} {'jitter':>9} {'stagger':>9}")
        for kind in KINDS:
            counts = Counter()
            for seed in range(trials):
                observation, _truth = generate_observation(
                    kind, seed=seed, detection_probability=detection)
                scores = pri_evidence(observation)
                counts[PRI_MODELS[max(range(len(scores)), key=lambda i: scores[i])]] += 1
            print(f"{kind:<14} " + " ".join(f"{counts[m]:>9}" for m in PRI_MODELS))
    print("\nInterpretation: Rows are generated patterns; columns are current")
    print("model-score winners. Ramp, sinusoidal and bursts have no exact")
    print("corresponding model. This is NOT a calibrated confusion matrix.")


if __name__ == "__main__":
    main()

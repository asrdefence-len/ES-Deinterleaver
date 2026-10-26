"""Compare statistical constant/jitter fits with missed pulses.

Run: python compare_pri_likelihood.py
This deliberately tests only the first two hypotheses. Stagger/change
likelihoods will follow once this baseline is validated.
"""
from collections import Counter
from testbed.complex_pri_scenarios import generate_observation
from testbed.pri_likelihood import fit_constant_jitter


def main():
    print("Missing-pulse-aware constant vs jitter likelihood (BIC)")
    print("Pd is supplied as an experimental assumption, not inferred.")
    for pd in (1.0, .95, .7):
        print(f"\nPd={pd:.0%}")
        for kind in ("constant", "jitter"):
            wins = Counter()
            for seed in range(5):
                obs, _ = generate_observation(
                    kind, seed=seed, count=90, detection_probability=pd)
                result = fit_constant_jitter(obs, detection_probability=pd)
                wins[result["winner"]] += 1
            print(f"  {kind:<10} constant={wins['constant']} jitter={wins['jitter']}")
    print("\nOutlier handling: a 1% broad contamination component")
    print("downweights improbable timing intervals; it does not discard them.")
    print("Not yet integrated with IMM, stagger, or change-point models.")


if __name__ == "__main__":
    main()

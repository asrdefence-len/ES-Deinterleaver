"""Stage A: analytically inspect the information supplied by joint nondetections.

Run: python joint_detection_diagnostic.py

A constant Pd shared by emitters gives identical likelihood whenever two
hypotheses explain the same number of detections and expected pulse slots.
This is an identifiability result, not a failure to apply the likelihood.
"""
import math


def detection_log_likelihood(detected, missed, pd):
    if not 0 < pd <= 1:
        raise ValueError("Pd must be in (0,1]")
    if detected < 0 or missed < 0:
        raise ValueError("Counts must be nonnegative")
    if pd == 1 and missed:
        return -math.inf
    return detected * math.log(pd) + (missed * math.log1p(-pd) if missed else 0.0)


def score(detected_by_emitter, missed_by_emitter, pd_by_emitter):
    return sum(detection_log_likelihood(d, m, p)
               for d, m, p in zip(detected_by_emitter,
                                   missed_by_emitter, pd_by_emitter))


def main():
    cases = [
        ("Same Pd, one detection and one miss either way",
         (1, 0), (0, 1), (0, 1), (1, 0), (.7, .7)),
        ("Different Pd, same observed event",
         (1, 0), (0, 1), (0, 1), (1, 0), (.99, .60)),
        ("Same Pd, different numbers of missed slots",
         (1, 0), (0, 1), (1, 0), (0, 2), (.7, .7)),
    ]
    for title, det_a, miss_a, det_b, miss_b, pd in cases:
        la = score(det_a, miss_a, pd)
        lb = score(det_b, miss_b, pd)
        odds = math.exp(la-lb)
        print(title)
        print("  hypothesis A: detections", det_a, "misses", miss_a,
              "logL", round(la, 6))
        print("  hypothesis B: detections", det_b, "misses", miss_b,
              "logL", round(lb, 6))
        print("  likelihood ratio A/B:", round(odds, 6))
        print()


if __name__ == "__main__":
    main()

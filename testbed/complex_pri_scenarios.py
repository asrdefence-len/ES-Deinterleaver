"""Deterministic complex-PRI observation generator for model stress testing.

All schedules are synthetic. The returned intervals are *detected-pulse*
intervals; missed pulses produce sums of transmitted intervals. Truth is
returned separately, never attached to the observation.
"""
import random
from testbed.pulse_train_observation import PulseTrainObservation


def transmitted_pris(kind, count, seed=1):
    rng = random.Random(seed)
    if kind == "constant":
        return [1000.0] * count
    if kind == "jitter":
        return [max(200.0, rng.gauss(1000.0, 75.0)) for _ in range(count)]
    if kind == "stagger3":
        return ([900.0, 1000.0, 1100.0] * (count // 3 + 1))[:count]
    if kind == "stagger5":
        return ([780.0, 950.0, 1130.0, 870.0, 1270.0] *
                (count // 5 + 1))[:count]
    if kind == "change":
        return [1000.0 if i < count // 2 else 750.0 for i in range(count)]
    if kind == "ramp":
        return [900.0 + 200.0 * i / max(1, count - 1) for i in range(count)]
    if kind == "sinusoidal":
        import math
        return [1000.0 + 100.0 * math.sin(2 * math.pi * i / 17)
                for i in range(count)]
    if kind == "bursts":
        return [700.0 if (i // 12) % 2 else 1100.0 for i in range(count)]
    raise ValueError("Unknown PRI schedule: " + kind)


KINDS = ("constant", "jitter", "stagger3", "stagger5", "change",
         "ramp", "sinusoidal", "bursts")


def generate_observation(kind, seed=1, detection_probability=0.95,
                         toa_sigma_s=50e-9, count=180):
    if not 0 < detection_probability <= 1:
        raise ValueError("Invalid detection probability")
    rng = random.Random(seed)
    intervals = transmitted_pris(kind, count, seed)
    transmitted = [0.0]
    for pri_us in intervals:
        transmitted.append(transmitted[-1] + pri_us * 1e-6)
    measured = [max(0.0, t + rng.gauss(0, toa_sigma_s))
                for t in transmitted if rng.random() < detection_probability]
    measured.sort()
    if len(measured) < 3:
        raise ValueError("Too few detections; increase count or probability")
    observed_intervals = tuple(b-a for a, b in zip(measured, measured[1:]))
    mean_pri = sum(observed_intervals) / len(observed_intervals)
    obs = PulseTrainObservation(
        schema_version=1, train_id=f"{kind}-{seed}",
        start_toa_s=measured[0], end_toa_s=measured[-1],
        pri_s=mean_pri, pri_sigma_s=2**0.5 * toa_sigma_s,
        frequency_hz=9.4e9, frequency_sigma_hz=100e3,
        aoa_deg=45.0, aoa_sigma_deg=0.3,
        pulse_width_s=10e-6, pulse_width_sigma_s=0.1e-6,
        pulse_count=len(measured), pdw_ids=tuple(range(len(measured))),
        toa_sequence_s=tuple(measured),
        frequency_sequence_hz=(9.4e9,) * len(measured),
        pri_intervals_s=observed_intervals)
    return obs, {"kind": kind, "transmitted_intervals_us": tuple(intervals),
                 "transmitted_count": len(transmitted),
                 "detected_count": len(measured)}

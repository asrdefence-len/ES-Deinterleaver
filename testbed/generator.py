"""Deterministic PDW observations; simulator truth is kept out of observation objects."""
from dataclasses import dataclass
import random

@dataclass(frozen=True)
class PDW:
    pdw_id: int
    toa_s: float
    frequency_hz: float
    pulse_width_s: float
    aoa_deg: float
    amplitude_dbfs: float

def generate(scenario):
    rng = random.Random(int(scenario.get("seed", 0)))
    m = scenario["measurement"]
    duration = float(scenario["duration_s"])
    records = []
    for e in scenario["emitters"]:
        pri = float(e["pri_s"])
        if pri <= 0 or float(e["pw_s"]) <= 0 or float(e["pw_s"]) >= pri:
            raise ValueError("Require 0 < pulse width < PRI")
        t = float(e.get("start_s", 0))
        while t < duration:
            if rng.random() < float(m.get("detection_probability", 1)):
                toa = max(0.0, t + rng.gauss(0, float(m.get("toa_sigma_s", 0))))
                records.append((toa, e["id"],
                    float(e["frequency_hz"]) + rng.gauss(0, float(m.get("frequency_sigma_hz", 0))),
                    max(1e-9, float(e["pw_s"]) + rng.gauss(0, float(m.get("pw_sigma_s", 0)))),
                    float(e["aoa_deg"]) + rng.gauss(0, float(m.get("aoa_sigma_deg", 0))),
                    float(e["amplitude_dbfs"]) + rng.gauss(0, float(m.get("amplitude_sigma_db", 0)))))
            t += pri
    records.sort(key=lambda row: row[0])
    observations = []
    truth = {}
    for idx, (toa, emitter, freq, pw, aoa, amp) in enumerate(records, 1):
        observations.append(PDW(idx, toa, freq, pw, aoa, amp))
        truth[idx] = emitter
    return observations, truth

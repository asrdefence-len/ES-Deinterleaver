"""Algorithm interface and deliberately simple frequency-only baseline."""
from dataclasses import dataclass

@dataclass
class _Track:
    track_id: str
    mean_frequency_hz: float
    count: int = 1

class FrequencyGateDeinterleaver:
    def __init__(self, gate_hz=500_000):
        self.gate_hz = float(gate_hz)
        self.tracks = []

    def process(self, pdws):
        assignments = {}
        for p in pdws:
            candidates = [(abs(p.frequency_hz - t.mean_frequency_hz), t)
                          for t in self.tracks]
            candidates.sort(key=lambda item: item[0])
            if candidates and candidates[0][0] <= self.gate_hz:
                track = candidates[0][1]
                track.count += 1
                track.mean_frequency_hz += (p.frequency_hz - track.mean_frequency_hz) / track.count
            else:
                track = _Track(f"T{len(self.tracks)+1}", p.frequency_hz)
                self.tracks.append(track)
            assignments[p.pdw_id] = track.track_id
        return assignments

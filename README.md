# ES Deinterleaver Testbed

Standalone, truth-scored experimental testbed for emitter deinterleaving. Independent of S2B-ESM.

## First milestone
- Deterministic multi-emitter PDW scenario generator (fixed RF/PRI/PW/AOA with measurement noise and missed pulses).
- Ground truth held separately from the observations supplied to the algorithm.
- Replaceable frequency-gated baseline deinterleaver.
- Permutation-invariant scoring of emitter assignments, purity, completeness and fragmentation.
- CSV exports and a simple truth-versus-estimate plot.

## Quick start

```bash
python3 -m pip install -r requirements.txt
python3 run_experiment.py --scenario scenarios/baseline.json --output results/baseline
python3 -m unittest discover -s tests -v
```

The output folder contains `observations.csv`, `truth.csv`, `assignments.csv`, `metrics.json`, and `assignments.png`.

**Important:** The baseline is deliberately simple: it does not solve PRI ambiguity, frequency hopping, overlapping IQ pulses, or waveform-assisted association. AOA is simulated, not measured by hardware. Scoring uses truth only after deinterleaving.

## Pulse association inside the tracking pipeline

The [integrated fixed-PRI baseline](docs/pulse_association_pipeline.md)
connects pulse-train discovery to the existing emitter identity manager.

```bash
python3 -m pip install -r requirements-experiments.txt
python3 run_pulse_association.py turing_data/config_101.h5 \
  --window 0 --blobs 9 11 12 --output results/pulse_association_focus
```

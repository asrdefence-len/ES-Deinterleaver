# Pulse association inside the emitter-tracking pipeline

Pulse association is part of tracking: it establishes which pulse detections
support a timing hypothesis, while emitter identity tracking relates those
observations over time. This first integration puts the existing components
behind one `PulseAssociationTrackingPipeline.process_window` interface.

## What it connects

1. Existing AOA/RF grouping supplies candidate blobs without using truth.
2. `FixedPRIPulseAssociation` reuses `deinterleaver_v0.discover` inside each
   blob, producing temporary pulse-train hypotheses and unexplained PDWs.
3. Each candidate is converted into the existing `PulseTrainObservation`
   schema with SI units, ordered TOAs/RFs and globally unique PDW references.
4. The pipeline feeds those observations in time order to the existing
   `EmitterTrackManager`, whose identity state survives between windows.

No OT or frequency-density splitting is used. Existing grouping and V0
discovery are not changed. The identity manager's bearing gate now measures
circular angular distance, so +179 and -179 degrees are two degrees apart.

## Run using the usual workflow

From the DEINT repository on `main`, in your normal Python environment:

```bash
git pull
python3 -m pip install -r requirements-experiments.txt
python3 -u run_pulse_association.py turing_data/config_101.h5 \
  --window 0 --blobs 9 11 12 --output results/pulse_association_focus
```

This selects only those three blobs. Reported coverage uses the selected
PDWs as denominator; it is not a whole-window or whole-recording result.
IDs refer to the default AOA + multiscale RF grouping. To use AOA + RF/PW
HDBSCAN instead, add `--grouping hdbscan` and select the resulting IDs.

To process every blob in two consecutive windows and carry identity state:

```bash
python3 -u run_pulse_association.py turing_data/config_101.h5 \
  --window 0 --windows 2 --output results/pulse_association_first2
```

Blob IDs are local to a window; they do not identify the same emitter in
the next window. Therefore `--blobs` deliberately requires a one-window run.

The default input scales are seconds = raw TOA x 1e-6, Hz = raw frequency x
1e6, and seconds = raw pulse width x 1e-6; AOA is in degrees. These match
the Turing microsecond/MHz convention. For other recordings, explicitly
set `--toa-scale`, `--frequency-scale`, and `--pw-scale` as appropriate.

The existing V0 search is retained: PRI 20–200 microseconds, timing
tolerance 2 microseconds, histogram bin width 0.1 microseconds, minimum
20 pulses, maximum four candidate trains per blob. Command-line parameters
can change these settings; none are inferred from transmitter metadata.

## Outputs and interpretation

| File | Meaning |
|---|---|
| `pulse_assignments.csv` | Every selected PDW: temporary train ID, provisional emitter ID or explicit unexplained/pending status |
| `trains.csv` | PRI, phase epoch, residual, observed slot support, features and source PDW IDs |
| `windows.csv` | Selected/associated/unexplained counts, conditional train purity when labels exist, provisional identity count |
| `train_evaluation.csv` | Truth composition for each train, used only after all inference is complete |
| `identity_snapshot.json` | Existing manager's provisional emitter hypotheses and pending observations |
| `config.json` | Run settings and input-unit scales |

PDW IDs are recording-row index + 1, preserving a reference to the original
observation. Every selected PDW is either in one candidate train or retained
as unexplained. No whole-blob assignment is substituted for pulse evidence.
No association probability or uncertainty estimate is invented. Empty
slots are not reported as a known missed-pulse count: they can also reflect
hopping or scan gaps. Observed TOA gaps in the observation schema remain
inter-detection intervals, not a claim that every gap is one transmitted PRI.

## Deliberate limits of this first integration

- Association is batch within a completed window. This is not yet a
  pulse-by-pulse causal implementation or a validated real-time system.
- V0 is greedy fixed-PRI discovery. Harmonic/subharmonic ambiguity, jitter,
  stagger, phase resets and competing memberships are not resolved generally.
- The identity manager uses RF/AOA/PW and time continuity. It allows PRI
  changes when those other features remain compatible, but is a deterministic
  baseline, not an MHT/IMM. It does not feed predictions back into V0 yet.
- Overlapping train observations and widely separated RF modes are not
  automatically combined into one emitter. Multiple provisional IDs may
  correspond to one physical radar; one ID does not establish correct identity.
- Discovery is rerun in each window; persistent identity is maintained by
  the manager, not by a carried V0 pulse-lattice state. Windows must be
  disjoint and chronological, with no repeated PDW/window IDs.
- Train purity must be read with association coverage and fragmentation.
  It describes only assigned pulses and cannot establish persistent identity
  accuracy. Truth and transmitter configurations never enter inference.

This gives a runnable baseline for locating the next issue: train discovery,
ambiguous pulse membership, or association of several trains/modes with one
persistent emitter. It does not claim to have solved all three.

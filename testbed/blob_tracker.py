"""Stateful whole-blob association hypotheses; no PDW processing.

One track may receive overlapping blobs and several RF modes. A new RF
mode and a new emitter remain competing explanations. Continuous states use
Kalman covariance and joint NIS/chi-square gating. Track existence is a
Bernoulli probability updated by measurement likelihood ratios and declared
missed-detection opportunities. Model parameters still require calibration.
Individual PDWs never cross this interface.
"""
from dataclasses import dataclass, replace, asdict
import math

from scipy.stats import chi2

from testbed.blob_observation import BlobObservation
from testbed.pri_histogram import GapPRI


@dataclass(frozen=True)
class RFMode:
    low_hz: float
    median_hz: float
    high_hz: float
    phase_epoch_s: float | None = None
    timing_rms_residual_s: float | None = None


@dataclass(frozen=True)
class BlobTrack:
    track_id: str
    last_end_s: float
    aoa_deg: float
    aoa_spread_deg: float
    pw_median_s: float
    pw_spread_s: float
    rf_modes: tuple[RFMode, ...]
    recent_blob_ids: tuple[str, ...]
    aoa_variance_deg2: float
    pw_variance_s2: float
    existence_log_odds: float
    updates: int = 1
    pri_s: float | None = None
    pri_variance_s2: float | None = None
    phase_epoch_s: float | None = None
    timing_rms_residual_s: float | None = None
    last_evidence_window_id: str | None = None
    last_scored_window_id: str | None = None
    window_best_log_likelihood_ratio: float = 0.
    last_nis: float | None = None
    last_nis_dof: int | None = None
    last_gate_threshold: float | None = None
    last_log_likelihood_ratio: float | None = None

    pri_histogram: GapPRI | None = None

    @property
    def identity_log_odds(self):
        """Compatibility alias; existence is now the probabilistic score."""
        return self.existence_log_odds


@dataclass(frozen=True)
class BlobHypothesis:
    tracks: tuple[BlobTrack, ...] = ()
    recent_assignments: tuple[tuple[str, str | None], ...] = ()
    cost: float = 0.


class BlobEmitterTracker:
    """Bounded beam over associations of whole blob observations.

    Existing-mode match, new-mode match, new emitter and capacity-unresolved
    explanations are considered without modifying input membership. Limits
    bound beam size, active tracks, modes, matching branches and history.
    """

    def __init__(self, beam_width=16, max_tracks=64, max_modes=8,
                 max_match_branches=4, history_limit=256, track_history_limit=16,
                 max_gap_s=.3, rf_gate_hz=5e6, birth_cost=3.,
                 ambiguity_margin=5.,
                 gate_probability=.9973, detection_probability=.65,
                 survival_probability=.999,
                 expected_clutter_per_gate=.05,
                 known_mode_probability=.90,
                 detection_opportunity_interval_s=.10,
                 aoa_measurement_floor_deg=.25, pw_measurement_floor_s=.01e-6,
                 aoa_process_variance_deg2_s=.04,
                 pw_process_variance_s2_s=(.01e-6) ** 2,
                 pri_measurement_floor_s=1e-6,
                 pri_process_variance_s2_s=(.10e-6) ** 2,
                 timing_gate_s=5e-6,
                 birth_existence_confidence=.60,
                 archive_confidence=.30,
                 archive_after_s=.50, archive_limit=512,
                 consolidation_min_updates=3,
                 consolidation_parent_updates=10,
                 consolidation_gate_sigma=3., consistency_sink=None):
        limits = (beam_width, max_tracks, max_modes, max_match_branches,
                  history_limit, track_history_limit, archive_limit,
                  consolidation_min_updates, consolidation_parent_updates)
        scales = (max_gap_s, rf_gate_hz,
                  aoa_measurement_floor_deg, pw_measurement_floor_s,
                  aoa_process_variance_deg2_s, pw_process_variance_s2_s,
                  pri_measurement_floor_s, pri_process_variance_s2_s,
                  timing_gate_s, archive_after_s)
        scales = scales + (detection_opportunity_interval_s,)
        scales = scales + (consolidation_gate_sigma,)
        costs = (birth_cost, ambiguity_margin)
        if (any(not isinstance(v, int) or v < 1 for v in limits)
                or any(not math.isfinite(v) or v <= 0 for v in scales)
                or any(not math.isfinite(v) or v < 0 for v in costs)
                or not 0 < birth_existence_confidence < 1
                or not 0 < archive_confidence < 1
                or not 0 < gate_probability < 1
                or not 0 < detection_probability < 1
                or not 0 < survival_probability <= 1
                or not math.isfinite(expected_clutter_per_gate)
                or expected_clutter_per_gate <= 0
                or not 0 < known_mode_probability < 1):
            raise ValueError("Invalid blob tracker configuration")
        self.beam_width = beam_width
        self.max_tracks = max_tracks
        self.max_modes = max_modes
        self.max_match_branches = max_match_branches
        self.history_limit = history_limit
        self.track_history_limit = track_history_limit
        self.max_gap_s = max_gap_s
        self.rf_gate_hz = rf_gate_hz
        self.birth_cost = birth_cost
        self.ambiguity_margin = ambiguity_margin
        self.gate_probability = gate_probability
        self.detection_probability = detection_probability
        self.survival_probability = survival_probability
        self.expected_clutter_per_gate = expected_clutter_per_gate
        self.known_mode_probability = known_mode_probability
        self.detection_opportunity_interval_s = detection_opportunity_interval_s
        self.aoa_measurement_floor_deg = aoa_measurement_floor_deg
        self.pw_measurement_floor_s = pw_measurement_floor_s
        self.aoa_q = aoa_process_variance_deg2_s
        self.pw_q = pw_process_variance_s2_s
        self.pri_measurement_floor_s = pri_measurement_floor_s
        self.pri_q = pri_process_variance_s2_s
        self.timing_gate_s = timing_gate_s
        self.birth_existence_confidence = birth_existence_confidence
        self.archive_confidence = archive_confidence
        self.archive_after_s = archive_after_s
        self.archive_limit = archive_limit
        self.consolidation_min_updates = consolidation_min_updates
        self.consolidation_parent_updates = consolidation_parent_updates
        self.consolidation_gate_sigma = consolidation_gate_sigma
        self.consistency_sink = consistency_sink
        self.hypotheses = [BlobHypothesis()]
        self.archived_tracks = {}
        self.consolidated_aliases = {}
        self.consolidation_events = []
        self._last_start_s = float("-inf")
        self._recent_ids = ()
        self._sequence = 0
        self._current_window_id = None
        self._last_detection_time_s = None

    @staticmethod
    def _pw_spread(obs):
        return (obs.pw_p95_s - obs.pw_p05_s) / 3.29

    @staticmethod
    def _logit(probability):
        return math.log(probability / (1. - probability))

    @staticmethod
    def _logistic(value):
        if value >= 0:
            return 1. / (1. + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1. + exp_value)

    @staticmethod
    def _kalman_scalar(mean, variance, measurement, measurement_variance):
        gain = variance / (variance + measurement_variance)
        return (mean + gain * (measurement - mean),
                (1. - gain) * variance, gain)

    def _predicted_existence_log_odds(self, log_odds):
        """Bernoulli survival prediction for one detection opportunity."""
        predicted = self.survival_probability * self._logistic(log_odds)
        predicted = min(1. - 1e-12, max(1e-12, predicted))
        return self._logit(predicted)

    def _measurement_variances(self, obs):
        # A blob mean becomes more precise with support, but the floors retain
        # sensor/systematic uncertainty and prevent huge blobs becoming exact.
        n = max(1, obs.pulse_count)
        aoa = max(self.aoa_measurement_floor_deg ** 2,
                  obs.aoa_spread_deg ** 2 / n)
        pw = max(self.pw_measurement_floor_s ** 2,
                 self._pw_spread(obs) ** 2 / n)
        return aoa, pw

    def _predicted_covariance(self, track, obs):
        dt = max(0., obs.start_s - track.last_end_s)
        return (track.aoa_variance_deg2 + self.aoa_q * dt,
                track.pw_variance_s2 + self.pw_q * dt, dt)

    @staticmethod
    def _timing_match(obs, track, predicted_pri_variance, pri_floor, timing_gate,
                      reference_phase_epoch=None, reference_timing_residual=None):
        if obs.pri_s is None or track.pri_s is None:
            return 0., None
        observed_variance = max(obs.pri_sigma_s ** 2, pri_floor ** 2)
        if (obs.pri_histogram is None) != (track.pri_histogram is None):
            return 0., None  # Different PRI statistics must not be compared.
        if obs.pri_histogram is not None:
            innovation = obs.pri_s - track.pri_s
            variance = predicted_pri_variance + observed_variance
            z = abs(innovation) / math.sqrt(variance)
            return z * z, {"pri_z": z, "phase_z": 0.,
                "innovation_s": innovation, "innovation_variance_s2": variance,
                "phase_residual_s": None, "observed_harmonic": 1,
                "track_harmonic": 1, "pri_model": "local_gap_humps_v1"}
        candidates = []
        for observed_harmonic in range(1, 5):
            for track_harmonic in range(1, 5):
                observed_base = obs.pri_s / observed_harmonic
                track_base = track.pri_s / track_harmonic
                variance = (observed_variance / observed_harmonic ** 2
                            + predicted_pri_variance / track_harmonic ** 2)
                z = abs(observed_base - track_base) / math.sqrt(variance)
                candidates.append((z, observed_harmonic, track_harmonic,
                                   observed_base, track_base, variance))
        (pri_z, observed_harmonic, track_harmonic, observed_base,
         track_base, variance) = min(candidates)
        base_pri = .5 * (observed_base + track_base)
        phase_residual = None
        phase_z = 0.
        if obs.phase_epoch_s is not None and reference_phase_epoch is not None:
            phase_difference = obs.phase_epoch_s - reference_phase_epoch
            phase_residual = abs((phase_difference + base_pri / 2.) % base_pri
                                 - base_pri / 2.)
            phase_scale = max(timing_gate, obs.timing_rms_residual_s or 0.,
                              reference_timing_residual or 0.)
            phase_z = phase_residual / phase_scale
        return pri_z * pri_z, {
            "pri_z": pri_z, "phase_z": phase_z,
            "innovation_s": observed_base - track_base,
            "innovation_variance_s2": variance,
            "phase_residual_s": phase_residual,
            "observed_harmonic": observed_harmonic,
            "track_harmonic": track_harmonic}

    def _compatibility(self, obs, track, *, include_rejected=False):
        if obs.start_s - track.last_end_s > self.max_gap_s:
            return None
        predicted_aoa, predicted_pw, _ = self._predicted_covariance(track, obs)
        measured_aoa, measured_pw = self._measurement_variances(obs)
        aoa_innovation = (obs.aoa_deg - track.aoa_deg + 180) % 360 - 180
        pw_innovation = obs.pw_median_s - track.pw_median_s
        aoa_s = predicted_aoa + measured_aoa
        pw_s = predicted_pw + measured_pw
        gaps = [max(0., obs.frequency_p05_hz - mode.high_hz,
                        mode.low_hz - obs.frequency_p95_hz) for mode in track.rf_modes]
        mode_index = min(range(len(gaps)), key=gaps.__getitem__)
        known_mode = gaps[mode_index] <= self.rf_gate_hz
        reference_mode = track.rf_modes[mode_index] if known_mode else None
        predicted_pri = (None if track.pri_variance_s2 is None else
            track.pri_variance_s2 + self.pri_q * max(0., obs.start_s - track.last_end_s))
        _, timing = self._timing_match(obs, track,
            predicted_pri or self.pri_measurement_floor_s ** 2,
            self.pri_measurement_floor_s, self.timing_gate_s,
            None if reference_mode is None else reference_mode.phase_epoch_s,
            None if reference_mode is None else reference_mode.timing_rms_residual_s)
        components = ((aoa_innovation, aoa_s), (pw_innovation, pw_s))
        if timing is not None:
            components += ((timing["innovation_s"],
                            timing["innovation_variance_s2"]),)
        nis = sum(innovation * innovation / variance
                  for innovation, variance in components)
        dof = len(components)
        threshold = float(chi2.ppf(self.gate_probability, dof))
        if nis > threshold and not include_rejected:
            return None

        # In whitened innovation coordinates, target-origin measurements have
        # a standard normal density. Clutter is modeled as uniform within the
        # chi-square gate with an expected count per gate. Their ratio is
        # dimensionless and can therefore update Bernoulli existence odds.
        gate_volume = (math.pi ** (dof / 2.)
            / math.gamma(dof / 2. + 1.) * threshold ** (dof / 2.))
        log_target_density = -.5 * (nis + dof * math.log(2. * math.pi))
        log_clutter_density = math.log(self.expected_clutter_per_gate / gate_volume)
        mode_probability = (self.known_mode_probability if known_mode
                            else 1. - self.known_mode_probability)
        log_likelihood_ratio = (math.log(self.detection_probability)
            + log_target_density - log_clutter_density
            + math.log(mode_probability))
        existence = max(1e-12, self._logistic(track.existence_log_odds))
        cost = -math.log(existence) - log_likelihood_ratio
        statistics = {"nis": nis, "dof": dof, "gate_threshold": threshold,
            "log_likelihood_ratio": log_likelihood_ratio,
            "aoa_innovation_deg": aoa_innovation,
            "aoa_innovation_variance_deg2": aoa_s,
            "pw_innovation_s": pw_innovation,
            "pw_innovation_variance_s2": pw_s,
            "pri_innovation_s": None if timing is None else timing["innovation_s"],
            "pri_innovation_variance_s2": (None if timing is None else
                timing["innovation_variance_s2"]),
            "known_rf_mode": known_mode,
            "gate_passed": nis <= threshold,
            "timing": timing,
            "axes": {
                "aoa_deg": self._axis_statistics(aoa_innovation, predicted_aoa, measured_aoa),
                "pw_s": self._axis_statistics(pw_innovation, predicted_pw, measured_pw),
                **({} if timing is None else {"pri_base_s": self._axis_statistics(
                    timing["innovation_s"],
                    predicted_pri / timing["track_harmonic"] ** 2,
                    max(obs.pri_sigma_s ** 2, self.pri_measurement_floor_s ** 2)
                        / timing["observed_harmonic"] ** 2)})}}
        return cost, mode_index if known_mode else None, statistics

    @staticmethod
    def _axis_statistics(innovation, predicted, measured):
        return {"innovation": innovation, "P_predicted": predicted,
                "R": measured, "S": predicted + measured,
                "K": predicted / (predicted + measured)}

    def _emit_consistency(self, track, obs, statistics, outcome, rank,
                          updated=None, window_id=None, event_time_s=None):
        if self.consistency_sink is None:
            return
        self.consistency_sink({
            "schema_version": 1, "prior_hypothesis_rank": rank,
            "track_id": track.track_id,
            "window_id": obs.window_id if obs else window_id,
            "observation_id": obs.observation_id if obs else None,
            "time_s": obs.start_s if obs else event_time_s,
            "outcome": outcome,
            "existence_before": self._logistic(track.existence_log_odds),
            "existence_after": self._logistic(
                (updated or track).existence_log_odds),
            "nis": statistics["nis"] if statistics else None,
            "dof": statistics["dof"] if statistics else None,
            "gate_threshold": statistics["gate_threshold"] if statistics else None,
            "gate_probability": self.gate_probability,
            "axes": statistics["axes"] if statistics else {},
            "timing": statistics["timing"] if statistics else None})

    def _new_track(self, obs, track_id):
        aoa_variance, pw_variance = self._measurement_variances(obs)
        return BlobTrack(track_id, obs.end_s, obs.aoa_deg, obs.aoa_spread_deg,
            obs.pw_median_s, self._pw_spread(obs),
            (RFMode(obs.frequency_p05_hz, obs.frequency_median_hz, obs.frequency_p95_hz,
                    obs.phase_epoch_s, obs.timing_rms_residual_s),),
            (obs.observation_id,), aoa_variance, pw_variance,
            self._logit(self.birth_existence_confidence), 1,
            obs.pri_s,
            (None if obs.pri_s is None else max(obs.pri_sigma_s ** 2,
                                                self.pri_measurement_floor_s ** 2)),
            obs.phase_epoch_s, obs.timing_rms_residual_s, obs.window_id,
            obs.window_id, pri_histogram=obs.pri_histogram)

    def _updated_track(self, track, obs, mode_index, statistics):
        mode = RFMode(obs.frequency_p05_hz, obs.frequency_median_hz,
                      obs.frequency_p95_hz, obs.phase_epoch_s,
                      obs.timing_rms_residual_s)
        modes = list(track.rf_modes)
        if mode_index is None:
            modes.append(mode)
            modes = modes[-self.max_modes:]
        else:
            modes[mode_index] = mode
        predicted_aoa, predicted_pw, dt = self._predicted_covariance(track, obs)
        measured_aoa, measured_pw = self._measurement_variances(obs)
        aoa_innovation = (obs.aoa_deg - track.aoa_deg + 180) % 360 - 180
        aoa, aoa_variance, _ = self._kalman_scalar(track.aoa_deg,
            predicted_aoa, track.aoa_deg + aoa_innovation, measured_aoa)
        aoa = (aoa + 180.) % 360. - 180.
        pw, pw_variance, _ = self._kalman_scalar(track.pw_median_s,
            predicted_pw, obs.pw_median_s, measured_pw)
        log_lr = statistics["log_likelihood_ratio"]
        if obs.window_id != track.last_scored_window_id:
            existence = (self._predicted_existence_log_odds(
                track.existence_log_odds) + log_lr)
            window_best = log_lr
        else:
            improvement = max(0., log_lr - track.window_best_log_likelihood_ratio)
            existence = track.existence_log_odds + improvement
            window_best = max(track.window_best_log_likelihood_ratio, log_lr)
        existence = max(-20., min(20., existence))
        pri, pri_variance = track.pri_s, track.pri_variance_s2
        if obs.pri_s is not None:
            observed_variance = max(obs.pri_sigma_s ** 2,
                                    self.pri_measurement_floor_s ** 2)
            if (pri is None or
                    (obs.pri_histogram is None) != (track.pri_histogram is None)):
                pri, pri_variance = obs.pri_s, observed_variance
            else:
                predicted_pri = pri_variance + self.pri_q * dt
                timing = statistics["timing"]
                th, oh = timing["track_harmonic"], timing["observed_harmonic"]
                base, base_variance, _ = self._kalman_scalar(
                    pri / th, predicted_pri / th ** 2,
                    obs.pri_s / oh, observed_variance / oh ** 2)
                pri, pri_variance = base * th, base_variance * th ** 2
        return replace(track, last_end_s=max(track.last_end_s, obs.end_s),
            aoa_deg=aoa, aoa_spread_deg=obs.aoa_spread_deg,
            pw_median_s=pw, pw_spread_s=self._pw_spread(obs),
            rf_modes=tuple(modes),
            recent_blob_ids=(track.recent_blob_ids + (obs.observation_id,))[-self.track_history_limit:],
            aoa_variance_deg2=aoa_variance, pw_variance_s2=pw_variance,
            existence_log_odds=existence, updates=track.updates + 1,
            pri_s=pri, pri_variance_s2=pri_variance,
            phase_epoch_s=(obs.phase_epoch_s if obs.phase_epoch_s is not None
                           else track.phase_epoch_s),
            timing_rms_residual_s=(obs.timing_rms_residual_s
                if obs.timing_rms_residual_s is not None
                else track.timing_rms_residual_s),
            last_evidence_window_id=obs.window_id,
            last_scored_window_id=obs.window_id,
            window_best_log_likelihood_ratio=window_best,
            last_nis=statistics["nis"], last_nis_dof=statistics["dof"],
            last_gate_threshold=statistics["gate_threshold"],
            last_log_likelihood_ratio=log_lr,
            pri_histogram=(obs.pri_histogram if obs.pri_s is not None
                           else track.pri_histogram))

    def _track_consolidation_compatible(self, parent, child):
        if (parent.updates < self.consolidation_parent_updates
                or child.updates < self.consolidation_min_updates
                or child.updates >= parent.updates):
            return False
        aoa_difference = abs((child.aoa_deg - parent.aoa_deg + 180.) % 360. - 180.)
        aoa_scale = math.sqrt(parent.aoa_variance_deg2
            + child.aoa_variance_deg2 + 2. * self.aoa_measurement_floor_deg ** 2)
        pw_scale = math.sqrt(parent.pw_variance_s2 + child.pw_variance_s2
            + 2. * self.pw_measurement_floor_s ** 2)
        if (aoa_difference > self.consolidation_gate_sigma * aoa_scale
                or abs(child.pw_median_s - parent.pw_median_s)
                   > self.consolidation_gate_sigma * pw_scale):
            return False
        rf_overlap = any(max(0., left.low_hz - right.high_hz,
                             right.low_hz - left.high_hz) <= self.rf_gate_hz
                         for left in parent.rf_modes for right in child.rf_modes)
        if not rf_overlap:
            return False
        if parent.pri_s is not None and child.pri_s is not None:
            if (parent.pri_histogram is None) != (child.pri_histogram is None):
                return False
            harmonics = (1,) if parent.pri_histogram is not None else range(1, 5)
            candidates = []
            for parent_harmonic in harmonics:
                for child_harmonic in harmonics:
                    left = parent.pri_s / parent_harmonic
                    right = child.pri_s / child_harmonic
                    variance = ((parent.pri_variance_s2 or 0.) / parent_harmonic ** 2
                        + (child.pri_variance_s2 or 0.) / child_harmonic ** 2
                        + 2. * self.pri_measurement_floor_s ** 2)
                    candidates.append(abs(left - right) / math.sqrt(variance))
            if min(candidates) > self.consolidation_gate_sigma:
                return False
        return True

    def _merge_tracks(self, parent, child):
        def fuse(mean_a, variance_a, mean_b, variance_b):
            weight_a, weight_b = 1. / variance_a, 1. / variance_b
            return ((weight_a * mean_a + weight_b * mean_b) / (weight_a + weight_b),
                    1. / (weight_a + weight_b))

        angle_measurement = parent.aoa_deg + (
            (child.aoa_deg - parent.aoa_deg + 180.) % 360. - 180.)
        aoa, aoa_variance = fuse(parent.aoa_deg, parent.aoa_variance_deg2,
                                  angle_measurement, child.aoa_variance_deg2)
        aoa = (aoa + 180.) % 360. - 180.
        pw, pw_variance = fuse(parent.pw_median_s, parent.pw_variance_s2,
                               child.pw_median_s, child.pw_variance_s2)
        modes = list(parent.rf_modes)
        for child_mode in child.rf_modes:
            if not any(max(0., mode.low_hz - child_mode.high_hz,
                           child_mode.low_hz - mode.high_hz) <= self.rf_gate_hz
                       for mode in modes):
                modes.append(child_mode)
        modes = modes[-self.max_modes:]
        pri, pri_variance = parent.pri_s, parent.pri_variance_s2
        if pri is None:
            pri, pri_variance = child.pri_s, child.pri_variance_s2
        elif child.pri_s is not None:
            ratio = (1 if parent.pri_histogram is not None else
                     max(1, min(4, round(child.pri_s / pri))))
            measurement = child.pri_s / ratio
            measurement_variance = (child.pri_variance_s2
                or self.pri_measurement_floor_s ** 2) / ratio ** 2
            pri, pri_variance, _ = self._kalman_scalar(
                pri, pri_variance or self.pri_measurement_floor_s ** 2,
                measurement, measurement_variance)
        latest = child if child.last_end_s >= parent.last_end_s else parent
        return replace(parent,
            pri_histogram=(latest.pri_histogram if latest.pri_s is not None
                           else parent.pri_histogram or child.pri_histogram),
            last_end_s=max(parent.last_end_s, child.last_end_s),
            aoa_deg=aoa, aoa_variance_deg2=aoa_variance,
            aoa_spread_deg=max(parent.aoa_spread_deg, child.aoa_spread_deg),
            pw_median_s=pw, pw_variance_s2=pw_variance,
            pw_spread_s=max(parent.pw_spread_s, child.pw_spread_s),
            rf_modes=tuple(modes),
            recent_blob_ids=(parent.recent_blob_ids + child.recent_blob_ids)[
                -self.track_history_limit:],
            existence_log_odds=max(parent.existence_log_odds,
                                    child.existence_log_odds),
            updates=parent.updates + child.updates,
            pri_s=pri, pri_variance_s2=pri_variance,
            phase_epoch_s=latest.phase_epoch_s,
            timing_rms_residual_s=latest.timing_rms_residual_s,
            last_evidence_window_id=latest.last_evidence_window_id,
            last_scored_window_id=latest.last_scored_window_id,
            window_best_log_likelihood_ratio=max(
                parent.window_best_log_likelihood_ratio,
                child.window_best_log_likelihood_ratio),
            last_nis=latest.last_nis, last_nis_dof=latest.last_nis_dof,
            last_gate_threshold=latest.last_gate_threshold,
            last_log_likelihood_ratio=latest.last_log_likelihood_ratio)

    def _consolidate_hypothesis(self, hypothesis):
        tracks = list(hypothesis.tracks)
        aliases = []
        changed = True
        while changed:
            changed = False
            for child_index, child in sorted(enumerate(tracks),
                    key=lambda item: (item[1].updates, item[1].track_id)):
                candidates = [(parent.updates, parent.track_id, parent_index, parent)
                    for parent_index, parent in enumerate(tracks)
                    if parent_index != child_index
                    and self._track_consolidation_compatible(parent, child)]
                if not candidates:
                    continue
                _, _, parent_index, parent = max(candidates)
                merged = self._merge_tracks(parent, child)
                tracks[parent_index] = merged
                del tracks[child_index]
                assignments = tuple((observation_id,
                    parent.track_id if track_id == child.track_id else track_id)
                    for observation_id, track_id in hypothesis.recent_assignments)
                hypothesis = replace(hypothesis, tracks=tuple(tracks),
                                     recent_assignments=assignments)
                aliases.append((child.track_id, parent.track_id, child.updates))
                changed = True
                break
        return replace(hypothesis, tracks=tuple(tracks)), aliases

    def _apply_missed_detection(self, window_id, time_s=None):
        """Apply one Bernoulli missed-detection update per track/window."""
        updated_hypotheses = []
        for rank, hypothesis in enumerate(self.hypotheses):
            tracks = []
            for track in hypothesis.tracks:
                if (track.last_scored_window_id == window_id
                        or track.last_evidence_window_id == window_id):
                    tracks.append(track)
                    continue
                predicted_log_odds = self._predicted_existence_log_odds(
                    track.existence_log_odds)
                missed = replace(track,
                    existence_log_odds=max(-20., predicted_log_odds
                        + math.log(1. - self.detection_probability)),
                    last_scored_window_id=window_id,
                    window_best_log_likelihood_ratio=0.)
                tracks.append(missed)
                self._emit_consistency(track, None, None, "missed", rank,
                                       missed, window_id, time_s)
            updated_hypotheses.append(replace(hypothesis, tracks=tuple(tracks)))
        self.hypotheses = updated_hypotheses

    def complete_window(self, window_id, time_s):
        """Declare one detection opportunity complete and score any misses."""
        if not isinstance(window_id, str) or not window_id:
            raise ValueError("Window ID must be a non-empty string")
        if self._current_window_id is not None and self._current_window_id != window_id:
            self._apply_missed_detection(self._current_window_id, time_s)
        self._apply_missed_detection(window_id, time_s)
        self._current_window_id = None
        self._last_detection_time_s = (time_s if self._last_detection_time_s is None
            else max(self._last_detection_time_s, time_s))
        self.coast_to(time_s)

    def advance_to(self, time_s):
        """Insert expected opportunities; return pre-archive states for monitoring."""
        if not math.isfinite(time_s):
            raise ValueError("Advance time must be finite")
        if self._last_detection_time_s is None:
            self._last_detection_time_s = time_s
            return []
        elapsed = time_s - self._last_detection_time_s
        if elapsed < 0:
            raise ValueError("Detection-opportunity time cannot move backwards")
        count = int(elapsed / self.detection_opportunity_interval_s)
        samples = []
        for _ in range(count):
            opportunity_time = (self._last_detection_time_s
                                + self.detection_opportunity_interval_s)
            self._apply_missed_detection(f"TIME:{opportunity_time:.9f}", opportunity_time)
            self._last_detection_time_s = opportunity_time
            samples.append((opportunity_time,
                tuple(self.hypotheses[0].tracks) if self.hypotheses else ()))
            self.coast_to(opportunity_time)
        return samples

    def process(self, obs):
        if not isinstance(obs, BlobObservation):
            raise TypeError("Tracker input must be a BlobObservation, not PDWs")
        if obs.observation_id in self._recent_ids:
            raise ValueError("Duplicate recent blob observation")
        if obs.start_s < self._last_start_s:
            raise ValueError("Blob observations must arrive in start-time order")
        if self._current_window_id is not None and self._current_window_id != obs.window_id:
            self._apply_missed_detection(self._current_window_id, obs.start_s)
        self._current_window_id = obs.window_id
        self.coast_to(obs.start_s)
        self._sequence += 1
        new_track_id = f"E{self._sequence}"
        branches = []
        for rank, hypothesis in enumerate(self.hypotheses):
            # Expired identities disappear from active inference, not from the
            # caller's persisted output. IDs are never reused.
            tracks = tuple(t for t in hypothesis.tracks
                           if obs.start_s - t.last_end_s <= self.max_gap_s)
            def record(updated_tracks, track_id, extra_cost):
                assignments = (hypothesis.recent_assignments +
                               ((obs.observation_id, track_id),))[-self.history_limit:]
                branches.append(BlobHypothesis(tuple(updated_tracks), assignments,
                                               hypothesis.cost + extra_cost))
            matches = []
            diagnostics = {}
            for i, track in enumerate(tracks):
                compatible = self._compatibility(obs, track,
                    include_rejected=self.consistency_sink is not None)
                if compatible is not None:
                    diagnostics[track.track_id] = compatible[2]
                    if not compatible[2]["gate_passed"]:
                        compatible = None
                if compatible is not None:
                    matches.append((compatible[0], track.track_id, i,
                                    compatible[1], compatible[2]))
            updated_by_id = {}
            for cost, _, index, mode_index, statistics in sorted(matches,
                    key=lambda item: (item[0], item[1]))[:self.max_match_branches]:
                updated = list(tracks)
                updated[index] = self._updated_track(
                    tracks[index], obs, mode_index, statistics)
                updated_by_id[tracks[index].track_id] = updated[index]
                record(updated, tracks[index].track_id, cost)
            for track in tracks:
                stats = diagnostics.get(track.track_id)
                if stats is not None:
                    updated_track = updated_by_id.get(track.track_id)
                    outcome = ("accepted" if updated_track else
                               "not_selected" if stats["gate_passed"] else "rejected")
                    self._emit_consistency(track, obs, stats, outcome, rank, updated_track)
            if len(tracks) < self.max_tracks:
                record(tracks + (self._new_track(obs, new_track_id),), new_track_id, self.birth_cost)
            else:
                # Do not force a blob into an incompatible track at capacity.
                record(tracks, None, self.birth_cost)
        # Deterministic deduplication; retain alternative blob associations.
        unique = {}
        for branch in branches:
            key = (branch.tracks, branch.recent_assignments)
            if key not in unique or branch.cost < unique[key].cost:
                unique[key] = branch
        ranked = sorted(unique.values(), key=lambda h: (h.cost,
            repr(h.recent_assignments)))[:self.beam_width]
        offset = ranked[0].cost
        self.hypotheses = [replace(h, cost=h.cost - offset) for h in ranked]
        consolidated = []
        for index, hypothesis in enumerate(self.hypotheses):
            hypothesis, aliases = self._consolidate_hypothesis(hypothesis)
            consolidated.append(hypothesis)
            if index == 0:
                for child_id, parent_id, updates in aliases:
                    self.consolidated_aliases[child_id] = parent_id
                    self.consolidation_events.append({"time_s": obs.end_s,
                        "child_track_id": child_id, "parent_track_id": parent_id,
                        "child_updates": updates})
        self.hypotheses = consolidated
        self._last_start_s = obs.start_s
        self._recent_ids = (self._recent_ids + (obs.observation_id,))[-self.history_limit:]
        return self.association_options(obs.observation_id)

    def coast_to(self, time_s):
        """Advance lifecycle time and archive stale/lost tracks.

        Archived identities retain bounded audit metadata but no longer enter
        association gates or consume active-track capacity.
        """
        if not math.isfinite(time_s):
            raise ValueError("Coast time must be finite")
        if time_s < self._last_start_s:
            raise ValueError("Tracker lifecycle time cannot move backwards")
        dropped = {}
        updated_hypotheses = []
        for hypothesis in self.hypotheses:
            active = []
            for track in hypothesis.tracks:
                age = max(0., time_s - track.last_end_s)
                confidence = self._logistic(track.existence_log_odds)
                expired = age > self.max_gap_s
                lost_and_stale = (age >= self.archive_after_s
                                  and confidence < self.archive_confidence)
                if expired or lost_and_stale:
                    dropped[track.track_id] = (track, confidence,
                        "MAX_GAP" if expired else "LOST_CONFIDENCE")
                else:
                    active.append(track)
            updated_hypotheses.append(replace(hypothesis, tracks=tuple(active)))
        self.hypotheses = updated_hypotheses
        surviving = {track.track_id for hypothesis in self.hypotheses
                     for track in hypothesis.tracks}
        for track_id, (track, confidence, reason) in dropped.items():
            if track_id in surviving:
                continue
            identity_confidence = self._logistic(track.existence_log_odds)
            self.archived_tracks[track_id] = {
                "track_id": track_id, "archived_at_s": time_s,
                "reason": reason, "last_end_s": track.last_end_s,
                "final_track_confidence": confidence,
                "final_identity_confidence": identity_confidence,
                # Compatibility alias for readers of earlier result files.
                "final_existence_confidence": confidence,
                "updates": track.updates,
                "recent_blob_ids": list(track.recent_blob_ids),
                "state": {"aoa_deg": track.aoa_deg,
                          "pw_median_s": track.pw_median_s,
                          "pri_s": track.pri_s,
                          "last_pri_histogram": (None if track.pri_histogram is None
                              else asdict(track.pri_histogram)),
                          "rf_modes_hz": [mode.median_hz for mode in track.rf_modes]},
            }
        if len(self.archived_tracks) > self.archive_limit:
            ordered = sorted(self.archived_tracks.values(),
                             key=lambda row: (row["archived_at_s"], row["track_id"]))
            self.archived_tracks = {row["track_id"]: row
                                    for row in ordered[-self.archive_limit:]}

    def plausible_hypotheses(self):
        return [h for h in self.hypotheses if h.cost <= self.ambiguity_margin]

    def association_options(self, observation_id):
        options = {dict(h.recent_assignments).get(observation_id)
                   for h in self.plausible_hypotheses()
                   if observation_id in dict(h.recent_assignments)}
        if not options:
            return {"status": "OUTSIDE_RETAINED_HISTORY", "candidate_track_ids": []}
        real = sorted(o for o in options if o is not None)
        status = ("UNRESOLVED_CAPACITY" if options == {None} else
                  "PROVISIONAL" if len(options) == 1 and None not in options else "AMBIGUOUS")
        return {"status": status, "candidate_track_ids": real}

    def association_metrics(self, observation_id):
        """Return relative per-track evidence for one blob observation.

        These normalized beam weights are useful monitoring scores, not
        calibrated probabilities.
        """
        weighted = {}
        for hypothesis in self.plausible_hypotheses():
            assignments = dict(hypothesis.recent_assignments)
            if observation_id not in assignments:
                continue
            track_id = assignments[observation_id]
            key = "UNRESOLVED" if track_id is None else track_id
            weighted[key] = weighted.get(key, 0.) + math.exp(-hypothesis.cost)
        total = sum(weighted.values())
        confidences = ({key: value / total for key, value in sorted(weighted.items())}
                       if total else {})
        return {"relative_association_confidence": confidences,
                "best_confidence": max(confidences.values(), default=0.),
                "confidence_is_calibrated_probability": False}

    def candidate_cost(self, obs, track_id):
        """Return the current compatibility cost, or None when hard-gated."""
        if not isinstance(obs, BlobObservation):
            raise TypeError("Candidate input must be a BlobObservation")
        for hypothesis in self.hypotheses:
            track = next((item for item in hypothesis.tracks
                          if item.track_id == track_id), None)
            if track is not None:
                compatible = self._compatibility(obs, track)
                return None if compatible is None else compatible[0]
        return None

    def track_monitor(self, track, time_s=None):
        """Expose Kalman covariance, NIS gates and Bernoulli existence."""
        time_s = track.last_end_s if time_s is None else time_s
        dt = max(0., time_s - track.last_end_s)
        aoa_variance = track.aoa_variance_deg2 + self.aoa_q * dt
        pw_variance = track.pw_variance_s2 + self.pw_q * dt
        track_confidence = self._logistic(track.existence_log_odds)
        scalar_gate_sigma = math.sqrt(float(chi2.ppf(
            self.gate_probability, 1)))
        return {"track_id": track.track_id, "last_end_s": track.last_end_s,
            "updates": track.updates,
            "track_confidence": track_confidence,
            "existence_probability": track_confidence,
            "lifecycle_status": ("CONFIRMED" if track_confidence >= .8 else
                "LOST" if track_confidence < .3 else
                "TENTATIVE" if track.updates == 1 else "COASTING"),
            "pri_model": ("local_gap_humps_v1" if track.pri_histogram is not None
                          else "fixed_train" if track.pri_s is not None else None),
            "last_pri_histogram": (None if track.pri_histogram is None
                                   else asdict(track.pri_histogram)),
            # Compatibility aliases for existing result readers.
            "identity_confidence": track_confidence,
            "existence_confidence": track_confidence,
            "last_innovation": {"nis": track.last_nis,
                "degrees_of_freedom": track.last_nis_dof,
                "chi_square_gate": track.last_gate_threshold,
                "log_likelihood_ratio": track.last_log_likelihood_ratio},
            "state": {"aoa_deg": track.aoa_deg,
                      "pw_median_s": track.pw_median_s,
                      "pri_s": track.pri_s,
                      "rf_modes_hz": [m.median_hz for m in track.rf_modes]},
            "covariance": {"aoa_variance_deg2": aoa_variance,
                           "pw_variance_s2": pw_variance,
                           "pri_variance_s2": (None if track.pri_variance_s2 is None
                               else track.pri_variance_s2 + self.pri_q * dt)},
            "gates": {"probability": self.gate_probability,
                      "sigma": scalar_gate_sigma,
                      "aoa_center_deg": track.aoa_deg,
                      "aoa_half_width_deg": scalar_gate_sigma * math.sqrt(
                          aoa_variance + self.aoa_measurement_floor_deg ** 2),
                      "pw_center_s": track.pw_median_s,
                      "pw_half_width_s": scalar_gate_sigma * math.sqrt(
                          pw_variance + self.pw_measurement_floor_s ** 2),
                      "pri_center_s": track.pri_s,
                      "pri_half_width_s": (None if track.pri_variance_s2 is None
                          else scalar_gate_sigma * math.sqrt(
                              track.pri_variance_s2 + self.pri_q * dt
                              + self.pri_measurement_floor_s ** 2))}}

    def snapshot(self):
        return {
            "retained_hypotheses": len(self.hypotheses),
            "plausible_hypotheses": len(self.plausible_hypotheses()),
            "relative_costs_are_probabilities": False,
            "observations_processed": self._sequence,
            "associations": {oid: self.association_options(oid) for oid in self._recent_ids},
            "hypotheses": [{"relative_cost": h.cost,
                "active_tracks": [{"track_id": t.track_id,
                    "aoa_deg": t.aoa_deg, "pw_median_s": t.pw_median_s,
                    "recent_blob_ids": list(t.recent_blob_ids),
                    "rf_modes_hz": [[m.low_hz, m.median_hz, m.high_hz] for m in t.rf_modes],
                    "monitor": self.track_monitor(t)}
                    for t in h.tracks],
                "recent_assignments": dict(h.recent_assignments)} for h in self.hypotheses],
            "association_metrics": {oid: self.association_metrics(oid)
                                    for oid in self._recent_ids},
            "archived_tracks": list(self.archived_tracks.values()),
            "archived_track_count": len(self.archived_tracks),
            "consolidated_aliases": dict(self.consolidated_aliases),
            "consolidation_events": list(self.consolidation_events[-self.archive_limit:]),
            "limits": {name: getattr(self, name) for name in (
                "beam_width", "max_tracks", "max_modes", "max_match_branches",
                "history_limit", "track_history_limit", "archive_limit")},
        }

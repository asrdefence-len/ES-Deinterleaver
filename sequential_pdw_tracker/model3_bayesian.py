"""Bayesian multi-hypothesis association around the Model 2 emitter HMM.

Every chronological PDW is explained by one of three mutually exclusive event
sources: an existing emitter, a new emitter, or clutter.  Association remains a
posterior distribution over global hypotheses; the diagnostic MAP action is
never fed back as a hard decision.  The only approximation in this first
milestone is bounded top-K global-hypothesis pruning.

The event model is a marked Poisson process.  Existing emitters have Gamma-
Poisson pulse-rate posteriors and Model 2 supplies their RF/AOA/PW mark density.
Birth and clutter use explicit rates and uniform mark densities over declared
sensor support.  Track existence is the marginal posterior mass of global
hypotheses containing that track.  Discrete association proposals are scored
and pruned before the conditional HMM state is materialized, preserving the
same bounded posterior while avoiding updates for discarded branches.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

from .hmm_emitter import HMMModel1Config
from .hmm_emitter_model2 import HMMEmitterModel2, HMMModel2Config
from .model import PDW


def _logsumexp(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return -math.inf
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


@dataclass(frozen=True)
class BayesianModel3Config:
    """Generative assumptions for the sequential marked point process."""

    clutter_rate_hz: float = 100.0
    birth_event_rate_hz: float = 5.0
    track_rate_prior_shape: float = 2.0
    track_rate_prior_exposure_s: float = 0.002
    frequency_min_hz: float = 0.5e9
    frequency_max_hz: float = 18.0e9
    pulse_width_min_s: float = 5e-9
    pulse_width_max_s: float = 300e-6
    max_global_hypotheses: int = 16

    def __post_init__(self) -> None:
        positive = (
            self.clutter_rate_hz,
            self.birth_event_rate_hz,
            self.track_rate_prior_shape,
            self.track_rate_prior_exposure_s,
            self.max_global_hypotheses,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("Bayesian Model 3 rates and capacity must be positive")
        if self.frequency_max_hz <= self.frequency_min_hz:
            raise ValueError("frequency support must have positive width")
        if self.pulse_width_min_s <= 0.0:
            raise ValueError("pulse-width support must be positive")
        if self.pulse_width_max_s <= self.pulse_width_min_s:
            raise ValueError("pulse-width support must have positive width")


@dataclass(frozen=True)
class _BayesianTrack:
    track_id: str
    model: HMMEmitterModel2
    birth_toa_s: float
    last_assigned_toa_s: float
    pulse_count: int
    rate_shape: float
    rate_prior_exposure_s: float

    def expected_rate_hz(self, scene_toa_s: float) -> float:
        exposure_s = self.rate_prior_exposure_s + max(
            scene_toa_s - self.birth_toa_s,
            0.0,
        )
        return self.rate_shape / exposure_s


@dataclass
class _GlobalHypothesis:
    log_weight: float = 0.0
    tracks: dict[str, _BayesianTrack] = field(default_factory=dict)
    last_scene_toa_s: float | None = None
    last_action: str = "initial"


@dataclass(frozen=True)
class _EventExplanation:
    """One joint association and waveform-state explanation."""

    reported_action: str
    log_intensity: float
    create_rf_state: bool = False
    create_pulse_width_state: bool = False


@dataclass(frozen=True)
class _ChildProposal:
    """A scored joint latent branch before HMM materialization."""

    parent: _GlobalHypothesis
    explanation: _EventExplanation
    log_weight: float


@dataclass(frozen=True)
class BayesianModel3Update:
    pdw_id: int
    toa_s: float
    association_probabilities: dict[str, float]
    track_existence_probabilities: dict[str, float]
    map_action_for_reporting: str
    retained_global_hypotheses: int
    retained_posterior_mass: float


class BayesianModel3Tracker:
    """Bounded multi-hypothesis Bayesian PDW tracker.

    There is deliberately no confirmation threshold, AOA gate, birth
    suppression rule, or forced nearest-track update.  A track is represented
    by its marginal existence probability across retained global hypotheses.
    """

    def __init__(
        self,
        config: BayesianModel3Config | None = None,
        model_config: HMMModel2Config | None = None,
    ):
        self.config = config or BayesianModel3Config()
        self.model_config = model_config or HMMModel2Config(
            model1=HMMModel1Config(max_rf_states=64),
        )
        self._hypotheses = [_GlobalHypothesis()]
        self._last_retained_mass = 1.0
        self._sharing_metrics = {
            "track_mark_requests": 0,
            "track_mark_evaluations": 0,
            "track_update_requests": 0,
            "track_state_updates": 0,
            "birth_state_constructions": 0,
            "global_child_proposals": 0,
            "retained_child_materializations": 0,
        }

    def _uniform_mark_log_density(self, pdw: PDW) -> float:
        if not self.config.frequency_min_hz <= pdw.frequency_hz <= self.config.frequency_max_hz:
            return -math.inf
        if not self.config.pulse_width_min_s <= pdw.pulse_width_s <= self.config.pulse_width_max_s:
            return -math.inf
        frequency_width = self.config.frequency_max_hz - self.config.frequency_min_hz
        log_pw_width = math.log(
            self.config.pulse_width_max_s / self.config.pulse_width_min_s
        )
        return -math.log(frequency_width) - math.log(360.0) - math.log(log_pw_width)

    def _track_mark_explanations(
        self,
        track_id: str,
        track: _BayesianTrack,
        pdw: PDW,
    ) -> tuple[_EventExplanation, ...]:
        """Enumerate known/new RF and PW explanations for one track.

        The alternatives partition the track mark density.  Unlike the prior
        aggregate calculation, the new-state indicators remain explicit so
        that a retained branch can perform the matching conditional update.
        """
        evidence = track.model.predictive_evidence(pdw)
        frequency_width = self.config.frequency_max_hz - self.config.frequency_min_hz
        log_pw_width = math.log(
            self.config.pulse_width_max_s / self.config.pulse_width_min_s
        )
        rf_new_probability = self.model_config.model1.new_state_prior
        pw_new_probability = self.model_config.new_state_prior
        can_create_rf, can_create_pw = track.model.waveform_state_capacity()

        if can_create_rf:
            rf_options = (
                (
                    False,
                    math.log1p(-rf_new_probability) + evidence.rf_log_evidence,
                ),
                (
                    True,
                    math.log(rf_new_probability) - math.log(frequency_width),
                ),
            )
        else:
            # Once representation capacity is exhausted, the occupied-state
            # model owns all RF prior mass rather than silently discarding the
            # unavailable new-state alternative.
            rf_options = ((False, evidence.rf_log_evidence),)

        if can_create_pw:
            pw_options = (
                (
                    False,
                    math.log1p(-pw_new_probability)
                    + evidence.pulse_width_log_evidence,
                ),
                (
                    True,
                    math.log(pw_new_probability) - math.log(log_pw_width),
                ),
            )
        else:
            pw_options = ((False, evidence.pulse_width_log_evidence),)

        action = f"track:{track_id}"
        return tuple(
            _EventExplanation(
                reported_action=action,
                log_intensity=(
                    evidence.aoa_log_evidence + rf_log_density + pw_log_density
                ),
                create_rf_state=create_rf_state,
                create_pulse_width_state=create_pw_state,
            )
            for create_rf_state, rf_log_density in rf_options
            for create_pw_state, pw_log_density in pw_options
        )

    def _event_log_intensities(
        self,
        hypothesis: _GlobalHypothesis,
        pdw: PDW,
        mark_cache: dict[int, tuple[_EventExplanation, ...]],
    ) -> list[_EventExplanation]:
        background_log_density = self._uniform_mark_log_density(pdw)
        values = [
            _EventExplanation(
                reported_action="clutter",
                log_intensity=(
                    math.log(self.config.clutter_rate_hz) + background_log_density
                ),
            ),
            _EventExplanation(
                reported_action=f"birth:B{pdw.pdw_id}",
                log_intensity=(
                    math.log(self.config.birth_event_rate_hz)
                    + background_log_density
                ),
            ),
        ]
        for track_id, track in hypothesis.tracks.items():
            self._sharing_metrics["track_mark_requests"] += 1
            cache_key = id(track)
            if cache_key not in mark_cache:
                mark_cache[cache_key] = self._track_mark_explanations(
                    track_id,
                    track,
                    pdw,
                )
                self._sharing_metrics["track_mark_evaluations"] += 1
            predicted_rate = track.expected_rate_hz(pdw.toa_s)
            values.extend(
                _EventExplanation(
                    reported_action=explanation.reported_action,
                    log_intensity=(
                        math.log(predicted_rate) + explanation.log_intensity
                    ),
                    create_rf_state=explanation.create_rf_state,
                    create_pulse_width_state=(
                        explanation.create_pulse_width_state
                    ),
                )
                for explanation in mark_cache[cache_key]
            )
        return values

    def _total_unmarked_intensity(
        self,
        hypothesis: _GlobalHypothesis,
        elapsed_s: float,
    ) -> float:
        total = self.config.clutter_rate_hz + self.config.birth_event_rate_hz
        total += sum(
            track.expected_rate_hz(
                (hypothesis.last_scene_toa_s or track.birth_toa_s) + elapsed_s
            )
            for track in hypothesis.tracks.values()
        )
        return total

    def _child_for_action(
        self,
        parent: _GlobalHypothesis,
        pdw: PDW,
        explanation: _EventExplanation,
        log_weight: float,
        update_cache: dict[tuple[int, bool, bool], _BayesianTrack],
        birth_track: _BayesianTrack | None,
    ) -> _GlobalHypothesis:
        action = explanation.reported_action
        tracks = parent.tracks
        if action.startswith("track:"):
            track_id = action.split(":", 1)[1]
            track = parent.tracks[track_id]
            cache_key = (
                id(track),
                explanation.create_rf_state,
                explanation.create_pulse_width_state,
            )
            updated = update_cache.get(cache_key)
            if updated is None:
                model = track.model.clone_active()
                model.process_active_features_conditioned(
                    pdw,
                    create_rf_state=explanation.create_rf_state,
                    create_pulse_width_state=(
                        explanation.create_pulse_width_state
                    ),
                )
                updated = _BayesianTrack(
                    track_id=track.track_id,
                    model=model,
                    birth_toa_s=track.birth_toa_s,
                    last_assigned_toa_s=pdw.toa_s,
                    pulse_count=track.pulse_count + 1,
                    rate_shape=track.rate_shape + 1.0,
                    rate_prior_exposure_s=track.rate_prior_exposure_s,
                )
                update_cache[cache_key] = updated
                self._sharing_metrics["track_state_updates"] += 1
            tracks = dict(parent.tracks)
            tracks[track_id] = updated
        elif action.startswith("birth:"):
            track_id = action.split(":", 1)[1]
            if birth_track is None:
                raise RuntimeError("birth proposal was not materialized")
            if birth_track.track_id != track_id:
                raise RuntimeError("shared birth state has the wrong identity")
            tracks = dict(parent.tracks)
            tracks[track_id] = birth_track
        return _GlobalHypothesis(
            log_weight=log_weight,
            tracks=tracks,
            last_scene_toa_s=pdw.toa_s,
            last_action=action,
        )

    def _birth_track(self, pdw: PDW) -> _BayesianTrack:
        model = HMMEmitterModel2(self.model_config)
        model.process_active_features(pdw)
        self._sharing_metrics["birth_state_constructions"] += 1
        return _BayesianTrack(
            track_id=f"B{pdw.pdw_id}",
            model=model,
            birth_toa_s=pdw.toa_s,
            last_assigned_toa_s=pdw.toa_s,
            pulse_count=1,
            rate_shape=self.config.track_rate_prior_shape + 1.0,
            rate_prior_exposure_s=self.config.track_rate_prior_exposure_s,
        )

    def process(self, pdw: PDW) -> BayesianModel3Update:
        if pdw.frequency_hz <= 0.0 or pdw.pulse_width_s <= 0.0:
            raise ValueError("PDW frequency and pulse width must be positive")
        proposals: list[_ChildProposal] = []
        mark_cache: dict[int, tuple[_EventExplanation, ...]] = {}
        for parent in self._hypotheses:
            if parent.last_scene_toa_s is not None and pdw.toa_s < parent.last_scene_toa_s:
                raise ValueError("PDWs must arrive in non-decreasing TOA order")
            elapsed_s = (
                0.0 if parent.last_scene_toa_s is None
                else pdw.toa_s - parent.last_scene_toa_s
            )
            survival_log_factor = -self._total_unmarked_intensity(
                parent, elapsed_s,
            ) * elapsed_s
            intensities = self._event_log_intensities(
                parent,
                pdw,
                mark_cache,
            )
            for explanation in intensities:
                if not math.isfinite(explanation.log_intensity):
                    continue
                if explanation.reported_action.startswith("track:"):
                    self._sharing_metrics["track_update_requests"] += 1
                proposals.append(_ChildProposal(
                    parent=parent,
                    explanation=explanation,
                    log_weight=(
                        parent.log_weight
                        + survival_log_factor
                        + explanation.log_intensity
                    ),
                ))
        self._sharing_metrics["global_child_proposals"] += len(proposals)
        if not proposals:
            raise ValueError("PDW lies outside declared birth and clutter support")

        # Rao-Blackwellized execution order: score and prune the discrete
        # association variable first.  The conditional RF/AOA/PW state is
        # copied and updated only for proposals that survive top-K pruning.
        # No probability or branch ordering changes relative to eager
        # materialization because child state is absent from the current-event
        # weight.
        full_log_normalizer = _logsumexp([
            proposal.log_weight for proposal in proposals
        ])
        proposals.sort(key=lambda item: item.log_weight, reverse=True)
        retained_proposals = proposals[:self.config.max_global_hypotheses]
        normalized_log_weights = [
            proposal.log_weight - full_log_normalizer
            for proposal in retained_proposals
        ]
        retained_mass = sum(math.exp(value) for value in normalized_log_weights)
        retained_log_normalizer = _logsumexp(normalized_log_weights)

        update_cache: dict[tuple[int, bool, bool], _BayesianTrack] = {}
        birth_track: _BayesianTrack | None = None
        retained: list[_GlobalHypothesis] = []
        for proposal, normalized_log_weight in zip(
            retained_proposals,
            normalized_log_weights,
            strict=True,
        ):
            if (
                proposal.explanation.reported_action.startswith("birth:")
                and birth_track is None
            ):
                birth_track = self._birth_track(pdw)
            retained.append(self._child_for_action(
                proposal.parent,
                pdw,
                proposal.explanation,
                normalized_log_weight - retained_log_normalizer,
                update_cache,
                birth_track,
            ))
        self._sharing_metrics["retained_child_materializations"] += len(retained)
        self._hypotheses = retained
        self._last_retained_mass = retained_mass

        association: dict[str, float] = {}
        existence: dict[str, float] = {}
        for hypothesis in retained:
            probability = math.exp(hypothesis.log_weight)
            association[hypothesis.last_action] = (
                association.get(hypothesis.last_action, 0.0) + probability
            )
            for track_id in hypothesis.tracks:
                existence[track_id] = existence.get(track_id, 0.0) + probability
        association_total = sum(association.values())
        association = {
            action: probability / association_total
            for action, probability in association.items()
        }
        existence = {
            track_id: min(max(probability, 0.0), 1.0)
            for track_id, probability in existence.items()
        }
        map_action = max(association, key=association.get)
        return BayesianModel3Update(
            pdw_id=pdw.pdw_id,
            toa_s=pdw.toa_s,
            association_probabilities=association,
            track_existence_probabilities=existence,
            map_action_for_reporting=map_action,
            retained_global_hypotheses=len(retained),
            retained_posterior_mass=retained_mass,
        )

    def snapshots(self) -> list[dict]:
        existence: dict[str, float] = {}
        representatives: dict[str, tuple[float, _BayesianTrack]] = {}
        branch_counts: dict[str, int] = {}
        for hypothesis in self._hypotheses:
            probability = math.exp(hypothesis.log_weight)
            for track_id, track in hypothesis.tracks.items():
                existence[track_id] = existence.get(track_id, 0.0) + probability
                branch_counts[track_id] = branch_counts.get(track_id, 0) + 1
                if track_id not in representatives or probability > representatives[track_id][0]:
                    representatives[track_id] = (probability, track)
        rows = []
        scene_toa_s = self._hypotheses[0].last_scene_toa_s
        if scene_toa_s is None:
            scene_toa_s = 0.0
        for track_id in sorted(existence, key=lambda item: (-existence[item], item)):
            _probability, representative = representatives[track_id]
            rows.append({
                "track_id": track_id,
                "existence_probability": existence[track_id],
                "conditional_branch_count": branch_counts[track_id],
                "representative_pulse_count": representative.pulse_count,
                "representative_expected_rate_hz": representative.expected_rate_hz(
                    scene_toa_s
                ),
                "birth_toa_s": representative.birth_toa_s,
                "last_assigned_toa_s": representative.last_assigned_toa_s,
                "representative_model": representative.model.snapshot(),
            })
        return rows

    def summary(self) -> dict:
        return {
            "model": (
                "Rao-Blackwellized Bayesian Model 3 marked-Poisson "
                "multi-hypothesis tracker"
            ),
            "config": asdict(self.config),
            "emitter_model_config": asdict(self.model_config),
            "global_hypotheses": len(self._hypotheses),
            "last_retained_posterior_mass": self._last_retained_mass,
            "posterior_representation": (
                "shared-state hypothesis DAG; joint association and Bayesian "
                "known/new RF/PW-state proposals are pruned before conditional "
                "RF/AOA/PW HMM materialization"
            ),
            "waveform_state_birth": (
                "posterior competition over known/new RF and PW alternatives; "
                "Model 2 distance gates are inactive in Model 3"
            ),
            "sharing_metrics": dict(self._sharing_metrics),
            "hard_association_feedback": False,
            "confirmation_threshold": None,
            "track_existence": {
                row["track_id"]: row["existence_probability"]
                for row in self.snapshots()
            },
            "known_boundary": (
                "track death/survival branching is not implemented in this first "
                "association-and-birth milestone"
            ),
        }

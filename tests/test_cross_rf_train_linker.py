from testbed.cross_rf_train_linker import CrossRFTrainLinker
from testbed.pulse_association_pipeline import AssociatedTrain
from testbed.pulse_train_observation import PulseTrainObservation


def train(name, *, pri=85.1e-6, phase=0., rf=10e9, aoa=32.5, pw=.27e-6,
          start=0., end=.1):
    obs = PulseTrainObservation(1, name, start, end, pri, None, rf, None,
        aoa, None, pw, None, 3, (1, 2, 3))
    return AssociatedTrain(obs, 0, phase, 1e-6, 3, 10)


def test_links_rf_modes_with_equal_and_harmonic_pris_and_retains_alternatives():
    linker = CrossRFTrainLinker()
    families = linker.link([
        train("a", phase=0., rf=10e9),
        train("b", phase=10e-6, rf=10.2e9),
        train("c", pri=170.2e-6, phase=20e-6, rf=10.4e9),
    ])
    assert len(families) == 1
    family = families[0]
    assert family.train_ids == ("a", "b", "c")
    assert family.base_pri_s == 85.1e-6
    assert family.phase_cluster_count == 3
    assert family.retained_hypotheses[0] == (("a", "b", "c"),)
    assert family.retained_hypotheses[1] == (("a",), ("b",), ("c",))
    assert any(link.harmonic_left == 1 and link.harmonic_right == 2
               for link in family.links)


def test_incompatible_aoa_or_pulse_width_stays_in_separate_family():
    linker = CrossRFTrainLinker()
    families = linker.link([
        train("a"), train("bearing", aoa=50.), train("pw", pw=2e-6)])
    assert {family.train_ids for family in families} == {
        ("a",), ("bearing",), ("pw",)}


def test_phase_alignment_is_evidence_not_a_required_gate():
    linker = CrossRFTrainLinker()
    aligned = linker.compare(train("a", phase=0.), train("b", phase=1e-6))
    separated = linker.compare(train("a", phase=0.), train("c", phase=15e-6))
    assert aligned.phase_aligned is True
    assert separated.phase_aligned is False


def test_equal_pris_use_the_simplest_one_to_one_harmonic_description():
    link = CrossRFTrainLinker().compare(train("a"), train("b"))
    assert (link.harmonic_left, link.harmonic_right) == (1, 1)
    assert link.base_pri_s == 85.1e-6

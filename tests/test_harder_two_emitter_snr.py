import pytest
from harder_two_emitter_snr import SCENARIOS, run


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_scenarios_normalize_and_repeat(scenario):
    a=run(scenario,trials=3,pd=.7,snr_db=10.,
          pri_prior_sigma_s=.2e-6,epoch_prior_sigma_s=1e-6)
    b=run(scenario,trials=3,pd=.7,snr_db=10.,
          pri_prior_sigma_s=.2e-6,epoch_prior_sigma_s=1e-6)
    assert a==b
    assert 0 <= a["accuracy"] <= 1
    assert 0 <= a["ambiguous_fraction"] <= 1
    assert sum(x["n"] for x in a["calibration"]) == a["pdws"]


def test_coincident_pulses_are_not_perfectly_distinguishable():
    r=run("coincident",trials=3,pd=1.,snr_db=10.)
    assert r["accuracy"] < 1
    assert r["ambiguous_fraction"] > 0


def test_invalid_scenario():
    with pytest.raises(ValueError):
        run("unknown")

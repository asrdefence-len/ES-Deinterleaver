from two_emitter_snr_baseline import association, experiment, toa_sigma


def test_low_snr_increases_uncertainty():
    assert toa_sigma(5) > toa_sigma(20) > toa_sigma(35)


def test_probabilities_normalize():
    models = [(0., .001, 80), (.00012, .00075, 80)]
    for snr in (5, 20, 35):
        p = association(.001, snr, models)
        assert abs(sum(p)-1) < 1e-12


def test_high_snr_discriminates_more():
    models = [(0., .001, 80), (.00012, .00075, 80)]
    assert association(.001, 35, models)[0] > association(.001, 0, models)[0]


def test_reproducible():
    assert experiment(3, .7, 30, 10) == experiment(3, .7, 30, 10)

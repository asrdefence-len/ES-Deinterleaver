"""NRL-style deinterleaving performance metric accounting."""
import pytest
from paired_pdw_quality import experiment


def test_disjoint_reporting_partition():
    r=experiment(trials=2,count=10)
    for arm in ("quality_aware","quality_blind"):
        x=r[arm]
        assert (x["nrl_style_reporting_correct_fraction"]+
                x["nrl_style_reporting_miscorrelated_fraction"]+
                x["nrl_style_reporting_ambiguous_fraction"])==pytest.approx(1.)
        assert x["nrl_style_reporting_ambiguous_fraction"]==pytest.approx(
            1-x["assignment_coverage"])


def test_final_map_track_purity_bounded():
    r=experiment(trials=2,count=10)
    for arm in ("quality_aware","quality_blind"):
        x=r[arm]
        assert .5<=x["final_map_track_purity"]<=1.
        assert sum(map(sum,x["final_map_track_contingency"]))==x["detected_pdws"]

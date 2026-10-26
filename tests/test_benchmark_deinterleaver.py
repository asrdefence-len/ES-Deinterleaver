"""Smoke tests for repeatable, non-mutating benchmark harness."""
import pytest
from benchmark_deinterleaver import samples, measure


def test_joint_samples_are_ordered():
    x=samples("joint",20)
    assert all(x[i][0]<x[i+1][0] for i in range(len(x)-1))


def test_joint_benchmark_smoke():
    r=measure("joint",beam=2,n=6,repeats=1)
    assert r["throughput_pdws_s"]>0
    assert r["p99_latency_us"]>=r["p95_latency_us"]


def test_single_benchmark_smoke():
    r=measure("single",beam=2,n=6,repeats=1)
    assert r["throughput_pdws_s"]>0


def test_invalid_length():
    with pytest.raises(ValueError):
        measure("joint",beam=1,n=1)

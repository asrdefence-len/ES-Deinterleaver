"""V0.1 conditional PRI-jitter assessment for V0 pulse-train candidates.

Preserves V0 discovery and frozen AOA/RF grouping. Uses truth only for evaluation
inside V0 output. A fixed-phase V0 gate is selection-biased against jitter:
UNRESOLVED is not proof of zero jitter. No stagger or cross-blob association.

Run: python -u deinterleaver_v01.py turing_data/config_101.h5 --window 0 --blob 11
"""
import argparse
import numpy as np
import h5py
from scipy.optimize import minimize
from scipy.stats import chi2
from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window
from deinterleaver_v0 import discover


def assess(toa, nominal, toa_sigma, max_jitter_fraction=0.25):
    """Compare fixed and independent interval-jitter models conditional on V0 membership.

    For observed adjacent TOAs, gap d spans k nominal pulse intervals.
    Var(d-k*mu)=k*sigma_jitter**2 + 2*toa_sigma**2.
    k is inferred from nominal period and held fixed in the likelihood fit;
    alternate k and stagger models are not evaluated.
    """
    t=np.sort(np.asarray(toa,float))
    if len(t)<8:
        return {"status":"UNRESOLVED","reason":"too few assigned pulses"}
    d=np.diff(t)
    k=np.maximum(1,np.rint(d/nominal)).astype(int)
    if np.any(d<=0):
        return {"status":"UNRESOLVED","reason":"nonpositive TOA differences"}
    # Gaps near half-integer multiples are ambiguous about missing-pulse counts.
    gap_ambiguity=float(np.mean(np.abs(d/nominal-k)>0.35))
    mu=float(np.sum(d)/np.sum(k))
    max_sd=max_jitter_fraction*nominal
    floor=max(2*toa_sigma**2,1e-10)
    def nll(mu0,sd):
        var=k*sd*sd+floor
        residual=d-k*mu0
        return float(0.5*np.sum(np.log(2*np.pi*var)+residual*residual/var))
    fixed_nll=nll(mu,0.)
    result=minimize(lambda z:nll(z[0],z[1]),[mu,max(.01*nominal,min(max_sd,np.std(d-k*mu)/np.sqrt(np.mean(k))))],
                    bounds=[(max(.01,nominal*.5),nominal*1.5),(0.,max_sd)],
                    method="L-BFGS-B")
    if not result.success:
        return {"status":"UNRESOLVED","reason":"likelihood optimiser did not converge"}
    mu_j,sd=map(float,result.x)
    jitter_nll=nll(mu_j,sd)
    n=len(d)
    # Extra jitter parameter penalised using BIC. This is a model-selection
    # heuristic, NOT a calibrated confidence or formal zero-jitter test.
    bic_fixed=2*fixed_nll+np.log(n)
    bic_jitter=2*jitter_nll+2*np.log(n)
    delta=bic_fixed-bic_jitter
    # Profile-likelihood one-sided upper limit (approximate, conditional on k).
    # For variance parameter at zero boundary, chi-square approximation is rough.
    threshold=jitter_nll+0.5*chi2.ppf(.95,1)
    grid=np.linspace(0,max_sd,251)
    prof=[]
    for s in grid:
        # Weighted least-squares closed form for mu at fixed sd.
        weights=1/(k*s*s+floor)
        m=np.sum(weights*k*d)/np.sum(weights*k*k)
        prof.append(nll(m,s))
    valid=grid[np.asarray(prof)<=threshold]
    upper=float(valid[-1]) if len(valid) else float("nan")
    at_cap=(sd>=.99*max_sd or (len(valid) and upper>=.99*max_sd))
    if gap_ambiguity>.10 or at_cap:
        status="UNRESOLVED"
    elif delta>6 and sd>0:
        status="JITTERED_CANDIDATE"
    elif delta< -2:
        status="FIXED_COMPATIBLE"
    else:
        status="UNRESOLVED"
    return dict(status=status,mu=mu_j,sd=sd,jitter_pct=100*sd/mu_j,
                upper_pct=100*upper/mu_j,delta_bic=delta,
                gaps=n,mean_missing=float(np.mean(k-1)),
                ambiguous_gap_fraction=gap_ambiguity,at_cap=bool(at_cap))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--window",type=int,default=0)
    ap.add_argument("--blob",type=int,default=11)
    ap.add_argument("--toa-sigma",type=float,default=0.25,
                    help="Assumed single-pulse TOA measurement std, raw units; sensitivity-test this")
    ap.add_argument("--pri-min",type=float,default=20.)
    ap.add_argument("--pri-max",type=float,default=200.)
    ap.add_argument("--tolerance",type=float,default=2.)
    ap.add_argument("--min-pulses",type=int,default=20)
    ap.add_argument("--max-tracks",type=int,default=4)
    ap.add_argument("--bin-width",type=float,default=.1)
    a=ap.parse_args()
    if not (a.toa_sigma>0 and a.pri_min>0 and a.pri_max>a.pri_min
            and a.tolerance>0 and a.bin_width>0):
        ap.error("Invalid parameters")
    with h5py.File(a.file) as h:
        start=a.window*10000
        stop=min(start+10000,len(h["data"]))
        if start<0 or stop<=start: ap.error("Window outside dataset")
        names=[v.decode() if isinstance(v,bytes) else str(v)
               for v in h["metadata/feature_names"][:]]
        x=np.asarray(h["data"][start:stop],float)
    pops,npop=batch_aoa_populations(x[:,names.index("AoA")])
    blobs=group_window(x,names,pops,npop,8,.20,6.,8)
    original=np.flatnonzero(blobs==a.blob)
    if not len(original): ap.error("Blob not found")
    toa=x[original,names.index("ToA")]
    tracks,unexplained=discover(toa,a.pri_min,a.pri_max,a.tolerance,
                               a.min_pulses,a.max_tracks,a.bin_width)
    print(f"V0.1 window={a.window} blob={a.blob} PDWs={len(toa)} "
          f"V0_trains={len(tracks)} unexplained={len(unexplained)} "
          f"assumed_toa_sigma={a.toa_sigma}")
    for tr in tracks:
        r=assess(toa[tr.pdw_indices],tr.period,a.toa_sigma)
        print(f" {tr.id} V0_period={tr.period:.4f} assigned={len(tr.pdw_indices)} "
              f"slots={tr.occupied_slots}/{tr.span_slots} model={r['status']}")
        if 'mu' in r:
            print(f"    mean_PRI={r['mu']:.4f} jitter_sd={r['sd']:.4f} "
                  f"jitter={r['jitter_pct']:.2f}% approx_upper95={r['upper_pct']:.2f}% "
                  f"delta_BIC_fixed_minus_jitter={r['delta_bic']:+.2f} "
                  f"gaps={r['gaps']} mean_missing_per_gap={r['mean_missing']:.2f} "
                  f"ambiguous_gap_fraction={r['ambiguous_gap_fraction']:.2%} "
                  f"parameter_cap={r['at_cap']}")
        else:
            print(f"    reason={r['reason']}")
    print("CAUTION: Jitter assessment is conditional on V0 fixed-phase-selected pulses "
          "and inferred integer gaps. It can miss genuinely jittered pulses and "
          "cannot prove zero jitter. Upper limits are approximate and depend on "
          "the assumed TOA error. No truth labels used for model fitting.")


if __name__=="__main__":
    main()

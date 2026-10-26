"""Experimental fixed-PRI extraction from measured TOAs; never uses truth."""
from bisect import bisect_left
from statistics import median
from testbed.pri_discovery import difference_histogram, candidate_peaks
from testbed.tracks import EmitterTrack, PRIType

class PRIGuidedDeinterleaver:
    def __init__(self, tolerance_s=8e-6, min_train_length=15, max_missing=3):
        self.tolerance_s=tolerance_s
        self.min_train_length=min_train_length
        self.max_missing=max_missing
        self.tracks=[]

    def process(self, pdws):
        pdws=sorted(pdws,key=lambda p:p.toa_s)
        if not pdws: return {}
        times=[p.toa_s for p in pdws]
        available={p.pdw_id for p in pdws}
        assignments={}
        histogram=difference_histogram(pdws,0.0005,0.0015,0.000002)
        peaks=candidate_peaks(histogram,top_n=12,min_spacing_s=0.000012)
        pris=[p.interval_s for p in peaks if p.count>=max(6,len(pdws)//30)]
        while len(available)>=self.min_train_length:
            best=None
            for pri in pris:
                anchors=[p for p in pdws if p.pdw_id in available and p.toa_s<=pdws[0].toa_s+3*pri]
                for anchor in anchors:
                    train=self._extract(pdws,times,available,anchor,pri)
                    if len(train)<self.min_train_length: continue
                    refined=self._fit(train,pri)
                    if abs(refined-pri)>12e-6: continue
                    retry=self._extract(pdws,times,available,anchor,refined)
                    if len(retry)>=len(train): train=retry
                    residual=median([abs((b.toa_s-a.toa_s)-max(1,round((b.toa_s-a.toa_s)/refined))*refined) for a,b in zip(train,train[1:])])
                    score=(len(train),-residual)
                    if best is None or score>best[0]: best=(score,refined,train)
            if best is None: break
            _,pri,train=best
            track=EmitterTrack(f"T{len(self.tracks)+1}")
            for p in train:
                track.add_pdw(p)
                assignments[p.pdw_id]=track.track_id
                available.remove(p.pdw_id)
            track.pri_model.kind=PRIType.FIXED
            track.pri_model.mean_s=pri
            track.pri_model.confidence=min(0.9,len(train)/100)
            self.tracks.append(track)
        return assignments

    def _extract(self,pdws,times,available,anchor,pri):
        selected=[anchor]
        used={anchor.pdw_id}
        current=anchor.toa_s
        while True:
            found=None
            for k in range(1,self.max_missing+2):
                target=current+k*pri
                if target>times[-1]+self.tolerance_s: break
                i=bisect_left(times,target)
                options=[p for p in pdws[max(0,i-4):min(len(pdws),i+5)]
                         if p.pdw_id in available and p.pdw_id not in used and abs(p.toa_s-target)<=self.tolerance_s]
                if options:
                    found=min(options,key=lambda p:abs(p.toa_s-target))
                    break
            if found is None: break
            selected.append(found)
            used.add(found.pdw_id)
            current=found.toa_s
        return selected

    @staticmethod
    def _fit(train,nominal):
        if len(train)<2: return nominal
        index=[0]
        for a,b in zip(train,train[1:]):
            index.append(index[-1]+max(1,round((b.toa_s-a.toa_s)/nominal)))
        return median([(b.toa_s-a.toa_s)/(ib-ia) for a,ia in zip(train[:-1],index[:-1])
                       for b,ib in zip(train[1:],index[1:]) if ib>ia])

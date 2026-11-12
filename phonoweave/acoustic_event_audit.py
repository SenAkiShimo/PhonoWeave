"""Post-hoc diagnostics for OTO-free candidate events.

This module may read label/OTO-linked metadata *only after* a frozen scan;
its output never serves as parameters or labels for the scan. Do not present
these diagnostics as independent held-out phoneme-boundary accuracy.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import zipfile

import numpy as np


def _read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def reviewed_regions(events_csv:Path,summary_json:Path,review_manifest:dict,reviewed_csv:Path):
    summary=json.loads(Path(summary_json).read_text(encoding='utf8'))
    if summary['archive_sha256']!=review_manifest['source']['archive_sha256']:
        raise ValueError('review reference WAV archive does not match scanned bank')
    rows=_read_csv(events_csv)
    ann=_read_csv(reviewed_csv)
    annotations={(r['left_surface'],r['right_surface']):r for r in ann}
    found=[]
    for group in review_manifest['review_groups']:
        right=group['right_surface']
        for rec in group['clip_records']:
            left=rec['left_surface']
            r=annotations[(left,right)]
            if r['original_alias']!=rec['original_alias'] or r['source_wav']!=rec['source_wav']:
                raise ValueError('manual reviewed annotation does not match source manifest')
            if r['target_sound_present']!='yes':
                raise ValueError('unconfirmed target in review data')
            lo=rec['oto_midpoint_ms']-rec['excerpt_oto_anchor_ms']+float(r['boundary_onset_ms_in_excerpt'])
            hi=rec['oto_midpoint_ms']-rec['excerpt_oto_anchor_ms']+float(r['boundary_offset_ms_in_excerpt'])
            if lo>=hi:raise ValueError('bad manual transition interval')
            wav=rec['source_wav']
            bucket={kind:np.asarray([float(x['time_ms']) for x in rows
                                   if x['wav']==wav and x['kind']==kind])
                    for kind in ('internal_acoustic_change','activity_onset','activity_offset')}
            internal=bucket['internal_acoustic_change']
            if len(internal)==0:dist=float('inf');nearest=None
            else:
                gaps=np.maximum(np.maximum(lo-internal,0),internal-hi)
                n=int(np.argmin(gaps));dist=float(gaps[n]);nearest=float(internal[n])
            found.append({'left_surface':left,'right_surface':right,'source_wav':wav,
                'reference_start_ms':round(lo,3),'reference_end_ms':round(hi,3),
                'closest_acoustic_event_ms':nearest,
                'distance_to_reviewed_transition_ms':round(dist,3),
                'event_inside_interval':bool(dist==0),
                'within_10ms_of_interval':bool(dist<=10),
                'activity_edge_inside_interval':any(np.any((a>=lo)&(a<=hi)) for a in
                    [bucket['activity_onset'],bucket['activity_offset']]),
                'status':'debugged_dev_set_NOT_unseen_gold'})
    return {'manual_regions':len(found),
            'internal_inside':sum(r['event_inside_interval'] for r in found),
            'internal_within_10ms':sum(r['within_10ms_of_interval'] for r in found),
            'activity_edge_inside':sum(r['activity_edge_inside_interval'] for r in found),
            'status':'diagnostic_only_development_regions_NOT_phone_boundary_accuracy',
            'rows':found}


def _nearest_matches(a,b,tolerance):
    if not len(a) or not len(b):return 0
    ix=np.searchsorted(b,a)
    right=np.abs(a-b[np.clip(ix,0,len(b)-1)])
    left=np.abs(a-b[np.clip(ix-1,0,len(b)-1)])
    return int(np.sum(np.minimum(left,right)<=tolerance))


def pitch_style_repeatability(results_root:Path,reference='mang_D4',tol_ms=30):
    root=Path(results_root)
    def bank_data(bank):
        rows=_read_csv(root/bank/'events.csv')
        sources=_read_csv(root/bank/'sources.csv')
        durations={r['source_wav']:float(r['duration_ms']) for r in sources}
        by={}
        for r in rows:
            if r['kind']=='internal_acoustic_change':
                by.setdefault(r['wav'],[]).append(float(r['time_ms']))
        return {w:np.asarray(sorted(p)) for w,p in by.items()},durations
    base,durations=bank_data(reference)
    report=[]
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name==reference or not (d/'events.csv').exists():continue
        other,other_durations=bank_data(d.name)
        count=matched=control=0
        for wav,a in base.items():
            b=other.get(wav)
            if b is None:continue
            dur=durations[wav]
            if abs(dur-other_durations[wav])>15.0:
                raise ValueError('source template durations differ >15ms')
            count+=len(a)
            matched+=_nearest_matches(a,b,tol_ms)
            # Circular shift deliberately misaligns the source template. This
            # is a NEGATIVE CONTROL, not a p-value / independent sample.
            bad=np.sort((b+500.0)%dur)
            control+=_nearest_matches(a,bad,tol_ms)
        if not count:continue
        report.append({'reference_bank':reference,'compared_bank':d.name,
            'reference_events':count,'matched_within_30ms':matched,
            'matched_fraction':round(matched/count,4),
            'shifted_control_fraction':round(control/count,4),
            'source_templates_are_related':True,
            'status':'same_DiffSinger_source_template_consistency_NOT_independent_accuracy'})
    return {'comparison_count':len(report),'rows':report,
            'status':'repeatability_diagnostic_NOT_phoneme_boundary_accuracy'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--review-zip',type=Path)
    parser.add_argument('--reviewed-csv',type=Path)
    a=parser.parse_args(argv)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    agreement=pitch_style_repeatability(a.results_root)
    (a.output/'repeatability.json').write_text(json.dumps(agreement,indent=2,ensure_ascii=False))
    if a.review_zip and a.reviewed_csv:
        with zipfile.ZipFile(a.review_zip) as z:
            manifest=json.loads(z.read('review_manifest.json'))
        reviewed=reviewed_regions(a.results_root/'mang_D4/events.csv',
                 a.results_root/'mang_D4/summary.json',manifest,a.reviewed_csv)
        (a.output/'manual_dev_region_check.json').write_text(
                json.dumps(reviewed,indent=2,ensure_ascii=False))
        with (a.output/'manual_dev_region_check.csv').open('w',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(reviewed['rows'][0]));w.writeheader();w.writerows(reviewed['rows'])
    print(json.dumps({'repeatability':agreement['rows'],
           'reviewed': reviewed['manual_regions'] if a.review_zip and a.reviewed_csv else 0},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())

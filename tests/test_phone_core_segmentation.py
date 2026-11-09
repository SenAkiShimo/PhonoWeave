"""Synthetic regression tests; no claims of real phonetic accuracy."""
from __future__ import annotations
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import wave
import zipfile

import numpy as np
import pytest

from phonoweave.phone_core_segmentation import (
    TARGETS,_crop,_frames,_oto_rows,_load_review_pack,_run,
    frication_event,_ranges,_textgrid,
)

RATE=44100


def _wav(b):
    out=io.BytesIO()
    with wave.open(out,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(RATE)
        w.writeframes(b)
    return out.getvalue()


def _synth_six(tmp_path, *,other_pitch=False):
    """Independent *synthetic* reference pack, not DiffSinger audio."""
    def sound(sh_locs,style_seed):
        duration=4.5
        t=np.arange(round(duration*RATE))/RATE
        data=.09*np.sin(2*np.pi*220*t).astype(float)
        for j,at in enumerate(sh_locs):
            i=round((at-.240+.250)*RATE)
            a=i;b=min(len(t),i+round(.260*RATE))
            rng=np.random.default_rng(100+j+style_seed)
            data[a:b]=.07*rng.normal(size=b-a)
        return (np.clip(data,-.85,.85)*32767).astype('<i2').tobytes()

    waveforms={
       'vcsection2.wav': _wav(sound([],1)),
       'vcsection3.wav': _wav(sound([1.,2.],4)),
       'ingi0ir.wav':_wav(sound([1.],11)),
    }
    anchors={('i','m'):1.0,('ir','m'):2.,('i0','m'):3.,
             ('i','sh'):1.,('ir','sh'):1.,('i0','sh'):2.}
    def bank(pitch:str,seed=0):
        zpath=tmp_path/f'{pitch}.zip';root=f'fake{pitch}/'
        lines=[]
        for (left,right),wav in TARGETS.items():
            center=anchors[(left,right)]*1000
            offset=center-70
            lines.append(f'{wav}={pitch}{left} {right},{offset},20,70,90,50')
        oto=('\n'.join(lines)+'\n').encode()
        with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr(root+'oto.ini',oto)
            for name,content in waveforms.items():z.writestr(root+name,content)
        return dict(style='mang',pitch=pitch,archive=str(zpath),root=root,prefix=pitch),oto,lines

    ref,refoto,lines=bank('D4')
    banks=[ref]
    if other_pitch:
        bank2,_,_=bank('G3',1);banks.append(bank2)
    review_pack=tmp_path/'review.zip';records=[];csvrows=[]
    source_sha={wav:sha256(waveforms[wav]).hexdigest() for wav in waveforms}
    with zipfile.ZipFile(review_pack,'w',zipfile.ZIP_DEFLATED) as z:
        for n,((left,right),wav) in enumerate(TARGETS.items(),1):
            anchor=anchors[(left,right)]*1000
            # Ensure excerpt and reviewer fixture are exactly source-derived.
            pcm=waveforms[wav]
            with wave.open(io.BytesIO(pcm),'rb') as w: arr=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2')
            start=round((anchor-240)*RATE/1000);end=start+round(.600*RATE)
            excerpt=_wav(arr[start:end].tobytes())
            rel=f'right_{right}/{left}.wav'
            z.writestr(rel,excerpt)
            records.append(dict(left_surface=left,source_wav=wav,
                source_wav_sha256=source_sha[wav],oto_line=n,
                original_alias=f'D4{left} {right}',excerpt_filename=rel,
                excerpt_sha256=sha256(excerpt).hexdigest()))
            csvrows.append(dict(right_surface=right,left_surface=left,
                original_alias=f'D4{left} {right}',source_wav=wav,
                oto_line=n,boundary_onset_ms_in_excerpt='220',
                boundary_offset_ms_in_excerpt='291',target_sound_present='yes'))
        group=[]
        for right in ('m','sh'):
            group.append(dict(right_surface=right,
                              clip_records=[r for r in records if (r['left_surface'],right) in TARGETS and TARGETS[(r['left_surface'],right)]==r['source_wav'] and r['original_alias'].endswith(' '+right)]))
        manifest=dict(review_groups=group,source={
            'archive_sha256':sha256(Path(ref['archive']).read_bytes()).hexdigest(),
            'oto_sha256':sha256(refoto).hexdigest()})
        z.writestr('review_manifest.json',json.dumps(manifest))
    csvfile=tmp_path/'review.csv'
    with csvfile.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    config=dict(schema='phonoweave.multisubbank_input.v1',banks=banks)
    return config,csvfile,review_pack


def test_acoustic_frication_is_an_event_not_a_phoneme_boundary():
    t=np.arange(round(.600*RATE))/RATE
    y=.075*np.sin(2*np.pi*235*t)
    rng=np.random.default_rng(212)
    y[t>=.254]=.095*rng.normal(size=np.sum(t>=.254))
    event=frication_event(y,RATE)
    assert event and 238<event['event_center_ms']<277
    assert event['evidence']=='persistent_highband_frication_not_phoneme_boundary'


def test_m_like_voiced_audio_does_not_fabricate_frication():
    t=np.arange(round(.600*RATE))/RATE
    y=.11*np.sin(2*np.pi*205*t)
    assert frication_event(y,RATE) is None


def test_range_is_conservative_and_not_exact_phoneme():
    left,transition,right=_ranges(220.,285.)
    assert left==(163.,203.) and transition==(220.,285.) and right==(303.,343.)
    assert not (left[1]>=transition[0] or transition[1]>=right[0])
    assert _ranges(220.,228.) is None


def test_textgrid_export_is_three_tiers_not_a_lying_full_alignment(tmp_path):
    path=tmp_path/'a.TextGrid'
    _textgrid(path,(160,200),(220,280),(300,340),'i','sh')
    s=path.read_text()
    assert s.count('class = "IntervalTier"')==3
    assert 'uncertain_transition' in s
    assert 'name = "left_nucleus_core"' in s
    assert 'name = "right_consonant_core"' in s
    assert 'xmin = 0.160000' in s


def test_oto_parser_rejects_nonfinite_and_preserves_original_alias():
    good=_oto_rows(b'a.wav=D4i sh,930,60,70,90,50\n')
    assert good[0]['anchor_ms']==1000
    assert good[0]['alias']=='D4i sh'
    with pytest.raises(ValueError,match='malformed'): _oto_rows(b'a.wav=D4i sh,nan,60,70,90,50')


def test_synthetic_source_linked_segmentation_end_to_end(tmp_path):
    config,review,pack=_synth_six(tmp_path)
    out=tmp_path/'segs'
    result=_run(config,review,out,review_pack=pack)
    assert result['total_targets']==6
    assert result['human_reviewed_bands']==6
    assert result['phoneme_point_boundaries_confirmed']==0
    assert result['whole_phone_boundaries_confirmed']==0
    assert result['no_auto_split_merge']
    for target in result['segments']:
        assert target['D4_review_pack_excerpt_exactly_verified'] is True
        d=out/'banks'/target['bank']/f"{target['left_alias']}_to_{target['right_phone']}"
        assert (d/'excerpt_600ms.wav').exists()
        assert (d/'left_nucleus_core.wav').exists()
        assert (d/'uncertain_transition.wav').exists()
        assert (d/'right_consonant_core.wav').exists()
        assert (d/'phones.TextGrid').exists()
        assert target['phone_borders_verified'] is False
    with pytest.raises(FileExistsError):_run(config,review,out,review_pack=pack)


def test_multi_pitch_candidates_never_pretend_human_verified(tmp_path):
    config,review,pack=_synth_six(tmp_path,other_pitch=True)
    result=_run(config,review,tmp_path/'segments',review_pack=pack)
    assert result['total_targets']==12
    other=[x for x in result['segments'] if x['bank']=='mang_G3']
    assert len(other)==6
    assert all(x['status']=='provisional_auto_segment' for x in other)
    assert all(x['D4_review_pack_excerpt_exactly_verified'] is None for x in other)
    assert all(x['transition_interval_provenance'].startswith('nasal_') for x in other if x['right_phone']=='m')
    assert all(x['phone_borders_verified'] is False for x in other)


def test_source_mutation_invalidates_author_review_pack(tmp_path):
    config,review,pack=_synth_six(tmp_path)
    bad=Path(config['banks'][0]['archive'])
    original=bad.read_bytes()
    bad.write_bytes(original+b'garbage')
    with pytest.raises(ValueError,match='source ZIP hash'):
        _run(config,review,tmp_path/'out',review_pack=pack)
    assert not (tmp_path/'out').exists()


def test_review_csv_line_or_alias_tamper_rejected(tmp_path):
    config,review,pack=_synth_six(tmp_path)
    raw=review.read_text().replace('D4i sh','D4x sh',1)
    review.write_text(raw)
    with pytest.raises(ValueError,match='review alias identity'):
        _run(config,review,tmp_path/'out',review_pack=pack)
    assert not (tmp_path/'out').exists()


def test_reviews_have_exact_fixed_six_targets(tmp_path):
    config,review,pack=_synth_six(tmp_path)
    review.write_text('\n'.join(review.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError,match='six human-reviewed targets'):
        _run(config,review,tmp_path/'out',review_pack=pack)


def test_all_sh_aliases_yield_source_linked_phone_cores(tmp_path):
    config,review,pack=_synth_six(tmp_path,other_pitch=True)
    out=tmp_path/'segments'
    result=_run(config,review,out,review_pack=pack)
    scan=result['all_sh_phone_core_scan']
    assert scan['oto_sh_alias_candidates']==6
    assert scan['acoustic_sh_cores_extracted']==6
    assert scan['phonetic_boundaries_certified']==0
    assert len(list((out/'all_sh_phone_cores').rglob('sh_candidate_core_40ms.wav')))==6
    idx=(out/'all_sh_phone_cores/sh_phone_core_index.csv').read_text()
    assert 'left_surface_unverified' in idx and 'OTO' not in idx.splitlines()[0]


def test_sh_core_preserves_exact_source_pcm_bytes(tmp_path):
    config,review,pack=_synth_six(tmp_path)
    out=tmp_path/'segments';_run(config,review,out,review_pack=pack)
    source_zip=zipfile.ZipFile(config['banks'][0]['archive'])
    sh=(out/'all_sh_phone_cores/mang_D4/i_to_sh_OTO4')
    outaudio=sh/'sh_candidate_core_40ms.wav'
    metadata=json.loads((sh/'provenance.json').read_text())
    with wave.open(str(outaudio)) as w:
        actual=w.readframes(w.getnframes())
    with wave.open(io.BytesIO(source_zip.read('fakeD4/vcsection3.wav'))) as w:
        rate=w.getframerate();src=w.readframes(w.getnframes())
    a,b=metadata['core_interval_ms_in_source']
    aa=round(a*rate/1000);bb=round(b*rate/1000)
    assert actual==src[2*aa:2*bb]
    assert metadata['phonetic_boundary_verified'] is False

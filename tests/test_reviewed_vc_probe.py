"""Synthetic software tests, not evidence of actual phonetic efficacy."""
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import wave
import numpy as np
import pytest

from phonoweave.reviewed_vc_probe import compile_probe, export_probe, load_review, SCHEMA


def _fake_pair(tmp_path):
    root=tmp_path/'clips';root.mkdir()
    src=[]; csv_rows=[]
    for right in ('m','sh'):
        group=[]
        for i,left in enumerate(('i','ir','i0')):
            rate=44100;t=np.arange(int(rate*.6))/rate
            signal=.15*np.sin(2*np.pi*(220+i*10)*t)
            if right=='sh':
                rng=np.random.default_rng(i)
                gate=(t>.28)&(t<.40)
                signal[gate]=rng.normal(0,.09,gate.sum())
            with io.BytesIO() as b:
                with wave.open(b,'wb') as f:
                    f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate)
                    f.writeframes((signal*32767).astype('<i2').tobytes())
                wav=b.getvalue()
            rel=f'right_{right}/{left}.wav'
            path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(wav)
            item={'excerpt_filename':rel,'left_surface':left,'source_wav':f'{right}.wav',
              'excerpt_oto_anchor_ms':240,'original_alias':f'D4{left} {right}',
              'oto_line':100+i,'excerpt_sha256':sha256(wav).hexdigest()}
            group.append(item)
            csv_rows.append({'right_surface':right,'left_surface':left,'original_alias':item['original_alias'],
              'source_wav':item['source_wav'],'oto_line':str(item['oto_line']),
              'boundary_onset_ms_in_excerpt':'215.0','boundary_offset_ms_in_excerpt':'290.0',
              'target_sound_present':'yes','listening_comments':'','reviewer':'synthetic_tester'})
        src.append({'right_surface':right,'clip_records':group})
    manifest={'schema':'phonoweave.vc_boundary_feasibility_gate.v0.1',
      'status':'auditory_boundary_review_required_no_split_share_evidence',
      'review_groups':src,'source':{'archive_sha256':'a'*64,'oto_sha256':'b'*64}}
    (root/'review_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    annotation=tmp_path/'annotations.csv'
    with annotation.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    return root,annotation


def test_compile_has_six_and_only_descriptive(tmp_path):
    root,anno=_fake_pair(tmp_path)
    report=compile_probe(root,anno)
    assert report['schema']==SCHEMA
    assert len(report['interval_features'])==6
    assert len(report['within_consonant_pair_distances'])==6
    assert report['research_decision']['go_for_automatic_split_share'] is False
    assert report['review']['independent_recording_sessions_verified'] is False
    assert all(x['sensitivity_grid_size']==81 for x in report['within_consonant_pair_distances'])
    assert compile_probe(root,anno)==report


def test_missing_human_yes_is_rejected(tmp_path):
    root,anno=_fake_pair(tmp_path)
    rows=list(csv.DictReader(anno.open()))
    rows[0]['target_sound_present']='uncertain'
    with anno.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match='nonaffirmative'):
        compile_probe(root,anno)


def test_hash_and_identity_fail_closed(tmp_path):
    root,anno=_fake_pair(tmp_path)
    (root/'right_m/i.wav').write_bytes(b'altered')
    with pytest.raises(ValueError,match='SHA-256'):
        compile_probe(root,anno)


def test_reviewed_interval_outside_bounds_rejected(tmp_path):
    root,anno=_fake_pair(tmp_path)
    rows=list(csv.DictReader(anno.open()))
    rows[2]['boundary_offset_ms_in_excerpt']='999'
    with anno.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match='outside extractable'):
        compile_probe(root,anno)


def test_no_overwrite_and_json_export(tmp_path):
    root,anno=_fake_pair(tmp_path)
    result=compile_probe(root,anno)
    dst=tmp_path/'result'
    export_probe(result,dst)
    assert json.loads((dst/'reviewed_vc_diagnostics.json').read_text())['schema']==SCHEMA
    with pytest.raises(FileExistsError,match='already exists'):
        export_probe(result,dst)


def test_source_pack_cannot_be_changed_by_report(tmp_path):
    root,anno=_fake_pair(tmp_path)
    result=compile_probe(root,anno)
    with pytest.raises(ValueError,match="outside original"):
        export_probe(result,root/"report",source_pack=root)

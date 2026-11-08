"""Source-verified, offline blind listening page for frozen note-phrase previews.

No listener ratings are generated; user explicitly supplies each preference.
K=3 higher-budget reference is excluded from the blind equal-budget trials.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import shutil
import tempfile
import wave

SCHEMA = 'phonoweave.diffsinger_note_phrase_blind_listening.v1'
RENDER_SCHEMA = 'phonoweave.diffsinger_note_phrase_ab.v1'


def _safe_rel(root: Path, rel: str) -> Path:
    if not isinstance(rel,str) or rel.startswith('/') or '..' in Path(rel).parts:
        raise ValueError('unsafe rendered WAV path')
    p=(root/rel).resolve(strict=True)
    if not p.is_relative_to(root) or not p.is_file():
        raise ValueError('source WAV leaves frozen render root')
    return p


def _csv_escape(s: str) -> str:
    return '"'+s.replace('"','""')+'"'


def _html_page(data:dict) -> str:
    packed=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    return '''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PhonoWeave · 六音符拼接盲听</title>
<style>:root{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f6f7f9;color:#182233}body{max-width:800px;margin:30px auto;padding:0 17px}h1{font-size:25px}p{line-height:1.65}.hint{color:#546477}.note{padding:12px 15px;border-radius:9px;background:#eaf0f8;margin:18px 0}.trial{background:white;border:1px solid #dce1e8;border-radius:13px;padding:20px;margin:16px 0}audio{width:100%;margin:4px 0 12px}select,button,textarea{font:inherit;padding:7px;border:1px solid #b5c0cd;border-radius:7px}textarea{width:96%;min-height:48px}.player{display:grid;grid-template-columns:1fr 1fr;gap:16px}.player audio{min-width:0}.pref{display:flex;flex-wrap:wrap;gap:16px;margin:9px 0}.pref label{cursor:pointer}button{padding:10px 18px;background:#1b3558;color:white;cursor:pointer}#progress{font-weight:600}@media(max-width:650px){.player{grid-template-columns:1fr}}</style></head><body>
<h1>PhonoWeave · 多音符 VC 共享盲听</h1>
<p class="note"><b>这里比较的都是相同预算的 K=2 策略。</b> 每个短乐句包含六个音符；两段试听由同一个声音库、音阶、固定乐谱和相同音频处理器生成，只有 VC 共享方式不同。你可以反复听，也可以选「无法判断」。<br><b>它不是</b>与独立 DiffSinger 原唱的质量测试，也不是正式论文听觉实验。</p>
<p class="hint">不需要做满所有题。如果声音差别太小，选「听不出差异」是有意义的反馈。耳机和安静的环境会更方便。</p>
<div id="progress"></div><div id="trials"></div>
<p><button id="export">导出我的试听记录 CSV</button></p>
<p class="hint">这里不会上传任何音频或结果。所有播放和导出均在你的浏览器本地完成。</p>
<script id="data" type="application/json">__DATA__</script>
<script>
const report=JSON.parse(document.getElementById('data').textContent);
const root=document.getElementById('trials');const state={};
for(const [index,trial] of report.trials.entries()){
 const area=document.createElement('section');area.className='trial';
 const h=document.createElement('h3');h.textContent=(index+1)+' / '+report.trials.length+' · '+trial.style+' '+trial.pitch+' · '+trial.phrase;area.append(h);
 const players=document.createElement('div');players.className='player';
 for(const side of ['A','B']){const panel=document.createElement('div');const title=document.createElement('b');title.textContent='试听 '+side;panel.append(title);const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=trial[side+'_wav'];panel.append(audio);players.append(panel)}area.append(players);
 const p=document.createElement('p');p.textContent='哪一段的连接更自然、听起来更连贯？';area.append(p);
 const options=document.createElement('div');options.className='pref';
 for(const [value,label] of [['A','A 更好'],['B','B 更好'],['tie','差不多'],['uncertain','无法判断']]){
  const item=document.createElement('label');const control=document.createElement('input');control.type='radio';control.name=trial.trial_id;control.value=value;control.addEventListener('change',()=>{state[trial.trial_id]={...state[trial.trial_id],choice:value};refresh()});item.append(control,document.createTextNode(' '+label));options.append(item)
 }area.append(options);
 const desc=document.createElement('textarea');desc.placeholder='可选：哪里出现了跳音、断裂、拖尾或发音不清？';desc.addEventListener('input',()=>{state[trial.trial_id]={...state[trial.trial_id],notes:desc.value}});area.append(desc);
 root.append(area)
}
function refresh(){document.getElementById('progress').textContent='已作答 '+Object.values(state).filter(x=>x.choice).length+'/'+report.trials.length+' 题（不必全部完成）'}refresh();
function esc(v){return '"'+String(v??'').replaceAll('"','""')+'"'}
document.getElementById('export').addEventListener('click',()=>{
 const head=['trial_id','style','pitch','phrase','preference','notes'];const lines=[head.join(',')];
 for(const t of report.trials){const r=state[t.trial_id]||{};lines.push([t.trial_id,t.style,t.pitch,t.phrase,r.choice||'',r.notes||''].map(esc).join(','))}
 const blob=new Blob([lines.join('\\r\\n')+'\\r\\n'],{type:'text/csv;charset=utf-8'});
 const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='vc_note_phrase_listening.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
</script></body></html>'''.replace('__DATA__',packed)


def package_blind_listening(render_root:Path, output:Path, *,focus_banks:tuple[str,...]=('mang_D4','lu_C4')) -> dict:
    root=Path(render_root).expanduser().resolve(strict=True)
    target=Path(output).expanduser().resolve()
    if target.exists():raise FileExistsError('listening output already exists')
    if root==target or root in target.parents or target in root.parents:
        raise ValueError('blind package must remain separate from frozen rendered source')
    report=json.loads((root/'render_manifest.json').read_text(encoding='utf-8'))
    if report.get('schema')!=RENDER_SCHEMA or not report.get('experiment_gates',{}).get('all_K2_vc_budgets_matched'):
        raise ValueError('only a validated frozen K2 note-phrase render is accepted')
    bykey={}
    for row in report.get('renders',[]):
        key=(row['bank'],row['score_phrase'],row['policy'])
        if key in bykey:raise ValueError('duplicate render identity')
        bykey[key]=row
    bank_policies=defaultdict(set)
    for row in report['budget_ledger']:
        if row['policy'].startswith('k2_'):
            bank_policies[row['bank']].add((row['policy'],row['VC_source_ms'],row['total_deduplicated_source_ms']))
    trial_rows=[];private=[];selected={};n=0
    comparisons=[('k2_keep_i_ir','k2_keep_i_i0'),('k2_keep_i_ir','k2_keep_ir_i0'),('k2_keep_i_i0','k2_keep_ir_i0')]
    for bank in focus_banks:
        eligible=bank_policies[bank]
        if len(eligible)!=3 or len({(b,c) for _,b,c in eligible})!=1:
            raise ValueError('selected bank lacks exact equal-budget K2 policies')
        for phrase in ('phrase_a','phrase_b'):
            for first,second in comparisons:
                n+=1;trial_id=f'trial_{n:02d}'
                one,two=bykey[(bank,phrase,first)],bykey[(bank,phrase,second)]
                for row in [one,two]:
                    p=_safe_rel(root,row['rendered_wav'])
                    if sha256(p.read_bytes()).hexdigest()!=row['wav_sha256']:
                        raise ValueError('rendered WAV hash mismatch')
                    with wave.open(str(p)) as w:
                        if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getnframes()<w.getframerate():
                            raise ValueError('invalid blind input WAV')
                # Deterministic blinded side assignment (score hash + trial id).
                swap=int(sha256((report['score_sha256']+trial_id).encode()).hexdigest(),16)%2
                left,right=(two,one) if swap else (one,two)
                idrow={'trial_id':trial_id,'style':one['style'],'pitch':one['pitch'],
                       'phrase':phrase,'A_wav':f'audio/{trial_id}_A.wav',
                       'B_wav':f'audio/{trial_id}_B.wav'}
                trial_rows.append(idrow)
                selected[idrow['A_wav']]=root/left['rendered_wav']
                selected[idrow['B_wav']]=root/right['rendered_wav']
                private.append({**idrow,'A_policy':left['policy'],'B_policy':right['policy'],
                                'A_sha256':left['wav_sha256'],'B_sha256':right['wav_sha256']})
    public={'schema':SCHEMA,'status':'blind_preference_ui_without_listener_results',
            'source_score_sha256':report['score_sha256'],
            'trials':trial_rows,'scientific_effect_inferred':False}
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pw-blind-note-',dir=target.parent) as temp:
        stage=Path(temp)/'listening';stage.mkdir()
        for rel,src in selected.items():
            dst=stage/rel;dst.parent.mkdir(exist_ok=True,parents=True)
            shutil.copyfile(src,dst)
        (stage/'index.html').write_text(_html_page(public),encoding='utf-8')
        (stage/'blind_trials.json').write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        research=stage/'ResearchOnly';research.mkdir()
        with (research/'answer_key_PRIVATE.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(private[0]));writer.writeheader();writer.writerows(private)
        (stage/'README.txt').write_text('Open index.html to hear randomized same-budget six-note phrase pairs.\\n'
            'ResearchOnly/answer_key_PRIVATE.csv is not for listeners.\\n'
            'No recordings or results are uploaded to any server.\\n'
            'These are note-timed synthetic-source previews, NOT independent DiffSinger reference quality tests.\\n',encoding='utf-8')
        if target.exists():raise FileExistsError('output appeared concurrently')
        os.rename(stage,target)
    return public


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description='Source-verified note-phrase K2 blind audio package')
    p.add_argument('--renders',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bank',action='append',dest='banks',help='focus bank id, e.g. mang_D4; repeat to include several')
    args=p.parse_args(argv)
    try:
        result=package_blind_listening(args.renders,args.output,focus_banks=tuple(args.banks or ('mang_D4','lu_C4')))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        p.error(str(exc))
    print(f'Blind K2 listening comparisons: {len(result["trials"])}; no outcomes inferred')
    return 0


if __name__=='__main__':raise SystemExit(main())

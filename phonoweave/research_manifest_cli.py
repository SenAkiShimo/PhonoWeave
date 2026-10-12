"""Build a stable, local-only research manifest from a versioned relative cohort.

Never persist local absolute paths in the Git repository. Never overwrite
an existing manifest without an explicit --force flag.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

DEFAULT_COHORT=Path(__file__).resolve().parent.parent/"research"/"cohorts"/"mandarin_cvv_eight.v1.json"
DEFAULT_OUTPUT=Path.home()/".config"/"phonoweave"/"voicebanks.json"
DEFAULT_SINGERS=Path.home()/"Library"/"OpenUtau"/"Singers"


def generate(cohort:Path=DEFAULT_COHORT, singers:Path=DEFAULT_SINGERS,
             output:Path=DEFAULT_OUTPUT, *,force:bool=False)->Path:
    cohort=json.loads(cohort.expanduser().resolve(strict=True).read_text("utf-8"))
    if cohort.get("schema")!="phonoweave.voicebank_replication_cohort.v1":
        raise ValueError("unrecognized cohort schema")
    root=singers.expanduser().resolve(strict=True)
    banks=[]
    missing=[]
    for entry in cohort["voicebanks"]:
        rel=Path(entry["relative_dir"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("cohort path must remain relative to singer root")
        candidate=(root/rel).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("voicebank path escapes singer root")
        if not candidate.is_dir():
            missing.append(str(rel))
        else:
            banks.append({
                "bank_id":entry["bank_id"],
                "speaker_group_id":entry["speaker_group_id"],
                "path":str(candidate),
            })
    if missing:
        raise FileNotFoundError("missing required singer directories: "+", ".join(missing))
    ids=[b["bank_id"] for b in banks]
    speakers=[b["speaker_group_id"] for b in banks]
    if len(set(ids))!=len(ids) or len(set(speakers))!=len(speakers):
        raise ValueError("duplicate bank or speaker IDs")
    path=output.expanduser()
    if path.exists() and not force:
        # Idempotent: keep the existing file unchanged if it matches.
        existing=json.loads(path.read_text("utf-8"))
        if existing.get("voicebanks")==banks and existing.get("schema")=="phonoweave.voicebank_replication_manifest.v1":
            return path
        raise FileExistsError(f"{path} already exists with different contents; use --force to replace")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({
        "schema":"phonoweave.voicebank_replication_manifest.v1",
        "voicebanks":banks,
    },ensure_ascii=False,indent=2)+"\n","utf-8")
    return path


def main(argv=None):
    p=argparse.ArgumentParser(description="Generate stable local-only PhonoWeave replication manifest")
    p.add_argument("--cohort",type=Path,default=DEFAULT_COHORT)
    p.add_argument("--singers-root",type=Path,default=DEFAULT_SINGERS)
    p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--force",action="store_true")
    a=p.parse_args(argv)
    try:
        path=generate(a.cohort,a.singers_root,a.output,force=a.force)
    except (FileNotFoundError,FileExistsError,ValueError) as exc:
        p.error(str(exc))
    print("Manifest:",path)


if __name__=="__main__":
    main()

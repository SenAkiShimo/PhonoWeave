"""Prepare isolated three-alias singer for manual OpenUtau A/B rendering."""
from __future__ import annotations
import argparse
from pathlib import Path
from .synthesis_relevance_ab_staging import stage_trial

def main(argv=None):
    p=argparse.ArgumentParser(description="Stage matched/cross CV splice aliases in new temporary UTAU voicebank")
    p.add_argument("voicebank",type=Path)
    p.add_argument("--pilot",required=True,type=Path)
    p.add_argument("--output-singer",required=True,type=Path)
    p.add_argument("--context",choices=("plain","rounded"),default="plain")
    p.add_argument("--subbank")
    p.add_argument("--target-id")
    args=p.parse_args(argv)
    result=stage_trial(args.voicebank,args.pilot,args.output_singer,
        context=args.context,subbank=args.subbank,target_id=args.target_id)
    print("Singer:",args.output_singer)
    print("Target:",result["target_alias"],result["pitch_subbank"])
    print("Matched donor:",result["donors"]["matched"]["alias"])
    print("Cross donor:",result["donors"]["cross"]["alias"])
    print("OpenUtau Default phonemizer aliases: PW_N, PW_A, PW_B")
    print("The source singer was not modified; inspect splice boundaries before interpreting audio.")

if __name__=="__main__":
    main()

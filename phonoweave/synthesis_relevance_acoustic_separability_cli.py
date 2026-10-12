"""Cross-voicebank objective Mandarin fricative separability study."""
from __future__ import annotations
import argparse
from pathlib import Path
from .synthesis_relevance_acoustic_separability import evaluate_manifest

def main(argv=None):
    parser=argparse.ArgumentParser(description="WAV-grouped objective fricative-context separability")
    parser.add_argument("manifest",type=Path)
    parser.add_argument("--base",choices=("sh","s","x"),default="sh")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    result=evaluate_manifest(args.manifest,args.output,base=args.base)
    print("Evaluated:",result["evaluated_banks"],"Speaker groups:",result["speaker_group_count"])
    for bank in result["banks"]:
        cv=bank["grouped_cv_all_subbanks"]
        paired=bank["paired_same_WAV_all_subbanks"]
        print(bank["bank_id"],"events",bank["extraction"]["events"],
              "WAVs",bank["extraction"]["physical_wavs"],
              "grouped_BA",cv.get("balanced_accuracy"),"CV_status",cv["status"],
              "paired_WAVs",paired.get("paired_wavs"),
              "paired_p",paired.get("paired_sign_flip_p"))
    print("Saved:",args.output)

if __name__=="__main__":
    main()

"""Prepare a randomized-label OpenUtau consonant-cut pilot singer."""
from __future__ import annotations
import argparse
from pathlib import Path
from .synthesis_relevance_boundary_ablation import stage_boundary_ablation

def main(argv=None):
    p=argparse.ArgumentParser(description="Stage six blinded same/cross context OTO cut offsets")
    p.add_argument("voicebank",type=Path)
    p.add_argument("--previous-singer",type=Path,required=True)
    p.add_argument("--output-singer",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261011)
    a=p.parse_args(argv)
    result=stage_boundary_ablation(a.voicebank,a.previous_singer,
                                    a.output_singer,seed=a.seed)
    print("Singer:",result["output"])
    print("Test aliases:",", ".join(result["aliases"]))
    print("Reference:",result["reference_alias"])
    print("Reveal key (keep closed until listening):",result["reveal_key"])
    print("Left duration:",round(result["left_ms"],2),"ms")
    print("Review provisional OTO consonant cut positions before interpreting results.")

if __name__=="__main__":
    main()

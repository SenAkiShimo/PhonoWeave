from __future__ import annotations
import argparse
from pathlib import Path
from .calibration_carrier_trigger_review import review_trigger_segments

def main(argv=None):
    p=argparse.ArgumentParser(description="Export two flagged a-carrier trigger clips for manual listening")
    p.add_argument("session",type=Path)
    result=review_trigger_segments(p.parse_args(argv).session)
    for row in result["rows"]:
        print(f"{row['wav']} {row['spoken_pattern']}: candidate={row['candidate_ms']}ms, "
              f"carrier_cue={row['nominal_carrier_cue_ms']}ms, previous={row['previous_token']}; "
              f"clip={row['clip_filename']}")
    print("Saved: analysis/calibration_carrier_trigger_review_v1.8.json")
if __name__=="__main__": main()

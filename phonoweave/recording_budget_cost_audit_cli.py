"""Read-only OTO segment-yield and recording-duration cost audit."""
from __future__ import annotations
import argparse
from pathlib import Path
from .recording_budget_cost_audit import audit_cost_manifest

def main(argv=None):
    parser=argparse.ArgumentParser(description="Audit segment and duration burden of frozen WAV selection")
    parser.add_argument("manifest",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    report=audit_cost_manifest(args.manifest,args.output)
    print("Audited:",report["audited_banks"],"Excluded:",report["excluded_banks"])
    for bank in report["banks"]:
        if bank["status"] != "audited":
            print(bank["bank_id"],"EXCLUDED",bank["reason"])
            continue
        print(bank["bank_id"],"WAVs",bank["unique_eligible_wavs"],
              "segments",bank["eligible_oto_acoustic_segments"],
              "duration_sec",bank["total_eligible_wav_duration_seconds"])
        for row in bank["summary"]:
            if row["budget_fraction"] in (0.25,0.50,0.75):
                print(" ",row["policy"],"{:.0%}".format(row["budget_fraction"]),
                      "segments={:.1%}".format(row["mean_segment_fraction"]),
                      "duration={:.1%}".format(row["mean_duration_fraction"]))
    print("Saved:",args.output)

if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
from pathlib import Path
from .calibration_noise_floor_contrast import run_contrast_audit

def main(argv=None):
    p=argparse.ArgumentParser(description="Audit high-band noise-floor contrast")
    p.add_argument("session",type=Path)
    r=run_contrast_audit(p.parse_args(argv).session)
    print("Session:",r["session_id"])
    for scenario,s in r["summary"].items():
        print(f"{scenario}: paired={s['paired']}/{s['targets']} "
              f"median_peak_rise_paired={s['median_high_rise_peak_db_paired']}dB "
              f"median_peak_rise_unpaired={s['median_high_rise_peak_db_unpaired']}dB")
    print("Failures:",r["failures"])
    print("Saved: analysis/calibration_noise_floor_contrast_v2.4.json")
if __name__=="__main__":
    main()

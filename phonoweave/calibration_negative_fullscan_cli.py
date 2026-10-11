from __future__ import annotations
import argparse
from pathlib import Path
from .calibration_negative_fullscan import fullscan

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("root",nargs="?",type=Path,default=Path.home()/"Downloads"/"PhonoWeaveCalibration")
    result=fullscan(p.parse_args(argv).root)
    for category,metrics in result["summary"].items():
        print(f"{category}: {metrics}")
    print("Failures:",result["failures"])
    print("Saved: negative_controls/negative_fullscan_v2.0.json")
if __name__=="__main__":
    main()

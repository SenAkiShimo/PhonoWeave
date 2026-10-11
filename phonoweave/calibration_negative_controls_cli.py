from __future__ import annotations
import argparse
from pathlib import Path
from .calibration_negative_controls import write_negative_corpus_report

def main(argv=None):
    p=argparse.ArgumentParser(description="Screen independent negative-control recordings")
    p.add_argument("root",type=Path,nargs="?",default=Path.home()/"Downloads"/"PhonoWeaveCalibration")
    result=write_negative_corpus_report(p.parse_args(argv).root)
    print("Recordings:",result["total_recordings"])
    for category,stats in result["summary"].items():
        print(category,stats)
    print("Failures:",result["failures"])
    print("Report:",result["output"])
if __name__=="__main__":main()

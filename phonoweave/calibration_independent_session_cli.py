"""CLI for frozen independent-session onset evaluation."""
from __future__ import annotations
import argparse
from pathlib import Path
from .calibration_independent_session import evaluate_independent_session, ALLOWED_DEVICES, ALLOWED_ENVIRONMENTS

def main(argv=None):
    p=argparse.ArgumentParser(description="Evaluate frozen onset strategies on a new recording session")
    p.add_argument("session",type=Path)
    p.add_argument("--development-session",type=Path,required=True)
    p.add_argument("--device",choices=ALLOWED_DEVICES,required=True)
    p.add_argument("--environment",choices=ALLOWED_ENVIRONMENTS,required=True)
    args=p.parse_args(argv)
    r=evaluate_independent_session(args.session,args.development_session,device=args.device,environment=args.environment)
    print("Session:",r["session_id"])
    print("Targets:",r["targets"],"onset labels:",r["annotated_onsets"],"CV labels:",r["annotated_cv_boundaries"])
    print("Paired:",r["paired_candidates"])
    for name in ("baseline","always_backtrace","selective"):
        s=r[name]
        print(f"{name}: matched={s['matched']}/{s['eligible']} MAE={s['conditional_mae_ms']}ms P90={s['conditional_p90_ms']}ms >40ms={s['errors_over_40ms']}")
    print("CV:",r["cv_boundary"])
    print("Paired comparison:",r["paired_comparison"])
    print("Saved: analysis/calibration_independent_session_v3.0.json")
if __name__=="__main__":
    main()

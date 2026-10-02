"""Run known faults in synthetic disposable classifiers, without user data."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from app.services.fault_validation import run_fault_suite
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True, help='New output directory; existing files are never replaced')
args = parser.parse_args()
report = run_fault_suite(args.output)
print(f"{sum(r['passed'] for r in report['cases'])}/{len(report['cases'])} fault/control cases passed. Report: {args.output / 'fault-validation.html'}")
raise SystemExit(0 if report['passed'] else 1)

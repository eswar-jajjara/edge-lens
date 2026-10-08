"""Create an offline comparison from EdgeLens exported report JSON files."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from app.services.report_comparison import build_comparison, comparison_html, comparison_csv

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reports', type=Path, nargs='+', required=True)
parser.add_argument('--references', type=Path)
parser.add_argument('--title', default='EdgeLens conversion comparison')
parser.add_argument('--output', type=Path, required=True, help='New comparison directory; existing files are not overwritten')
args = parser.parse_args()
if args.output.exists():
    parser.error('Choose a new output directory to preserve previous evidence.')
def read(path):
    if path.stat().st_size > 8 * 1024 * 1024:
        parser.error('Each JSON file must be at most 8 MiB.')
    return json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite JSON values are not supported.')))
try:
    result = build_comparison([read(p) for p in args.reports], read(args.references) if args.references else None, args.title)
    html, csv = comparison_html(result), comparison_csv(result)
except (ValueError, OSError, KeyError, TypeError) as exc:
    parser.error(str(exc))
args.output.mkdir(parents=True)
(args.output / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
(args.output / 'comparison.html').write_text(html, encoding='utf-8')
(args.output / 'comparison.csv').write_text(csv, encoding='utf-8-sig')
print('Saved: ' + str(args.output.resolve() / 'comparison.html'))

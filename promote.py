"""
promote.py — Champion/challenger promotion (CLI)
=================================================
  python promote.py                 # compare challengers with the champion, move the alias if one wins
  python promote.py --dry-run       # show the decision without changing the registry
  python promote.py --min-gain 1.0  # require a bigger validation-RMSE improvement

Rules live in registry.py.
"""

import argparse
import json
import sys

from config import PROMOTION_MIN_GAIN
from registry import format_decision, promote


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Promote the best RUL model to champion")
    p.add_argument("--min-gain", type=float, default=PROMOTION_MIN_GAIN,
                   help=f"validation RMSE cycles a challenger must win by (default {PROMOTION_MIN_GAIN})")
    p.add_argument("--dry-run", action="store_true", help="decide but do not move the champion alias")
    p.add_argument("--decision-out", default=None, help="write the decision as JSON to this path")
    args = p.parse_args(argv)

    decision = promote(min_gain=args.min_gain, dry_run=args.dry_run)
    print(format_decision(decision))
    if args.decision_out:
        with open(args.decision_out, "w") as f:
            json.dump(decision, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

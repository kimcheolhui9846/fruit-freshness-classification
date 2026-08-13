"""Apply the frozen Phase 9.6 decision rule to two OOF metric files.

The thresholds are the protocol's, fixed before the run. This script computes
the verdict so it cannot be decided after the number is seen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

ADVANCE_MACRO_F1 = 0.9112
TOP1_GUARDRAIL = 0.9466
NEXT_PHASE_ON_ADVANCE = "Owner decision, informed by the result"
NEXT_PHASE_ON_HOLD = "H1 exhausted; Phase 9.7 is H2 augmentation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the frozen loss-001 decision rule."
    )
    parser.add_argument(
        "--baseline-metrics",
        required=True,
        help="Baseline development_oof_metrics.json.",
    )
    parser.add_argument(
        "--experiment-metrics",
        required=True,
        help="loss-001 development_oof_metrics.json.",
    )
    parser.add_argument("--output", required=True, help="Path for the verdict JSON.")
    return parser


def apply_decision(baseline_metrics: dict, experiment_metrics: dict) -> dict:
    """Compute the verdict. Both conditions must hold; neither is negotiable."""
    base = baseline_metrics["metrics"]
    experiment = experiment_metrics["metrics"]
    clears_macro = experiment["macro_f1"] >= ADVANCE_MACRO_F1
    clears_top1 = experiment["top1_accuracy"] >= TOP1_GUARDRAIL
    advance = clears_macro and clears_top1
    return {
        "outcome": "ADVANCE" if advance else "NOT_ADVANCED",
        "advance_macro_f1_threshold": ADVANCE_MACRO_F1,
        "top1_guardrail": TOP1_GUARDRAIL,
        "baseline_macro_f1": base["macro_f1"],
        "experiment_macro_f1": experiment["macro_f1"],
        "macro_f1_delta": experiment["macro_f1"] - base["macro_f1"],
        "baseline_top1": base["top1_accuracy"],
        "experiment_top1": experiment["top1_accuracy"],
        "top1_delta": experiment["top1_accuracy"] - base["top1_accuracy"],
        "clears_macro_f1": clears_macro,
        "clears_top1_guardrail": clears_top1,
        "next_phase": NEXT_PHASE_ON_ADVANCE if advance else NEXT_PHASE_ON_HOLD,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(
            f"{output} already exists; refusing to overwrite a recorded verdict."
        )

    verdict = apply_decision(
        json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8")),
        json.loads(Path(args.experiment_metrics).read_text(encoding="utf-8")),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Outcome: {verdict['outcome']}")
    print(
        f"Macro F1 {verdict['experiment_macro_f1']:.4f} "
        f"(delta {verdict['macro_f1_delta']:+.4f}, threshold {ADVANCE_MACRO_F1})"
    )
    print(
        f"Top-1    {verdict['experiment_top1']:.4f} "
        f"(delta {verdict['top1_delta']:+.4f}, guardrail {TOP1_GUARDRAIL})"
    )
    print(f"Next: {verdict['next_phase']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The frozen measurement floor for post-holdout development comparisons.

Every constant here is measured rather than chosen. They come from the three
Phase 9.6a replicates of the identical baseline recipe on the identical
frozen folds, which are the only direct evidence this project has about how
much its own measurements move between runs.

Seeding does not lower these numbers. Fixing a seed pins one draw from the
same distribution; it does not narrow the distribution. The replicates
therefore remain the right estimate of how far a result would move at a
different seed, even though they predate seeding.

The protocol is docs/postholdout-measurement-floor-protocol.md.
"""

from __future__ import annotations


REPLICATE_MACRO_F1 = (0.901167, 0.912041, 0.901858)
REPLICATE_TOP1 = (0.956598, 0.957936, 0.956016)

MACRO_F1_MEAN = 0.905022
MACRO_F1_STDEV = 0.006089

MDE_MACRO_F1 = 0.012177
MDE_TOP1 = 0.001969
MDE_FRESHPOTATO_F1 = 0.147833

VALIDITY_ENVELOPE = (0.892845, 0.917199)

ADVANCED = "ADVANCED"
BELOW_RESOLUTION = "BELOW_RESOLUTION"
REGRESSED = "REGRESSED"


def classify_effect(candidate: float, baseline: float, *, mde: float) -> dict:
    """Classify a single-run difference against a frozen minimum detectable effect.

    Three verdicts, because two would not cover the line. An improvement at
    or above the floor advances; a decline at or beyond it is a measured
    regression; anything between is a result this project cannot separate
    from the seed it happened to draw.
    """
    if mde <= 0:
        raise ValueError("mde must be positive; a zero floor advances every gain.")

    difference = candidate - baseline
    if difference >= mde:
        verdict = ADVANCED
    elif difference <= -mde:
        verdict = REGRESSED
    else:
        verdict = BELOW_RESOLUTION

    return {
        "verdict": verdict,
        "difference": difference,
        "mde": mde,
        "candidate": candidate,
        "baseline": baseline,
    }


def within_validity_envelope(macro_f1: float) -> bool:
    """Test a deterministic baseline against the pre-registered envelope.

    Falling inside is consistent with A_STRICT's deterministic kernel
    selection not having changed results materially. It is not proof: an
    envelope built from three samples is wide, and a real shift smaller than
    the envelope would pass unnoticed.
    """
    low, high = VALIDITY_ENVELOPE
    return low <= macro_f1 <= high

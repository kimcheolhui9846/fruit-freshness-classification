"""Compare two completed training output directories for bit-exactness.

Phase 9.7 verification. This script trains nothing, constructs no model, and
touches no dataset. It reads the trusted local training state each run wrote
and reports whether the two runs are identical. The frozen adoption ladder is
in docs/postholdout-determinism-protocol.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


TRAINING_STATE_FILENAME = "training_state.pt"
_COMPARED_FIELDS = ("model_state_dict", "ema_state_dict", "completed_fold_histories")


def digest_tensor_mapping(mapping: dict) -> str:
    """Hash a state dict in sorted key order, including dtype and shape.

    Key length is hashed alongside the key so that {"ab": x, "c": y} cannot
    collide with {"a": x, "bc": y}, and shape is hashed separately from the
    buffer so that a reshaped tensor is not read as unchanged.
    """
    hasher = hashlib.sha256()
    for key in sorted(mapping):
        value = mapping[key]
        hasher.update(str(len(key)).encode("utf-8"))
        hasher.update(key.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            hasher.update(str(value.dtype).encode("utf-8"))
            hasher.update(str(tuple(value.shape)).encode("utf-8"))
            hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            hasher.update(repr(value).encode("utf-8"))
    return hasher.hexdigest()


def digest_history(histories) -> str:
    """Hash recorded per-fold metric histories with full float precision."""
    encoded = json.dumps(histories, sort_keys=True, default=repr)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_state(directory: Path) -> dict:
    path = Path(directory) / TRAINING_STATE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"No training state in {directory}: {path.name} is absent."
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def compare_runs(first_directory, second_directory) -> dict:
    """Compare two completed runs field by field, in a fixed order."""
    first = _load_state(Path(first_directory))
    second = _load_state(Path(second_directory))

    digests = {}
    first_mismatch = None
    for field in _COMPARED_FIELDS:
        if field == "completed_fold_histories":
            pair = [digest_history(first.get(field)), digest_history(second.get(field))]
        else:
            pair = [
                digest_tensor_mapping(first.get(field, {})),
                digest_tensor_mapping(second.get(field, {})),
            ]
        digests[field] = pair
        if first_mismatch is None and pair[0] != pair[1]:
            first_mismatch = field

    return {
        "bit_exact": first_mismatch is None,
        "first_directory": str(first_directory),
        "second_directory": str(second_directory),
        "model_digests": digests["model_state_dict"],
        "ema_digests": digests["ema_state_dict"],
        "history_digests": digests["completed_fold_histories"],
        "first_mismatch": first_mismatch,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two completed training runs for bit-exactness.",
    )
    parser.add_argument("--first", required=True, help="First run output directory.")
    parser.add_argument("--second", required=True, help="Second run output directory.")
    parser.add_argument("--output", required=True, help="Verification record JSON path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(
            f"Verification record already exists: {output_path}. "
            "A verification result is evidence and is not overwritten."
        )

    result = compare_runs(Path(args.first), Path(args.second))
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if result["bit_exact"]:
        print("BIT_EXACT: the two runs are identical.")
        return 0
    print(f"NOT_BIT_EXACT: first mismatching field is {result['first_mismatch']}.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

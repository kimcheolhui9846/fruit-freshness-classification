"""Materialize the blinded Phase 9.5 review set from the frozen protocol.

Writes position-named images and a sealed key. No model is constructed, and no
locked-test index may enter the review set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.datasets.label_audit import select_review_set

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTROL_SAMPLE_SEED = 20260813
PRESENTATION_ORDER_SEED = 20260813
SUBJECT_COUNT = 347
CONTROL_COUNT = 150
SHEET_COLUMNS = 4
SHEET_ROWS = 4
CELL_PIXELS = 256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize the blinded Phase 9.5 label audit review set."
    )
    parser.add_argument(
        "--split-manifest",
        default="configs/splits/deep3-postholdout-research-01.json",
        help="Frozen Phase 9 split manifest.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Empty local-only directory for the review set and sealed key.",
    )
    return parser


def partition_outputs(output_dir: str | Path) -> dict[str, Path]:
    """Place the key outside review/ so blinding survives a careless reviewer."""
    root = Path(output_dir)
    return {
        "root": root,
        "review_dir": root / "review",
        "sheet_dir": root / "review" / "contact_sheets",
        "key_path": root / "review_set_key.json",
        "template_path": root / "judgment_template.csv",
    }


def _load_development(split_manifest: Path) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    return (
        np.asarray(manifest["development_indices"], dtype=np.int64),
        np.asarray(manifest["locked_test_indices"], dtype=np.int64),
    )


def _write_contact_sheets(images, positions, sheet_dir: Path) -> None:
    from PIL import Image, ImageDraw

    sheet_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = SHEET_COLUMNS * SHEET_ROWS
    for sheet_index in range(0, len(positions), per_sheet):
        chunk = positions[sheet_index : sheet_index + per_sheet]
        sheet = Image.new(
            "RGB",
            (SHEET_COLUMNS * CELL_PIXELS, SHEET_ROWS * (CELL_PIXELS + 24)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for cell, position in enumerate(chunk):
            thumb = images[position].convert("RGB").resize((CELL_PIXELS, CELL_PIXELS))
            x = (cell % SHEET_COLUMNS) * CELL_PIXELS
            y = (cell // SHEET_COLUMNS) * (CELL_PIXELS + 24)
            sheet.paste(thumb, (x, y))
            draw.text((x + 4, y + CELL_PIXELS + 4), f"{position:03d}", fill="black")
        sheet.save(sheet_dir / f"sheet_{sheet_index // per_sheet:02d}.jpg", quality=92)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = partition_outputs(REPOSITORY_ROOT / args.output_dir)
    if outputs["root"].exists() and any(outputs["root"].iterdir()):
        raise SystemExit(f"Output directory is not empty: {outputs['root']}")

    from scripts.freeze_postholdout_split import _reconstruct_canonical_pool_with_images

    dataset, labels, class_names, _ = _reconstruct_canonical_pool_with_images()
    development_indices, locked_test_indices = _load_development(
        REPOSITORY_ROOT / args.split_manifest
    )

    selection = select_review_set(
        development_indices,
        labels[development_indices],
        class_names,
        control_seed=CONTROL_SAMPLE_SEED,
        order_seed=PRESENTATION_ORDER_SEED,
        subject_count=SUBJECT_COUNT,
        control_count=CONTROL_COUNT,
    )
    presentation = selection["presentation"]

    locked = set(locked_test_indices.tolist())
    leaked = sorted(locked.intersection(presentation.tolist()))
    if leaked:
        raise SystemExit(f"Locked-test indices reached the review set: {leaked[:5]}")

    outputs["review_dir"].mkdir(parents=True, exist_ok=True)
    images = {}
    for position, source_index in enumerate(presentation.tolist()):
        image = dataset[int(source_index)]["image"]
        images[position] = image
        image.convert("RGB").save(outputs["review_dir"] / f"{position:03d}.jpg", quality=95)

    _write_contact_sheets(images, list(range(len(presentation))), outputs["sheet_dir"])

    subject = set(selection["subject_indices"].tolist())
    key = {
        "schema_version": 1,
        "control_sample_seed": CONTROL_SAMPLE_SEED,
        "presentation_order_seed": PRESENTATION_ORDER_SEED,
        "review_set_count": int(len(presentation)),
        "presentation_indices_sha256": hashlib.sha256(
            presentation.astype("<i8").tobytes()
        ).hexdigest(),
        "entries": [
            {
                "position": position,
                "source_index": int(source_index),
                "group": "SUBJECT" if int(source_index) in subject else "CONTROL",
            }
            for position, source_index in enumerate(presentation.tolist())
        ],
    }
    outputs["key_path"].write_text(json.dumps(key, indent=2, sort_keys=True), encoding="utf-8")

    with outputs["template_path"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["position", "judgment"])
        for position in range(len(presentation)):
            writer.writerow([f"{position:03d}", ""])

    print(f"Review set: {len(presentation)} images")
    print(f"Sealed key: {outputs['key_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

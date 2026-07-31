"""Label metadata persistence helpers."""

import json
import os


def save_label_names(label_names, output_dir: str) -> None:
    """Save ordered label names with the notebook's existing JSON settings."""
    with open(os.path.join(output_dir, "label_names.json"), "w", encoding="utf-8") as file:
        json.dump(label_names, file, ensure_ascii=False)

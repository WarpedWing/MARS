# rubric_loader.py
import json
import re
from collections import defaultdict
from pathlib import Path


def load_rubrics(rubric_dir: Path):
    """Load all rubric JSONs and build a token search index."""
    rubrics = {}
    token_index = defaultdict(set)

    for f in rubric_dir.glob("*.rubric.json"):
        try:
            data = json.loads(f.read_text())
            sys_hint = data.get("system_hint", "unknown").lower()
            hw_hint = data.get("hardware_hint", "unknown").lower()
            rubrics[f.stem] = data

            for table in data.get("tables", {}):
                tokens = re.findall(r"[A-Za-z0-9]+", table.lower())
                for token in tokens:
                    token_index[token].add((f.stem, sys_hint, hw_hint, table))
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    print(f"Loaded {len(rubrics)} rubrics, indexed {len(token_index)} unique tokens")
    return rubrics, token_index

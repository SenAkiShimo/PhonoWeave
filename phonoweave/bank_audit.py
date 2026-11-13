from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

from .mandarin import normalize_alias
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


KINDS = ("cv", "initial", "vc", "end")
_WORD = re.compile(r"^[a-zü0-9]+$")
_END_MARK = "r"


def _layer_name(root: Path, oto_path: Path) -> str:
    directory = oto_path.parent
    return "." if directory == root else str(directory.relative_to(root))


def classify(alias: str) -> tuple[str, tuple[str, ...]] | None:
    tokens = alias.strip().lower().split()
    if not tokens or not all(t == "-" or _WORD.match(t) for t in tokens):
        return None
    if len(tokens) == 1:
        return ("cv", (tokens[0],))
    if len(tokens) == 2:
        first, second = tokens
        if first == "-":
            return ("initial", (second,))
        if second == _END_MARK:
            return ("end", (first,))
        return ("vc", (first, second))
    return None


def audit_bank(root: Path) -> dict:
    root = root.expanduser().resolve()
    entries, _ = load_voicebank(root)
    affixes = affix_pairs(load_prefix_maps(root))

    items: dict[str, dict[str, set[tuple[str, ...]]]] = defaultdict(
        lambda: {kind: set() for kind in KINDS}
    )
    unrecognized: Counter = Counter()
    for entry in entries:
        layer = _layer_name(root, entry.oto_path)
        parsed = classify(normalize_alias(entry.alias, affixes))
        if parsed is None:
            unrecognized[layer] += 1
            continue
        kind, key = parsed
        items[layer][kind].add(key)

    layers = sorted(items)
    census = {layer: {kind: len(items[layer][kind]) for kind in KINDS} for layer in layers}

    union = {kind: set().union(*(items[layer][kind] for layer in layers)) if layers else set() for kind in KINDS}
    cross_layer = {
        kind: {layer: sorted(union[kind] - items[layer][kind]) for layer in layers}
        for kind in KINDS
    }

    cv_all = {key[0] for key in union["cv"]}
    initial_all = {key[0] for key in union["initial"]}
    finals = {key[0] for key in union["vc"]} | {key[0] for key in union["end"]}
    labels: dict[str, set[str]] = defaultdict(set)
    for final, label in union["vc"]:
        labels[label].add(final)
    label_rows = [
        {
            "label": label,
            "finals": len(covered),
            "missing_finals": sorted(finals - covered),
        }
        for label, covered in sorted(labels.items(), key=lambda item: (-len(item[1]), item[0]))
    ]
    systematic = [row for row in label_rows if row["finals"] * 2 >= len(finals)]
    partial = [row for row in label_rows if row["finals"] * 2 < len(finals)]

    return {
        "layers": layers,
        "census": census,
        "unrecognized": dict(unrecognized),
        "finals": sorted(finals),
        "systematic_labels": systematic,
        "partial_labels": [
            {"label": row["label"], "finals": row["finals"]} for row in partial
        ],
        "cv_without_initial": sorted(cv_all - initial_all),
        "initial_without_cv": sorted(initial_all - cv_all),
        "finals_without_end": sorted(finals - {key[0] for key in union["end"]}),
        "cross_layer_missing": {
            kind: {layer: [" ".join(key) for key in rows] for layer, rows in per.items() if rows}
            for kind, per in cross_layer.items()
        },
    }


def format_audit(audit: dict, limit: int = 30) -> str:
    lines = ["Alias census (distinct entries per pitch layer):"]
    for layer in audit["layers"]:
        row = audit["census"][layer]
        lines.append(
            f"  {layer}: CV={row['cv']}, initial CV={row['initial']}, "
            f"VC={row['vc']}, end={row['end']}"
        )
    if audit["unrecognized"]:
        lines.append(f"  Unrecognized aliases: {audit['unrecognized']}")

    lines.append("")
    lines.append(f"Finals seen in VC or end entries ({len(audit['finals'])}): {' '.join(audit['finals'])}")
    lines.append("VC labels covering at least half of the finals:")
    for row in audit["systematic_labels"]:
        missing = row["missing_finals"]
        tail = f", missing finals: {' '.join(missing)}" if missing else ""
        lines.append(f"  {row['label']}: {row['finals']} finals{tail}")
    if audit["partial_labels"]:
        lines.append("VC labels covering fewer than half of the finals:")
        for row in audit["partial_labels"][:limit]:
            lines.append(f"  {row['label']}: {row['finals']} finals")

    lines.append("")
    lines.append(f"Syllables with a plain CV but no initial CV: {' '.join(audit['cv_without_initial']) or 'none'}")
    lines.append(f"Syllables with an initial CV but no plain CV: {' '.join(audit['initial_without_cv']) or 'none'}")
    lines.append(f"Finals without an end entry: {' '.join(audit['finals_without_end']) or 'none'}")

    lines.append("")
    lines.append("Entries present in some pitch layers but missing from others:")
    any_gap = False
    for kind in KINDS:
        for layer, rows in audit["cross_layer_missing"][kind].items():
            any_gap = True
            shown = " | ".join(rows[:limit])
            more = f" (+{len(rows) - limit} more)" if len(rows) > limit else ""
            lines.append(f"  {layer} {kind}: {len(rows)} missing: {shown}{more}")
    if not any_gap:
        lines.append("  none")
    return "\n".join(lines)

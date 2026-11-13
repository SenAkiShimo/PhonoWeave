from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .bank_audit import audit_bank
from .mandarin import classify_alias, structure_for
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


def derive_final_classes(root: Path) -> dict[tuple[str, str], str]:
    """Map each (onset, final) to the VC final class the voicebank itself uses.

    The class is the WAV file name that holds the syllable's CV entries, kept only
    when it is one of the finals appearing in the voicebank's VC and end entries.
    """
    root = root.expanduser().resolve()
    vc_finals = set(audit_bank(root)["finals"])
    entries, _ = load_voicebank(root)
    affixes = affix_pairs(load_prefix_maps(root))

    votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for entry in entries:
        observation = classify_alias(entry, affixes)
        if observation is None:
            continue
        structure = structure_for(observation)
        stem = entry.wav_path.stem.lower()
        if stem in vc_finals:
            votes[(structure.onset or "", structure.final)][stem] += 1

    mapping: dict[tuple[str, str], str] = {
        key: counter.most_common(1)[0][0] for key, counter in votes.items()
    }
    return mapping


def summarize_final_classes(mapping: dict[tuple[str, str], str]) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for (_, final), tail in mapping.items():
        grouped[final].add(tail)
    return {final: sorted(tails) for final, tails in sorted(grouped.items())}

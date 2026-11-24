from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .bank_audit import audit_bank
from .final_table import standard_final_class
from .mandarin import classify_alias, structure_for
from .oto import load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


def derive_final_classes(root: Path) -> dict[tuple[str, str], str]:
    """Map each (onset, final) to the VC final class the voicebank itself uses.

    The class comes from the standard Mandarin grouping (ia and ua with a, ian with
    en0, apical i with ir or i0, and so on) when the voicebank's own VC and end entries
    use that class. Otherwise the WAV file name that holds the syllable's CV entries
    is used, if it is one of the voicebank's finals.
    """
    root = root.expanduser().resolve()
    vc_finals = set(audit_bank(root)["finals"])
    entries, _ = load_voicebank(root)
    affixes = affix_pairs(load_prefix_maps(root))

    votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        observation = classify_alias(entry, affixes)
        if observation is None:
            continue
        structure = structure_for(observation)
        stem = entry.wav_path.stem.lower()
        seen.add((structure.onset or "", structure.final))
        if stem in vc_finals:
            votes[(structure.onset or "", structure.final)][stem] += 1

    mapping: dict[tuple[str, str], str] = {}
    for onset, final in seen:
        standard = standard_final_class(onset, final)
        if standard in vc_finals:
            mapping[(onset, final)] = standard
        elif votes.get((onset, final)):
            mapping[(onset, final)] = votes[(onset, final)].most_common(1)[0][0]
    return mapping


def summarize_final_classes(mapping: dict[tuple[str, str], str]) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for (_, final), tail in mapping.items():
        grouped[final].add(tail)
    return {final: sorted(tails) for final, tails in sorted(grouped.items())}

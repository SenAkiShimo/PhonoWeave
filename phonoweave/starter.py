from __future__ import annotations

from pathlib import Path

from .final_table import standard_final_class
from .mandarin import classify_alias
from .oto import OtoEntry
from .reclist import STOP_ONSETS, Syllable, _with_er, syllables_from_observations
from .synthesis_inventory import SynthesisInventory, SynthesisUnit


_TABLE = {
    "b": "a o ai ei ao an en ang eng i ian iao ie in ing u",
    "p": "a o ai ei ao ou an en ang eng i ian iao ie in ing u",
    "m": "a o e ai ei ao ou an en ang eng i ian iao ie iu in ing u",
    "f": "a o ei ou an en ang eng u",
    "d": "a e ai ei ao ou an en ang eng ong i ia ian iao ie iu ing u uo ui uan un",
    "t": "a e ai ao ou an ang eng ong i ian iao ie ing u uo ui uan un",
    "n": "a e ai ei ao ou an en ang eng ong i ian iang iao ie in iu ing u uo uan v ve",
    "l": "a o e ai ei ao ou an ang eng ong i ia ian iang iao ie in iu ing u uo uan un v ve",
    "g": "a e ai ei ao ou an en ang eng ong u ua uo uai ui uan un uang",
    "k": "a e ai ao ou an en ang eng ong u ua uo uai ui uan un uang",
    "h": "a e ai ei ao ou an en ang eng ong u ua uo uai ui uan un uang",
    "j": "i ia ian iang iao ie in iu ing iong u uan ue un",
    "q": "i ia ian iang iao ie in iu ing iong u uan ue un",
    "x": "i ia ian iang iao ie in iu ing iong u uan ue un",
    "zh": "a e ai ei ao ou an en ang eng ong i u ua uo uai ui uan un uang",
    "ch": "a e ai ao ou an en ang eng ong i u uo uai ui uan un uang",
    "sh": "a e ai ei ao ou an en ang eng i u ua uo uai ui uan un uang",
    "r": "e ao ou an en ang eng ong i u ua uo ui uan un",
    "z": "a e ai ei ao ou an en ang eng ong i u uo ui uan un",
    "c": "a e ai ao ou an en ang eng ong i u uo ui uan un",
    "s": "a e ai ao ou an en ang eng ong i u uo ui uan un",
    "y": "a e ao ou an ang ong i in ing u uan ue un",
    "w": "a o ai ei an en ang eng u",
}

def _entries() -> list[OtoEntry]:
    path = Path(".")
    return [
        OtoEntry(path, path, f"{onset}{final}", 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        for onset, finals in _TABLE.items()
        for final in finals.split()
    ]


def starter_syllables() -> tuple[Syllable, ...]:
    observations = [
        observation
        for entry in _entries()
        if (observation := classify_alias(entry)) is not None
    ]
    base = _with_er(syllables_from_observations(observations))
    return tuple(
        Syllable(item.text, item.onset, item.final, standard_final_class(item.onset, item.final))
        for item in base
    )


def _unit(unit_id: str, base: str, *contexts: str) -> SynthesisUnit:
    return SynthesisUnit(
        id=unit_id,
        base_unit=base,
        contexts=tuple(contexts),
        source_group=unit_id,
        canonical_context=None,
        decision="default_rule",
        confidence="prior",
        acoustic_evidence="not_tested",
        synthesis_evidence="not_tested",
    )


def starter_inventory() -> SynthesisInventory:
    units: list[SynthesisUnit] = []
    for base in ("m", "n", "l"):
        units.append(_unit(f"{base}_front", base, "internal:i_series", "initial:i_series"))
        units.append(_unit(f"{base}_plain", base, "plain", "rounded"))
    for base in ("s", "sh"):
        units.append(_unit(f"{base}_plain", base, "plain"))
        units.append(_unit(f"{base}_rounded", base, "rounded"))
    units.append(_unit("h_rounded", "h", "internal:rounded", "initial:rounded"))
    units.append(_unit("h_other", "h", "internal:other", "initial:other"))
    for base in ("j", "q", "x"):
        units.append(_unit(f"{base}_front", base, "plain"))
        units.append(_unit(f"{base}_rounded", base, "rounded"))
    return SynthesisInventory(
        speaker_id="starter",
        language="zh",
        source_voicebank=Path("."),
        units=tuple(units),
    )


def starter_shared_connection(stop_vc: str) -> frozenset[str]:
    return STOP_ONSETS if stop_vc == "shared" else frozenset()

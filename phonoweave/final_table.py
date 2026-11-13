from __future__ import annotations

_CLASSES = {
    "ia": "a", "ua": "a", "uai": "ai", "uan": "an", "iang": "ang", "uang": "ang",
    "iao": "ao", "ie": "e0", "ve": "e0", "ue": "e0", "ui": "ei", "un": "en",
    "ian": "en0", "van": "en0", "ueng": "eng", "uo": "o", "iong": "ong", "iu": "ou",
}
_RETROFLEX = {"zh", "ch", "sh", "r"}
_DENTAL = {"z", "c", "s"}


def standard_final_class(onset: str, final: str) -> str:
    if final == "i":
        if onset in _RETROFLEX:
            return "ir"
        if onset in _DENTAL:
            return "i0"
    return _CLASSES.get(final, final)

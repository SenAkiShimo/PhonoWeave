from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationPrompt:
    base_unit: str
    context_family: str
    syllable: str
    role_scope: str
    carrier: str
    repeats: int

    @property
    def spoken_pattern(self) -> str:
        if self.role_scope == "internal":
            return f"{self.carrier} {self.syllable} {self.carrier}"
        return f"{self.syllable} {self.carrier} {self.syllable} {self.carrier}"


@dataclass(frozen=True)
class CalibrationProtocol:
    name: str
    version: str
    prompts: tuple[CalibrationPrompt, ...]
    automatic_supplement_rounds: int
    stop_rule: str


_BASE_PROMPTS = (
    CalibrationPrompt("zh", "plain", "zha", "all", "a", 3),
    CalibrationPrompt("zh", "rounded", "zhua", "all", "a", 3),
    CalibrationPrompt("ch", "plain", "cha", "all", "a", 3),
    CalibrationPrompt("ch", "rounded", "chua", "all", "a", 3),
    CalibrationPrompt("z", "plain", "za", "all", "a", 3),
    CalibrationPrompt("z", "rounded", "zu", "all", "a", 3),
    CalibrationPrompt("c", "plain", "cai", "all", "a", 3),
    CalibrationPrompt("c", "rounded", "cu", "all", "a", 3),
    CalibrationPrompt("j", "plain", "ji", "all", "a", 3),
    CalibrationPrompt("j", "rounded", "jue", "all", "a", 3),
    CalibrationPrompt("q", "plain", "qi", "all", "a", 3),
    CalibrationPrompt("q", "rounded", "quan", "all", "a", 3),
    CalibrationPrompt("x", "plain", "xi", "all", "a", 3),
    CalibrationPrompt("x", "rounded", "xue", "all", "a", 3),
    CalibrationPrompt("sh", "plain", "sha", "all", "a", 3),
    CalibrationPrompt("sh", "rounded", "shu", "all", "a", 3),
    CalibrationPrompt("r", "front", "ri", "all", "a", 3),
    CalibrationPrompt("r", "plain", "ran", "all", "a", 3),
    CalibrationPrompt("r", "rounded", "ru", "all", "a", 3),
    CalibrationPrompt("m", "i_series", "mian", "internal", "a", 3),
    CalibrationPrompt("m", "u_series", "mu", "internal", "a", 3),
    CalibrationPrompt("m", "other", "ma", "internal", "a", 3),
    CalibrationPrompt("n", "i_series", "nian", "internal", "a", 3),
    CalibrationPrompt("n", "u_series", "nu", "internal", "a", 3),
    CalibrationPrompt("n", "v_series", "nv", "internal", "a", 3),
    CalibrationPrompt("n", "other", "na", "internal", "a", 3),
    CalibrationPrompt("b", "i_series", "bian", "internal", "a", 3),
    CalibrationPrompt("b", "u_series", "bu", "internal", "a", 3),
    CalibrationPrompt("b", "other", "ba", "internal", "a", 3),
    CalibrationPrompt("p", "i_series", "pian", "internal", "a", 3),
    CalibrationPrompt("p", "u_series", "pu", "internal", "a", 3),
    CalibrationPrompt("p", "other", "pa", "internal", "a", 3),
    CalibrationPrompt("d", "i_series", "dian", "internal", "a", 3),
    CalibrationPrompt("d", "u_series", "du", "internal", "a", 3),
    CalibrationPrompt("d", "other", "da", "internal", "a", 3),
    CalibrationPrompt("t", "i_series", "tian", "internal", "a", 3),
    CalibrationPrompt("t", "u_series", "tu", "internal", "a", 3),
    CalibrationPrompt("t", "other", "ta", "internal", "a", 3),
    CalibrationPrompt("g", "u_series", "gu", "internal", "a", 3),
    CalibrationPrompt("g", "other", "ga", "internal", "a", 3),
    CalibrationPrompt("k", "u_series", "ku", "internal", "a", 3),
    CalibrationPrompt("k", "other", "ka", "internal", "a", 3),
    CalibrationPrompt("f", "rounded", "fo", "internal", "a", 3),
    CalibrationPrompt("f", "other", "fa", "internal", "a", 3),
    CalibrationPrompt("h", "rounded", "hu", "internal", "a", 3),
    CalibrationPrompt("h", "other", "ha", "internal", "a", 3),
)


def live_calibration_protocol() -> CalibrationProtocol:
    return CalibrationProtocol(
        name="live_calibration",
        version="0.1",
        prompts=_BASE_PROMPTS,
        automatic_supplement_rounds=1,
        stop_rule="reanalyze_once_then_freeze_if_still_unresolved",
    )

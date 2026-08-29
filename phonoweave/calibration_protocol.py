from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationPrompt:
    base_unit: str
    context_family: str
    syllable: str
    role_scope: str
    carrier: str
    repeats: int = 1

    @property
    def spoken_pattern(self) -> str:
        if self.role_scope == "internal":
            return (
                f"{self.carrier} {self.syllable} {self.carrier} "
                f"{self.syllable} {self.carrier}"
            )
        return f"{self.syllable} {self.carrier} {self.syllable} {self.carrier}"

    @property
    def target_occurrences(self) -> int:
        return self.spoken_pattern.split().count(self.syllable)


@dataclass(frozen=True)
class CalibrationProtocol:
    name: str
    version: str
    prompts: tuple[CalibrationPrompt, ...]
    automatic_supplement_rounds: int
    stop_rule: str


_BASE_PROMPTS = (
    CalibrationPrompt("zh", "plain", "zha", "all", "a"),
    CalibrationPrompt("zh", "rounded", "zhua", "all", "a"),
    CalibrationPrompt("ch", "plain", "cha", "all", "a"),
    CalibrationPrompt("ch", "rounded", "chua", "all", "a"),
    CalibrationPrompt("z", "plain", "za", "all", "a"),
    CalibrationPrompt("z", "rounded", "zu", "all", "a"),
    CalibrationPrompt("c", "plain", "cai", "all", "a"),
    CalibrationPrompt("c", "rounded", "cu", "all", "a"),
    CalibrationPrompt("j", "plain", "ji", "all", "a"),
    CalibrationPrompt("j", "rounded", "jue", "all", "a"),
    CalibrationPrompt("q", "plain", "qi", "all", "a"),
    CalibrationPrompt("q", "rounded", "quan", "all", "a"),
    CalibrationPrompt("x", "plain", "xi", "all", "a"),
    CalibrationPrompt("x", "rounded", "xue", "all", "a"),
    CalibrationPrompt("sh", "plain", "sha", "all", "a"),
    CalibrationPrompt("sh", "rounded", "shu", "all", "a"),
    CalibrationPrompt("r", "front", "ri", "all", "a"),
    CalibrationPrompt("r", "plain", "ran", "all", "a"),
    CalibrationPrompt("r", "rounded", "ru", "all", "a"),
    CalibrationPrompt("m", "i_series", "mian", "internal", "a"),
    CalibrationPrompt("m", "u_series", "mu", "internal", "a"),
    CalibrationPrompt("m", "other", "ma", "internal", "a"),
    CalibrationPrompt("n", "i_series", "nian", "internal", "a"),
    CalibrationPrompt("n", "u_series", "nu", "internal", "a"),
    CalibrationPrompt("n", "v_series", "nv", "internal", "a"),
    CalibrationPrompt("n", "other", "na", "internal", "a"),
    CalibrationPrompt("b", "i_series", "bian", "internal", "a"),
    CalibrationPrompt("b", "u_series", "bu", "internal", "a"),
    CalibrationPrompt("b", "other", "ba", "internal", "a"),
    CalibrationPrompt("p", "i_series", "pian", "internal", "a"),
    CalibrationPrompt("p", "u_series", "pu", "internal", "a"),
    CalibrationPrompt("p", "other", "pa", "internal", "a"),
    CalibrationPrompt("d", "i_series", "dian", "internal", "a"),
    CalibrationPrompt("d", "u_series", "du", "internal", "a"),
    CalibrationPrompt("d", "other", "da", "internal", "a"),
    CalibrationPrompt("t", "i_series", "tian", "internal", "a"),
    CalibrationPrompt("t", "u_series", "tu", "internal", "a"),
    CalibrationPrompt("t", "other", "ta", "internal", "a"),
    CalibrationPrompt("g", "u_series", "gu", "internal", "a"),
    CalibrationPrompt("g", "other", "ga", "internal", "a"),
    CalibrationPrompt("k", "u_series", "ku", "internal", "a"),
    CalibrationPrompt("k", "other", "ka", "internal", "a"),
    CalibrationPrompt("f", "rounded", "fo", "internal", "a"),
    CalibrationPrompt("f", "other", "fa", "internal", "a"),
    CalibrationPrompt("h", "rounded", "hu", "internal", "a"),
    CalibrationPrompt("h", "other", "ha", "internal", "a"),
)


def live_calibration_protocol() -> CalibrationProtocol:
    return CalibrationProtocol(
        name="live_calibration",
        version="0.1",
        prompts=_BASE_PROMPTS,
        automatic_supplement_rounds=1,
        stop_rule="reanalyze_once_then_freeze_if_still_unresolved",
    )

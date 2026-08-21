from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np

from .oto import OtoEntry


@dataclass(frozen=True)
class AudioSegment:
    samples: np.ndarray
    sample_rate: int
    start_ms: float
    end_ms: float


class AudioReadError(RuntimeError):
    pass


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if sample_width == 3:
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            data[:, 0].astype(np.int32)
            | (data[:, 1].astype(np.int32) << 8)
            | (data[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float64) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    raise AudioReadError(f"unsupported PCM sample width: {sample_width}")


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getcomptype() != "NONE":
                raise AudioReadError(f"compressed WAV is not supported: {path}")
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            raw = wav.readframes(wav.getnframes())
    except (wave.Error, OSError) as exc:
        raise AudioReadError(str(exc)) from exc

    samples = _decode_pcm(raw, sample_width)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def consonant_segment(
    entry: OtoEntry,
    edge_trim: float = 0.12,
    min_duration_ms: float = 28.0,
) -> AudioSegment:
    if entry.preutterance <= 0:
        raise AudioReadError("preutterance is not positive")

    samples, sample_rate = read_wav(entry.wav_path)
    start_ms = max(0.0, entry.offset)
    end_ms = entry.offset + entry.preutterance
    duration = end_ms - start_ms
    if duration < min_duration_ms:
        raise AudioReadError(f"consonant region is too short: {duration:.1f} ms")

    trim = min(duration * edge_trim, 15.0)
    start_ms += trim
    end_ms -= trim
    if end_ms - start_ms < min_duration_ms * 0.6:
        raise AudioReadError("trimmed consonant region is too short")

    start = max(0, int(round(start_ms * sample_rate / 1000.0)))
    end = min(len(samples), int(round(end_ms * sample_rate / 1000.0)))
    if end <= start:
        raise AudioReadError("consonant region is outside the WAV")

    return AudioSegment(
        samples=samples[start:end],
        sample_rate=sample_rate,
        start_ms=start_ms,
        end_ms=end_ms,
    )

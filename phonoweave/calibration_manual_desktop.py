from __future__ import annotations

import argparse
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import read_wav
from .calibration_manual_labels import (
    ManualPrompt,
    load_manual_labels,
    resolve_dev_selection,
    save_manual_label,
)


DISPLAY_BEFORE_PREV_CUE_MS = 80.0
DISPLAY_AFTER_NEXT_CUE_MS = 80.0


@dataclass
class Step:
    prompt: ManualPrompt
    occurrence: int


class ManualAnchorDesktop:
    def __init__(self, session_dir: Path) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.session_dir = session_dir.expanduser().resolve()
        self.prompts = resolve_dev_selection(self.session_dir)
        self.steps = [
            Step(prompt, occurrence)
            for prompt in self.prompts
            for occurrence in (1, 2)
        ]
        self.labels = load_manual_labels(self.session_dir).get("labels", {})
        self.index = 0
        self.audio_cache: dict[int, tuple[np.ndarray, int]] = {}
        self.play_process: subprocess.Popen[bytes] | None = None
        self.temp_files: list[Path] = []

        self.root = tk.Tk()
        self.root.title("PhonoWeave — Manual Anchor")
        self.root.geometry("1180x690")
        self.root.minsize(920, 560)

        self.header = ttk.Frame(self.root, padding=(12, 10))
        self.header.pack(fill="x")
        self.progress_var = tk.StringVar()
        self.title_var = tk.StringVar()
        self.meta_var = tk.StringVar()
        ttk.Label(self.header, textvariable=self.progress_var, width=12).pack(side="left")
        ttk.Label(self.header, textvariable=self.title_var, font=("TkDefaultFont", 19, "bold")).pack(side="left", padx=(8, 18))
        ttk.Label(self.header, textvariable=self.meta_var).pack(side="left")

        self.canvas = tk.Canvas(self.root, background="#fbfbfb", highlightthickness=1, highlightbackground="#999")
        self.canvas.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<Configure>", lambda event: self.draw())

        self.readout_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.readout_var, padding=(12, 2)).pack(fill="x")

        controls = ttk.Frame(self.root, padding=(12, 8))
        controls.pack(fill="x")
        ttk.Button(controls, text="◀ 上一个", command=self.prev).pack(side="left")
        ttk.Button(controls, text="▶ 这一段   [Space]", command=self.play_window).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="▶ 整条", command=self.play_full).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="不确定   [U]", command=self.toggle_uncertain).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="清除   [Delete]", command=self.clear).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="下一个 ▶", command=self.next).pack(side="right")

        footer = ttk.Frame(self.root, padding=(12, 6))
        footer.pack(fill="x")
        self.status_var = tk.StringVar(value="点击波形即可保存，并自动进入下一条。")
        ttk.Label(footer, textvariable=self.status_var).pack(side="left")
        ttk.Label(footer, text="蓝=当前 beep   灰=前/后 beep   红=标记").pack(side="right")

        self.root.bind("<space>", lambda event: self.play_window())
        self.root.bind("<Key-u>", lambda event: self.toggle_uncertain())
        self.root.bind("<Key-U>", lambda event: self.toggle_uncertain())
        self.root.bind("<Delete>", lambda event: self.clear())
        self.root.bind("<BackSpace>", lambda event: self.clear())
        self.root.bind("<Left>", lambda event: self.prev())
        self.root.bind("<Right>", lambda event: self.next())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._jump_to_first_unlabeled()
        self.refresh()

    def _jump_to_first_unlabeled(self) -> None:
        for i, step in enumerate(self.steps):
            if self._label(step) is None:
                self.index = i
                return

    def _step(self) -> Step:
        return self.steps[self.index]

    def _row(self, step: Step) -> dict[str, object] | None:
        row = self.labels.get(str(step.prompt.prompt_index))
        return row if isinstance(row, dict) else None

    def _label(self, step: Step) -> dict[str, object] | None:
        row = self._row(step)
        occurrences = row.get("occurrences") if row else None
        if not isinstance(occurrences, dict):
            return None
        label = occurrences.get(str(step.occurrence))
        return label if isinstance(label, dict) else None

    def _audio(self, prompt: ManualPrompt) -> tuple[np.ndarray, int]:
        cached = self.audio_cache.get(prompt.prompt_index)
        if cached is not None:
            return cached
        audio = read_wav(self.session_dir / "recordings" / prompt.wav)
        self.audio_cache[prompt.prompt_index] = audio
        return audio

    def _bounds_ms(self, step: Step) -> tuple[float, float]:
        o = step.occurrence - 1
        start = step.prompt.prev_cues_ms[o] - DISPLAY_BEFORE_PREV_CUE_MS
        end = step.prompt.next_cues_ms[o] + DISPLAY_AFTER_NEXT_CUE_MS
        samples, sample_rate = self._audio(step.prompt)
        duration = len(samples) * 1000.0 / sample_rate
        return max(0.0, start), min(duration, end)

    def _label_limits_ms(self, step: Step) -> tuple[float, float]:
        cue = step.prompt.cues_ms[step.occurrence - 1]
        return (
            cue + step.prompt.label_min_ms_after_cue(step.occurrence),
            cue + step.prompt.label_max_ms_after_cue(step.occurrence),
        )

    def refresh(self) -> None:
        step = self._step()
        done = sum(1 for item in self.steps if self._label(item) is not None)
        self.progress_var.set(f"{self.index + 1}/{len(self.steps)}  ({done} done)")
        self.title_var.set(f"{step.prompt.syllable}   {step.occurrence}/2")
        self.meta_var.set(f"{step.prompt.base_unit} · {step.prompt.context_family} · {step.prompt.class_name}")
        label = self._label(step)
        if label is None:
            self.readout_var.set("—")
        else:
            status = "?" if label.get("status") == "uncertain" else "OK"
            self.readout_var.set(f"{float(label['anchor_ms_after_cue']):.1f} ms relative to cue   {status}")
        self.draw()

    def draw(self) -> None:
        if not hasattr(self, "canvas"):
            return
        c = self.canvas
        c.delete("all")
        width = max(10, c.winfo_width())
        height = max(10, c.winfo_height())
        step = self._step()
        samples, sample_rate = self._audio(step.prompt)
        start_ms, end_ms = self._bounds_ms(step)
        start = max(0, int(start_ms * sample_rate / 1000.0))
        end = min(len(samples), int(end_ms * sample_rate / 1000.0))
        region = samples[start:end]
        if not len(region):
            return

        mid = height / 2.0
        bins = min(width, len(region))
        edges = np.linspace(0, len(region), bins + 1, dtype=int)
        scale = max(1e-5, float(np.percentile(np.abs(region), 99.5)))
        points: list[float] = []
        for x in range(bins):
            chunk = region[edges[x] : edges[x + 1]]
            if not len(chunk):
                amp = 0.0
            else:
                amp = float(np.max(np.abs(chunk))) / scale
            y1 = mid - min(1.0, amp) * height * 0.42
            y2 = mid + min(1.0, amp) * height * 0.42
            points.extend((x, y1, x, y2))
        for i in range(0, len(points), 4):
            c.create_line(points[i], points[i + 1], points[i + 2], points[i + 3], fill="#222")

        def x_for_ms(value_ms: float) -> float:
            return (value_ms - start_ms) / max(1e-9, end_ms - start_ms) * width

        o = step.occurrence - 1
        for cue_ms, color, dash in (
            (step.prompt.prev_cues_ms[o], "#999", (3, 3)),
            (step.prompt.cues_ms[o], "#245b97", None),
            (step.prompt.next_cues_ms[o], "#999", (3, 3)),
        ):
            x = x_for_ms(cue_ms)
            c.create_line(x, 0, x, height, fill=color, width=2, dash=dash)

        minimum, maximum = self._label_limits_ms(step)
        c.create_line(x_for_ms(minimum), 0, x_for_ms(minimum), height, fill="#c8c8c8", dash=(2, 4))
        c.create_line(x_for_ms(maximum), 0, x_for_ms(maximum), height, fill="#c8c8c8", dash=(2, 4))

        label = self._label(step)
        if label is not None:
            absolute = step.prompt.cues_ms[o] + float(label["anchor_ms_after_cue"])
            x = x_for_ms(absolute)
            c.create_line(x, 0, x, height, fill="#b23232", width=3)

    def _click(self, event: object) -> None:
        step = self._step()
        width = max(1, self.canvas.winfo_width())
        start_ms, end_ms = self._bounds_ms(step)
        absolute_ms = start_ms + float(event.x) / width * (end_ms - start_ms)  # type: ignore[attr-defined]
        minimum, maximum = self._label_limits_ms(step)
        if absolute_ms < minimum or absolute_ms > maximum:
            self.status_var.set("这个点已经进了相邻 token；如果目标声母确实在那里，先告诉我这一项。")
            return
        cue = step.prompt.cues_ms[step.occurrence - 1]
        relative = absolute_ms - cue
        self._save(relative, "ok")
        if self.index < len(self.steps) - 1:
            self.index += 1
        self.refresh()

    def _save(self, relative_ms: float | None, status: str) -> None:
        step = self._step()
        payload = save_manual_label(
            self.session_dir,
            step.prompt.prompt_index,
            step.occurrence,
            relative_ms,
            status,
        )
        labels = payload.get("labels", {})
        self.labels = labels if isinstance(labels, dict) else {}
        self.status_var.set("saved")

    def toggle_uncertain(self) -> None:
        step = self._step()
        label = self._label(step)
        if label is None:
            self.status_var.set("先点一个位置，再按 U。")
            return
        status = "ok" if label.get("status") == "uncertain" else "uncertain"
        self._save(float(label["anchor_ms_after_cue"]), status)
        self.refresh()

    def clear(self) -> None:
        self._save(None, "unset")
        self.refresh()

    def prev(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.refresh()

    def next(self) -> None:
        if self.index < len(self.steps) - 1:
            self.index += 1
            self.refresh()

    def _stop_playback(self) -> None:
        if self.play_process is not None and self.play_process.poll() is None:
            self.play_process.terminate()
        self.play_process = None

    def _write_clip(self, start_ms: float, end_ms: float) -> Path:
        step = self._step()
        samples, sample_rate = self._audio(step.prompt)
        start = max(0, int(round(start_ms * sample_rate / 1000.0)))
        end = min(len(samples), int(round(end_ms * sample_rate / 1000.0)))
        clip = np.asarray(samples[start:end], dtype=np.float64)
        peak = max(1.0, float(np.max(np.abs(clip))))
        pcm = np.clip(clip / peak, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype("<i2")
        handle = tempfile.NamedTemporaryFile(prefix="phonoweave-anchor-", suffix=".wav", delete=False)
        path = Path(handle.name)
        handle.close()
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm16.tobytes())
        self.temp_files.append(path)
        return path

    def play_window(self) -> None:
        self._stop_playback()
        start_ms, end_ms = self._bounds_ms(self._step())
        path = self._write_clip(start_ms, end_ms)
        self.play_process = subprocess.Popen(["afplay", str(path)])

    def play_full(self) -> None:
        self._stop_playback()
        path = self.session_dir / "recordings" / self._step().prompt.wav
        self.play_process = subprocess.Popen(["afplay", str(path)])

    def close(self) -> None:
        self._stop_playback()
        for path in self.temp_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonoweave-manual-anchor-desktop")
    parser.add_argument("session", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    app = ManualAnchorDesktop(args.session)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

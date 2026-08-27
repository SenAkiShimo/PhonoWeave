from __future__ import annotations

import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .gui_model import (
    GuiAnalysisSnapshot,
    GuiDecisionRow,
    analyze_for_gui,
    write_cached_profile,
    write_cached_synthesis_inventory,
)


class PhonoWeaveGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PhonoWeave")
        self.root.geometry("1040x680")
        self.root.minsize(900, 560)

        self.snapshot: GuiAnalysisSnapshot | None = None
        self.rows: dict[str, GuiDecisionRow] = {}
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()

        self.voicebank_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Choose a voicebank to begin.")
        self.summary_var = tk.StringVar(value="No analysis yet")

        self._build()
        self.root.after(100, self._poll_events)

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x")

        title = ttk.Label(header, text="PhonoWeave", font=("TkDefaultFont", 20, "bold"))
        title.pack(side="left")
        ttk.Label(
            header,
            text="Speaker realization inventory",
        ).pack(side="left", padx=(12, 0), pady=(7, 0))

        picker = ttk.Frame(outer)
        picker.pack(fill="x", pady=(16, 10))
        ttk.Label(picker, text="Voicebank").pack(side="left")
        self.voicebank_entry = ttk.Entry(picker, textvariable=self.voicebank_var)
        self.voicebank_entry.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(picker, text="Browse", command=self._browse).pack(side="left")
        self.analyze_button = ttk.Button(picker, text="Analyze", command=self._start_analysis)
        self.analyze_button.pack(side="left", padx=(8, 0))

        summary = ttk.Frame(outer)
        summary.pack(fill="x", pady=(0, 10))
        ttk.Label(summary, textvariable=self.summary_var).pack(side="left")
        ttk.Label(summary, textvariable=self.status_var).pack(side="right")

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, padding=(0, 0, 8, 0))
        right = ttk.Frame(body, padding=(8, 0, 0, 0))
        body.add(left, weight=3)
        body.add(right, weight=2)

        columns = ("base", "class", "decision", "confidence")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("base", text="Onset")
        self.tree.heading("class", text="Class")
        self.tree.heading("decision", text="Decision")
        self.tree.heading("confidence", text="Confidence")
        self.tree.column("base", width=70, anchor="center", stretch=False)
        self.tree.column("class", width=110, anchor="w")
        self.tree.column("decision", width=180, anchor="w")
        self.tree.column("confidence", width=100, anchor="w")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._select_row)

        ttk.Label(right, text="Details", font=("TkDefaultFont", 14, "bold")).pack(anchor="w")
        self.detail = tk.Text(right, wrap="word", height=20, borderwidth=1, relief="solid")
        self.detail.pack(fill="both", expand=True, pady=(8, 0))
        self.detail.configure(state="disabled")

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(12, 0))
        self.profile_button = ttk.Button(
            footer,
            text="Export Speaker Profile",
            command=self._export_profile,
            state="disabled",
        )
        self.profile_button.pack(side="left")
        self.inventory_button = ttk.Button(
            footer,
            text="Export Synthesis Inventory",
            command=self._export_inventory,
            state="disabled",
        )
        self.inventory_button.pack(side="left", padx=(8, 0))

    def _browse(self) -> None:
        path = filedialog.askdirectory(title="Choose OpenUtau voicebank")
        if path:
            self.voicebank_var.set(path)

    def _set_busy(self, busy: bool) -> None:
        self.analyze_button.configure(state="disabled" if busy else "normal")
        self.profile_button.configure(
            state="disabled" if busy or self.snapshot is None else "normal"
        )
        self.inventory_button.configure(
            state="disabled" if busy or self.snapshot is None else "normal"
        )

    def _start_analysis(self) -> None:
        text = self.voicebank_var.get().strip()
        if not text:
            messagebox.showerror("PhonoWeave", "Choose a voicebank folder first.")
            return
        root = Path(text).expanduser()
        if not root.is_dir():
            messagebox.showerror("PhonoWeave", "The selected voicebank folder does not exist.")
            return

        self.snapshot = None
        self.rows.clear()
        self.tree.delete(*self.tree.get_children())
        self._show_detail("Analysis is running.\n\nThe first full pass may take a while.")
        self.summary_var.set("Analyzing voicebank…")
        self.status_var.set("Running")
        self._set_busy(True)

        threading.Thread(target=self._analyze_worker, args=(root,), daemon=True).start()

    def _analyze_worker(self, root: Path) -> None:
        try:
            snapshot = analyze_for_gui(root)
        except Exception as exc:
            self.events.put(("error", exc))
            return
        self.events.put(("done", snapshot))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "done":
                    self._analysis_done(payload)
                elif kind == "error":
                    self._analysis_error(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _analysis_done(self, payload: object) -> None:
        snapshot = payload
        if not isinstance(snapshot, GuiAnalysisSnapshot):
            self._analysis_error(RuntimeError("invalid GUI analysis result"))
            return
        self.snapshot = snapshot
        self.rows = {row.base_unit: row for row in snapshot.rows}

        for row in snapshot.rows:
            self.tree.insert(
                "",
                "end",
                iid=row.base_unit,
                values=(row.base_unit, row.class_name, row.decision, row.confidence),
            )

        self.summary_var.set(
            f"{len(snapshot.rows)} onsets · "
            f"{len(snapshot.synthesis_inventory.units)} synthesis units"
        )
        self.status_var.set(
            f"analyzed {snapshot.analyzed_count} · "
            f"experimental {snapshot.experimental_count} · "
            f"unsupported {snapshot.unsupported_count}"
        )
        self._set_busy(False)

        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])
            self._render_row(self.rows[children[0]])

    def _analysis_error(self, payload: object) -> None:
        self.snapshot = None
        self.summary_var.set("Analysis failed")
        self.status_var.set("Error")
        self._set_busy(False)
        messagebox.showerror("PhonoWeave", str(payload))

    def _select_row(self, _event: object = None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        row = self.rows.get(selected[0])
        if row is not None:
            self._render_row(row)

    def _render_row(self, row: GuiDecisionRow) -> None:
        lines = [
            row.base_unit,
            "",
            f"Class: {row.class_name}",
            f"Decision: {row.decision}",
            f"Confidence: {row.confidence}",
            "",
            f"Acoustic evidence: {row.acoustic_evidence}",
            f"Synthesis evidence: {row.synthesis_evidence}",
            "",
            "Realization groups",
        ]
        for group, context in zip(row.groups, row.contexts, strict=False):
            lines.append(f"  {group}  ·  {context}")
        if row.notes:
            lines.extend(("", "Analysis notes"))
            lines.extend(f"  {note}" for note in row.notes)
        self._show_detail("\n".join(lines))

    def _show_detail(self, text: str) -> None:
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", text)
        self.detail.configure(state="disabled")

    def _export_profile(self) -> None:
        if self.snapshot is None:
            return
        path = filedialog.asksaveasfilename(
            title="Export Speaker Profile",
            defaultextension=".yaml",
            initialfile="speaker_profile.yaml",
            filetypes=(("YAML", "*.yaml"), ("All files", "*")),
        )
        if not path:
            return
        output = write_cached_profile(self.snapshot, Path(path))
        self.status_var.set(f"Saved {output.name}")

    def _export_inventory(self) -> None:
        if self.snapshot is None:
            return
        path = filedialog.asksaveasfilename(
            title="Export Synthesis Inventory",
            defaultextension=".yaml",
            initialfile="synthesis_inventory.yaml",
            filetypes=(("YAML", "*.yaml"), ("All files", "*")),
        )
        if not path:
            return
        output = write_cached_synthesis_inventory(self.snapshot, Path(path))
        self.status_var.set(f"Saved {output.name}")


def main() -> int:
    root = tk.Tk()
    PhonoWeaveGui(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

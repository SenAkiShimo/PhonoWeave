from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import subprocess
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .calibration_io import create_calibration_session, protocol_payload, save_calibration_recording
from .calibration_page import HTML as CALIBRATION_HTML
from .calibration_protocol import live_calibration_protocol
from .gui_model import GuiAnalysisSnapshot, analyze_for_gui
from .gui_page_remake import HTML
from .profile import profile_yaml
from .synthesis_inventory import synthesis_inventory_yaml


@dataclass
class _GuiState:
    snapshot: GuiAnalysisSnapshot | None = None
    calibration_root: Path = Path.home() / "Downloads" / "PhonoWeaveCalibration"
    calibration_session_id: str | None = None


_STATE = _GuiState()
_STATE_LOCK = threading.Lock()


def _main_page_html() -> str:
    marker = '<button class="toolbtn primary" id="analyze"></button>'
    calibration = (
        marker
        + '<a class="toolbtn" href="/calibration" '
        + 'title="Record a live speaker calibration">'
        + 'Live Calibration / 现场校准</a>'
    )
    return HTML.replace(marker, calibration, 1)


def _snapshot_payload(snapshot: GuiAnalysisSnapshot) -> dict[str, Any]:
    split_supported = [row.base_unit for row in snapshot.rows if row.decision == "split_recommended"]
    return {
        "voicebank": str(snapshot.root),
        "summary": {
            "onsets": len(snapshot.rows),
            "synthesis_units": len(snapshot.synthesis_inventory.units),
            "analyzed": snapshot.analyzed_count,
            "experimental": snapshot.experimental_count,
            "unsupported": snapshot.unsupported_count,
            "unresolved": sum(row.decision == "unresolved" for row in snapshot.rows),
            "split_supported": len(split_supported),
            "split_supported_units": split_supported,
            "supplement_items": sum(len(row.evidence_gap.diagnostic_items) for row in snapshot.rows if row.evidence_gap is not None),
        },
        "rows": [
            {
                "base_unit": row.base_unit,
                "class_name": row.class_name,
                "acoustic_evidence": row.acoustic_evidence,
                "synthesis_evidence": row.synthesis_evidence,
                "decision": row.decision,
                "confidence": row.confidence,
                "groups": [{"id": group.id, "contexts": list(group.contexts)} for group in row.groups],
                "notes": list(row.notes),
                "evidence_gap": None if row.evidence_gap is None else {
                    "gap_type": row.evidence_gap.gap_type,
                    "priority": row.evidence_gap.priority,
                    "recommended_action": row.evidence_gap.recommended_action,
                    "role_scope": row.evidence_gap.role_scope,
                    "context_families": list(row.evidence_gap.context_families),
                    "rationale": list(row.evidence_gap.rationale),
                    "diagnostic_items": [
                        {
                            "syllable": item.syllable,
                            "context_family": item.context_family,
                            "replicate": item.replicate,
                            "existing_observations": item.existing_observations,
                            "reclist_line": item.reclist_line,
                        }
                        for item in row.evidence_gap.diagnostic_items
                    ],
                },
            }
            for row in snapshot.rows
        ],
    }


def _diagnostic_manifest_csv(snapshot: GuiAnalysisSnapshot) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("base_unit", "class_name", "current_decision", "gap_type", "priority", "recommended_action", "role_scope", "target_context", "syllable", "replicate", "existing_observations", "reclist_line", "purpose"))
    for row in snapshot.rows:
        gap = row.evidence_gap
        if gap is None:
            continue
        for item in gap.diagnostic_items:
            writer.writerow((row.base_unit, row.class_name, row.decision, gap.gap_type, gap.priority, gap.recommended_action, gap.role_scope or "all", item.context_family, item.syllable, item.replicate, item.existing_observations, item.reclist_line, "diagnostic_evidence_only"))
    return output.getvalue()


def _pick_folder(language: str = "en") -> str:
    if platform.system() != "Darwin":
        raise RuntimeError("Folder picker is currently available on macOS only.")
    prompt = "选择 OpenUtau 声库" if language == "zh" else "Choose OpenUtau voicebank"
    script = (f'set chosenFolder to choose folder with prompt "{prompt}"\n' 'return POSIX path of chosenFolder')
    completed = subprocess.run(["osascript", "-e", script], check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip()
        if "User canceled" in message or "-128" in message:
            return ""
        raise RuntimeError(message or "Could not open the folder picker.")
    return completed.stdout.strip().rstrip("/")


class _Handler(BaseHTTPRequestHandler):
    server_version = "PhonoWeaveGuiRemake/0.3"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK, headers: dict[str, str] | None = None) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("request body must be a JSON object")
        return parsed

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send_bytes(_main_page_html().encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/calibration":
            self._send_bytes(CALIBRATION_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/calibration/protocol":
            self._json(protocol_payload(live_calibration_protocol()))
            return
        if path == "/api/export/profile":
            with _STATE_LOCK:
                snapshot = _STATE.snapshot
            if snapshot is None:
                self._json({"error": "No completed analysis is available."}, HTTPStatus.CONFLICT)
                return
            self._send_bytes(profile_yaml(snapshot.profile).encode("utf-8"), "application/yaml; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="speaker_profile.yaml"'})
            return
        if path == "/api/export/inventory":
            with _STATE_LOCK:
                snapshot = _STATE.snapshot
            if snapshot is None:
                self._json({"error": "No completed analysis is available."}, HTTPStatus.CONFLICT)
                return
            self._send_bytes(synthesis_inventory_yaml(snapshot.synthesis_inventory).encode("utf-8"), "application/yaml; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="current_synthesis_inventory.yaml"'})
            return
        if path == "/api/export/supplement-reclist":
            with _STATE_LOCK:
                snapshot = _STATE.snapshot
            if snapshot is None:
                self._json({"error": "No completed analysis is available."}, HTTPStatus.CONFLICT)
                return
            self._send_bytes(snapshot.supplement_reclist.encode("utf-8"), "text/plain; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="diagnostic_supplement_reclist.txt"'})
            return
        if path == "/api/export/diagnostic-manifest":
            with _STATE_LOCK:
                snapshot = _STATE.snapshot
            if snapshot is None:
                self._json({"error": "No completed analysis is available."}, HTTPStatus.CONFLICT)
                return
            self._send_bytes(_diagnostic_manifest_csv(snapshot).encode("utf-8"), "text/csv; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="diagnostic_supplement_manifest.csv"'})
            return
        self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/pick-folder":
            try:
                payload = self._read_json()
                language = "zh" if payload.get("lang") == "zh" else "en"
                selected = _pick_folder(language)
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._json({"path": selected})
            return
        if path == "/api/analyze":
            try:
                payload = self._read_json()
                raw_path = str(payload.get("path", "")).strip()
                if not raw_path:
                    raise ValueError("Choose a voicebank first.")
                root = Path(raw_path).expanduser().resolve()
                if not root.is_dir():
                    raise ValueError("The selected voicebank folder does not exist.")
                snapshot = analyze_for_gui(root)
                with _STATE_LOCK:
                    _STATE.snapshot = snapshot
                self._json(_snapshot_payload(snapshot))
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if path == "/api/calibration/start":
            try:
                protocol = live_calibration_protocol()
                with _STATE_LOCK:
                    session_id, session_dir = create_calibration_session(protocol, _STATE.calibration_root)
                    _STATE.calibration_session_id = session_id
                self._json({"session_id": session_id, "session_dir": str(session_dir), "protocol": protocol_payload(protocol)})
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if path == "/api/calibration/recording":
            try:
                query = parse_qs(parsed.query)
                index = int(query.get("index", ["-1"])[0])
                sample_rate = int(query.get("sample_rate", ["0"])[0])
                protocol = live_calibration_protocol()
                if index < 0 or index >= len(protocol.prompts):
                    raise ValueError("invalid calibration prompt index")
                if sample_rate <= 0:
                    raise ValueError("invalid sample rate")
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 64 * 1024 * 1024:
                    raise ValueError("invalid recording size")
                wav_bytes = self.rfile.read(length)
                with _STATE_LOCK:
                    session_id = _STATE.calibration_session_id
                    root = _STATE.calibration_root
                if session_id is None:
                    raise ValueError("no active calibration session")
                output = save_calibration_recording(root, session_id, index, protocol.prompts[index], wav_bytes, sample_rate)
                self._json({"saved": str(output), "index": index})
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonoweave-gui")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/"
    print(f"PhonoWeave GUI remake: {url}")
    print(f"Live calibration: {url}calibration")
    print("Press Ctrl-C to stop.")
    if not args.no_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

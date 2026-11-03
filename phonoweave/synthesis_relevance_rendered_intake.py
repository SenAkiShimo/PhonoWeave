"""Provenance-gated external singing renderer intake and blind A/B packaging.

A rendering engine is NOT invoked here. This adapter fixes S01-only eligible
donor pools, binds S02 targets to the frozen controlled pilot, audits user-
supplied renderer WAVs, and blinds labels without touching original sources.
Declaration fields are not independent proof of correct renderer routing.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import wave

from .synthesis_relevance_controlled_corpus import FINALS, PAIRS, REPEATS
from .synthesis_relevance_controlled_evaluation import evaluate_controlled_manifest

PLAN_SCHEMA = "phonoweave.synthesis_relevance.rendered_ab_plan.v1"
RECEIPT_SCHEMA = "phonoweave.synthesis_relevance.external_render_receipt.v1"
BLIND_SCHEMA = "phonoweave.synthesis_relevance.blind_ab_session.v1"
KEY_SCHEMA = "phonoweave.synthesis_relevance.blind_ab_key.v1"
CHOICES = ("A", "B", "tie", "unusable", "unrated")
_HEX = re.compile(r"[a-fA-F0-9]{64}\Z")


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def _canonical(data: dict) -> str:
    return sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _new_outside_sources(output: Path, sources: tuple[Path, ...]) -> Path:
    path = output.expanduser().resolve()
    if any(path == root or root in path.parents for root in sources):
        raise ValueError("output must be outside every source or rendered-audio root")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    return path


def create_render_plan(manifest: dict, proxy: dict, corpus: Path) -> dict:
    """Bind each held-out S02 target to the exact allowed S01 donor pools."""
    corpus = corpus.expanduser().resolve(strict=True)
    expected = evaluate_controlled_manifest(manifest, corpus)
    if expected["status"] != "evaluable_acoustic_proxy":
        raise ValueError("48-take controlled corpus is not audit-ready")
    for field in ("schema", "source_manifest_sha256", "source_audio_sha256_by_take",
                  "rankings", "train_only_utility", "n_heldout_WAVs"):
        if proxy.get(field) != expected.get(field):
            raise ValueError(f"stale or incompatible frozen spectral proxy: {field}")
    if (proxy.get("protocol") != expected.get("protocol")
            or proxy.get("outcome_status") != "S02_spectral_proxy_only"):
        raise ValueError("invalid S01/S02 evaluation provenance")

    train, heldout = {}, {}
    for row in manifest["recordings"]:
        if row["session_id"] == "S01":
            train.setdefault(row["final"], []).append(row["take_id"])
        elif row["session_id"] == "S02":
            heldout[row["take_id"]] = row
        else:
            raise ValueError("unexpected session identity")
    if (len(heldout) != 24 or set(train) != set(FINALS)
            or any(len(takes) != REPEATS for takes in train.values())):
        raise ValueError("controlled train/test grid mismatch")
    by_pair = {f: pair for pair in PAIRS for f in pair}
    tasks = []
    for take_id, row in sorted(heldout.items()):
        pair = by_pair[row["final"]]
        pool = sorted(train[pair[0]] + train[pair[1]])
        specific = sorted(train[row["final"]])
        if len(pool) != 6 or len(specific) != 3 or take_id in pool:
            raise ValueError("invalid S01 donor pool")
        tasks.append({
            "target_take_id": take_id,
            "target_final": row["final"],
            "source_reference_sha256": row["wav_sha256"],
            "allowed_S01_pooled_donor_take_ids": pool,
            "allowed_S01_split_donor_take_ids": specific,
            "render_wavs": {
                "pooled": f"{take_id}/pooled.wav",
                "split": f"{take_id}/split.wav",
            },
        })
    return {
        "schema": PLAN_SCHEMA,
        "status": "render_tasks_only_no_audio_generated",
        "source_manifest_sha256": expected["source_manifest_sha256"],
        "source_proxy_sha256": _canonical(proxy),
        "source_audio_sha256_by_take": dict(sorted(expected["source_audio_sha256_by_take"].items())),
        "speaker_id": manifest["speaker_id"],
        "pitch_subbank": manifest["pitch_subbank"],
        "train_session": "S01",
        "heldout_session": "S02",
        "tasks": tasks,
        "task_count": len(tasks),
        "no_S02_source_in_allowed_donor_pools": True,
        "condition_definitions": {
            "pooled": "source samples only from six S01 takes in the corresponding fixed final pair",
            "split": "source samples only from three same-final S01 takes",
        },
        "external_rendering_required": True,
        "production_alias_export": "disabled",
        "source_recordings_modified": False,
    }


def make_receipt_template(plan: dict) -> dict:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("task_count") != 24:
        raise ValueError("invalid frozen render plan")
    return {
        "schema": RECEIPT_SCHEMA,
        "render_plan_sha256": _canonical(plan),
        "engine_name": "",
        "engine_version": "",
        "resampler_name": "",
        "resampler_version": "",
        "score_template_sha256": "",
        "render_configuration_sha256": "",
        "same_score_and_render_settings_between_arms_confirmed": False,
        "S02_audio_not_used_as_render_donor_or_conditioning_confirmed": False,
        "trial_receipts": [
            {
                "target_take_id": task["target_take_id"],
                "pooled_donor_take_ids_used": [],
                "split_donor_take_ids_used": [],
                "pooled_wav_sha256": "",
                "split_wav_sha256": "",
            }
            for task in plan["tasks"]
        ],
    }


def write_render_plan(plan: dict, output: Path, *, corpus: Path) -> Path:
    output = _new_outside_sources(output, (corpus.expanduser().resolve(strict=True),))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pw-render-plan-", dir=output.parent) as d:
        stage = Path(d) / "plan"
        stage.mkdir()
        for name, data in (("render_plan.json", plan),
                           ("render_receipt_template.json", make_receipt_template(plan))):
            (stage / name).write_text(
                json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8")
        (stage / "README.txt").write_text(
            "External renderer required. Fill the receipt with actual S01 donor IDs,\n"
            "engine metadata, fixed score/configuration hashes and output WAV hashes.\n"
            "Do not render from held-out S02 audio; it is reference-only.\n"
            "This program cannot independently verify the renderer's internal routing.\n",
            encoding="utf-8")
        if output.exists():
            raise FileExistsError("output appeared during staging")
        os.rename(stage, output)
    return output


def _render_audio_info(path: Path) -> tuple[int, int, float]:
    try:
        with wave.open(str(path), "rb") as wav:
            rate, count = wav.getframerate(), wav.getnframes()
            if (wav.getcomptype() != "NONE" or wav.getnchannels() != 1
                    or wav.getsampwidth() not in (2, 3)
                    or rate not in (44100, 48000, 96000)):
                raise ValueError("rendered output must be mono PCM16/24 at 44.1/48/96 kHz")
            duration = count / rate
            if not 0.15 <= duration <= 30:
                raise ValueError("rendered WAV has out-of-range duration")
            if not wav.readframes(min(count, 4096)):
                raise ValueError("empty rendered WAV")
            return rate, count, duration
    except (wave.Error, EOFError, OSError) as exc:
        raise ValueError(f"invalid rendered WAV: {path.name}") from exc


def _check_receipt(plan: dict, receipt: dict, render_root: Path) -> list[dict]:
    """Check declared donor membership and actual WAV byte hashes separately."""
    template = make_receipt_template(plan)
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("render_plan_sha256") != template["render_plan_sha256"]:
        raise ValueError("renderer receipt does not match frozen plan")
    for field in ("engine_name", "engine_version", "resampler_name", "resampler_version"):
        value = receipt.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > 120:
            raise ValueError(f"missing or invalid renderer metadata: {field}")
    for field in ("score_template_sha256", "render_configuration_sha256"):
        if not isinstance(receipt.get(field), str) or not _HEX.fullmatch(receipt[field]):
            raise ValueError(f"missing valid SHA-256 renderer declaration: {field}")
    for flag in ("same_score_and_render_settings_between_arms_confirmed",
                 "S02_audio_not_used_as_render_donor_or_conditioning_confirmed"):
        if receipt.get(flag) is not True:
            raise ValueError(f"renderer has not affirmed {flag}")
    rows = receipt.get("trial_receipts")
    if not isinstance(rows, list) or len(rows) != len(plan["tasks"]):
        raise ValueError("renderer receipt must specify all 24 targets exactly")
    by_id = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid renderer trial receipt")
        key = row.get("target_take_id")
        if key in by_id or key is None:
            raise ValueError("duplicate or missing target in renderer receipt")
        by_id[key] = row
    if set(by_id) != {x["target_take_id"] for x in plan["tasks"]}:
        raise ValueError("renderer receipt changed heldout target population")
    files = []
    for task in plan["tasks"]:
        row = by_id[task["target_take_id"]]
        actual = {}
        for arm in ("pooled", "split"):
            donors = row.get(f"{arm}_donor_take_ids_used")
            allowed = set(task[f"allowed_S01_{arm}_donor_take_ids"])
            if (not isinstance(donors, list) or not donors
                    or not all(isinstance(x, str) for x in donors)
                    or len(set(donors)) != len(donors)
                    or not set(donors).issubset(allowed)):
                raise ValueError(f"unapproved or missing S01 donor references: {task['target_take_id']}/{arm}")
            rel = Path(task["render_wavs"][arm])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("invalid render path")
            path = (render_root / rel).resolve(strict=True)
            if not path.is_relative_to(render_root):
                raise ValueError("rendered WAV escapes render directory")
            digest = _digest(path)
            if row.get(f"{arm}_wav_sha256") != digest:
                raise ValueError(f"render WAV checksum mismatch: {task['target_take_id']}/{arm}")
            if digest == task["source_reference_sha256"]:
                raise ValueError("rendered WAV identical to heldout reference")
            actual[arm] = {"file": path, "sha256": digest, "format": _render_audio_info(path)}
        if actual["pooled"]["format"][:2] != actual["split"]["format"][:2]:
            raise ValueError("paired renders have different sample rate or frame count")
        if actual["pooled"]["sha256"] == actual["split"]["sha256"]:
            raise ValueError("paired render conditions have identical WAV bytes")
        files.append({"task": task, "renders": actual, "receipt": row})
    return files


def verify_frozen_corpus(plan: dict, corpus: Path) -> None:
    """Recheck all 48 frozen pilot sources before publishing blind media."""
    root = corpus.expanduser().resolve(strict=True)
    hashes = plan.get("source_audio_sha256_by_take")
    expected = {
        f"{sid}_sh{final}_{take:02d}":
        f"audio/{sid}/sh{final}_{take:02d}.wav"
        for sid in ("S01", "S02")
        for final in FINALS
        for take in range(1, REPEATS + 1)
    }
    if not isinstance(hashes, dict) or set(hashes) != set(expected):
        raise ValueError("plan must freeze all 48 canonical source WAV hashes")
    by_id = {x.get("target_take_id"): x for x in plan["tasks"]}
    if len(by_id) != 24 or set(by_id) != {k for k in expected if k.startswith("S02_")}:
        raise ValueError("render plan does not enumerate the frozen S02 targets")
    for take_id, rel in expected.items():
        digest = hashes[take_id]
        if not isinstance(digest, str) or not _HEX.fullmatch(digest):
            raise ValueError("invalid frozen source audio SHA-256")
        path = (root / rel).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("frozen audio path escapes corpus")
        if _digest(path) != digest:
            raise ValueError(f"frozen source audio checksum mismatch: {take_id}")
        if take_id.startswith("S02_") and by_id[take_id].get("source_reference_sha256") != digest:
            raise ValueError("heldout reference hash differs from frozen source")
    for task in plan["tasks"]:
        tid, final = task["target_take_id"], task.get("target_final")
        if final not in FINALS or not tid.startswith(f"S02_sh{final}_"):
            raise ValueError("render target final/identity mismatch")
        pair = next(p for p in PAIRS if final in p)
        pooled = sorted(f"S01_sh{f}_{n:02d}" for f in pair for n in range(1, REPEATS + 1))
        split = sorted(f"S01_sh{final}_{n:02d}" for n in range(1, REPEATS + 1))
        if (task.get("allowed_S01_pooled_donor_take_ids") != pooled
                or task.get("allowed_S01_split_donor_take_ids") != split):
            raise ValueError("render donor pools do not match frozen S01 protocol")


def stage_blind_session(
    plan: dict, receipt: dict, render_root: Path,
    blind_output: Path, key_output: Path, *, corpus: Path, seed_hex: str,
) -> tuple[dict, dict]:
    """Publish shareable participant-only audio and a separate private answer key."""
    if plan.get("schema") != PLAN_SCHEMA or plan.get("task_count") != 24:
        raise ValueError("invalid rendering plan")
    corpus = corpus.expanduser().resolve(strict=True)
    render_root = render_root.expanduser().resolve(strict=True)
    if render_root == corpus or corpus in render_root.parents:
        raise ValueError("rendered WAVs must not be stored in the source capture corpus")
    blind_output = _new_outside_sources(blind_output, (corpus, render_root))
    key_output = _new_outside_sources(key_output, (corpus, render_root))
    if (key_output == blind_output or blind_output in key_output.parents
            or key_output in blind_output.parents):
        raise ValueError("blinded package and private key must be completely separate")
    if not isinstance(seed_hex, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", seed_hex):
        raise ValueError("blinding seed must be 32 random bytes, given as hex")
    verify_frozen_corpus(plan, corpus)
    files = _check_receipt(plan, receipt, render_root)
    rng = random.Random(int(seed_hex, 16))
    indices = list(range(len(files)))
    rng.shuffle(indices)
    # Fixed 12/12 pooled-on-A allocation and independently shuffled trials.
    pooled_a = set(indices[:len(indices) // 2])
    rng.shuffle(indices)
    participant = {
        "schema": BLIND_SCHEMA,
        "status": "externally_rendered_WAVs_ready_for_blinded_listening",
        "instructions": "Listen to A and B; select A, B, tie or unusable. No option is identified.",
        "trials": [],
    }
    private = {
        "schema": KEY_SCHEMA,
        "render_plan_sha256": _canonical(plan),
        "render_receipt_sha256": _canonical(receipt),
        "blinding_seed_hex": seed_hex,
        "trials": [],
        "renderer_route_status": "source_donors_declared_by_external_renderer_not_independently_verified",
    }
    copied = []
    for n, index in enumerate(indices, 1):
        item = files[index]
        task, arms = item["task"], item["renders"]
        trial_id = f"AB-{n:03d}"
        cond_a, cond_b = ("pooled", "split") if index in pooled_a else ("split", "pooled")
        names = {label: f"media/{trial_id}_{label}.wav" for label in ("A", "B")}
        participant["trials"].append({"trial_id": trial_id, "A": names["A"], "B": names["B"]})
        private["trials"].append({
            "trial_id": trial_id,
            "target_take_id": task["target_take_id"],
            "source_reference_sha256": task["source_reference_sha256"],
            "A_condition": cond_a, "B_condition": cond_b,
            "A_sha256": arms[cond_a]["sha256"],
            "B_sha256": arms[cond_b]["sha256"],
        })
        for label, condition in (("A", cond_a), ("B", cond_b)):
            copied.append((arms[condition]["file"], names[label],
                           arms[condition]["sha256"]))
    private["participant_manifest_sha256"] = _canonical(participant)
    response = {
        "schema": "phonoweave.synthesis_relevance.blind_ab_response.v1",
        "participant_manifest_sha256": private["participant_manifest_sha256"],
        "listener_id": "",
        "votes": [{"trial_id": row["trial_id"], "choice": "unrated"}
                  for row in participant["trials"]],
    }
    blind_output.parent.mkdir(parents=True, exist_ok=True)
    key_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pw-blind-", dir=blind_output.parent) as tmp:
        stage = Path(tmp) / "participant"
        (stage / "media").mkdir(parents=True)
        for src, rel, digest in copied:
            dst = stage / rel
            shutil.copyfile(src, dst)
            if _digest(dst) != digest:
                raise ValueError("rendered WAV changed during blind package copying")
        for filename, data in (("manifest.json", participant),
                               ("response_template.json", response)):
            (stage / filename).write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # The key is NEVER copied into the participant package.
        key_created = False
        try:
            with key_output.open("x", encoding="utf-8") as stream:
                key_created = True
                json.dump(private, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            if blind_output.exists():
                raise FileExistsError("blinded output appeared during staging")
            os.rename(stage, blind_output)
        except BaseException:
            if key_created:
                key_output.unlink(missing_ok=True)
            raise
    return participant, private

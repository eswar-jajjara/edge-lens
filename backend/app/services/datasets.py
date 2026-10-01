"""Bounded ingestion of labelled ImageNet-compatible image archives."""
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import stat
from uuid import uuid4
import zipfile
import zlib

MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
MAX_EXPANDED_BYTES = 200 * 1024 * 1024
MAX_ENTRY_BYTES = 10 * 1024 * 1024
MAX_IMAGES = 1000
MAX_LABEL_BYTES = 1024 * 1024


class DatasetValidationError(ValueError):
    """The archive does not follow the supported dataset contract."""


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DatasetValidationError("labels.json contains duplicate class names")
        result[key] = value
    return result


def _safe_parts(name: str) -> tuple[str, ...]:
    if not name or "\\" in name or "\x00" in name or ":" in name or name.startswith("/"):
        raise DatasetValidationError("Archive entry paths must be safe relative paths")
    parts = (name[:-1] if name.endswith("/") else name).split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise DatasetValidationError("Archive entry paths must not contain traversal or empty segments")
    return tuple(parts)


def _read_entry(archive: zipfile.ZipFile, entry: zipfile.ZipInfo, limit: int) -> bytes:
    with archive.open(entry) as stream:
        value = stream.read(limit + 1)
    if len(value) > limit or len(value) != entry.file_size:
        raise DatasetValidationError("An archive entry exceeds its allowed size")
    return value


def ingest_dataset(content: bytes, name: str, data_dir: Path) -> dict:
    """Validate then save a ZIP without extracting client-controlled paths.

    Root labels.json maps class folders to torchvision ImageNet indices.
    The model runner decodes the actual image pixels before evaluation.
    """
    if not isinstance(content, bytes) or not content or len(content) > MAX_ARCHIVE_BYTES:
        raise DatasetValidationError("Upload a non-empty ZIP no larger than 50 MiB")
    entries = []
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            members = archive.infolist()
            if len(members) > 11001:
                raise DatasetValidationError("Archive contains too many entries")
            seen, expanded, images, label_entry = set(), 0, [], None
            for member in members:
                if member.orig_filename != member.filename:
                    raise DatasetValidationError("Archive entry paths must not contain null characters")
                parts = _safe_parts(member.filename)
                canonical = "/".join(parts).casefold()
                if canonical in seen:
                    raise DatasetValidationError("Archive contains duplicate entry paths")
                seen.add(canonical)
                if member.flag_bits & 1:
                    raise DatasetValidationError("Encrypted ZIP entries are not supported")
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}):
                    raise DatasetValidationError("Archive links and special files are not supported")
                expanded += member.file_size
                if member.file_size > MAX_ENTRY_BYTES or expanded > MAX_EXPANDED_BYTES:
                    raise DatasetValidationError("Archive exceeds the 10 MiB entry or 200 MiB expanded limit")
                if member.is_dir():
                    continue
                if member.filename == "labels.json":
                    label_entry = member
                elif len(parts) >= 2 and PurePosixPath(member.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    images.append((member, parts[0]))
                else:
                    raise DatasetValidationError("Use class folders containing JPG/PNG images and one root labels.json")
            if label_entry is None:
                raise DatasetValidationError("A root labels.json mapping class folders to model class indices is required")
            if not 2 <= len(images) <= MAX_IMAGES:
                raise DatasetValidationError(f"Dataset must contain between 2 and {MAX_IMAGES} images")
            if label_entry.file_size > MAX_LABEL_BYTES:
                raise DatasetValidationError("labels.json is too large")
            labels = json.loads(_read_entry(archive, label_entry, MAX_LABEL_BYTES).decode("utf-8"), object_pairs_hook=_unique_keys)
            if not isinstance(labels, dict) or not labels or len(labels) > 10000:
                raise DatasetValidationError("labels.json must be a non-empty object of class folders and integer labels")
            label_values = set()
            for folder, label in labels.items():
                if len(_safe_parts(folder)) != 1 or folder.endswith("/"):
                    raise DatasetValidationError("Each label key must be one class folder name")
                if type(label) is not int or not 0 <= label <= 9999:
                    raise DatasetValidationError("Class labels must be integer indices between 0 and 9999; the runner checks the model class count")
                if label in label_values:
                    raise DatasetValidationError("Each class folder must map to a different class index")
                label_values.add(label)
            for member, folder in images:
                if folder not in labels:
                    raise DatasetValidationError(f"Image folder {folder!r} has no entry in labels.json")
                payload = _read_entry(archive, member, MAX_ENTRY_BYTES)
                suffix = PurePosixPath(member.filename).suffix.lower()
                signature_ok = payload.startswith(b"\x89PNG\r\n\x1a\n") if suffix == ".png" else payload.startswith(b"\xff\xd8\xff")
                if not signature_ok:
                    raise DatasetValidationError(f"Image {member.filename!r} does not have a matching JPG/PNG signature")
                entries.append({"path": member.filename, "label": labels[folder], "sha256": hashlib.sha256(payload).hexdigest()})
    except DatasetValidationError:
        raise
    except (zipfile.BadZipFile, UnicodeError, json.JSONDecodeError, RuntimeError, EOFError, zlib.error, NotImplementedError, OSError, RecursionError) as exc:
        raise DatasetValidationError("The ZIP or labels.json is invalid or unsupported") from exc

    dataset_id = "ds_" + uuid4().hex
    destination = Path(data_dir).resolve() / "datasets"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{dataset_id}.zip"
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(content)
    except Exception:
        if created and path.exists():
            path.unlink()
        raise
    return {
        "id": dataset_id, "name": str(name).strip()[:200] or "Image dataset",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(content).hexdigest(), "image_count": len(entries),
        "class_count": len({entry["label"] for entry in entries}), "path": str(path),
        "entries": sorted(entries, key=lambda entry: entry["path"]), "labels": labels,
    }

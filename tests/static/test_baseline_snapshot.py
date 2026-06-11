from hashlib import sha256
from pathlib import Path


ARCHIVE_SHA256 = (
    "f34897e5745c6380dfd819bf87c8a016"
    "aebb8e9ffe7a0025304015fa7b0f0411"
)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def test_verified_archive_is_present() -> None:
    assert digest(Path("VGGT-SLAM-version1.0.zip")) == ARCHIVE_SHA256


def test_modified_baseline_artifacts_are_absent() -> None:
    root = Path("VGGT-SLAM-version1.0")
    for name in ("main_offline.py", "findings.md", "progress.md", "task_plan.md"):
        assert not (root / name).exists()
    assert not (root / "vggt").exists()


def test_temporary_baseline_directory_is_absent() -> None:
    assert not Path("VGGT-SLAM-version1.0 2").exists()

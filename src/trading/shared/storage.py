"""Filesystem layout for data/ and artifacts/ (both gitignored)."""

from pathlib import Path

from trading.shared.config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_DIR = PROJECT_ROOT / "configs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_parquet(df, path: Path) -> None:
    """Write via a temp file + rename so a crash never leaves a half-written partition."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)

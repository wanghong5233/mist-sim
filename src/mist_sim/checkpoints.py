"""Bounded snapshots within one active run; final historical runs are immutable."""
from pathlib import Path


def trim_snapshots(directory, pattern, keep, protected=()):
    directory = Path(directory).resolve()
    protected = {Path(path).resolve() for path in protected}
    candidates = sorted(directory.glob(pattern), key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
    removable = [path for path in candidates if path.resolve() not in protected]
    for path in removable[max(0, keep - len(protected)):]:
        if path.is_symlink() or path.resolve().parent != directory:
            raise ValueError("Snapshot outside checkpoint directory")
        path.unlink()

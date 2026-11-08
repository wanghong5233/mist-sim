"""Check staged files and reachable history before publishing the code repository."""
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


SECRET_PATTERNS = (
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"(?m)^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?i)[a-z]:[\\/]Users[\\/][^\s\"<>]+"),
)


def private_path(name):
    path = PurePosixPath(name)
    parts = [part.lower() for part in path.parts]
    if name in ("data/README.md", "references/README.md"):
        return False
    if parts[0] in {"data", "results", "runs", "archives", "tmp", "logs", "checkpoints", "wandb", "tensorboard"}:
        return True
    if any(part in {".venv", ".vscode", ".idea", "__pycache__"} for part in parts):
        return True
    filename = path.name.lower()
    return (parts[0] == "references" or filename in {"setup.md", "experiment_protocol.md"}
            or (filename.startswith(".env") and filename != ".env.example")
            or "credentials" in filename or "secrets" in filename
            or path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors", ".npz", ".pem", ".key",
                                       ".pdf", ".tex", ".docx", ".zip", ".tar", ".gz", ".log", ".aux", ".out"})


def inspect_repository(root):
    def git(*arguments, input_bytes=None):
        return subprocess.run(["git", "-C", str(root), *arguments], input=input_bytes,
                              capture_output=True, check=True).stdout

    if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != root.resolve():
        raise ValueError("Run this check in the independent mist-sim repository, not its parent")
    problems, inspected = [], set()

    def inspect(name, content):
        if private_path(name):
            problems.append({"file": name, "reason": "private asset path"})
        if len(content) > 1024 * 1024:
            problems.append({"file": name, "reason": "file exceeds 1 MiB; review before publishing"})
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            problems.append({"file": name, "reason": "credential or personal filesystem path pattern"})

    staged = git("ls-files", "--cached", "-z").decode("utf-8").split("\0")
    for name in filter(None, staged):
        inspect(name, git("show", ":" + name))
    objects = git("rev-list", "--objects", "--all").decode("utf-8").splitlines()
    for entry in objects:
        object_id, separator, name = entry.partition(" ")
        if not separator:
            continue
        if git("cat-file", "-t", object_id).strip() != b"blob":
            continue
        if object_id in inspected:
            if private_path(name):
                problems.append({"file": name, "reason": "private asset path in history"})
            continue
        inspected.add(object_id)
        inspect(name, git("cat-file", "blob", object_id))
    return {"staged_files": len(list(filter(None, staged))), "historical_blobs": len(inspected),
            "issues": problems, "passed": not problems}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    report = inspect_repository(Path(__file__).resolve().parents[2])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

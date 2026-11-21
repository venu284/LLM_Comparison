"""S1.7 audit sandbox: run one untrusted response against one test inside Docker.

Model responses and dataset test code are untrusted. They are never executed on the host. Each
cell runs in a fresh container with:

- `--network none`;
- a non-root user (65534, `nobody`);
- `--read-only`, with tmpfs work directories (`/work`, `/tmp`);
- CPU, memory (swap equal to memory), PID and wall-time limits;
- `--cap-drop ALL` and `no-new-privileges`;
- **no mount** of any host path, and no environment passed through. Files reach the container
  only as a JSON payload on stdin.

`check_args` refuses any command line missing one of these, or carrying a mount, volume or env
flag. `run` calls it before every start. Images are referenced by digest only.

The payload is {"files": {name: text}, "cmd": [...], "timeout": seconds}. A small boot program
inside the container writes the files to /work, runs `cmd` there with the timeout, and prints one
JSON line {"returncode", "stdout", "stderr", "timed_out", "elapsed_s"}.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass

USER = "65534:65534"
BOOT = r"""
import json, os, subprocess, sys, time
p = json.loads(sys.stdin.read())
os.makedirs("/work", exist_ok=True)
for name, text in p["files"].items():
    path = os.path.join("/work", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
t0 = time.time()
try:
    r = subprocess.run(p["cmd"], cwd="/work", capture_output=True, text=True, timeout=p["timeout"])
    out = {"returncode": r.returncode, "stdout": r.stdout[-20000:], "stderr": r.stderr[-20000:],
           "timed_out": False}
except subprocess.TimeoutExpired as e:
    out = {"returncode": None, "stdout": (e.stdout or b"")[-20000:].decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")[-20000:],
           "stderr": "", "timed_out": True}
out["elapsed_s"] = round(time.time() - t0, 3)
print("@@RESULT@@" + json.dumps(out))
"""
RESULT_MARK = "@@RESULT@@"
FORBIDDEN = ("-v", "--volume", "--mount", "-e", "--env", "--env-file", "--privileged", "--network=host",
             "--pid", "--ipc", "--userns", "--device", "--cap-add")


@dataclass(frozen=True)
class Spec:
    image: str                  # repo@sha256:... (digest required)
    platform: str = "linux/amd64"
    cpus: float = 1.0
    memory: str = "2g"
    pids: int = 256
    tmpfs_size: str = "512m"
    wall_s: float = 120.0       # host-side limit on the whole container
    env: tuple[tuple[str, str], ...] = (("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
                                       ("HOME", "/work"), ("TMPDIR", "/tmp"),
                                       ("MPLCONFIGDIR", "/tmp"), ("PYTHONDONTWRITEBYTECODE", "1"))


def docker_args(spec: Spec, name: str, python: str = "python3") -> list[str]:
    args = ["docker", "run", "--rm", "-i", "--name", name, "--platform", spec.platform,
            "--network", "none", "--read-only",
            "--tmpfs", f"/work:rw,exec,size={spec.tmpfs_size},uid=65534,gid=65534",
            "--tmpfs", f"/tmp:rw,exec,size={spec.tmpfs_size},uid=65534,gid=65534",
            "--user", USER, "--cpus", str(spec.cpus), "--memory", spec.memory,
            "--memory-swap", spec.memory, "--pids-limit", str(spec.pids),
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--workdir", "/work"]
    # Fixed, non-secret variables only: `env -i` starts from an empty environment inside the
    # container (nothing from the host or the image), then sets these.
    envs = [f"{k}={v}" for k, v in spec.env]
    return args + ["--entrypoint", "env", spec.image, "-i", *envs, python, "-c", BOOT]


def check_args(args: list[str]) -> None:
    """Refuse to start unless every isolation flag is present and nothing exposes the host."""
    def has(flag, value=None):
        for i, a in enumerate(args):
            if a == flag and (value is None or (i + 1 < len(args) and args[i + 1] == value)):
                return True
            if value is not None and a == f"{flag}={value}":
                return True
        return False

    image_at = next((i for i, a in enumerate(args) if "@sha256:" in a), None)
    head = args[:image_at] if image_at is not None else args
    missing = [f for f, v in (("--network", "none"), ("--read-only", None), ("--user", USER),
                              ("--cpus", None), ("--memory", None), ("--memory-swap", None),
                              ("--pids-limit", None), ("--cap-drop", "ALL"),
                              ("--security-opt", "no-new-privileges"), ("--rm", None))
               if not has(f, v)]
    if image_at is None:
        missing.append("image@sha256 digest")
    if not any(a.startswith("/work:") for a in head):
        missing.append("--tmpfs /work")
    bad = [a for a in head if a in FORBIDDEN or a.startswith(("--volume=", "--mount=", "--env=",
                                                              "-v=", "--network=host"))]
    if missing or bad:
        raise PermissionError(f"sandbox refused: missing {missing}, forbidden {bad}")


def run(spec: Spec, payload: dict, python: str = "python3") -> dict:
    name = f"s17-{uuid.uuid4().hex[:12]}"
    args = docker_args(spec, name, python)
    check_args(args)
    t0 = time.time()
    try:
        r = subprocess.run(args, input=json.dumps(payload), capture_output=True, text=True,
                           timeout=spec.wall_s)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        return {"status": "timeout", "host_timeout": True, "elapsed_s": round(time.time() - t0, 3),
                "log": "host wall-time limit"}
    line = next((l for l in r.stdout.splitlines() if l.startswith(RESULT_MARK)), None)
    if line is None:
        return {"status": "error", "host_timeout": False, "elapsed_s": round(time.time() - t0, 3),
                "log": f"no result line; docker exit {r.returncode}\n{r.stderr[-4000:]}"}
    res = json.loads(line[len(RESULT_MARK):])
    res["host_timeout"] = False
    res["container_s"] = round(time.time() - t0, 3)
    return res

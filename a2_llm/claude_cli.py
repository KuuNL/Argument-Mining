"""One A2 model call: Claude Sonnet 5.5 through Claude Code's scripted mode, on the user's Claude Pro login (D052).

    claude.exe -p --model claude-sonnet-5-5 --effort E --safe-mode --tools "" --disallowedTools "mcp__*"
      --permission-prompts none --no-session-persistence --system-prompt-file S --json-schema J --output-format json

The user message goes in on stdin, and the call runs from an empty folder outside the repo.
--safe-mode skips CLAUDE.md, hooks, plugins and memory (--bare would too, but ignores subscription logins);
--tools "" makes it a pure text task; --system-prompt-file replaces Claude Code's own prompt entirely.
The Windows binary is called from WSL through interop (verified: a JSON schema survives the argument passing).
Environment (forwarded to Windows through WSLENV):
- CLAUDE_CODE_DISABLE_TERMINAL_TITLE=1 skips Claude Code's background Haiku request that names the session,
  so every token comes from Sonnet 5.5 (the A2.1 smoke test saw one ~4k-token Haiku call per run without it);
- CLAUDE_CODE_EFFORT_LEVEL pins the effort; it outranks --effort and the user's own effortLevel settings.
"""
import json
import os
import re
import subprocess
from pathlib import Path

CLAUDE = os.environ.get("A2_CLAUDE", "/mnt/c/Users/Kunal/.local/bin/claude.exe")
WORKDIR = Path(os.environ.get("A2_WORKDIR", "/mnt/c/Users/Kunal/AppData/Local/Temp/dialam-a2-cwd"))
MODEL = "claude-sonnet-5-5"
LIMIT = re.compile(r"usage limit|limit reached|hit your .*limit|rate.?limit|resets? (at|in)", re.I)


class UsageLimit(Exception):
    """The subscription's usage window is used up: stop the run (the user restarts it)."""


def winpath(p: Path) -> str:
    return subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, check=True).stdout.strip()


def version() -> str:
    return subprocess.run([CLAUDE, "--version"], capture_output=True, text=True).stdout.strip()


def call(system_file: Path, user: str, schema: dict, effort: str, timeout: int = 900) -> dict:
    """The result JSON plus `_error`: None, or why the call itself failed (no or unusable output). Raises UsageLimit.
    Whether the answer is acceptable (model, schema) is judged separately, from the stored result (runner.evaluate)."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    cmd = [CLAUDE, "-p", "--model", MODEL, "--effort", effort, "--safe-mode", "--tools", "",
           "--disallowedTools", "mcp__*", "--permission-prompts", "none", "--no-session-persistence",
           "--system-prompt-file", winpath(system_file), "--json-schema", json.dumps(schema, separators=(",", ":")),
           "--output-format", "json"]
    env = {**os.environ, "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1", "CLAUDE_CODE_EFFORT_LEVEL": effort}
    env["WSLENV"] = ":".join(x for x in (os.environ.get("WSLENV"), "CLAUDE_CODE_DISABLE_TERMINAL_TITLE/w",
                                        "CLAUDE_CODE_EFFORT_LEVEL/w") if x)        # /w: WSL -> Windows only
    try:
        proc = subprocess.run(cmd, input=user.encode("utf-8"), capture_output=True, cwd=WORKDIR, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"_error": f"timeout after {timeout} s"}
    out = proc.stdout.decode("utf-8", "replace").strip()
    try:
        res = json.loads(out.splitlines()[-1]) if out else {}
    except json.JSONDecodeError:
        res = {}
    if not res:
        err = proc.stderr.decode("utf-8", "replace")[-600:]
        if LIMIT.search(err + out):
            raise UsageLimit((err + out).strip()[-300:])
        return {"_error": f"exit {proc.returncode}: {err} {out[-300:]}".strip()}
    if res.get("is_error"):
        text = f"{res.get('subtype', '')}: {res.get('result', '')}"
        if LIMIT.search(text):
            raise UsageLimit(text.strip()[:300])
        return {**res, "_error": text[:500]}
    return {**res, "_error": None}


# Sonnet 5.5 list prices per million tokens (API-equivalent cost; Claude Code reports its own figure
# with costBasis "unknown" for this model). Claude Code writes the prompt cache with the 1-hour TTL.
PRICE = {"input": 2.0, "cache_5m": 2.5, "cache_1h": 4.0, "cache_read": 0.2, "output": 10.0}


def api_usd(res: dict) -> float:
    u = res.get("usage") or {}
    c = u.get("cache_creation") or {}
    return round((u.get("input_tokens", 0) * PRICE["input"] + c.get("ephemeral_5m_input_tokens", 0) * PRICE["cache_5m"]
                  + c.get("ephemeral_1h_input_tokens", 0) * PRICE["cache_1h"]
                  + u.get("cache_read_input_tokens", 0) * PRICE["cache_read"] + u.get("output_tokens", 0) * PRICE["output"]) / 1e6, 6)

"""Vercel AI Gateway client for Jev (D073, D075).

POST /v1/evaluate {"model", "state", "questions"} -> {"model", "answers", "usage", "providerMetadata"}; GET /v1/credits.
The key is AI_GATEWAY_API_KEY (WSL ~/.profile, never in the repo). Rate limits and overloads (429, 529, 5xx) and
dropped connections are retried with exponential backoff (honouring Retry-After). Any other status raises Refused,
which stops a run: 402 / 403 mean no credit or no access to the model, and 4xx a request we built wrongly.
"""
import os
import time

import httpx

BASE = "https://ai-gateway.vercel.sh"
MODEL = "typesafe-ai/jev"
RETRY = {429, 500, 502, 503, 504, 529}


class Refused(Exception):
    pass


def _headers() -> dict:
    key = os.environ.get("AI_GATEWAY_API_KEY")
    if not key:
        raise Refused("AI_GATEWAY_API_KEY is not set (it lives in WSL ~/.profile; run under `bash -lc`)")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def evaluate(state, questions: dict, model: str = MODEL, attempts: int = 6, timeout: float = 300.0) -> dict:
    """{"status", "seconds" (the successful attempt), "attempts", "response"}."""
    body = {"model": model, "state": state, "questions": questions}
    error = None
    for i in range(attempts):
        t0 = time.time()
        wait = min(60.0, 2.0 ** i)
        try:
            r = httpx.post(f"{BASE}/v1/evaluate", json=body, headers=_headers(), timeout=timeout)
        except httpx.TransportError as e:
            error = f"transport: {e!r}"
        else:
            if r.status_code == 200:
                return {"status": 200, "seconds": round(time.time() - t0, 3), "attempts": i + 1, "response": r.json()}
            if r.status_code not in RETRY:
                raise Refused(f"HTTP {r.status_code}: {r.text[:800]}")
            error = f"HTTP {r.status_code}: {r.text[:200]}"
            try:
                wait = max(wait, float(r.headers.get("retry-after", 0)))
            except ValueError:
                pass
        time.sleep(wait)
    raise RuntimeError(f"gave up after {attempts} attempts: {error}")


def credits() -> dict:
    """{"balance", "total_used"} in dollars, as strings."""
    r = httpx.get(f"{BASE}/v1/credits", headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def cost(response: dict) -> float:
    """The gateway's charge for one answered request, in dollars."""
    return float(((response.get("providerMetadata") or {}).get("gateway") or {}).get("cost") or 0)


def input_tokens(response: dict) -> int:
    u = response.get("usage") or {}
    return int(u.get("inputTokens", u.get("input_tokens", 0)) or 0)
